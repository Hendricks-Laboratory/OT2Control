'''Pure execution-plan contract for one supervised well-mix calibration.

This module has no controller, reader, socket, robot, spreadsheet-I/O, or
filesystem imports. It turns already-normalized calibration settings plus
injected run-specific identities into one strictly validated, reviewable plan.
A later controller stage may execute this plan, but no function here can cause
a physical action.
'''

import re

from auto_mix_calibration import (
    MIX_CALIBRATION_ASSAY_PROFILE_HEADER,
    MIX_CALIBRATION_CYCLE_COUNT,
    MIX_CALIBRATION_FINAL_WELL_VOLUME_UL,
    MIX_CALIBRATION_MODE_OFF,
    MIX_CALIBRATION_P300_VOLUME_UL,
    MIX_CALIBRATION_TARGET_READER_WELL_HEADER,
    MIX_CALIBRATION_WELL_CLEARANCE_MM,
    get_auto_mix_calibration_profile_mode,
    get_auto_mix_calibration_profile_transfer_volumes,
)
from auto_mix_calibration_protocol import (
    MIX_CALIBRATION_SCHEMA_VERSION,
    validate_mix_calibration_preflight_request,
    validate_mix_calibration_request,
)


MIX_CALIBRATION_PLAN_VERSION = 4
MIX_CALIBRATION_BASELINE_SCAN_INDEX = 0
MIX_CALIBRATION_EXPECTED_SCAN_COUNT = 1 + MIX_CALIBRATION_CYCLE_COUNT
_IDENTIFIER_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')


class AutoMixCalibrationPlanError(ValueError):
    '''Raised when a calibration-only run plan is incomplete or ambiguous.'''


def _require_identifier(value, field_name):
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.match(value):
        raise AutoMixCalibrationPlanError(
            '{} must contain only letters, digits, dots, hyphens, or '
            'underscores and begin with a letter or digit.'.format(field_name)
        )
    return value


def _require_nonnegative_integer(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutoMixCalibrationPlanError(
            '{} must be a nonnegative integer.'.format(field_name)
        )
    return value


def _require_container_name(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise AutoMixCalibrationPlanError(
            '{} must be a nonblank canonical source-container name.'
            .format(field_name)
        )
    return value.strip()


def _require_logical_wellname(value):
    if not isinstance(value, str) or not value.strip().startswith('autowell'):
        raise AutoMixCalibrationPlanError(
            'logical_wellname must be one calibration-only Auto product well.'
        )
    return value.strip()


def resolve_auto_mix_calibration_target(reader_well, reader_location_mapping):
    '''Resolve one user-entered reader well into one guarded Pi identity.'''
    if not isinstance(reader_well, str) or not reader_well.strip():
        raise AutoMixCalibrationPlanError('Calibration reader well is blank.')
    normalized_reader_well = reader_well.strip().upper()
    try:
        mapped = reader_location_mapping[normalized_reader_well]
    except (KeyError, TypeError, AttributeError):
        raise AutoMixCalibrationPlanError(
            'Calibration reader well {} is not present in the established '
            'plate-reader mapping.'.format(normalized_reader_well)
        )
    if not isinstance(mapped, tuple) or len(mapped) != 2:
        raise AutoMixCalibrationPlanError(
            'Calibration reader mapping for {} is malformed.'
            .format(normalized_reader_well)
        )
    robot_well, labware_name = mapped
    if (not isinstance(robot_well, str) or not robot_well.strip()
            or not isinstance(labware_name, str)):
        raise AutoMixCalibrationPlanError(
            'Calibration reader mapping for {} is incomplete.'
            .format(normalized_reader_well)
        )
    normalized_labware = labware_name.strip().lower()
    if not normalized_labware.startswith('platereader'):
        raise AutoMixCalibrationPlanError(
            'Calibration reader well {} does not resolve to a plate-reader '
            'labware position.'.format(normalized_reader_well)
        )
    deck_text = normalized_labware[len('platereader'):]
    try:
        deck_pos = int(deck_text)
    except (TypeError, ValueError):
        raise AutoMixCalibrationPlanError(
            'Calibration reader mapping for {} has an invalid deck position.'
            .format(normalized_reader_well)
        )
    if deck_pos not in (4, 7):
        raise AutoMixCalibrationPlanError(
            'Calibration reader well {} resolves to unsupported deck position '
            '{}.'.format(normalized_reader_well, deck_pos)
        )
    return {
        'reader_well': normalized_reader_well,
        'expected_loc': robot_well.strip().upper(),
        'expected_deck_pos': deck_pos,
        'expected_labware_name': 'platereader{}'.format(deck_pos),
    }


def resolve_auto_mix_calibration_deck_target(
        deck_reader_well, legacy_header_reader_well, reader_location_mapping):
    '''Resolve the Deck Positions target and reject stale duplicate Headers.

    ``deck_reader_well`` is the existing controller-preserved, reader-facing
    first usable plate well. It is the only routing authority. Older
    workbooks may contain the former duplicate Header field; it is accepted
    only when it resolves to the same reader-facing well and is never used to
    select a different physical target.
    '''
    target = resolve_auto_mix_calibration_target(
        deck_reader_well, reader_location_mapping
    )
    legacy_reader_well = None
    if legacy_header_reader_well is not None:
        legacy_target = resolve_auto_mix_calibration_target(
            legacy_header_reader_well, reader_location_mapping
        )
        legacy_reader_well = legacy_target['reader_well']
        if legacy_reader_well != target['reader_well']:
            raise AutoMixCalibrationPlanError(
                'Legacy calibration Header well {} conflicts with Deck '
                'Positions first usable reader well {}. Remove the legacy '
                'Header row or make it match before calibration preflight.'
                .format(legacy_reader_well, target['reader_well'])
            )
    target['target_source'] = 'deck_positions_first_usable'
    target['legacy_header_reader_well'] = legacy_reader_well
    return target


def _build_scan_step(scan_index, cycle_index, wellname, target, prerequisite):
    if scan_index == MIX_CALIBRATION_BASELINE_SCAN_INDEX:
        reason = 'baseline_after_second_transfer_before_mix'
    else:
        reason = 'post_mix_cycle_{:02d}'.format(cycle_index)
    return {
        'step_id': 'scan-{:02d}'.format(scan_index),
        'step_type': 'selected_well_reader_scan',
        'scan_index': scan_index,
        'cycle_index': cycle_index,
        'observation_reason': reason,
        'wellname': wellname,
        'reader_well': target['reader_well'],
        'expected_deck_pos': target['expected_deck_pos'],
        'expected_loc': target['expected_loc'],
        'requires_step_id': prerequisite,
        'record_in_aggregate': False,
    }


def build_auto_mix_calibration_run_plan(
        settings, calibration_id, logical_wellname, source_container_names,
        reader_location_mapping, plate_mapping_revision, plate_generation,
        deck_reader_well):
    '''Build one profile-locked transfer, baseline, then mix/scan plan.

    ``source_container_names`` deliberately uses first/second keys. The current
    already-deployed Pi preflight packet still calls its two identity-only
    fields ``base_chemical_name`` and ``trigger_chemical_name``; this pure plan
    maps first->base and second->trigger only at that wire-compatibility edge.
    Transfer order is governed solely by this plan's explicit first/second
    steps, not by those historical packet field names.
    '''
    if not isinstance(settings, dict):
        raise AutoMixCalibrationPlanError('Calibration settings must be a dict.')
    mode = settings.get('auto_mix_calibration_mode')
    if mode in (None, MIX_CALIBRATION_MODE_OFF):
        raise AutoMixCalibrationPlanError(
            'A calibration run plan requires enabled calibration mode.'
        )
    profile = settings.get(MIX_CALIBRATION_ASSAY_PROFILE_HEADER)
    if not isinstance(profile, str) or not profile:
        raise AutoMixCalibrationPlanError('Calibration assay profile is blank.')
    try:
        first_volume, second_volume = (
            get_auto_mix_calibration_profile_transfer_volumes(profile)
        )
        expected_mode = get_auto_mix_calibration_profile_mode(profile)
    except ValueError as error:
        raise AutoMixCalibrationPlanError(str(error))
    if mode != expected_mode:
        raise AutoMixCalibrationPlanError(
            'Calibration mode does not permit the selected assay profile.'
        )
    if (settings.get('auto_mix_calibration_first_volume_uL') != first_volume
            or settings.get('auto_mix_calibration_second_volume_uL')
            != second_volume):
        raise AutoMixCalibrationPlanError(
            'Calibration settings do not match the reviewed assay profile.'
        )
    calibration_id = _require_identifier(calibration_id, 'calibration_id')
    wellname = _require_logical_wellname(logical_wellname)
    if not isinstance(source_container_names, dict):
        raise AutoMixCalibrationPlanError(
            'source_container_names must contain first and second identities.'
        )
    if set(source_container_names) != set(('first', 'second')):
        raise AutoMixCalibrationPlanError(
            'source_container_names must contain exactly first and second.'
        )
    first_container = _require_container_name(
        source_container_names['first'], 'first source container'
    )
    second_container = _require_container_name(
        source_container_names['second'], 'second source container'
    )
    if first_container.casefold() == second_container.casefold():
        raise AutoMixCalibrationPlanError(
            'The resolved first and second source containers must differ.'
        )
    target = resolve_auto_mix_calibration_deck_target(
        deck_reader_well,
        settings.get(MIX_CALIBRATION_TARGET_READER_WELL_HEADER),
        reader_location_mapping
    )
    plate_mapping_revision = _require_nonnegative_integer(
        plate_mapping_revision, 'plate_mapping_revision'
    )
    plate_generation = _require_nonnegative_integer(
        plate_generation, 'plate_generation'
    )
    preflight_id = '{}-preflight'.format(calibration_id)
    common_request = {
        'schema_version': MIX_CALIBRATION_SCHEMA_VERSION,
        'calibration_id': calibration_id,
        'calibration_mode': mode,
        'wellname': wellname,
        'expected_deck_pos': target['expected_deck_pos'],
        'expected_loc': target['expected_loc'],
        'expected_plate_mapping_revision': plate_mapping_revision,
        'expected_plate_generation': plate_generation,
        # Compatibility names are identity-only Pi preflight fields; they do
        # not select transfer ordering or volumes.
        'base_chemical_name': first_container,
        'trigger_chemical_name': second_container,
        'expected_final_well_volume_uL': MIX_CALIBRATION_FINAL_WELL_VOLUME_UL,
        'mix_volume_uL': MIX_CALIBRATION_P300_VOLUME_UL,
        'total_cycle_count': MIX_CALIBRATION_CYCLE_COUNT,
        'well_clearance_mm': MIX_CALIBRATION_WELL_CLEARANCE_MM,
    }
    preflight_request = validate_mix_calibration_preflight_request(
        dict(common_request, preflight_id=preflight_id)
    )
    transfer_steps = [
        {
            'step_id': 'transfer-first',
            'step_type': 'ordinary_transfer',
            'sequence_index': 1,
            'source_chemical_name': first_container,
            'destination_wellname': wellname,
            'volume_uL': first_volume,
            'requires_step_id': 'pi-preflight',
        },
        {
            'step_id': 'transfer-second',
            'step_type': 'ordinary_transfer',
            'sequence_index': 2,
            'source_chemical_name': second_container,
            'destination_wellname': wellname,
            'volume_uL': second_volume,
            'requires_step_id': 'transfer-first',
        },
    ]
    baseline_scan = _build_scan_step(
        MIX_CALIBRATION_BASELINE_SCAN_INDEX, 0, wellname, target,
        'transfer-second'
    )
    mix_steps = []
    scan_steps = [baseline_scan]
    for cycle_index in range(1, MIX_CALIBRATION_CYCLE_COUNT + 1):
        action_id = '{}-mix-{:02d}'.format(calibration_id, cycle_index)
        mix_request = validate_mix_calibration_request(dict(
            common_request,
            action_id=action_id,
            preflight_id=preflight_id,
            cycle_index=cycle_index
        ))
        mix_step_id = 'mix-{:02d}'.format(cycle_index)
        mix_steps.append({
            'step_id': mix_step_id,
            'step_type': 'pi_calibration_mix',
            'cycle_index': cycle_index,
            'request': mix_request,
            'requires_step_id': (
                baseline_scan['step_id'] if cycle_index == 1 else
                scan_steps[-1]['step_id']
            ),
            'automatic_retry_permitted': False,
        })
        scan_steps.append(_build_scan_step(
            cycle_index, cycle_index, wellname, target, mix_step_id
        ))
    execution_steps = [{
        'step_id': 'pi-preflight',
        'step_type': 'pi_calibration_preflight',
        'request': preflight_request,
        'automatic_retry_permitted': False,
    }]
    execution_steps.extend(transfer_steps)
    execution_steps.append(baseline_scan)
    for mix_step, scan_step in zip(mix_steps, scan_steps[1:]):
        execution_steps.extend((mix_step, scan_step))
    plan = {
        'plan_version': MIX_CALIBRATION_PLAN_VERSION,
        'calibration_id': calibration_id,
        'calibration_mode': mode,
        'assay_profile': profile,
        'normal_auto_execution_permitted': False,
        'normal_stability_execution_permitted': False,
        'target': target,
        'logical_wellname': wellname,
        'plate_mapping_revision': plate_mapping_revision,
        'plate_generation': plate_generation,
        'preflight_request': preflight_request,
        'transfer_steps': transfer_steps,
        'mix_steps': mix_steps,
        'scan_steps': scan_steps,
        'execution_steps': execution_steps,
        'expected_final_well_volume_uL': MIX_CALIBRATION_FINAL_WELL_VOLUME_UL,
        'expected_scan_count': MIX_CALIBRATION_EXPECTED_SCAN_COUNT,
    }
    validate_auto_mix_calibration_run_plan(plan)
    return plan


def validate_auto_mix_calibration_run_plan(plan):
    '''Reject altered plans before a later controller can dispatch one action.'''
    if not isinstance(plan, dict):
        raise AutoMixCalibrationPlanError('Calibration run plan must be a dict.')
    if plan.get('plan_version') != MIX_CALIBRATION_PLAN_VERSION:
        raise AutoMixCalibrationPlanError('Calibration plan version is invalid.')
    _require_identifier(plan.get('calibration_id'), 'plan.calibration_id')
    if (plan.get('normal_auto_execution_permitted') is not False
            or plan.get('normal_stability_execution_permitted') is not False):
        raise AutoMixCalibrationPlanError(
            'Calibration plan must remain isolated from Auto and stability.'
        )
    if not isinstance(plan.get('assay_profile'), str) or not plan['assay_profile']:
        raise AutoMixCalibrationPlanError('Calibration plan assay profile is invalid.')
    try:
        first_volume, second_volume = (
            get_auto_mix_calibration_profile_transfer_volumes(
                plan['assay_profile']
            )
        )
        expected_mode = get_auto_mix_calibration_profile_mode(
            plan['assay_profile']
        )
    except ValueError as error:
        raise AutoMixCalibrationPlanError(str(error))
    preflight = validate_mix_calibration_preflight_request(
        plan.get('preflight_request')
    )
    if plan.get('calibration_mode') != expected_mode:
        raise AutoMixCalibrationPlanError(
            'Calibration plan mode does not permit its assay profile.'
        )
    target = plan.get('target')
    if not isinstance(target, dict):
        raise AutoMixCalibrationPlanError('Calibration plan target is invalid.')
    if (target.get('expected_deck_pos') != preflight['expected_deck_pos']
            or target.get('expected_loc') != preflight['expected_loc']
            or target.get('expected_labware_name')
            != 'platereader{}'.format(preflight['expected_deck_pos'])
            or target.get('target_source') != 'deck_positions_first_usable'
            or (target.get('legacy_header_reader_well') is not None
                and (not isinstance(
                    target.get('legacy_header_reader_well'), str)
                     or target['legacy_header_reader_well']
                     != target.get('reader_well')))
            or not isinstance(target.get('reader_well'), str)
            or not target['reader_well'].strip()
            or plan.get('logical_wellname') != preflight['wellname']
            or plan.get('calibration_mode') != preflight['calibration_mode']
            or plan.get('plate_mapping_revision')
            != preflight['expected_plate_mapping_revision']
            or plan.get('plate_generation')
            != preflight['expected_plate_generation']):
        raise AutoMixCalibrationPlanError(
            'Calibration plan target identity does not match its Pi preflight.'
        )
    transfer_steps = plan.get('transfer_steps')
    expected_transfers = (
        (preflight['base_chemical_name'], first_volume),
        (preflight['trigger_chemical_name'], second_volume),
    )
    if not isinstance(transfer_steps, list) or len(transfer_steps) != 2:
        raise AutoMixCalibrationPlanError(
            'Calibration plan must contain exactly first then second transfer.'
        )
    for sequence_index, (step, expected) in enumerate(
            zip(transfer_steps, expected_transfers), start=1):
        expected_step_id = ('transfer-first' if sequence_index == 1
                            else 'transfer-second')
        expected_prerequisite = ('pi-preflight' if sequence_index == 1
                                 else 'transfer-first')
        if (not isinstance(step, dict)
                or step.get('step_id') != expected_step_id
                or step.get('step_type') != 'ordinary_transfer'
                or step.get('sequence_index') != sequence_index
                or step.get('source_chemical_name') != expected[0]
                or step.get('destination_wellname') != preflight['wellname']
                or step.get('volume_uL') != expected[1]
                or step.get('requires_step_id') != expected_prerequisite):
            raise AutoMixCalibrationPlanError(
                'Calibration transfer sequence is not the reviewed '
                'profile-locked first/second sequence.'
            )
    if (plan.get('expected_final_well_volume_uL')
            != MIX_CALIBRATION_FINAL_WELL_VOLUME_UL):
        raise AutoMixCalibrationPlanError(
            'Calibration plan final well volume is invalid.'
        )
    mix_steps = plan.get('mix_steps')
    scan_steps = plan.get('scan_steps')
    if (not isinstance(mix_steps, list)
            or len(mix_steps) != MIX_CALIBRATION_CYCLE_COUNT
            or not isinstance(scan_steps, list)
            or len(scan_steps) != MIX_CALIBRATION_EXPECTED_SCAN_COUNT):
        raise AutoMixCalibrationPlanError(
            'Calibration plan must contain one baseline and ten mix/scan cycles.'
        )
    baseline = scan_steps[0]
    if (baseline.get('scan_index') != MIX_CALIBRATION_BASELINE_SCAN_INDEX
            or baseline.get('cycle_index') != 0
            or baseline.get('observation_reason')
            != 'baseline_after_second_transfer_before_mix'
            or baseline.get('requires_step_id') != 'transfer-second'
            or baseline.get('wellname') != preflight['wellname']
            or baseline.get('reader_well') != target['reader_well']
            or baseline.get('expected_deck_pos')
            != preflight['expected_deck_pos']
            or baseline.get('expected_loc') != preflight['expected_loc']
            or baseline.get('record_in_aggregate') is not False):
        raise AutoMixCalibrationPlanError(
            'Calibration baseline scan is not positioned after second transfer.'
        )
    seen_action_ids = set()
    for cycle_index, (mix_step, scan_step) in enumerate(
            zip(mix_steps, scan_steps[1:]), start=1):
        request = validate_mix_calibration_request(mix_step.get('request'))
        if (request['calibration_id'] != preflight['calibration_id']
                or request['preflight_id'] != preflight['preflight_id']
                or request['cycle_index'] != cycle_index
                or request['action_id'] in seen_action_ids
                or mix_step.get('step_id') != 'mix-{:02d}'.format(cycle_index)
                or mix_step.get('step_type') != 'pi_calibration_mix'
                or mix_step.get('cycle_index') != cycle_index
                or mix_step.get('requires_step_id') != (
                    'scan-00' if cycle_index == 1
                    else 'scan-{:02d}'.format(cycle_index - 1)
                )
                or mix_step.get('automatic_retry_permitted') is not False):
            raise AutoMixCalibrationPlanError(
                'Calibration mix actions must be unique, ordered, and '
                'non-retryable.'
            )
        seen_action_ids.add(request['action_id'])
        if (scan_step.get('scan_index') != cycle_index
                or scan_step.get('cycle_index') != cycle_index
                or scan_step.get('observation_reason')
                != 'post_mix_cycle_{:02d}'.format(cycle_index)
                or scan_step.get('requires_step_id') != mix_step.get('step_id')
                or scan_step.get('wellname') != preflight['wellname']
                or scan_step.get('reader_well') != target['reader_well']
                or scan_step.get('expected_deck_pos')
                != preflight['expected_deck_pos']
                or scan_step.get('expected_loc') != preflight['expected_loc']
                or scan_step.get('record_in_aggregate') is not False):
            raise AutoMixCalibrationPlanError(
                'Each calibration mix must have exactly one subsequent raw '
                'selected-well scan.'
            )
    if plan.get('expected_scan_count') != MIX_CALIBRATION_EXPECTED_SCAN_COUNT:
        raise AutoMixCalibrationPlanError('Calibration scan count is invalid.')
    expected_execution_steps = [{
        'step_id': 'pi-preflight',
        'step_type': 'pi_calibration_preflight',
        'request': preflight,
        'automatic_retry_permitted': False,
    }]
    expected_execution_steps.extend(transfer_steps)
    expected_execution_steps.append(baseline)
    for mix_step, scan_step in zip(mix_steps, scan_steps[1:]):
        expected_execution_steps.extend((mix_step, scan_step))
    if plan.get('execution_steps') != expected_execution_steps:
        raise AutoMixCalibrationPlanError(
            'Calibration execution steps do not match the validated sequence.'
        )
    return plan
