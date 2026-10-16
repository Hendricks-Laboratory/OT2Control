'''Hardware-free execution boundary for the supervised mix calibration.

This module owns the *ordering* and fail-closed checks for the calibration
assay.  It deliberately does not import the controller, reader, Armchair, or
OT-2 code.  A later narrow controller adapter supplies those operations.  That
separation permits comprehensive synthetic tests of the no-retry sequence
without creating a route that can accidentally enter normal Auto execution.
'''

import copy

from auto_mix_calibration import MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM
from auto_mix_calibration_plan import validate_auto_mix_calibration_run_plan
from auto_mix_calibration_protocol import (
    validate_mix_calibration_acknowledgement,
    validate_mix_calibration_preflight_acknowledgement,
)
from auto_mix_calibration_resources import (
    build_auto_mix_calibration_resource_preflight_request,
    build_auto_mix_calibration_well_binding_request,
    validate_auto_mix_calibration_resource_preflight_acknowledgement,
    validate_auto_mix_calibration_well_binding_acknowledgement,
)


class AutoMixCalibrationExecutionError(RuntimeError):
    '''Raised before a partial calibration sequence can continue safely.'''


def _require_exact_wavelength_coverage(wavelengths):
    if not isinstance(wavelengths, (list, tuple, set)):
        raise AutoMixCalibrationExecutionError(
            'Calibration reader preflight must report supported wavelengths.'
        )
    numeric_wavelengths = set()
    for value in wavelengths:
        if isinstance(value, bool):
            raise AutoMixCalibrationExecutionError(
                'Calibration reader preflight has an invalid wavelength.'
            )
        try:
            numeric_wavelengths.add(int(value))
        except (TypeError, ValueError):
            raise AutoMixCalibrationExecutionError(
                'Calibration reader preflight has an invalid wavelength.'
            )
    missing = set(MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM).difference(
        numeric_wavelengths
    )
    if missing:
        raise AutoMixCalibrationExecutionError(
            'Calibration reader preflight lacks required wavelength(s): {}.'
            .format(', '.join(str(value) for value in sorted(missing)))
        )


def validate_auto_mix_calibration_resource_preflight(plan, result):
    '''Validate the all-resources barrier before the first liquid transfer.

    The injected controller adapter must assemble this report from its local
    source/reader view and a Pi-side **atomic** source-and-tip preflight.  It
    is intentionally stricter than the already-deployed calibration-mix
    packet, whose acknowledgement covers only the dedicated P300 mix tips and
    target plate identity.  A missing report is a hard stop, never an implied
    approval.
    '''
    plan = validate_auto_mix_calibration_run_plan(plan)
    required_keys = {
        'approved', 'message', 'source_requirements',
        'ordinary_transfer_tip_requirements',
        'dedicated_mix_tip_requirements', 'reader_wavelengths_nm',
        'plate_mapping_revision', 'plate_generation'
    }
    if not isinstance(result, dict) or set(result) != required_keys:
        raise AutoMixCalibrationExecutionError(
            'Calibration resource preflight returned an invalid schema.'
        )
    if not isinstance(result['approved'], bool):
        raise AutoMixCalibrationExecutionError(
            'Calibration resource preflight returned a non-boolean decision.'
        )
    if not isinstance(result['message'], str) or not result['message'].strip():
        raise AutoMixCalibrationExecutionError(
            'Calibration resource preflight returned no explanation.'
        )
    if not result['approved']:
        raise AutoMixCalibrationExecutionError(
            'Calibration resource preflight rejected the assay before liquid '
            'handling: {}.'.format(result['message'].strip())
        )
    if (result['plate_mapping_revision'] != plan['plate_mapping_revision']
            or result['plate_generation'] != plan['plate_generation']):
        raise AutoMixCalibrationExecutionError(
            'Calibration resource preflight plate identity is stale.'
        )
    expected_sources = list(zip(
        ('first', 'second'), plan['transfer_steps']
    ))
    if (not isinstance(result['source_requirements'], list)
            or len(result['source_requirements']) != len(expected_sources)):
        raise AutoMixCalibrationExecutionError(
            'Calibration resource preflight source allocation is incomplete.'
        )
    for report, (slot, step) in zip(
            result['source_requirements'], expected_sources):
        if not isinstance(report, dict) or set(report) != set((
                'slot', 'source_container', 'required_volume_uL',
                'available_volume_uL')):
            raise AutoMixCalibrationExecutionError(
                'Calibration source preflight has an invalid source record.'
            )
        if (report['slot'] != slot
                or report['source_container']
                != step['source_chemical_name']
                or report['required_volume_uL'] != step['volume_uL']):
            raise AutoMixCalibrationExecutionError(
                'Calibration resource preflight source allocation does not '
                'match the reviewed first/second transfer plan.'
            )
        try:
            available_volume = float(report['available_volume_uL'])
        except (TypeError, ValueError):
            raise AutoMixCalibrationExecutionError(
                'Calibration source preflight has an invalid available volume.'
            )
        if available_volume < step['volume_uL']:
            raise AutoMixCalibrationExecutionError(
                'Calibration source preflight approved an insufficient source.'
            )
    # The 5/20 uL first addition must be performed by the normal small-pipette
    # transfer pathway; the 180/195 uL second addition uses the normal P300
    # pathway.  The eleven dedicated P300 mixing tips remain separate so no
    # transfer tip is silently reclassified as a calibration-mixing tip.
    _validate_tip_requirements(
        result['ordinary_transfer_tip_requirements'],
        (('p20', 1), ('p300', 1)), 'ordinary-transfer'
    )
    _validate_tip_requirements(
        result['dedicated_mix_tip_requirements'],
        (('p300', 11),), 'dedicated-mix'
    )
    _require_exact_wavelength_coverage(result['reader_wavelengths_nm'])
    return copy.deepcopy(result)


def _validate_tip_requirements(requirements, expected, label):
    if not isinstance(requirements, list) or len(requirements) != len(expected):
        raise AutoMixCalibrationExecutionError(
            'Calibration {} tip preflight is incomplete.'.format(label)
        )
    for report, (pipette, required) in zip(requirements, expected):
        if not isinstance(report, dict) or set(report) != set((
                'pipette', 'required_new_tips', 'available_new_tips')):
            raise AutoMixCalibrationExecutionError(
                'Calibration {} tip preflight has an invalid record.'
                .format(label)
            )
        if (report['pipette'] != pipette
                or report['required_new_tips'] != required):
            raise AutoMixCalibrationExecutionError(
                'Calibration {} tip preflight does not match the reviewed '
                'assay requirements.'.format(label)
            )
        if (isinstance(report['available_new_tips'], bool)
                or not isinstance(report['available_new_tips'], int)
                or report['available_new_tips'] < required):
            raise AutoMixCalibrationExecutionError(
                'Calibration {} tip preflight approved insufficient tips.'
                .format(label)
            )


def execute_auto_mix_calibration_plan(plan, adapter):
    '''Execute one injected, already-reviewed calibration plan exactly once.

    ``adapter`` is intentionally a minimal capability interface:

    * ``preflight_resources(plan)`` -> strict all-resources report;
    * ``request_pi_preflight(request)`` -> Pi no-motion acknowledgement;
    * ``transfer(source, destination, volume_uL)`` -> ordinary transfer;
    * ``scan(scan_step)`` -> selected-well raw scan token/record;
    * ``request_pi_mix(request)`` -> one Pi mix acknowledgement.

    There are no automatic retries.  Any exception or rejected acknowledgement
    stops immediately and returns no permission to replay a transfer or mix.
    This function does not know how to open a socket, drive a reader, or move
    hardware; production wiring remains a later, separately audited adapter.
    '''
    plan = validate_auto_mix_calibration_run_plan(plan)
    required_methods = (
        'preflight_resources', 'request_pi_preflight', 'transfer', 'scan',
        'request_pi_mix'
    )
    for method_name in required_methods:
        if not callable(getattr(adapter, method_name, None)):
            raise AutoMixCalibrationExecutionError(
                'Calibration adapter lacks {}().'.format(method_name)
            )

    resource_preflight = validate_auto_mix_calibration_resource_preflight(
        plan, adapter.preflight_resources(copy.deepcopy(plan))
    )
    preflight_ack = validate_mix_calibration_preflight_acknowledgement(
        adapter.request_pi_preflight(copy.deepcopy(plan['preflight_request'])),
        plan['preflight_request']
    )
    if not preflight_ack['accepted']:
        raise AutoMixCalibrationExecutionError(
            'Pi rejected calibration before liquid handling: {}.'
            .format(preflight_ack['message'])
        )

    execution_record = {
        'calibration_id': plan['calibration_id'],
        'logical_wellname': plan['logical_wellname'],
        'resource_preflight': resource_preflight,
        'pi_preflight': preflight_ack,
        'completed_step_ids': ['pi-preflight'],
        'scan_records': [],
        'mix_acknowledgements': [],
    }
    for transfer_step in plan['transfer_steps']:
        adapter.transfer(
            transfer_step['source_chemical_name'],
            transfer_step['destination_wellname'],
            transfer_step['volume_uL']
        )
        execution_record['completed_step_ids'].append(
            transfer_step['step_id']
        )

    for scan_step in plan['scan_steps']:
        if scan_step['scan_index']:
            mix_step = plan['mix_steps'][scan_step['cycle_index'] - 1]
            acknowledgement = validate_mix_calibration_acknowledgement(
                adapter.request_pi_mix(copy.deepcopy(mix_step['request'])),
                mix_step['request']
            )
            if not acknowledgement['accepted']:
                raise AutoMixCalibrationExecutionError(
                    'Pi rejected calibration mix cycle {} before physical '
                    'execution: {}.'.format(
                        scan_step['cycle_index'], acknowledgement['message']
                    )
                )
            execution_record['mix_acknowledgements'].append(acknowledgement)
            execution_record['completed_step_ids'].append(mix_step['step_id'])
        scan_record = adapter.scan(copy.deepcopy(scan_step))
        execution_record['scan_records'].append(copy.deepcopy(scan_record))
        execution_record['completed_step_ids'].append(scan_step['step_id'])
    return execution_record


def _validate_completed_controller_preflight(
        plan, preflight_record, expected_reader_scan_protocol=None):
    '''Validate C2eA evidence before the C2eB liquid sequence starts.

    C2eA binds a previously vacant reader well. Replaying its v1/v2 packets
    immediately before transfer would therefore be unsafe: the second v1
    request correctly sees that the well is no longer vacant. C2eB consumes
    the accepted record instead, verifies every immutable identity field, and
    then permits the reviewed transfer/scan/mix sequence exactly once.
    '''
    plan = validate_auto_mix_calibration_run_plan(plan)
    if not isinstance(preflight_record, dict):
        raise AutoMixCalibrationExecutionError(
            'Calibration execution requires the accepted C2eA preflight '
            'record before any liquid handling.'
        )
    required = {
        'record_version', 'record_type', 'created_at_utc', 'calibration_id',
        'logical_wellname', 'assay_profile', 'target',
        'plate_mapping_revision', 'plate_generation',
        'reader_scan_protocol', 'reader_wavelength_coverage_status',
        'expected_scan_count', 'pi_snapshot', 'pi_preflight',
        'well_binding', 'resource_preflight',
        'physical_execution_started', 'automatic_retry_permitted'
    }
    actual_keys = set(preflight_record)
    # C2eA adds this convenience field only after atomically persisting the
    # otherwise complete record.
    actual_keys.discard('record_path')
    if actual_keys != required:
        raise AutoMixCalibrationExecutionError(
            'Calibration preflight record has an invalid schema.'
        )
    if (preflight_record.get('record_version') != 1
            or preflight_record.get('record_type')
            != 'auto_mix_calibration_preflight'):
        raise AutoMixCalibrationExecutionError(
            'Calibration preflight record is not a supported C2eA record.'
        )
    for field_name in (
            'calibration_id', 'logical_wellname', 'assay_profile',
            'plate_mapping_revision', 'plate_generation',
            'expected_scan_count'):
        if preflight_record.get(field_name) != plan.get(field_name):
            raise AutoMixCalibrationExecutionError(
                'Calibration preflight record {} does not match the '
                'reviewed run plan.'.format(field_name)
            )
    if preflight_record.get('target') != plan.get('target'):
        raise AutoMixCalibrationExecutionError(
            'Calibration preflight record target does not match the '
            'reviewed run plan.'
        )
    if preflight_record.get('physical_execution_started') is not False or \
            preflight_record.get('automatic_retry_permitted') is not False:
        raise AutoMixCalibrationExecutionError(
            'Calibration preflight record is not an execution-free, '
            'non-retryable C2eA record.'
        )
    if preflight_record.get('reader_wavelength_coverage_status') != (
            'deferred_until_first_raw_scan'):
        raise AutoMixCalibrationExecutionError(
            'Calibration preflight record has an unsupported reader '
            'coverage status.'
        )
    recorded_scan_protocol = preflight_record.get('reader_scan_protocol')
    if (not isinstance(recorded_scan_protocol, str)
            or not recorded_scan_protocol.strip()):
        raise AutoMixCalibrationExecutionError(
            'Calibration preflight record has no reader scan protocol.'
        )
    if expected_reader_scan_protocol is not None:
        if (not isinstance(expected_reader_scan_protocol, str)
                or not expected_reader_scan_protocol.strip()):
            raise AutoMixCalibrationExecutionError(
                'Calibration execution received an invalid reader scan '
                'protocol identity.'
            )
        if recorded_scan_protocol != expected_reader_scan_protocol.strip():
            raise AutoMixCalibrationExecutionError(
                'Calibration preflight record reader protocol does not match '
                'the controller-selected reader protocol.'
            )
    try:
        pi_preflight = validate_mix_calibration_preflight_acknowledgement(
            preflight_record['pi_preflight'], plan['preflight_request']
        )
        binding_request = build_auto_mix_calibration_well_binding_request(plan)
        well_binding = validate_auto_mix_calibration_well_binding_acknowledgement(
            preflight_record['well_binding'], binding_request
        )
        resource_request = build_auto_mix_calibration_resource_preflight_request(
            plan, binding_request['binding_id']
        )
        resource_preflight = (
            validate_auto_mix_calibration_resource_preflight_acknowledgement(
                preflight_record['resource_preflight'], resource_request
            )
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise AutoMixCalibrationExecutionError(
            'Calibration C2eA preflight evidence is invalid: {}.'.format(exc)
        )
    if not well_binding['accepted']:
        raise AutoMixCalibrationExecutionError(
            'Calibration well binding was not accepted: {}.'.format(
                well_binding['message']
            )
        )
    if not pi_preflight['accepted']:
        raise AutoMixCalibrationExecutionError(
            'Calibration Pi preflight was not accepted: {}.'.format(
                pi_preflight['message']
            )
        )
    if not resource_preflight['accepted']:
        raise AutoMixCalibrationExecutionError(
            'Calibration resource preflight was not accepted: {}.'.format(
                resource_preflight['message']
            )
        )
    return {
        'pi_preflight': copy.deepcopy(pi_preflight),
        'well_binding': copy.deepcopy(well_binding),
        'resource_preflight': copy.deepcopy(resource_preflight),
        'reader_scan_protocol': preflight_record['reader_scan_protocol'],
    }


def _validate_scan_record(scan_record, scan_step, require_qc_wavelengths):
    '''Validate one controller-owned unmerged scan record.

    Reader protocol wavelength coverage cannot be established without an
    actual raw reader output. The baseline consequently proves 520/730 nm
    coverage before the first mix, never by replaying either transfer.
    '''
    required = {
        'scan_index', 'cycle_index', 'reader_well', 'logical_wellname',
        'raw_scan_relative_path', 'scan_started_at_utc',
        'scan_completed_at_utc', 'wavelengths_nm'
    }
    if not isinstance(scan_record, dict) or set(scan_record) != required:
        raise AutoMixCalibrationExecutionError(
            'Calibration scan returned an invalid record schema.'
        )
    for field_name in ('scan_index', 'cycle_index'):
        if scan_record[field_name] != scan_step[field_name]:
            raise AutoMixCalibrationExecutionError(
                'Calibration scan {} does not match the reviewed scan step.'
                .format(field_name)
            )
    if (scan_record['reader_well'] != scan_step['reader_well']
            or scan_record['logical_wellname'] != scan_step['wellname']):
        raise AutoMixCalibrationExecutionError(
            'Calibration scan identity does not match the reviewed scan step.'
        )
    if (not isinstance(scan_record['raw_scan_relative_path'], str)
            or not scan_record['raw_scan_relative_path'].strip()
            or not isinstance(scan_record['scan_started_at_utc'], str)
            or not isinstance(scan_record['scan_completed_at_utc'], str)):
        raise AutoMixCalibrationExecutionError(
            'Calibration scan has incomplete immutable provenance.'
        )
    if require_qc_wavelengths:
        _require_exact_wavelength_coverage(scan_record['wavelengths_nm'])
    return copy.deepcopy(scan_record)


def execute_auto_mix_calibration_plan_after_preflight(
        plan, preflight_record, adapter, expected_reader_scan_protocol=None):
    '''Execute C2eB only after one accepted C2eA record.

    The adapter is deliberately limited to two ordinary transfers, selected-
    well raw reader scans, and ten Pi-side mix requests. It has no normal-Auto
    recipe, optimizer, aggregate-scan, or automatic retry capability.
    '''
    plan = validate_auto_mix_calibration_run_plan(plan)
    for method_name in ('transfer', 'scan', 'request_pi_mix'):
        if not callable(getattr(adapter, method_name, None)):
            raise AutoMixCalibrationExecutionError(
                'Calibration execution adapter lacks {}().'.format(
                    method_name
                )
            )
    preflight_evidence = _validate_completed_controller_preflight(
        plan, preflight_record, expected_reader_scan_protocol
    )
    execution_record = {
        'record_type': 'auto_mix_calibration_execution',
        'calibration_id': plan['calibration_id'],
        'logical_wellname': plan['logical_wellname'],
        'preflight_evidence': preflight_evidence,
        'completed_step_ids': ['pi-preflight'],
        'scan_records': [],
        'mix_acknowledgements': [],
        'automatic_retry_permitted': False,
    }
    for transfer_step in plan['transfer_steps']:
        adapter.transfer(
            transfer_step['source_chemical_name'],
            transfer_step['destination_wellname'],
            transfer_step['volume_uL']
        )
        execution_record['completed_step_ids'].append(
            transfer_step['step_id']
        )
    for scan_step in plan['scan_steps']:
        if scan_step['scan_index']:
            mix_step = plan['mix_steps'][scan_step['cycle_index'] - 1]
            try:
                acknowledgement = validate_mix_calibration_acknowledgement(
                    adapter.request_pi_mix(copy.deepcopy(mix_step['request'])),
                    mix_step['request']
                )
            except (ValueError, TypeError) as exc:
                raise AutoMixCalibrationExecutionError(
                    'Calibration mix acknowledgement is invalid: {}.'.format(
                        exc
                    )
                )
            if not acknowledgement['accepted']:
                raise AutoMixCalibrationExecutionError(
                    'Pi rejected calibration mix cycle {} before physical '
                    'execution: {}.'.format(
                        scan_step['cycle_index'], acknowledgement['message']
                    )
                )
            execution_record['mix_acknowledgements'].append(acknowledgement)
            execution_record['completed_step_ids'].append(mix_step['step_id'])
        scan_record = _validate_scan_record(
            adapter.scan(copy.deepcopy(scan_step)),
            scan_step,
            require_qc_wavelengths=(
                plan['calibration_mode'] == 'two_dye'
                and scan_step['scan_index'] == 0
            )
        )
        execution_record['scan_records'].append(scan_record)
        execution_record['completed_step_ids'].append(scan_step['step_id'])
    return execution_record
