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
