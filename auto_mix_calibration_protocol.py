'''Pure Lab-PC/Pi wire contract for supervised well-mix calibration.

This module deliberately has no controller, reader, socket, or robot imports.
It defines the exact packet and capability boundary for the calibration-only
150 uL P300 assay before a controller execution workflow is connected.
'''

import math

from auto_mix_calibration import (
    MIX_CALIBRATION_CYCLE_COUNT,
    MIX_CALIBRATION_FINAL_WELL_VOLUME_UL,
    MIX_CALIBRATION_MODE_TWO_DYE,
    MIX_CALIBRATION_MODE_WATER_SAFETY,
    MIX_CALIBRATION_P300_VOLUME_UL,
    MIX_CALIBRATION_WELL_CLEARANCE_MM,
    get_auto_mix_calibration_profile_mode,
    get_auto_mix_calibration_profile_transfer_volumes,
)


MIX_CALIBRATION_SCHEMA_VERSION = 2
MIX_CALIBRATION_PREFLIGHT_COMMAND = 'preflight_auto_mix_calibration'
MIX_CALIBRATION_PREFLIGHT_ACKNOWLEDGEMENT = 'auto_mix_calibration_preflight'
MIX_CALIBRATION_COMMAND = 'mix_auto_calibration_well'
MIX_CALIBRATION_ACKNOWLEDGEMENT = 'auto_calibration_well_mixed'
MIX_CALIBRATION_PROTOCOL_VERSION = 'auto-mix-calibration-v3'
MIX_CALIBRATION_REQUIRED_P300_TIPS = 11
MIX_CALIBRATION_TIP_POLICY = 'dedicated_discarded'


class AutoMixCalibrationProtocolError(ValueError):
    '''Raised when calibration-only communication is incomplete or unsafe.'''


def _require_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise AutoMixCalibrationProtocolError(
            '{} must be a nonempty string.'.format(field_name)
        )
    return value.strip()


def _require_nonnegative_integer(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutoMixCalibrationProtocolError(
            '{} must be a nonnegative integer.'.format(field_name)
        )
    return value


def _require_exact_number(value, expected, field_name):
    if isinstance(value, bool):
        raise AutoMixCalibrationProtocolError(
            '{} must equal {}.'.format(field_name, expected)
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise AutoMixCalibrationProtocolError(
            '{} must equal {}.'.format(field_name, expected)
        )
    if not math.isfinite(numeric) or not math.isclose(
            numeric, expected, rel_tol=0, abs_tol=1e-9):
        raise AutoMixCalibrationProtocolError(
            '{} must equal {}.'.format(field_name, expected)
        )
    return expected


def _validate_common_request(request, identity_field):
    common_fields = {
        'schema_version', identity_field, 'calibration_id',
        'calibration_mode', 'wellname', 'expected_deck_pos', 'expected_loc',
        'expected_plate_mapping_revision', 'expected_plate_generation',
        'assay_profile', 'base_chemical_name', 'trigger_chemical_name',
        'first_transfer_volume_uL', 'second_transfer_volume_uL',
        'expected_final_well_volume_uL', 'mix_volume_uL',
        'total_cycle_count', 'well_clearance_mm'
    }
    if not isinstance(request, dict) or set(request) != common_fields:
        raise AutoMixCalibrationProtocolError(
            'Calibration request has an invalid schema.'
        )
    if request['schema_version'] != MIX_CALIBRATION_SCHEMA_VERSION:
        raise AutoMixCalibrationProtocolError(
            'Calibration request uses an unsupported schema.'
        )
    mode = _require_string(request['calibration_mode'], 'calibration_mode').lower()
    if mode not in (MIX_CALIBRATION_MODE_WATER_SAFETY,
                    MIX_CALIBRATION_MODE_TWO_DYE):
        raise AutoMixCalibrationProtocolError(
            'calibration_mode must be water_safety or two_dye.'
        )
    assay_profile = _require_string(
        request['assay_profile'], 'assay_profile'
    ).lower()
    try:
        expected_mode = get_auto_mix_calibration_profile_mode(assay_profile)
        expected_first_volume_uL, expected_second_volume_uL = (
            get_auto_mix_calibration_profile_transfer_volumes(assay_profile)
        )
    except ValueError as exc:
        raise AutoMixCalibrationProtocolError(
            'Calibration assay_profile is unsupported: {}.'.format(exc)
        )
    if mode != expected_mode:
        raise AutoMixCalibrationProtocolError(
            'Calibration assay_profile does not match calibration_mode.'
        )
    normalized = {
        'schema_version': MIX_CALIBRATION_SCHEMA_VERSION,
        identity_field: _require_string(request[identity_field], identity_field),
        'calibration_id': _require_string(
            request['calibration_id'], 'calibration_id'
        ),
        'calibration_mode': mode,
        'assay_profile': assay_profile,
        'wellname': _require_string(request['wellname'], 'wellname'),
        'expected_deck_pos': _require_nonnegative_integer(
            request['expected_deck_pos'], 'expected_deck_pos'
        ),
        'expected_loc': _require_string(
            request['expected_loc'], 'expected_loc'
        ).upper(),
        'expected_plate_mapping_revision': _require_nonnegative_integer(
            request['expected_plate_mapping_revision'],
            'expected_plate_mapping_revision'
        ),
        'expected_plate_generation': _require_nonnegative_integer(
            request['expected_plate_generation'],
            'expected_plate_generation'
        ),
        'base_chemical_name': _require_string(
            request['base_chemical_name'], 'base_chemical_name'
        ),
        'trigger_chemical_name': _require_string(
            request['trigger_chemical_name'], 'trigger_chemical_name'
        ),
        'first_transfer_volume_uL': _require_exact_number(
            request['first_transfer_volume_uL'], expected_first_volume_uL,
            'first_transfer_volume_uL'
        ),
        'second_transfer_volume_uL': _require_exact_number(
            request['second_transfer_volume_uL'], expected_second_volume_uL,
            'second_transfer_volume_uL'
        ),
        'expected_final_well_volume_uL': _require_exact_number(
            request['expected_final_well_volume_uL'],
            MIX_CALIBRATION_FINAL_WELL_VOLUME_UL,
            'expected_final_well_volume_uL'
        ),
        'mix_volume_uL': _require_exact_number(
            request['mix_volume_uL'], MIX_CALIBRATION_P300_VOLUME_UL,
            'mix_volume_uL'
        ),
        'total_cycle_count': request['total_cycle_count'],
        'well_clearance_mm': _require_exact_number(
            request['well_clearance_mm'], MIX_CALIBRATION_WELL_CLEARANCE_MM,
            'well_clearance_mm'
        )
    }
    if not normalized['wellname'].startswith('autowell'):
        raise AutoMixCalibrationProtocolError(
            'Calibration may address an Auto product well only.'
        )
    if normalized['expected_deck_pos'] not in (4, 7):
        raise AutoMixCalibrationProtocolError(
            'Calibration must address plate-reader deck position 4 or 7.'
        )
    if normalized['base_chemical_name'].casefold() == normalized[
            'trigger_chemical_name'].casefold():
        raise AutoMixCalibrationProtocolError(
            'Calibration base and trigger sources must be distinct.'
        )
    if (isinstance(normalized['total_cycle_count'], bool)
            or normalized['total_cycle_count'] != MIX_CALIBRATION_CYCLE_COUNT):
        raise AutoMixCalibrationProtocolError(
            'Calibration total_cycle_count must equal {}.'.format(
                MIX_CALIBRATION_CYCLE_COUNT
            )
        )
    return normalized


def validate_mix_calibration_preflight_request(request):
    '''Validate a non-mutating complete-calibration preflight packet.'''
    return _validate_common_request(request, 'preflight_id')


def validate_mix_calibration_preflight_acknowledgement(acknowledgement, request):
    '''Validate the Pi's no-motion readiness response before any transfer.'''
    request = validate_mix_calibration_preflight_request(request)
    required = {
        'schema_version', 'record_type', 'preflight_id', 'calibration_id',
        'accepted', 'message', 'wellname', 'expected_deck_pos',
        'expected_loc', 'base_chemical_name', 'trigger_chemical_name',
        'plate_mapping_revision', 'plate_generation', 'pipette_arm',
        'mix_volume_uL', 'total_cycle_count', 'well_clearance_mm',
        'required_new_tips', 'available_new_tips', 'tip_policy'
    }
    if not isinstance(acknowledgement, dict) or set(acknowledgement) != required:
        raise AutoMixCalibrationProtocolError(
            'Calibration preflight acknowledgement has an invalid schema.'
        )
    if (acknowledgement['schema_version'] != MIX_CALIBRATION_SCHEMA_VERSION
            or acknowledgement['record_type']
            != MIX_CALIBRATION_PREFLIGHT_ACKNOWLEDGEMENT
            or not isinstance(acknowledgement['accepted'], bool)):
        raise AutoMixCalibrationProtocolError(
            'Calibration preflight acknowledgement has an invalid record.'
        )
    _require_string(acknowledgement['message'], 'acknowledgement.message')
    for field_name in (
            'preflight_id', 'calibration_id', 'wellname',
            'expected_deck_pos', 'base_chemical_name',
            'trigger_chemical_name'):
        if acknowledgement[field_name] != request[field_name]:
            raise AutoMixCalibrationProtocolError(
                'Calibration preflight acknowledgement {} does not match '
                'the request.'.format(field_name)
            )
    if (not isinstance(acknowledgement['expected_loc'], str)
            or acknowledgement['expected_loc'].strip().upper()
            != request['expected_loc']):
        raise AutoMixCalibrationProtocolError(
            'Calibration preflight acknowledgement expected_loc does not '
            'match the request.'
        )
    if (acknowledgement['plate_mapping_revision']
            != request['expected_plate_mapping_revision']
            or acknowledgement['plate_generation']
            != request['expected_plate_generation']):
        raise AutoMixCalibrationProtocolError(
            'Calibration preflight acknowledgement plate identity does not '
            'match the request.'
        )
    if not acknowledgement['accepted']:
        return dict(acknowledgement)
    if acknowledgement['pipette_arm'] not in ('left', 'right'):
        raise AutoMixCalibrationProtocolError(
            'Accepted calibration preflight has no valid P300 arm.'
        )
    for field_name, expected in (
            ('mix_volume_uL', MIX_CALIBRATION_P300_VOLUME_UL),
            ('well_clearance_mm', MIX_CALIBRATION_WELL_CLEARANCE_MM)):
        _require_exact_number(acknowledgement[field_name], expected, field_name)
    if acknowledgement['total_cycle_count'] != MIX_CALIBRATION_CYCLE_COUNT:
        raise AutoMixCalibrationProtocolError(
            'Accepted calibration preflight has the wrong cycle count.'
        )
    if (acknowledgement['required_new_tips']
            != MIX_CALIBRATION_REQUIRED_P300_TIPS
            or isinstance(acknowledgement['available_new_tips'], bool)
            or not isinstance(acknowledgement['available_new_tips'], int)
            or acknowledgement['available_new_tips']
            < MIX_CALIBRATION_REQUIRED_P300_TIPS):
        raise AutoMixCalibrationProtocolError(
            'Accepted calibration preflight has an invalid tip readiness '
            'record.'
        )
    if acknowledgement['tip_policy'] != MIX_CALIBRATION_TIP_POLICY:
        raise AutoMixCalibrationProtocolError(
            'Accepted calibration preflight has an unexpected tip policy.'
        )
    return dict(acknowledgement)


def validate_mix_calibration_request(request):
    '''Validate exactly one discrete mix action between reader scans.'''
    required = set(_mix_request_fields())
    if not isinstance(request, dict) or set(request) != required:
        raise AutoMixCalibrationProtocolError(
            'Calibration mix request has an invalid schema.'
        )
    common = dict(request)
    cycle_index = common.pop('cycle_index')
    common.pop('action_id')
    preflight_id = common.pop('preflight_id')
    normalized = _validate_common_request(
        dict(common, action_id=request['action_id']), 'action_id'
    )
    if (isinstance(cycle_index, bool) or not isinstance(cycle_index, int)
            or cycle_index < 1
            or cycle_index > MIX_CALIBRATION_CYCLE_COUNT):
        raise AutoMixCalibrationProtocolError(
            'Calibration cycle_index must be from 1 through {}.'.format(
                MIX_CALIBRATION_CYCLE_COUNT
            )
        )
    normalized['cycle_index'] = cycle_index
    normalized['preflight_id'] = _require_string(preflight_id, 'preflight_id')
    return normalized


def validate_mix_calibration_acknowledgement(acknowledgement, request):
    '''Validate one completed/rejected Pi mix action without retry semantics.'''
    request = validate_mix_calibration_request(request)
    required = {
        'schema_version', 'record_type', 'action_id', 'calibration_id',
        'preflight_id', 'accepted', 'message', 'cycle_index', 'wellname',
        'expected_deck_pos', 'expected_loc', 'plate_mapping_revision',
        'plate_generation', 'pipette_arm', 'mix_volume_uL',
        'well_clearance_mm', 'required_new_tips', 'available_new_tips',
        'tip_policy', 'physical_execution_started', 'started_at_utc',
        'completed_at_utc'
    }
    if not isinstance(acknowledgement, dict) or set(acknowledgement) != required:
        raise AutoMixCalibrationProtocolError(
            'Calibration action acknowledgement has an invalid schema.'
        )
    if (acknowledgement['schema_version'] != MIX_CALIBRATION_SCHEMA_VERSION
            or acknowledgement['record_type'] != MIX_CALIBRATION_ACKNOWLEDGEMENT
            or not isinstance(acknowledgement['accepted'], bool)
            or not isinstance(acknowledgement['physical_execution_started'], bool)):
        raise AutoMixCalibrationProtocolError(
            'Calibration action acknowledgement has an invalid record.'
        )
    _require_string(acknowledgement['message'], 'acknowledgement.message')
    for field_name in (
            'action_id', 'calibration_id', 'preflight_id', 'cycle_index',
            'wellname', 'expected_deck_pos'):
        if acknowledgement[field_name] != request[field_name]:
            raise AutoMixCalibrationProtocolError(
                'Calibration action acknowledgement {} does not match the '
                'request.'.format(field_name)
            )
    if (not isinstance(acknowledgement['expected_loc'], str)
            or acknowledgement['expected_loc'].strip().upper()
            != request['expected_loc']):
        raise AutoMixCalibrationProtocolError(
            'Calibration action acknowledgement expected_loc does not match '
            'the request.'
        )
    if (acknowledgement['plate_mapping_revision']
            != request['expected_plate_mapping_revision']
            or acknowledgement['plate_generation']
            != request['expected_plate_generation']):
        raise AutoMixCalibrationProtocolError(
            'Calibration action acknowledgement plate identity does not '
            'match the request.'
        )
    if not acknowledgement['accepted']:
        if acknowledgement['physical_execution_started']:
            raise AutoMixCalibrationProtocolError(
                'Calibration action was rejected after physical execution '
                'began.'
            )
        return dict(acknowledgement)
    if (not acknowledgement['physical_execution_started']
            or acknowledgement['pipette_arm'] not in ('left', 'right')
            or acknowledgement['tip_policy'] != MIX_CALIBRATION_TIP_POLICY):
        raise AutoMixCalibrationProtocolError(
            'Accepted calibration action lacks required execution evidence.'
        )
    _require_exact_number(
        acknowledgement['mix_volume_uL'], MIX_CALIBRATION_P300_VOLUME_UL,
        'acknowledgement.mix_volume_uL'
    )
    _require_exact_number(
        acknowledgement['well_clearance_mm'],
        MIX_CALIBRATION_WELL_CLEARANCE_MM,
        'acknowledgement.well_clearance_mm'
    )
    for field_name in ('required_new_tips', 'available_new_tips'):
        if (isinstance(acknowledgement[field_name], bool)
                or not isinstance(acknowledgement[field_name], int)
                or acknowledgement[field_name] < 1):
            raise AutoMixCalibrationProtocolError(
                'Accepted calibration action has invalid {}.'.format(field_name)
            )
    for field_name in ('started_at_utc', 'completed_at_utc'):
        _require_string(acknowledgement[field_name], field_name)
    return dict(acknowledgement)


def _mix_request_fields():
    return (
        'schema_version', 'action_id', 'calibration_id', 'calibration_mode',
        'preflight_id', 'cycle_index', 'wellname', 'expected_deck_pos', 'expected_loc',
        'expected_plate_mapping_revision', 'expected_plate_generation',
        'assay_profile', 'base_chemical_name', 'trigger_chemical_name',
        'first_transfer_volume_uL', 'second_transfer_volume_uL',
        'expected_final_well_volume_uL', 'mix_volume_uL',
        'total_cycle_count', 'well_clearance_mm'
    )


def validate_mix_calibration_capability(snapshot, packet_registry):
    '''Fail closed unless the optional calibration Pi extension is exact.

    Ordinary Auto and ordinary pipette-stability operation deliberately never
    call this helper.  Stage 13G-C will call it only after calibration mode is
    explicitly enabled, preserving compatibility with existing Auto-main v10.
    '''
    if not isinstance(snapshot, dict):
        raise AutoMixCalibrationProtocolError(
            'Calibration requires an Auto-main state snapshot.'
        )
    if snapshot.get('auto_mix_calibration_protocol_version') != (
            MIX_CALIBRATION_PROTOCOL_VERSION):
        raise AutoMixCalibrationProtocolError(
            'Calibration requires Pi protocol {!r}.'.format(
                MIX_CALIBRATION_PROTOCOL_VERSION
            )
        )
    required_commands = (
        MIX_CALIBRATION_PREFLIGHT_COMMAND,
        MIX_CALIBRATION_COMMAND,
    )
    supported = snapshot.get('supported_commands')
    if not isinstance(supported, list) or any(
            command not in supported for command in required_commands):
        raise AutoMixCalibrationProtocolError(
            'Calibration Pi snapshot does not advertise its required commands.'
        )
    expected_packets = (
        (MIX_CALIBRATION_PREFLIGHT_COMMAND, b'\x21'),
        (MIX_CALIBRATION_PREFLIGHT_ACKNOWLEDGEMENT, b'\x22'),
        (MIX_CALIBRATION_COMMAND, b'\x23'),
        (MIX_CALIBRATION_ACKNOWLEDGEMENT, b'\x24'),
    )
    for packet_name, expected_code in expected_packets:
        if packet_registry.get(packet_name) != expected_code:
            raise AutoMixCalibrationProtocolError(
                'Calibration requires local packet {} with code {!r}.'.format(
                    packet_name, expected_code
                )
            )
    return True
