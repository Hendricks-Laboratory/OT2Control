'''Pure version-2 resource contract for supervised mix calibration.

The version-1 calibration packet proves only the dedicated P300 well-mix
setup.  This module adds the separate no-motion binding and resource-preflight
contract that a later paired Auto-main handler must satisfy before the
calibration executor may transfer either assay source.
'''

import copy

from auto_mix_calibration import get_auto_mix_calibration_profile_transfer_volumes
from auto_mix_calibration_plan import validate_auto_mix_calibration_run_plan
from auto_mix_calibration_protocol import (
    MIX_CALIBRATION_COMMAND,
    MIX_CALIBRATION_PREFLIGHT_COMMAND,
)


MIX_CALIBRATION_RESOURCE_SCHEMA_VERSION = 1
MIX_CALIBRATION_RESOURCE_PROTOCOL_VERSION = 'auto-mix-calibration-v2'
MIX_CALIBRATION_BIND_WELL_COMMAND = 'bind_auto_mix_calibration_well'
MIX_CALIBRATION_BIND_WELL_ACKNOWLEDGEMENT = (
    'auto_mix_calibration_well_bound'
)
MIX_CALIBRATION_RESOURCE_PREFLIGHT_COMMAND = (
    'preflight_auto_mix_calibration_resources'
)
MIX_CALIBRATION_RESOURCE_PREFLIGHT_ACKNOWLEDGEMENT = (
    'auto_mix_calibration_resources_preflight'
)


class AutoMixCalibrationResourceError(ValueError):
    '''Raised when calibration binding/resource evidence is incomplete.'''


def _require_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise AutoMixCalibrationResourceError(
            '{} must be a nonempty string.'.format(field_name)
        )
    return value.strip()


def _require_nonnegative_integer(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutoMixCalibrationResourceError(
            '{} must be a nonnegative integer.'.format(field_name)
        )
    return value


def _plan_identity(plan):
    plan = validate_auto_mix_calibration_run_plan(plan)
    target = plan['target']
    return {
        'calibration_id': plan['calibration_id'],
        'wellname': plan['logical_wellname'],
        'reader_well': target['reader_well'],
        'expected_deck_pos': target['expected_deck_pos'],
        'expected_loc': target['expected_loc'],
        'expected_plate_mapping_revision': plan['plate_mapping_revision'],
        'expected_plate_generation': plan['plate_generation'],
        'expected_final_well_volume_uL': (
            plan['expected_final_well_volume_uL']
        ),
    }


def build_auto_mix_calibration_well_binding_request(plan):
    '''Create the no-motion request to bind one vacant physical reader well.'''
    identity = _plan_identity(plan)
    request = dict(identity)
    request.update({
        'schema_version': MIX_CALIBRATION_RESOURCE_SCHEMA_VERSION,
        'binding_id': '{}-binding'.format(identity['calibration_id']),
    })
    return validate_auto_mix_calibration_well_binding_request(request)


def validate_auto_mix_calibration_well_binding_request(request):
    required = {
        'schema_version', 'binding_id', 'calibration_id', 'wellname',
        'reader_well', 'expected_deck_pos', 'expected_loc',
        'expected_plate_mapping_revision', 'expected_plate_generation',
        'expected_final_well_volume_uL'
    }
    if not isinstance(request, dict) or set(request) != required:
        raise AutoMixCalibrationResourceError(
            'Calibration well-binding request has an invalid schema.'
        )
    if request['schema_version'] != MIX_CALIBRATION_RESOURCE_SCHEMA_VERSION:
        raise AutoMixCalibrationResourceError(
            'Calibration well-binding request has an unsupported schema.'
        )
    normalized = {
        key: request[key] for key in required
    }
    for key in ('binding_id', 'calibration_id', 'wellname', 'reader_well',
                'expected_loc'):
        normalized[key] = _require_string(request[key], key)
    normalized['expected_loc'] = normalized['expected_loc'].upper()
    normalized['reader_well'] = normalized['reader_well'].upper()
    if not normalized['wellname'].startswith('autowell'):
        raise AutoMixCalibrationResourceError(
            'Calibration binding may address an Auto product well only.'
        )
    for key in ('expected_deck_pos', 'expected_plate_mapping_revision',
                'expected_plate_generation'):
        normalized[key] = _require_nonnegative_integer(request[key], key)
    if normalized['expected_deck_pos'] not in (4, 7):
        raise AutoMixCalibrationResourceError(
            'Calibration binding requires reader deck position 4 or 7.'
        )
    if request['expected_final_well_volume_uL'] != 200.0:
        raise AutoMixCalibrationResourceError(
            'Calibration binding requires the fixed 200 uL well volume.'
        )
    return normalized


def validate_auto_mix_calibration_well_binding_acknowledgement(
        acknowledgement, request):
    '''Validate Pi evidence that a selected reader location is vacant/bound.'''
    request = validate_auto_mix_calibration_well_binding_request(request)
    required = {
        'schema_version', 'record_type', 'binding_id', 'calibration_id',
        'accepted', 'message', 'wellname', 'reader_well',
        'expected_deck_pos', 'expected_loc', 'plate_mapping_revision',
        'plate_generation', 'well_was_vacant', 'binding_created'
    }
    if not isinstance(acknowledgement, dict) or set(acknowledgement) != required:
        raise AutoMixCalibrationResourceError(
            'Calibration well-binding acknowledgement has an invalid schema.'
        )
    if (acknowledgement['schema_version'] != MIX_CALIBRATION_RESOURCE_SCHEMA_VERSION
            or acknowledgement['record_type'] != (
                MIX_CALIBRATION_BIND_WELL_ACKNOWLEDGEMENT)
            or not isinstance(acknowledgement['accepted'], bool)
            or not isinstance(acknowledgement['well_was_vacant'], bool)
            or not isinstance(acknowledgement['binding_created'], bool)):
        raise AutoMixCalibrationResourceError(
            'Calibration well-binding acknowledgement has an invalid record.'
        )
    _require_string(acknowledgement['message'], 'acknowledgement.message')
    for key in ('binding_id', 'calibration_id', 'wellname', 'reader_well',
                'expected_deck_pos'):
        if acknowledgement[key] != request[key]:
            raise AutoMixCalibrationResourceError(
                'Calibration binding acknowledgement {} does not match the '
                'request.'.format(key)
            )
    if (not isinstance(acknowledgement['expected_loc'], str)
            or acknowledgement['expected_loc'].strip().upper()
            != request['expected_loc']
            or acknowledgement['plate_mapping_revision']
            != request['expected_plate_mapping_revision']
            or acknowledgement['plate_generation']
            != request['expected_plate_generation']):
        raise AutoMixCalibrationResourceError(
            'Calibration binding acknowledgement plate identity mismatches '
            'the request.'
        )
    if acknowledgement['accepted'] and not (
            acknowledgement['well_was_vacant']
            and acknowledgement['binding_created']):
        raise AutoMixCalibrationResourceError(
            'Accepted calibration binding lacks vacant-well evidence.'
        )
    if not acknowledgement['accepted'] and acknowledgement['binding_created']:
        raise AutoMixCalibrationResourceError(
            'Rejected calibration binding may not create state.'
        )
    return copy.deepcopy(acknowledgement)


def build_auto_mix_calibration_resource_preflight_request(plan, binding_id):
    '''Build an exact no-motion Pi source/tip preflight from a C2a plan.'''
    plan = validate_auto_mix_calibration_run_plan(plan)
    binding_id = _require_string(binding_id, 'binding_id')
    request = {
        'schema_version': MIX_CALIBRATION_RESOURCE_SCHEMA_VERSION,
        'resource_preflight_id': '{}-resources'.format(plan['calibration_id']),
        'binding_id': binding_id,
        'calibration_id': plan['calibration_id'],
        'assay_profile': plan['assay_profile'],
        'wellname': plan['logical_wellname'],
        'expected_deck_pos': plan['target']['expected_deck_pos'],
        'expected_loc': plan['target']['expected_loc'],
        'expected_plate_mapping_revision': plan['plate_mapping_revision'],
        'expected_plate_generation': plan['plate_generation'],
        'source_requirements': [
            {
                'slot': slot,
                'source_container': step['source_chemical_name'],
                'required_volume_uL': step['volume_uL'],
            }
            for slot, step in zip(('first', 'second'), plan['transfer_steps'])
        ],
        'ordinary_transfer_tip_requirements': [
            {'pipette': 'p20', 'required_new_tips': 1},
            {'pipette': 'p300', 'required_new_tips': 1},
        ],
        'dedicated_mix_tip_requirements': [
            {'pipette': 'p300', 'required_new_tips': 11},
        ],
    }
    return validate_auto_mix_calibration_resource_preflight_request(request)


def validate_auto_mix_calibration_resource_preflight_request(request):
    required = {
        'schema_version', 'resource_preflight_id', 'binding_id',
        'calibration_id', 'assay_profile', 'wellname', 'expected_deck_pos', 'expected_loc',
        'expected_plate_mapping_revision', 'expected_plate_generation',
        'source_requirements', 'ordinary_transfer_tip_requirements',
        'dedicated_mix_tip_requirements'
    }
    if not isinstance(request, dict) or set(request) != required:
        raise AutoMixCalibrationResourceError(
            'Calibration resource preflight request has an invalid schema.'
        )
    for key in ('resource_preflight_id', 'binding_id', 'calibration_id',
                'assay_profile', 'wellname', 'expected_loc'):
        _require_string(request[key], key)
    if not request['wellname'].startswith('autowell'):
        raise AutoMixCalibrationResourceError(
            'Calibration resource preflight requires an Auto product well.'
        )
    for key in ('expected_deck_pos', 'expected_plate_mapping_revision',
                'expected_plate_generation'):
        _require_nonnegative_integer(request[key], key)
    if request['expected_deck_pos'] not in (4, 7):
        raise AutoMixCalibrationResourceError(
            'Calibration resource preflight requires reader deck position.'
        )
    try:
        first_volume, second_volume = (
            get_auto_mix_calibration_profile_transfer_volumes(
                request['assay_profile']
            )
        )
    except ValueError as error:
        raise AutoMixCalibrationResourceError(str(error))
    _validate_exact_requirements(
        request['source_requirements'],
        (('first', first_volume), ('second', second_volume)), 'source'
    )
    _validate_exact_requirements(
        request['ordinary_transfer_tip_requirements'],
        (('p20', 1), ('p300', 1)), 'ordinary tip', key='pipette'
    )
    _validate_exact_requirements(
        request['dedicated_mix_tip_requirements'],
        (('p300', 11),), 'mix tip', key='pipette'
    )
    return copy.deepcopy(request)


def _validate_exact_requirements(records, expected, label, key='slot'):
    if not isinstance(records, list) or len(records) != len(expected):
        raise AutoMixCalibrationResourceError(
            'Calibration {} requirements are incomplete.'.format(label)
        )
    for record, (identity, required_volume) in zip(records, expected):
        expected_fields = (
            set(('slot', 'source_container', 'required_volume_uL'))
            if key == 'slot' else set(('pipette', 'required_new_tips'))
        )
        if (not isinstance(record, dict) or set(record) != expected_fields
                or record.get(key) != identity):
            raise AutoMixCalibrationResourceError(
                'Calibration {} requirement identity is invalid.'.format(label)
            )
        required = record.get('required_volume_uL',
                              record.get('required_new_tips'))
        if required != required_volume:
            raise AutoMixCalibrationResourceError(
                'Calibration {} requirement is not reviewed.'.format(label)
            )
        if key == 'slot':
            _require_string(record.get('source_container'),
                            'source_container')


def validate_auto_mix_calibration_resource_preflight_acknowledgement(
        acknowledgement, request):
    '''Validate a complete non-mutating Pi resource decision.

    A passed report must echo the immutable request and add positive source/tip
    availability evidence. A rejected report returns only its structured
    rejection and may not claim resource allocation.
    '''
    request = validate_auto_mix_calibration_resource_preflight_request(request)
    required = set(request).difference(set((
        'source_requirements', 'ordinary_transfer_tip_requirements',
        'dedicated_mix_tip_requirements'
    ))).union(set((
        'record_type', 'accepted', 'message', 'source_requirements',
        'ordinary_transfer_tip_requirements', 'dedicated_mix_tip_requirements'
    )))
    if not isinstance(acknowledgement, dict) or set(acknowledgement) != required:
        raise AutoMixCalibrationResourceError(
            'Calibration resource acknowledgement has an invalid schema.'
        )
    if (acknowledgement['schema_version'] != MIX_CALIBRATION_RESOURCE_SCHEMA_VERSION
            or acknowledgement['record_type'] != (
                MIX_CALIBRATION_RESOURCE_PREFLIGHT_ACKNOWLEDGEMENT)
            or not isinstance(acknowledgement['accepted'], bool)):
        raise AutoMixCalibrationResourceError(
            'Calibration resource acknowledgement has an invalid record.'
        )
    _require_string(acknowledgement['message'], 'acknowledgement.message')
    for key in set(request).difference(set((
            'source_requirements', 'ordinary_transfer_tip_requirements',
            'dedicated_mix_tip_requirements'))):
        if acknowledgement[key] != request[key]:
            raise AutoMixCalibrationResourceError(
                'Calibration resource acknowledgement {} does not match the '
                'request.'.format(key)
            )
    if not acknowledgement['accepted']:
        if any(acknowledgement[key] for key in (
                'source_requirements', 'ordinary_transfer_tip_requirements',
                'dedicated_mix_tip_requirements')):
            raise AutoMixCalibrationResourceError(
                'Rejected calibration resource preflight may not approve '
                'partial resource allocation.'
            )
        return copy.deepcopy(acknowledgement)
    _validate_acknowledged_sources(
        acknowledgement['source_requirements'], request['source_requirements']
    )
    _validate_acknowledged_tips(
        acknowledgement['ordinary_transfer_tip_requirements'],
        request['ordinary_transfer_tip_requirements']
    )
    _validate_acknowledged_tips(
        acknowledgement['dedicated_mix_tip_requirements'],
        request['dedicated_mix_tip_requirements']
    )
    return copy.deepcopy(acknowledgement)


def _validate_acknowledged_sources(records, requested):
    if not isinstance(records, list) or len(records) != len(requested):
        raise AutoMixCalibrationResourceError(
            'Accepted calibration resource preflight has incomplete sources.'
        )
    for record, request in zip(records, requested):
        required = set(request).union(set(('available_volume_uL',)))
        if not isinstance(record, dict) or set(record) != required:
            raise AutoMixCalibrationResourceError(
                'Accepted calibration source evidence is malformed.'
            )
        if any(record[key] != request[key] for key in request):
            raise AutoMixCalibrationResourceError(
                'Accepted calibration source evidence differs from request.'
            )
        try:
            available = float(record['available_volume_uL'])
        except (TypeError, ValueError):
            raise AutoMixCalibrationResourceError(
                'Accepted calibration source volume is invalid.'
            )
        if available < request['required_volume_uL']:
            raise AutoMixCalibrationResourceError(
                'Accepted calibration source volume is insufficient.'
            )


def _validate_acknowledged_tips(records, requested):
    if not isinstance(records, list) or len(records) != len(requested):
        raise AutoMixCalibrationResourceError(
            'Accepted calibration tip evidence is incomplete.'
        )
    for record, request in zip(records, requested):
        required = set(request).union(set(('available_new_tips',)))
        if not isinstance(record, dict) or set(record) != required:
            raise AutoMixCalibrationResourceError(
                'Accepted calibration tip evidence is malformed.'
            )
        if any(record[key] != request[key] for key in request):
            raise AutoMixCalibrationResourceError(
                'Accepted calibration tip evidence differs from request.'
            )
        if (isinstance(record['available_new_tips'], bool)
                or not isinstance(record['available_new_tips'], int)
                or record['available_new_tips'] < request['required_new_tips']):
            raise AutoMixCalibrationResourceError(
                'Accepted calibration tip count is insufficient.'
            )


def validate_auto_mix_calibration_resource_capability(snapshot, packet_registry):
    '''Require exact v2 capability without altering normal Auto compatibility.'''
    if not isinstance(snapshot, dict):
        raise AutoMixCalibrationResourceError(
            'Calibration resources require an Auto-main state snapshot.'
        )
    if snapshot.get('auto_mix_calibration_protocol_version') != (
            MIX_CALIBRATION_RESOURCE_PROTOCOL_VERSION):
        raise AutoMixCalibrationResourceError(
            'Calibration resources require Pi protocol {!r}.'.format(
                MIX_CALIBRATION_RESOURCE_PROTOCOL_VERSION
            )
        )
    required_commands = (
        MIX_CALIBRATION_PREFLIGHT_COMMAND,
        MIX_CALIBRATION_COMMAND,
        MIX_CALIBRATION_BIND_WELL_COMMAND,
        MIX_CALIBRATION_RESOURCE_PREFLIGHT_COMMAND,
    )
    supported = snapshot.get('supported_commands')
    if not isinstance(supported, list) or any(
            command not in supported for command in required_commands):
        raise AutoMixCalibrationResourceError(
            'Calibration resource commands are unavailable on the Pi.'
        )
    expected_packets = (
        (MIX_CALIBRATION_PREFLIGHT_COMMAND, b'\x21'),
        ('auto_mix_calibration_preflight', b'\x22'),
        (MIX_CALIBRATION_COMMAND, b'\x23'),
        ('auto_calibration_well_mixed', b'\x24'),
        (MIX_CALIBRATION_BIND_WELL_COMMAND, b'\x25'),
        (MIX_CALIBRATION_BIND_WELL_ACKNOWLEDGEMENT, b'\x26'),
        (MIX_CALIBRATION_RESOURCE_PREFLIGHT_COMMAND, b'\x27'),
        (MIX_CALIBRATION_RESOURCE_PREFLIGHT_ACKNOWLEDGEMENT, b'\x28'),
    )
    for name, code in expected_packets:
        if packet_registry.get(name) != code:
            raise AutoMixCalibrationResourceError(
                'Calibration resource packet {} is not registered as {!r}.'.format(
                    name, code
                )
            )
    return True
