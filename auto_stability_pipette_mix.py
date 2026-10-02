'''Pure controller/Pi contract for one targeted completed-well stability mix.

Stage 13C deliberately defines the wire format separately from controller or
robot execution.  It contains no hardware, socket, spreadsheet, or observer
imports, allowing exact request and acknowledgement validation without moving
an OT-2.  The controller records a durable intent before sending this request;
callers must never retry an action whose physical outcome is unknown.
'''

import datetime
import math


TARGETED_MIX_SCHEMA_VERSION = 1
TARGETED_MIX_COMMAND = 'mix_auto_completed_well'
TARGETED_MIX_ACKNOWLEDGEMENT = 'auto_completed_well_mixed'
TARGETED_MIX_RECORD_TYPE = 'auto_completed_well_mixed'
TARGETED_MIX_TIP_POLICY = 'dedicated_discarded'


class AutoStabilityPipetteMixContractError(ValueError):
    '''Raised when one Stage-13 targeted-mix packet is not exact and safe.'''


def _require_nonempty_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise AutoStabilityPipetteMixContractError(
            '{} must be a nonempty string.'.format(field_name)
        )
    return value.strip()


def _require_nonnegative_integer(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutoStabilityPipetteMixContractError(
            '{} must be a nonnegative integer.'.format(field_name)
        )
    return value


def _require_positive_finite_number(value, field_name):
    if isinstance(value, bool):
        raise AutoStabilityPipetteMixContractError(
            '{} must be a finite positive number.'.format(field_name)
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityPipetteMixContractError(
            '{} must be a finite positive number.'.format(field_name)
        )
    if not math.isfinite(numeric) or numeric <= 0:
        raise AutoStabilityPipetteMixContractError(
            '{} must be a finite positive number.'.format(field_name)
        )
    return numeric


def _require_utc_timestamp(value, field_name):
    text = _require_nonempty_string(value, field_name)
    try:
        timestamp = datetime.datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        raise AutoStabilityPipetteMixContractError(
            '{} must be an ISO-8601 timestamp.'.format(field_name)
        )
    if timestamp.tzinfo is None:
        raise AutoStabilityPipetteMixContractError(
            '{} must include a timezone.'.format(field_name)
        )
    return timestamp.astimezone(datetime.timezone.utc)


def validate_targeted_mix_request(request):
    '''Normalize and validate the exact one-well packet accepted by Auto-main.'''
    required = {
        'schema_version', 'action_id', 'wellname', 'expected_deck_pos',
        'expected_loc', 'expected_plate_mapping_revision',
        'expected_plate_generation', 'trigger_chemical_name',
        'mix_volume_uL', 'cycle_count'
    }
    if not isinstance(request, dict) or set(request) != required:
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability-mix request has an invalid schema.'
        )
    if request['schema_version'] != TARGETED_MIX_SCHEMA_VERSION:
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability-mix request uses an unsupported schema.'
        )

    normalized = {
        'schema_version': TARGETED_MIX_SCHEMA_VERSION,
        'action_id': _require_nonempty_string(request['action_id'], 'action_id'),
        'wellname': _require_nonempty_string(request['wellname'], 'wellname'),
        'expected_deck_pos': _require_nonnegative_integer(
            request['expected_deck_pos'], 'expected_deck_pos'
        ),
        'expected_loc': _require_nonempty_string(
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
        'trigger_chemical_name': _require_nonempty_string(
            request['trigger_chemical_name'], 'trigger_chemical_name'
        ),
        'mix_volume_uL': _require_positive_finite_number(
            request['mix_volume_uL'], 'mix_volume_uL'
        ),
        'cycle_count': _require_nonnegative_integer(
            request['cycle_count'], 'cycle_count'
        )
    }
    if not normalized['wellname'].startswith('autowell'):
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability mix may address Auto product wells only.'
        )
    if normalized['expected_deck_pos'] not in (4, 7):
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability mix must address plate-reader deck position 4 or 7.'
        )
    if normalized['cycle_count'] < 1 or normalized['cycle_count'] > 10:
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability-mix cycle_count must be from 1 through 10.'
        )
    return normalized


def validate_targeted_mix_acknowledgement(acknowledgement, request):
    '''Validate a Pi acknowledgement and require a completed physical mix.

    An explicit Pi rejection is structurally valid but still raises.  The
    controller records that response and terminates rather than resending the
    request.  A malformed response is rejected before it can be mistaken for
    evidence of a completed mix.
    '''
    request = validate_targeted_mix_request(request)
    required = {
        'schema_version', 'record_type', 'action_id', 'accepted', 'message',
        'wellname', 'expected_deck_pos', 'expected_loc',
        'trigger_chemical_name', 'plate_mapping_revision', 'plate_generation',
        'pipette_arm', 'mix_volume_uL', 'cycle_count', 'required_new_tips',
        'available_new_tips', 'tip_policy', 'physical_execution_started',
        'started_at_utc', 'completed_at_utc'
    }
    if not isinstance(acknowledgement, dict) or set(acknowledgement) != required:
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability-mix acknowledgement has an invalid schema.'
        )
    if acknowledgement['schema_version'] != TARGETED_MIX_SCHEMA_VERSION or \
            acknowledgement['record_type'] != TARGETED_MIX_RECORD_TYPE:
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability-mix acknowledgement uses an unsupported record.'
        )
    if not isinstance(acknowledgement['accepted'], bool) or \
            not isinstance(acknowledgement['physical_execution_started'], bool):
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability-mix acknowledgement has invalid status fields.'
        )
    _require_nonempty_string(acknowledgement['message'], 'acknowledgement.message')

    matching_fields = (
        'action_id', 'wellname', 'expected_deck_pos', 'expected_loc',
        'trigger_chemical_name'
    )
    for field_name in matching_fields:
        expected = request[field_name]
        received = acknowledgement[field_name]
        if field_name == 'expected_loc' and isinstance(received, str):
            received = received.strip().upper()
        if received != expected:
            raise AutoStabilityPipetteMixContractError(
                'Targeted stability-mix acknowledgement {} does not match '
                'the request.'.format(field_name)
            )
    if acknowledgement['plate_mapping_revision'] != request[
            'expected_plate_mapping_revision'] or acknowledgement[
                'plate_generation'] != request['expected_plate_generation']:
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability-mix acknowledgement plate identity does not '
            'match the request.'
        )

    if not acknowledgement['accepted']:
        if acknowledgement['physical_execution_started']:
            raise AutoStabilityPipetteMixContractError(
                'Targeted stability-mix acknowledgement rejected after '
                'physical execution began.'
            )
        if any(acknowledgement[field_name] is not None for field_name in (
                'pipette_arm', 'mix_volume_uL', 'cycle_count',
                'required_new_tips', 'available_new_tips', 'started_at_utc',
                'completed_at_utc')):
            raise AutoStabilityPipetteMixContractError(
                'Rejected targeted stability-mix acknowledgement contains '
                'unexpected execution details.'
            )
        if acknowledgement['tip_policy'] != TARGETED_MIX_TIP_POLICY:
            raise AutoStabilityPipetteMixContractError(
                'Rejected targeted stability-mix acknowledgement changed the '
                'tip policy.'
            )
        return dict(acknowledgement)

    if not acknowledgement['physical_execution_started']:
        raise AutoStabilityPipetteMixContractError(
            'Accepted targeted stability-mix acknowledgement does not confirm '
            'physical execution.'
        )
    if acknowledgement['pipette_arm'] not in ('left', 'right'):
        raise AutoStabilityPipetteMixContractError(
            'Accepted targeted stability-mix acknowledgement has an invalid '
            'pipette arm.'
        )
    acknowledged_volume = _require_positive_finite_number(
        acknowledgement['mix_volume_uL'], 'acknowledgement.mix_volume_uL'
    )
    if not math.isclose(
            acknowledged_volume, request['mix_volume_uL'], rel_tol=0,
            abs_tol=1e-9):
        raise AutoStabilityPipetteMixContractError(
            'Accepted targeted stability-mix acknowledgement volume does not '
            'match the request.'
        )
    if acknowledgement['cycle_count'] != request['cycle_count']:
        raise AutoStabilityPipetteMixContractError(
            'Accepted targeted stability-mix acknowledgement cycle count does '
            'not match the request.'
        )
    for field_name in ('required_new_tips', 'available_new_tips'):
        _require_nonnegative_integer(acknowledgement[field_name], field_name)
    if acknowledgement['available_new_tips'] < acknowledgement['required_new_tips']:
        raise AutoStabilityPipetteMixContractError(
            'Accepted targeted stability-mix acknowledgement reports an '
            'impossible tip count.'
        )
    if acknowledgement['tip_policy'] != TARGETED_MIX_TIP_POLICY:
        raise AutoStabilityPipetteMixContractError(
            'Accepted targeted stability-mix acknowledgement changed the '
            'required dedicated-tip policy.'
        )
    started_at = _require_utc_timestamp(
        acknowledgement['started_at_utc'], 'started_at_utc'
    )
    completed_at = _require_utc_timestamp(
        acknowledgement['completed_at_utc'], 'completed_at_utc'
    )
    if completed_at < started_at:
        raise AutoStabilityPipetteMixContractError(
            'Targeted stability-mix completion precedes its start.'
        )
    return dict(acknowledgement)
