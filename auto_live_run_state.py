'''Pure, versioned contract for Auto live-run recovery records.

This module deliberately contains no controller, robot, filesystem, cloud,
or workbook integration.  It defines the serializable records and permitted
lifecycle transitions that later recovery stages must use.  Keeping the
contract pure allows it to be tested without importing hardware-connected
code and prevents a partially implemented recovery feature from affecting an
Auto run.
'''

import copy
import datetime


LIVE_RUN_STATE_SCHEMA_VERSION = 1

MANIFEST_RECORD_TYPE = 'auto_live_run_manifest'
CURRENT_STATE_RECORD_TYPE = 'auto_live_run_current_state'
EVENT_RECORD_TYPE = 'auto_live_run_event'


LIFECYCLE_CREATED = 'created'
LIFECYCLE_READY_FOR_BATCH = 'ready_for_batch'
LIFECYCLE_PREFLIGHTING_BATCH = 'preflighting_batch'
LIFECYCLE_EXECUTING_BATCH = 'executing_batch'
LIFECYCLE_MEASURING_BATCH = 'measuring_batch'
LIFECYCLE_PROCESSING_BATCH = 'processing_batch'
LIFECYCLE_HELD_FOR_OPERATOR = 'held_for_operator'
# Preparation can physically begin before an experimental batch exists.  It
# therefore needs a separate terminal fault state rather than incorrectly
# claiming that a numbered batch was only partially completed.
LIFECYCLE_FAULTED_PREPARATION = 'faulted_preparation'
LIFECYCLE_FAULTED_PARTIAL_BATCH = 'faulted_partial_batch'
LIFECYCLE_FINALIZED = 'finalized'

LIFECYCLE_STATES = frozenset({
    LIFECYCLE_CREATED,
    LIFECYCLE_READY_FOR_BATCH,
    LIFECYCLE_PREFLIGHTING_BATCH,
    LIFECYCLE_EXECUTING_BATCH,
    LIFECYCLE_MEASURING_BATCH,
    LIFECYCLE_PROCESSING_BATCH,
    LIFECYCLE_HELD_FOR_OPERATOR,
    LIFECYCLE_FAULTED_PREPARATION,
    LIFECYCLE_FAULTED_PARTIAL_BATCH,
    LIFECYCLE_FINALIZED
})

# These states describe a run that must never silently resume.  A final
# archival transition remains permitted, but no same-phase durable milestone
# may reopen a faulted or finalized run.
TERMINAL_LIFECYCLE_STATES = frozenset({
    LIFECYCLE_FAULTED_PREPARATION,
    LIFECYCLE_FAULTED_PARTIAL_BATCH,
    LIFECYCLE_FINALIZED
})

ACTIVE_BATCH_REQUIRED_STATES = frozenset({
    LIFECYCLE_PREFLIGHTING_BATCH,
    LIFECYCLE_EXECUTING_BATCH,
    LIFECYCLE_MEASURING_BATCH,
    LIFECYCLE_PROCESSING_BATCH,
    LIFECYCLE_HELD_FOR_OPERATOR,
    LIFECYCLE_FAULTED_PARTIAL_BATCH
})


ALLOWED_LIFECYCLE_TRANSITIONS = {
    LIFECYCLE_CREATED: frozenset({LIFECYCLE_READY_FOR_BATCH}),
    LIFECYCLE_READY_FOR_BATCH: frozenset({
        LIFECYCLE_PREFLIGHTING_BATCH,
        # A completed batch may leave too little physical plate capacity for
        # the next unchanged batch.  That is a pre-execution operator hold,
        # not a partial batch fault.
        LIFECYCLE_HELD_FOR_OPERATOR,
        # Auto preparation may have begun before the first batch exists.  A
        # disconnect or unacknowledged failure there must be terminal and
        # must not fabricate an active experimental batch number.
        LIFECYCLE_FAULTED_PREPARATION,
        LIFECYCLE_FINALIZED
    }),
    LIFECYCLE_PREFLIGHTING_BATCH: frozenset({
        LIFECYCLE_EXECUTING_BATCH,
        LIFECYCLE_HELD_FOR_OPERATOR,
        LIFECYCLE_FINALIZED
    }),
    LIFECYCLE_EXECUTING_BATCH: frozenset({
        LIFECYCLE_MEASURING_BATCH,
        LIFECYCLE_FAULTED_PARTIAL_BATCH
    }),
    LIFECYCLE_MEASURING_BATCH: frozenset({
        LIFECYCLE_PROCESSING_BATCH,
        LIFECYCLE_FAULTED_PARTIAL_BATCH
    }),
    LIFECYCLE_PROCESSING_BATCH: frozenset({
        LIFECYCLE_READY_FOR_BATCH,
        LIFECYCLE_FINALIZED,
        LIFECYCLE_FAULTED_PARTIAL_BATCH
    }),
    LIFECYCLE_HELD_FOR_OPERATOR: frozenset({
        LIFECYCLE_PREFLIGHTING_BATCH,
        LIFECYCLE_FINALIZED
    }),
    LIFECYCLE_FAULTED_PREPARATION: frozenset({LIFECYCLE_FINALIZED}),
    LIFECYCLE_FAULTED_PARTIAL_BATCH: frozenset({LIFECYCLE_FINALIZED}),
    LIFECYCLE_FINALIZED: frozenset()
}


EVENT_TYPES = frozenset({
    'run_initialized',
    'input_snapshot_created',
    # A successful Pi compatibility assertion is durable provenance, not a
    # batch lifecycle transition. It is recorded after the controller connects
    # and before any recipe can be executed.
    'auto_main_compatibility_validated',
    # Grouped Auto-preparation milestones are durable provenance records
    # emitted before an experimental batch is dispatched.
    'auto_preparation_groups_reserved',
    'auto_preparation_groups_executed',
    'auto_preparation_sources_activated',
    'batch_preflight_requested',
    'batch_preflight_validated',
    'batch_preflight_rejected',
    'batch_execution_started',
    'batch_transfer_completed',
    'batch_measurement_completed',
    'batch_model_update_completed',
    'batch_completed',
    'hold_entered',
    # Stage 11C2 records a request and any rejected workbook response before
    # the pre-existing terminal/Pi recovery path can act on a valid response.
    'operator_action_workbook_activated',
    'operator_action_workbook_response_rejected',
    'operator_action_requested',
    'operator_action_rejected',
    'operator_action_applied',
    'fault_recorded',
    'cloud_sync_queued',
    'cloud_sync_completed',
    'cloud_sync_failed',
    'run_finalized'
})


OPERATOR_ACTION_TYPES = frozenset({
    # Stage 6: reweigh/refill only an already registered source at its
    # existing physical identity. New sources remain a later Stage 7 action.
    'refill_same_container',
    'replace_source',
    'register_backup_source',
    'replace_tip_rack',
    'replace_wellplate',
    'retry_cloud_sync'
})


class LiveRunStateContractError(ValueError):
    '''Raised when a proposed live-run record violates schema version 1.'''


def _require_mapping(record, record_name):
    if not isinstance(record, dict):
        raise LiveRunStateContractError(
            '{} must be a dictionary.'.format(record_name)
        )


def _require_exact_keys(record, expected_keys, record_name):
    actual_keys = set(record.keys())
    expected_keys = set(expected_keys)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)

    if missing or unexpected:
        details = []
        if missing:
            details.append('missing {}'.format(', '.join(missing)))
        if unexpected:
            details.append('unexpected {}'.format(', '.join(unexpected)))
        raise LiveRunStateContractError(
            '{} has {}.'.format(record_name, '; '.join(details))
        )


def _require_nonempty_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise LiveRunStateContractError(
            '{} must be a nonempty string.'.format(field_name)
        )


def _require_nonnegative_integer(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LiveRunStateContractError(
            '{} must be a nonnegative integer.'.format(field_name)
        )


def _require_utc_timestamp(value, field_name):
    _require_nonempty_string(value, field_name)
    try:
        parsed = datetime.datetime.fromisoformat(
            value.replace('Z', '+00:00')
        )
    except ValueError:
        raise LiveRunStateContractError(
            '{} must be an ISO-8601 timestamp.'.format(field_name)
        )

    if parsed.tzinfo is None:
        raise LiveRunStateContractError(
            '{} must include a UTC offset.'.format(field_name)
        )

    if parsed.utcoffset() != datetime.timedelta(0):
        raise LiveRunStateContractError(
            '{} must be expressed in UTC.'.format(field_name)
        )


def _require_sha256(value, field_name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise LiveRunStateContractError(
            '{} must be a lowercase SHA-256 hex digest.'.format(field_name)
        )


def _validate_common_record_fields(record, record_type, record_name):
    if record.get('schema_version') != LIVE_RUN_STATE_SCHEMA_VERSION:
        raise LiveRunStateContractError(
            '{} must use schema_version {}.'.format(
                record_name,
                LIVE_RUN_STATE_SCHEMA_VERSION
            )
        )

    if record.get('record_type') != record_type:
        raise LiveRunStateContractError(
            '{} has an invalid record_type.'.format(record_name)
        )

    _require_nonempty_string(record.get('run_id'), '{}.run_id'.format(record_name))


def validate_run_manifest(manifest):
    '''Validates one immutable run manifest without modifying it.'''
    _require_mapping(manifest, 'run manifest')
    _require_exact_keys(
        manifest,
        {
            'schema_version',
            'record_type',
            'run_id',
            'created_at_utc',
            'input_snapshot_sha256',
            'header_snapshot_sha256',
            'runtime_baseline_sha256',
            'git_branch',
            'git_commit'
        },
        'run manifest'
    )
    _validate_common_record_fields(
        manifest,
        MANIFEST_RECORD_TYPE,
        'run manifest'
    )
    _require_utc_timestamp(manifest['created_at_utc'], 'run manifest.created_at_utc')

    for field_name in (
        'input_snapshot_sha256',
        'header_snapshot_sha256',
        'runtime_baseline_sha256'
    ):
        _require_sha256(
            manifest[field_name],
            'run manifest.{}'.format(field_name)
        )

    for field_name in ('git_branch', 'git_commit'):
        _require_nonempty_string(
            manifest[field_name],
            'run manifest.{}'.format(field_name)
        )


def validate_current_state(state):
    '''Validates one mutable current-state record without modifying it.'''
    _require_mapping(state, 'current state')
    _require_exact_keys(
        state,
        {
            'schema_version',
            'record_type',
            'run_id',
            'revision',
            'lifecycle_state',
            'active_batch_number',
            'last_event_sequence',
            'hold_action_id',
            'fault_id'
        },
        'current state'
    )
    _validate_common_record_fields(
        state,
        CURRENT_STATE_RECORD_TYPE,
        'current state'
    )
    _require_nonnegative_integer(state['revision'], 'current state.revision')
    _require_nonnegative_integer(
        state['last_event_sequence'],
        'current state.last_event_sequence'
    )

    lifecycle_state = state['lifecycle_state']
    if lifecycle_state not in LIFECYCLE_STATES:
        raise LiveRunStateContractError(
            'current state.lifecycle_state is invalid: {!r}.'.format(
                lifecycle_state
            )
        )

    active_batch_number = state['active_batch_number']
    if active_batch_number is not None:
        _require_nonnegative_integer(
            active_batch_number,
            'current state.active_batch_number'
        )

    if (
        lifecycle_state in ACTIVE_BATCH_REQUIRED_STATES
        and active_batch_number is None
    ):
        raise LiveRunStateContractError(
            '{} requires active_batch_number.'.format(lifecycle_state)
        )

    if (
        lifecycle_state in {
            LIFECYCLE_CREATED,
            LIFECYCLE_READY_FOR_BATCH,
            LIFECYCLE_FAULTED_PREPARATION
        }
        and active_batch_number is not None
    ):
        raise LiveRunStateContractError(
            '{} must not retain active_batch_number.'.format(lifecycle_state)
        )

    for field_name in ('hold_action_id', 'fault_id'):
        value = state[field_name]
        if value is not None:
            _require_nonempty_string(
                value,
                'current state.{}'.format(field_name)
            )

    if (
        lifecycle_state == LIFECYCLE_HELD_FOR_OPERATOR
        and state['hold_action_id'] is None
    ):
        raise LiveRunStateContractError(
            'held_for_operator requires a hold_action_id.'
        )

    if (
        lifecycle_state in {
            LIFECYCLE_FAULTED_PREPARATION,
            LIFECYCLE_FAULTED_PARTIAL_BATCH
        }
        and state['fault_id'] is None
    ):
        raise LiveRunStateContractError(
            '{} requires a fault_id.'.format(lifecycle_state)
        )


def validate_event(event):
    '''Validates one append-only event record without modifying it.'''
    _require_mapping(event, 'event')
    _require_exact_keys(
        event,
        {
            'schema_version',
            'record_type',
            'run_id',
            'sequence',
            'timestamp_utc',
            'event_type',
            'state_revision',
            'payload'
        },
        'event'
    )
    _validate_common_record_fields(event, EVENT_RECORD_TYPE, 'event')
    _require_nonnegative_integer(event['sequence'], 'event.sequence')
    if event['sequence'] == 0:
        raise LiveRunStateContractError('event.sequence must be at least 1.')
    _require_utc_timestamp(event['timestamp_utc'], 'event.timestamp_utc')
    _require_nonnegative_integer(
        event['state_revision'],
        'event.state_revision'
    )

    if event['event_type'] not in EVENT_TYPES:
        raise LiveRunStateContractError(
            'event.event_type is invalid: {!r}.'.format(event['event_type'])
        )

    if not isinstance(event['payload'], dict):
        raise LiveRunStateContractError('event.payload must be a dictionary.')


def assert_valid_lifecycle_transition(previous_state, next_state):
    '''Checks one revisioned state transition without performing any I/O.

    A state transition must preserve run identity, increment the revision by
    exactly one, retain the prior event sequence or advance it, and follow the
    conservative lifecycle matrix above.  A finalized or partial-fault state
    cannot resume into an execution state.
    '''
    validate_current_state(previous_state)
    validate_current_state(next_state)

    if previous_state['run_id'] != next_state['run_id']:
        raise LiveRunStateContractError(
            'A lifecycle transition cannot change run_id.'
        )

    if next_state['revision'] != previous_state['revision'] + 1:
        raise LiveRunStateContractError(
            'A lifecycle transition must increment revision by exactly one.'
        )

    if (
        next_state['last_event_sequence']
        < previous_state['last_event_sequence']
    ):
        raise LiveRunStateContractError(
            'A lifecycle transition cannot decrease last_event_sequence.'
        )

    previous_lifecycle = previous_state['lifecycle_state']
    next_lifecycle = next_state['lifecycle_state']

    # A durable event may advance the state revision and event sequence
    # without changing lifecycle phase. This is required for auditable
    # milestones such as a successful preflight while the run remains in the
    # preflighting phase. It does not permit execution to resume from a fault.
    if previous_lifecycle == next_lifecycle:
        if previous_lifecycle in TERMINAL_LIFECYCLE_STATES:
            raise LiveRunStateContractError(
                '{} is terminal and cannot accept another lifecycle record.'.format(
                    previous_lifecycle
                )
            )
        return

    if next_lifecycle not in ALLOWED_LIFECYCLE_TRANSITIONS[
        previous_lifecycle
    ]:
        raise LiveRunStateContractError(
            'Invalid lifecycle transition: {} -> {}.'.format(
                previous_lifecycle,
                next_lifecycle
            )
        )


def make_initial_current_state(run_id):
    '''Builds the only valid revision-zero state for a new future run.

    Stage 0 exposes this pure helper for fixture construction. Stage 1 may use
    it when it adds durable writes, but this helper performs no write itself.
    '''
    state = {
        'schema_version': LIVE_RUN_STATE_SCHEMA_VERSION,
        'record_type': CURRENT_STATE_RECORD_TYPE,
        'run_id': run_id,
        'revision': 0,
        'lifecycle_state': LIFECYCLE_CREATED,
        'active_batch_number': None,
        'last_event_sequence': 0,
        'hold_action_id': None,
        'fault_id': None
    }
    validate_current_state(state)
    return copy.deepcopy(state)
