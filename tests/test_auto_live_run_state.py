'''Hardware-free schema and transition tests for Auto live-run state records.'''

import copy
import unittest

from auto_live_run_state import (
    CURRENT_STATE_RECORD_TYPE,
    EVENT_RECORD_TYPE,
    EVENT_TYPES,
    LIFECYCLE_CREATED,
    LIFECYCLE_EXECUTING_BATCH,
    LIFECYCLE_FAULTED_PARTIAL_BATCH,
    LIFECYCLE_FINALIZED,
    LIFECYCLE_HELD_FOR_OPERATOR,
    LIFECYCLE_READY_FOR_BATCH,
    MANIFEST_RECORD_TYPE,
    LIVE_RUN_STATE_SCHEMA_VERSION,
    LiveRunStateContractError,
    assert_valid_lifecycle_transition,
    make_initial_current_state,
    validate_current_state,
    validate_event,
    validate_run_manifest
)


class AutoLiveRunStateContractTests(unittest.TestCase):
    '''Verifies Stage 0's pure state contract without controller imports.'''

    RUN_ID = 'RTG_018_5D-20260729T123456Z'

    def _manifest(self):
        return {
            'schema_version': LIVE_RUN_STATE_SCHEMA_VERSION,
            'record_type': MANIFEST_RECORD_TYPE,
            'run_id': self.RUN_ID,
            'created_at_utc': '2026-07-29T12:34:56+00:00',
            'input_snapshot_sha256': 'a' * 64,
            'header_snapshot_sha256': 'b' * 64,
            'runtime_baseline_sha256': 'c' * 64,
            'git_branch': 'Auto-RTG',
            'git_commit': '0123456789abcdef'
        }

    def _event(self):
        return {
            'schema_version': LIVE_RUN_STATE_SCHEMA_VERSION,
            'record_type': EVENT_RECORD_TYPE,
            'run_id': self.RUN_ID,
            'sequence': 1,
            'timestamp_utc': '2026-07-29T12:34:57+00:00',
            'event_type': 'run_initialized',
            'state_revision': 0,
            'payload': {}
        }

    def _next_state(self, state, lifecycle_state, **changes):
        next_state = copy.deepcopy(state)
        next_state['revision'] += 1
        next_state['lifecycle_state'] = lifecycle_state
        next_state.update(changes)
        return next_state

    def test_valid_version_one_records_are_accepted(self):
        manifest = self._manifest()
        state = make_initial_current_state(self.RUN_ID)
        event = self._event()

        validate_run_manifest(manifest)
        validate_current_state(state)
        validate_event(event)

        self.assertEqual(state['record_type'], CURRENT_STATE_RECORD_TYPE)
        self.assertEqual(state['lifecycle_state'], LIFECYCLE_CREATED)

    def test_manifest_rejects_unknown_or_missing_top_level_fields(self):
        manifest = self._manifest()
        manifest['unreviewed_field'] = True

        with self.assertRaisesRegex(LiveRunStateContractError, 'unexpected'):
            validate_run_manifest(manifest)

        manifest = self._manifest()
        del manifest['git_commit']
        with self.assertRaisesRegex(LiveRunStateContractError, 'missing'):
            validate_run_manifest(manifest)

    def test_current_state_hold_and_fault_require_identifiers(self):
        state = make_initial_current_state(self.RUN_ID)
        held_state = self._next_state(
            state,
            LIFECYCLE_HELD_FOR_OPERATOR,
            active_batch_number=0
        )

        with self.assertRaisesRegex(LiveRunStateContractError, 'hold_action_id'):
            validate_current_state(held_state)

        faulted_state = self._next_state(
            state,
            LIFECYCLE_FAULTED_PARTIAL_BATCH,
            active_batch_number=0
        )
        with self.assertRaisesRegex(LiveRunStateContractError, 'fault_id'):
            validate_current_state(faulted_state)

    def test_transition_requires_identity_revision_and_permitted_state(self):
        created = make_initial_current_state(self.RUN_ID)
        ready = self._next_state(created, LIFECYCLE_READY_FOR_BATCH)
        assert_valid_lifecycle_transition(created, ready)

        skipped_revision = copy.deepcopy(ready)
        skipped_revision['revision'] += 1
        with self.assertRaisesRegex(LiveRunStateContractError, 'exactly one'):
            assert_valid_lifecycle_transition(created, skipped_revision)

        invalid_state = self._next_state(
            created,
            LIFECYCLE_EXECUTING_BATCH,
            active_batch_number=0
        )
        with self.assertRaisesRegex(LiveRunStateContractError, 'Invalid lifecycle'):
            assert_valid_lifecycle_transition(created, invalid_state)

    def test_same_lifecycle_event_transition_is_revisioned_and_permitted(self):
        created = make_initial_current_state(self.RUN_ID)
        event_state = self._next_state(
            created,
            LIFECYCLE_CREATED,
            last_event_sequence=1
        )

        assert_valid_lifecycle_transition(created, event_state)

    def test_faulted_partial_batch_cannot_resume_execution(self):
        created = make_initial_current_state(self.RUN_ID)
        ready = self._next_state(created, LIFECYCLE_READY_FOR_BATCH)
        faulted = self._next_state(
            ready,
            LIFECYCLE_FAULTED_PARTIAL_BATCH,
            active_batch_number=4,
            fault_id='transfer-interrupted-001'
        )

        # A direct ready-to-fault transition is deliberately invalid too; the
        # fixture is used only to validate the no-resume terminal boundary.
        validate_current_state(faulted)
        resumed = self._next_state(faulted, LIFECYCLE_EXECUTING_BATCH)
        with self.assertRaisesRegex(LiveRunStateContractError, 'Invalid lifecycle'):
            assert_valid_lifecycle_transition(faulted, resumed)

        finalized = self._next_state(faulted, LIFECYCLE_FINALIZED)
        assert_valid_lifecycle_transition(faulted, finalized)

    def test_event_requires_monotonic_nonzero_sequence_and_object_payload(self):
        event = self._event()
        validate_event(event)

        event['sequence'] = 0
        with self.assertRaisesRegex(LiveRunStateContractError, 'at least 1'):
            validate_event(event)

        event = self._event()
        event['payload'] = []
        with self.assertRaisesRegex(LiveRunStateContractError, 'dictionary'):
            validate_event(event)

    def test_auto_main_compatibility_event_is_accepted(self):
        '''The live Pi contract is auditable before any batch transition.'''
        event = self._event()
        event['event_type'] = 'auto_main_compatibility_validated'
        event['payload'] = {
            'runtime_role': 'Auto-main',
            'protocol_version': 'auto-main-state-v1'
        }

        validate_event(event)

    def test_auto_preparation_events_are_accepted(self):
        '''Stage 9 preparation milestones remain valid journal events.'''
        for event_type in (
                'auto_preparation_groups_reserved',
                'auto_preparation_groups_executed',
                'auto_preparation_sources_activated'):
            with self.subTest(event_type=event_type):
                self.assertIn(event_type, EVENT_TYPES)
                event = self._event()
                event['event_type'] = event_type
                validate_event(event)


if __name__ == '__main__':
    unittest.main()
