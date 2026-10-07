'''Hardware-free durability tests for the Auto Stage 1 live-run journal.'''

import ast
import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock

from auto_live_run_journal import (
    AutoLiveRunJournal,
    AutoLiveRunJournalError
)
from auto_live_run_state import (
    LIFECYCLE_EXECUTING_BATCH,
    LIFECYCLE_HELD_FOR_OPERATOR,
    LIFECYCLE_MEASURING_BATCH,
    LIFECYCLE_PREFLIGHTING_BATCH,
    LIFECYCLE_PROCESSING_BATCH,
    LIFECYCLE_READY_FOR_BATCH
)


class AutoLiveRunJournalTests(unittest.TestCase):
    '''Exercises local journal writes without importing controller hardware.''' 

    RUN_ID = 'RTG_018_5D-test-run'

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.run_state_directory = os.path.join(
            self.temporary_directory.name,
            'Run_State'
        )
        self.input_snapshot = {
            'worksheet_name': 'RTG_018_5D',
            'parsed_input_template': {'columns': ['reagent'], 'data': []}
        }
        self.header_snapshot = {
            'worksheet_name': 'RTG_018_5D',
            'header_rows': [['Header', 'Comment']]
        }
        self.runtime_baseline = {
            'variable_reagents': ['silver_nitrate'],
            'num_duplicates': 3
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _canonical_sha256(value):
        payload = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False
        ) + '\n'
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _initialize_journal(self):
        return AutoLiveRunJournal.initialize(
            run_state_directory=self.run_state_directory,
            run_id=self.RUN_ID,
            input_snapshot=self.input_snapshot,
            header_snapshot=self.header_snapshot,
            runtime_baseline=self.runtime_baseline,
            git_branch='Auto-RTG',
            git_commit='0123456789abcdef',
            created_at_utc='2026-07-29T12:34:56+00:00'
        )

    def _read_json(self, filename):
        with open(
            os.path.join(self.run_state_directory, filename),
            'r',
            encoding='utf-8'
        ) as input_file:
            return json.load(input_file)

    def _read_events(self):
        with open(
            os.path.join(
                self.run_state_directory,
                AutoLiveRunJournal.EVENTS_FILENAME
            ),
            'r',
            encoding='utf-8'
        ) as input_file:
            return [
                json.loads(line)
                for line in input_file
                if line.strip()
            ]

    def test_initialize_writes_hashed_snapshots_and_ready_state(self):
        journal = self._initialize_journal()
        manifest = self._read_json(AutoLiveRunJournal.MANIFEST_FILENAME)
        current_state = self._read_json(AutoLiveRunJournal.CURRENT_STATE_FILENAME)
        events = self._read_events()

        self.assertEqual(
            manifest['input_snapshot_sha256'],
            self._canonical_sha256(self.input_snapshot)
        )
        self.assertEqual(
            manifest['header_snapshot_sha256'],
            self._canonical_sha256(self.header_snapshot)
        )
        self.assertEqual(
            manifest['runtime_baseline_sha256'],
            self._canonical_sha256(self.runtime_baseline)
        )
        self.assertEqual(
            self._read_json(AutoLiveRunJournal.INPUT_SNAPSHOT_FILENAME),
            self.input_snapshot
        )
        self.assertEqual(current_state['lifecycle_state'], LIFECYCLE_READY_FOR_BATCH)
        self.assertEqual(current_state['revision'], 2)
        self.assertEqual(current_state['last_event_sequence'], 2)
        self.assertEqual(journal.current_state, current_state)
        self.assertEqual(
            [event['event_type'] for event in events],
            ['run_initialized', 'input_snapshot_created']
        )

    def test_batch_lifecycle_is_recorded_in_order(self):
        journal = self._initialize_journal()
        batch_payload = {'batch_number': 0, 'physical_well_count': 3}

        journal.record_transition(
            LIFECYCLE_PREFLIGHTING_BATCH,
            'batch_preflight_requested',
            batch_payload,
            active_batch_number=0
        )
        journal.record_event('batch_preflight_validated', batch_payload)
        journal.record_transition(
            LIFECYCLE_EXECUTING_BATCH,
            'batch_execution_started',
            batch_payload,
            active_batch_number=0
        )
        journal.record_transition(
            LIFECYCLE_MEASURING_BATCH,
            'batch_transfer_completed',
            batch_payload,
            active_batch_number=0
        )
        journal.record_transition(
            LIFECYCLE_PROCESSING_BATCH,
            'batch_measurement_completed',
            batch_payload,
            active_batch_number=0
        )
        journal.record_transition(
            LIFECYCLE_READY_FOR_BATCH,
            'batch_completed',
            {'batch_number': 0},
            active_batch_number=None
        )

        current_state = self._read_json(AutoLiveRunJournal.CURRENT_STATE_FILENAME)
        events = self._read_events()
        self.assertEqual(current_state['lifecycle_state'], LIFECYCLE_READY_FOR_BATCH)
        self.assertIsNone(current_state['active_batch_number'])
        self.assertEqual(
            [event['sequence'] for event in events],
            list(range(1, len(events) + 1))
        )
        self.assertEqual(
            [event['event_type'] for event in events[-6:]],
            [
                'batch_preflight_requested',
                'batch_preflight_validated',
                'batch_execution_started',
                'batch_transfer_completed',
                'batch_measurement_completed',
                'batch_completed'
            ]
        )

    def test_ready_batch_can_enter_a_durable_plate_replacement_hold(self):
        '''A full plate is a pre-execution hold, not an execution fault.'''
        journal = self._initialize_journal()
        hold_action_id = 'replace-plate-action-1'

        journal.record_transition(
            LIFECYCLE_HELD_FOR_OPERATOR,
            'hold_entered',
            {
                'hold_action_id': hold_action_id,
                'hold_reason': 'insufficient_remaining_plate_capacity'
            },
            active_batch_number=1,
            hold_action_id=hold_action_id
        )
        journal.record_event(
            'operator_action_applied',
            {
                'hold_action_id': hold_action_id,
                'applied_action': 'replace_wellplate'
            }
        )
        journal.record_transition(
            LIFECYCLE_PREFLIGHTING_BATCH,
            'batch_preflight_requested',
            {
                'batch_number': 1,
                'hold_action_id': hold_action_id,
                'preflight_retry_reason': 'accepted_wellplate_replacement'
            },
            active_batch_number=1
        )

        current_state = self._read_json(AutoLiveRunJournal.CURRENT_STATE_FILENAME)
        self.assertEqual(
            LIFECYCLE_PREFLIGHTING_BATCH,
            current_state['lifecycle_state']
        )
        self.assertEqual(1, current_state['active_batch_number'])
        self.assertIsNone(current_state['hold_action_id'])
        self.assertEqual(
            ['hold_entered', 'operator_action_applied', 'batch_preflight_requested'],
            [event['event_type'] for event in self._read_events()[-3:]]
        )

    def test_current_state_is_not_replaced_if_atomic_state_write_fails(self):
        journal = self._initialize_journal()
        state_before_failure = self._read_json(
            AutoLiveRunJournal.CURRENT_STATE_FILENAME
        )

        with mock.patch(
            'auto_live_run_journal.os.replace',
            side_effect=OSError('simulated state write failure')
        ):
            with self.assertRaises(AutoLiveRunJournalError):
                journal.record_event('batch_preflight_validated', {})

        self.assertEqual(
            self._read_json(AutoLiveRunJournal.CURRENT_STATE_FILENAME),
            state_before_failure
        )
        self.assertEqual(journal.current_state, state_before_failure)
        # The append-only event was fsynced first. A later recovery stage can
        # reconcile this conservative partial journal; Stage 1 fails closed.
        self.assertEqual(
            self._read_events()[-1]['event_type'],
            'batch_preflight_validated'
        )

    def test_initialize_refuses_to_overwrite_existing_journal_directory(self):
        self._initialize_journal()

        with self.assertRaisesRegex(
            AutoLiveRunJournalError,
            'Refusing to overwrite'
        ):
            self._initialize_journal()


class AutoLiveRunJournalControllerPlacementTests(unittest.TestCase):
    '''Checks Stage 1 hooks statically without importing controller.py.''' 

    @staticmethod
    def _method_node(class_node, method_name):
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return node
        raise AssertionError('AutoContr.{} was not found.'.format(method_name))

    @staticmethod
    def _call_names(method_node):
        names = []
        for node in ast.walk(method_node):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Attribute):
                names.append((function.attr, node.lineno))
            elif isinstance(function, ast.Name):
                names.append((function.id, node.lineno))
        return names

    def test_hooks_are_placed_before_connection_and_execution(self):
        controller_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'controller.py'
        )
        with open(controller_path, 'r', encoding='utf-8') as source_file:
            module = ast.parse(source_file.read(), filename=controller_path)

        auto_controller = next(
            node for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
        )
        run_calls = self._call_names(self._method_node(auto_controller, '_run'))
        create_samples_calls = self._call_names(
            self._method_node(auto_controller, '_create_samples')
        )

        run_call_lines = dict(run_calls)
        self.assertLess(
            run_call_lines['_initialize_auto_live_run_journal'],
            run_call_lines['create_connection']
        )
        sample_call_lines = {}
        for call_name, call_line in create_samples_calls:
            sample_call_lines.setdefault(call_name, []).append(call_line)
        self.assertLess(
            min(sample_call_lines['_record_auto_live_run_transition']),
            sample_call_lines['execute_protocol_df'][0]
        )
        self.assertIn('_record_auto_live_run_event', sample_call_lines)


if __name__ == '__main__':
    unittest.main()
