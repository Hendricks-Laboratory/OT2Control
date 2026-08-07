'''Source-level Stage 10B controller-fault contract checks.

The controller imports robot, plate-reader, and credential-capable modules, so
these checks deliberately inspect its AST rather than importing it.  They guard
the placement and fail-closed boundaries of the controller integration while
the pure evidence and lifecycle contracts remain separately unit tested.
'''

import ast
import copy
import os
import tempfile
import textwrap
import unittest
import uuid
import traceback

import pandas as pd

from auto_live_run_faults import (
    AutoLiveRunFaultError,
    AutoLiveRunFaultEvidenceWriter,
    build_fault_record,
    classify_fault_lifecycle
)
from auto_live_run_journal import AutoLiveRunJournal
from auto_live_run_state import (
    LIFECYCLE_EXECUTING_BATCH,
    LIFECYCLE_FAULTED_PARTIAL_BATCH,
    LIFECYCLE_PREFLIGHTING_BATCH
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


class AutoLiveRunFaultControllerContractTests(unittest.TestCase):
    '''Ensures Stage 10B never converts a partial-batch fault into completion.'''

    @classmethod
    def setUpClass(cls):
        with open(CONTROLLER_PATH, 'r', encoding='utf-8') as source_file:
            cls.source = source_file.read()
        tree = ast.parse(cls.source, filename=CONTROLLER_PATH)
        cls.auto_controller = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
        )
        cls.methods = {
            node.name: node
            for node in cls.auto_controller.body
            if isinstance(node, ast.FunctionDef)
        }

    def _method_source(self, name):
        return ast.get_source_segment(self.source, self.methods[name])

    def test_fault_methods_exist_once(self):
        for method_name in (
                '_get_auto_live_run_fault_descriptor',
                '_get_auto_live_run_fault_protocol_evidence',
                '_record_auto_live_run_fault',
                '_error_handler'):
            self.assertIn(method_name, self.methods)
            self.assertEqual(1, sum(
                1 for node in self.auto_controller.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            ))

    def test_fault_handler_bypasses_normal_completion_save(self):
        handler_source = self._method_source('_error_handler')
        self.assertIn('_record_auto_live_run_fault(error)', handler_source)
        self.assertIn('raise error', handler_source)
        self.assertIn('super()._error_handler(error)', handler_source)
        self.assertNotIn('self.close_connection(', handler_source)
        self.assertNotIn('_finalize_auto_run(', handler_source)

    def test_fault_record_precedes_no_automatic_resume(self):
        record_source = self._method_source('_record_auto_live_run_fault')
        self.assertIn('AutoLiveRunFaultEvidenceWriter.write(', record_source)
        self.assertIn("event_type='fault_recorded'", record_source)
        self.assertIn('journal.record_transition(', record_source)
        self.assertIn('human_review_required_no_automatic_resume', record_source)
        self.assertNotIn('_refresh_auto_live_run_mirror(', record_source)
        self.assertNotIn('_save_auto_model_checkpoint(', record_source)
        self.assertNotIn('_finalize_auto_run(', record_source)

    def test_preparation_marker_precedes_physical_request(self):
        phase_source = self._method_source('_execute_auto_preparation_phase')
        marker = phase_source.index(
            '_auto_live_run_preparation_execution_may_have_started = True'
        )
        execution = phase_source.index(
            '_request_auto_preparation_group_execution('
        )
        clear_marker = phase_source.index(
            '_auto_live_run_preparation_execution_may_have_started = False'
        )
        self.assertLess(marker, execution)
        self.assertGreater(clear_marker, execution)

    def test_planned_protocol_is_saved_before_execution_transition(self):
        sample_source = self._method_source('_create_samples')
        context = sample_source.index('_auto_live_run_active_batch_context =')
        transition = sample_source.index('LIFECYCLE_EXECUTING_BATCH')
        execution = sample_source.index('self.execute_protocol_df(model)')
        self.assertLess(context, transition)
        self.assertLess(transition, execution)


class AutoLiveRunFaultControllerIntegrationTests(unittest.TestCase):
    '''Exercises extracted Stage 10B methods without hardware imports.'''

    @classmethod
    def setUpClass(cls):
        with open(CONTROLLER_PATH, 'r', encoding='utf-8') as source_file:
            source = source_file.read()
        tree = ast.parse(source, filename=CONTROLLER_PATH)
        auto_controller = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
        )
        requested_names = {
            '_get_auto_live_run_fault_descriptor',
            '_get_auto_live_run_fault_protocol_evidence',
            '_record_auto_live_run_fault'
        }
        namespace = {
            'copy': copy,
            'traceback': traceback,
            'uuid': uuid,
            'AutoLiveRunFaultError': AutoLiveRunFaultError,
            'AutoLiveRunFaultEvidenceWriter': AutoLiveRunFaultEvidenceWriter,
            'build_fault_record': build_fault_record,
            'classify_fault_lifecycle': classify_fault_lifecycle
        }
        methods = {}
        for node in auto_controller.body:
            if not isinstance(node, ast.FunctionDef) or node.name not in requested_names:
                continue
            exec(
                textwrap.dedent(ast.get_source_segment(source, node)),
                namespace
            )
            methods[node.name] = namespace[node.name]
        cls.FaultController = type('FaultController', (), methods)

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.journal = AutoLiveRunJournal.initialize(
            run_state_directory=os.path.join(
                self.temporary_directory.name,
                'Run_State'
            ),
            run_id='stage-10b-controller-test',
            input_snapshot={'worksheet': 'input template'},
            header_snapshot={'target': 625.0},
            runtime_baseline={'branch': 'Auto-RTG'},
            git_branch='Auto-RTG',
            git_commit='0' * 40,
            created_at_utc='2026-08-07T12:00:00+00:00'
        )
        self.journal.record_transition(
            LIFECYCLE_PREFLIGHTING_BATCH,
            'batch_preflight_requested',
            {'batch_number': 0},
            active_batch_number=0
        )
        self.journal.record_transition(
            LIFECYCLE_EXECUTING_BATCH,
            'batch_execution_started',
            {'batch_number': 0},
            active_batch_number=0
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_batch_fault_writes_evidence_then_terminal_state(self):
        controller = self.FaultController()
        controller.auto_live_run_journal = self.journal
        controller.auto_live_run_fault_handled = False
        controller._auto_live_run_preparation_execution_may_have_started = False
        controller._auto_live_run_active_batch_context = {
            'batch_number': 0,
            'physical_well_count': 1,
            'wellnames': ['A1'],
            'planned_protocol_filename': 'planned_protocol_dataframe.csv'
        }
        controller._auto_live_run_active_protocol_dataframe = pd.DataFrame([
            {'op': 'transfer', 'chemical_name': 'silver_nitrate', 'A1': 10.0}
        ])

        record = controller._record_auto_live_run_fault(
            ConnectionError('robot acknowledgement lost')
        )

        self.assertTrue(controller.auto_live_run_fault_handled)
        self.assertEqual(
            LIFECYCLE_FAULTED_PARTIAL_BATCH,
            self.journal.current_state['lifecycle_state']
        )
        self.assertEqual(
            record['fault_record']['fault_id'],
            self.journal.current_state['fault_id']
        )
        self.assertTrue(os.path.isfile(os.path.join(
            record['evidence_directory'],
            'planned_protocol_dataframe.csv'
        )))
        with open(self.journal.events_path, 'r', encoding='utf-8') as events_file:
            final_event = events_file.read().strip().splitlines()[-1]
        self.assertIn('"event_type":"fault_recorded"', final_event)


if __name__ == '__main__':
    unittest.main()
