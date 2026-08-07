'''Hardware-free tests for the Stage 2 Live workbook and sync queue.''' 

import ast
import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from xml.etree import ElementTree

from auto_live_run_journal import AutoLiveRunJournal
from auto_live_run_sync import (
    AutoLiveRunSyncQueue,
    AutoLiveRunSyncQueueError
)
from auto_live_run_state import LIFECYCLE_FAULTED_PREPARATION
from auto_live_run_workbook import AutoLiveRunWorkbookRenderer


class _RecordingAdapter:
    '''Small fake cloud endpoint used only by isolated queue tests.'''

    def __init__(self, fail_at_queue_id=None):
        self.fail_at_queue_id = fail_at_queue_id
        self.received = []

    def upload_live_workbook(self, item, workbook_path):
        if item['queue_id'] == self.fail_at_queue_id:
            raise RuntimeError('simulated transient cloud failure')
        self.received.append((item['queue_id'], workbook_path))


class AutoLiveRunStage2Tests(unittest.TestCase):
    '''Exercises only local files and a fake remote adapter.''' 

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.run_directory = os.path.join(
            self.temporary_directory.name,
            'RTG_stage2'
        )
        self.run_state_directory = os.path.join(
            self.run_directory,
            'Run_State'
        )
        self.journal = AutoLiveRunJournal.initialize(
            run_state_directory=self.run_state_directory,
            run_id='RTG-stage2-test',
            input_snapshot={'worksheet_name': 'RTG'},
            header_snapshot={'worksheet_name': 'RTG', 'header_rows': []},
            runtime_baseline={'variable_reagents': ['silver_nitrate']},
            git_branch='Auto-RTG',
            git_commit='0123456789abcdef',
            created_at_utc='2026-07-29T12:34:56+00:00'
        )
        self.queue = AutoLiveRunSyncQueue.initialize(self.run_state_directory)

    def tearDown(self):
        self.temporary_directory.cleanup()

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
            return [json.loads(line) for line in input_file if line.strip()]

    def _render(self):
        return AutoLiveRunWorkbookRenderer.render(
            run_directory=self.run_directory,
            run_display_name='RTG_stage2',
            journal=self.journal,
            manifest=self._read_json(AutoLiveRunJournal.MANIFEST_FILENAME),
            runtime_baseline=self._read_json(
                AutoLiveRunJournal.RUNTIME_BASELINE_FILENAME
            ),
            events=self._read_events()
        )

    def test_renderer_creates_readable_workbook_and_initial_queue_item(self):
        result = self._render()
        queued_item = self.queue.enqueue_rendered_workbook(result)

        self.assertTrue(os.path.isfile(result['workbook_path']))
        self.assertTrue(zipfile.is_zipfile(result['workbook_path']))
        with zipfile.ZipFile(result['workbook_path']) as workbook:
            names = set(workbook.namelist())
            self.assertIn('xl/workbook.xml', names)
            self.assertIn('xl/worksheets/sheet1.xml', names)
            live_status_xml = workbook.read('xl/worksheets/sheet1.xml')
            workbook_xml = workbook.read('xl/workbook.xml')
            for name in names:
                if name.endswith('.xml'):
                    ElementTree.fromstring(workbook.read(name))
        self.assertIn(b'Auto Live Run Status', live_status_xml)
        self.assertIn(b'RTG-stage2-test', live_status_xml)
        self.assertNotIn(b'Fault Disposition', workbook_xml)
        self.assertEqual(queued_item['state_revision'], 2)
        self.assertEqual(len(self.queue.read_pending()), 1)

    def test_queue_preserves_per_revision_workbook_snapshots_and_replays_fifo(self):
        first_item = self.queue.enqueue_rendered_workbook(self._render())
        self.journal.record_event('batch_preflight_validated', {'batch_number': 0})
        second_item = self.queue.enqueue_rendered_workbook(self._render())

        pending = self.queue.read_pending()
        self.assertEqual(
            [item['queue_id'] for item in pending],
            [first_item['queue_id'], second_item['queue_id']]
        )
        self.assertNotEqual(
            pending[0]['local_workbook_relative_path'],
            pending[1]['local_workbook_relative_path']
        )
        for item in pending:
            snapshot_path = os.path.join(
                self.run_directory,
                item['local_workbook_relative_path']
            )
            with open(snapshot_path, 'rb') as input_file:
                self.assertEqual(
                    hashlib.sha256(input_file.read()).hexdigest(),
                    item['local_workbook_sha256']
                )

        adapter = _RecordingAdapter()
        replayed = self.queue.replay(adapter)
        self.assertEqual(
            [item['queue_id'] for item in replayed],
            [first_item['queue_id'], second_item['queue_id']]
        )
        self.assertEqual(
            [queue_id for queue_id, _ in adapter.received],
            [first_item['queue_id'], second_item['queue_id']]
        )
        self.assertEqual(self.queue.read_pending(), [])

    def test_renderer_adds_read_only_fault_disposition_after_terminal_fault(self):
        fault_id = 'auto-fault-stage2-display'
        self.journal.record_transition(
            LIFECYCLE_FAULTED_PREPARATION,
            'fault_recorded',
            {
                'fault_id': fault_id,
                'fault_scope': 'auto_preparation',
                'certainty': 'preparation_execution_unknown_or_partial',
                'disposition': 'human_review_required_no_automatic_resume',
                'evidence_directory': '/tmp/fault-evidence',
                'evidence_write_error': None,
                'last_known_event_sequence': 1
            },
            active_batch_number=None,
            fault_id=fault_id
        )

        result = self._render()
        with zipfile.ZipFile(result['workbook_path']) as workbook:
            workbook_xml = workbook.read('xl/workbook.xml')
            fault_sheet_xml = workbook.read('xl/worksheets/sheet6.xml')

        self.assertIn(b'Fault Disposition', workbook_xml)
        self.assertIn(b'AUTO IS FROZEN', fault_sheet_xml)
        self.assertIn(b'auto-fault-stage2-display', fault_sheet_xml)
        self.assertIn(b'preparation_execution_unknown_or_partial',
                      fault_sheet_xml)
        self.assertIn(b'DO NOT RESUME OR RE-RUN THIS BATCH', fault_sheet_xml)

    def test_failed_replay_preserves_the_failed_item_and_later_order(self):
        first_item = self.queue.enqueue_rendered_workbook(self._render())
        self.journal.record_event('batch_preflight_validated', {'batch_number': 0})
        second_item = self.queue.enqueue_rendered_workbook(self._render())

        adapter = _RecordingAdapter(fail_at_queue_id=first_item['queue_id'])
        with self.assertRaisesRegex(
            AutoLiveRunSyncQueueError,
            'stopped at {}'.format(first_item['queue_id'])
        ):
            self.queue.replay(adapter)

        self.assertEqual(
            [item['queue_id'] for item in self.queue.read_pending()],
            [first_item['queue_id'], second_item['queue_id']]
        )
        self.assertEqual(adapter.received, [])

    def test_duplicate_revision_is_not_queued_twice(self):
        result = self._render()
        first_item = self.queue.enqueue_rendered_workbook(result)
        self.assertIsNone(self.queue.enqueue_rendered_workbook(result))
        self.assertEqual(
            [item['queue_id'] for item in self.queue.read_pending()],
            [first_item['queue_id']]
        )


class AutoLiveRunStage2ControllerPlacementTests(unittest.TestCase):
    '''Guards controller integration without importing hardware dependencies.'''

    @staticmethod
    def _method_node(class_node, method_name):
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                return node
        raise AssertionError('AutoContr.{} was not found.'.format(method_name))

    @staticmethod
    def _call_lines(method_node, attribute_name):
        return [
            node.lineno
            for node in ast.walk(method_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attribute_name
        ]

    def test_live_workbook_refresh_follows_only_durable_journal_writes(self):
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

        initialize_method = self._method_node(
            auto_controller,
            '_initialize_auto_live_run_journal'
        )
        self.assertGreater(
            min(self._call_lines(
                initialize_method,
                '_initialize_auto_live_run_mirror'
            )),
            min(self._call_lines(initialize_method, 'initialize'))
        )

        for method_name, durable_call in (
            ('_record_auto_live_run_event', 'record_event'),
            ('_record_auto_live_run_transition', 'record_transition')
        ):
            method = self._method_node(auto_controller, method_name)
            self.assertGreater(
                min(self._call_lines(method, '_refresh_auto_live_run_mirror')),
                min(self._call_lines(method, durable_call))
            )


if __name__ == '__main__':
    unittest.main()
