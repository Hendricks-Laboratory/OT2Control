'''Hardware-free tests for the Stage 11C1 local action-workbook shell.'''

import ast
import os
import tempfile
import unittest
import zipfile

from auto_live_run_operator_actions import (
    AutoLiveRunOperatorActionWorkbook,
    AutoLiveRunOperatorActionWorkbookError
)


class AutoLiveRunOperatorActionWorkbookTests(unittest.TestCase):
    '''Confirms the separate action shell is local, atomic, and non-overwriting.'''

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.run_directory = os.path.join(
            self.temporary_directory.name,
            'RTG_018'
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _initialize(self):
        return AutoLiveRunOperatorActionWorkbook.initialize(
            run_directory=self.run_directory,
            run_display_name='RTG_018',
            run_id='RTG_018-abc123',
            initial_state_revision=4
        )

    def test_initialization_creates_separate_local_workbook(self):
        result = self._initialize()

        self.assertTrue(result['created'])
        self.assertEqual(result['run_id'], 'RTG_018-abc123')
        self.assertEqual(result['initial_state_revision'], 4)
        self.assertTrue(os.path.isfile(result['workbook_path']))
        self.assertTrue(result['workbook_path'].endswith(
            os.path.join('Live_Run', 'RTG_018_OPERATOR_ACTIONS.xlsx')
        ))

        with zipfile.ZipFile(result['workbook_path'], 'r') as archive:
            workbook_xml = archive.read('xl/workbook.xml').decode('utf-8')
            sheet_xml = archive.read('xl/worksheets/sheet3.xml').decode('utf-8')

        self.assertIn('Instructions', workbook_xml)
        self.assertIn('Active Request', workbook_xml)
        self.assertIn('Operator Response', workbook_xml)
        self.assertIn('NOT READ OR ACTED UPON BY AUTO', sheet_xml)

    def test_second_initialization_preserves_existing_operator_file(self):
        first = self._initialize()
        with open(first['workbook_path'], 'ab') as output_file:
            output_file.write(b'operator-edit-marker')

        second = self._initialize()

        self.assertFalse(second['created'])
        with open(second['workbook_path'], 'rb') as input_file:
            self.assertTrue(input_file.read().endswith(b'operator-edit-marker'))

    def test_rejects_invalid_identity_fields(self):
        with self.assertRaises(AutoLiveRunOperatorActionWorkbookError):
            AutoLiveRunOperatorActionWorkbook.initialize(
                self.run_directory,
                'RTG_018',
                '',
                0
            )
        with self.assertRaises(AutoLiveRunOperatorActionWorkbookError):
            AutoLiveRunOperatorActionWorkbook.initialize(
                self.run_directory,
                'RTG_018',
                'run-id',
                -1
            )

    def test_source_remains_local_and_dependency_free(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'auto_live_run_operator_actions.py'
        )
        with open(source_path, 'r', encoding='utf-8') as input_file:
            source = input_file.read()
        parsed = ast.parse(source, filename=source_path)
        imported_modules = []
        for node in parsed.body:
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or '')

        self.assertEqual(
            imported_modules,
            ['os', 'auto_live_run_workbook']
        )
        # Explanatory comments may legitimately state that credentials or
        # Google Drive are not accessed.  Guard the actual dependency surface
        # rather than matching ordinary documentation prose.
        for prohibited_module in (
            'googleapiclient', 'requests', 'google', 'oauth2client'
        ):
            self.assertNotIn(prohibited_module, imported_modules)


class AutoLiveRunOperatorActionControllerPlacementTests(unittest.TestCase):
    '''Confirms C1 creates only a shell before queue/status refresh behavior.'''

    def test_controller_initializes_shell_without_response_intake(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'controller.py'
        )
        with open(source_path, 'r', encoding='utf-8') as input_file:
            source = input_file.read()

        start = source.index('    def _initialize_auto_live_run_mirror(self):')
        end = source.index('    def _refresh_auto_live_run_mirror(self):', start)
        method_source = source[start:end]
        self.assertIn('AutoLiveRunOperatorActionWorkbook.initialize', method_source)
        self.assertIn('response intake is not active yet', method_source)
        self.assertNotIn('read_operator_response', method_source)
        self.assertNotIn('operator_action_applied', method_source)
