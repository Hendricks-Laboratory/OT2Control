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
from auto_live_run_workbook import AutoLiveRunWorkbookRenderer


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

    def _source_refill_request(self, expected_state_revision=5):
        return {
            'run_id': 'RTG_018-abc123',
            'request_id': 'source-refill-request-001',
            'expected_state_revision': expected_state_revision,
            'hold_action_id': 'hold-source-refill-001',
            'batch_number': 2,
            'permitted_actions': [
                'refill_same_container', 'retry_preflight', 'end_run'
            ],
            'same_container_refill_candidates': [
                {
                    'source_chemical_name': 'silver_nitrate',
                    'source_container_index': 0,
                    'source_loc': 'A1',
                    'source_deck_pos': 2
                }
            ]
        }

    @staticmethod
    def _write_response(workbook_path, request, response_values):
        sheet_payloads = (
            AutoLiveRunOperatorActionWorkbook._active_sheet_payloads(request)
        )
        response_rows = sheet_payloads[2][1]
        for row in response_rows:
            if row and row[0] in response_values:
                row[1] = response_values[row[0]]
        AutoLiveRunWorkbookRenderer.write_local_workbook(
            workbook_path, sheet_payloads
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

    def test_activation_refuses_to_overwrite_an_inactive_operator_edit(self):
        initialized = self._initialize()
        sheet_payloads = (
            AutoLiveRunOperatorActionWorkbook._inactive_sheet_payloads(
                'RTG_018-abc123', 4
            )
        )
        for row in sheet_payloads[2][1]:
            if row and row[0] == 'Operator note':
                row[1] = 'Do not overwrite this inactive operator edit.'
        AutoLiveRunWorkbookRenderer.write_local_workbook(
            initialized['workbook_path'], sheet_payloads
        )

        with self.assertRaisesRegex(
                AutoLiveRunOperatorActionWorkbookError,
                'inactive response'):
            AutoLiveRunOperatorActionWorkbook.activate_source_refill_request(
                initialized['workbook_path'], self._source_refill_request()
            )

    def test_active_request_requires_explicit_reissue_permission(self):
        initialized = self._initialize()
        request = self._source_refill_request()
        AutoLiveRunOperatorActionWorkbook.activate_source_refill_request(
            initialized['workbook_path'], request
        )

        with self.assertRaisesRegex(
                AutoLiveRunOperatorActionWorkbookError,
                'already has an active request'):
            AutoLiveRunOperatorActionWorkbook.activate_source_refill_request(
                initialized['workbook_path'], request
            )

        replacement = self._source_refill_request(expected_state_revision=6)
        replacement['request_id'] = 'source-refill-request-002'
        accepted = (
            AutoLiveRunOperatorActionWorkbook.activate_source_refill_request(
                initialized['workbook_path'],
                replacement,
                replace_active_request=True
            )
        )
        self.assertEqual('source-refill-request-002', accepted['request_id'])

    def test_resolved_request_preserves_audit_response_for_next_hold(self):
        initialized = self._initialize()
        request = self._source_refill_request()
        AutoLiveRunOperatorActionWorkbook.activate_source_refill_request(
            initialized['workbook_path'], request
        )
        AutoLiveRunOperatorActionWorkbook.resolve_source_refill_request(
            initialized['workbook_path'],
            request,
            {
                'requested_action': 'refill_same_container',
                'candidate_index': 0,
                'measured_total_mass_g': 14.25,
                'confirmation': 'REFILL',
                'operator_note': 'Terminal fallback was used.'
            },
            'same-container refill accepted via terminal'
        )

        sheets = AutoLiveRunOperatorActionWorkbook._read_workbook_sheets(
            initialized['workbook_path']
        )
        active_fields = AutoLiveRunOperatorActionWorkbook._sheet_field_values(
            sheets, 'Active Request'
        )
        response_fields = AutoLiveRunOperatorActionWorkbook._sheet_field_values(
            sheets, 'Operator Response'
        )
        self.assertEqual('resolved', active_fields['Request status'])
        self.assertEqual(
            'same-container refill accepted via terminal',
            active_fields['Resolution']
        )
        self.assertEqual('14.25', response_fields['Measured total mass (g)'])

        replacement = self._source_refill_request(expected_state_revision=6)
        replacement['request_id'] = 'source-refill-request-002'
        AutoLiveRunOperatorActionWorkbook.activate_source_refill_request(
            initialized['workbook_path'], replacement
        )

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

    def test_active_matching_refill_response_is_normalized(self):
        initialized = self._initialize()
        request = self._source_refill_request()
        AutoLiveRunOperatorActionWorkbook.activate_source_refill_request(
            initialized['workbook_path'], request
        )
        self._write_response(
            initialized['workbook_path'],
            request,
            {
                'Requested action': 'refill_same_container',
                'Candidate number': '1',
                'Measured total mass (g)': '14.25',
                'Confirmation': 'REFILL',
                'Operator note': 'Measured after refill.'
            }
        )

        response = (
            AutoLiveRunOperatorActionWorkbook.read_source_refill_response(
                initialized['workbook_path'], request
            )
        )

        self.assertEqual(response['requested_action'], 'refill_same_container')
        self.assertEqual(response['candidate_index'], 0)
        self.assertEqual(response['measured_total_mass_g'], 14.25)
        self.assertEqual(response['hold_action_id'], 'hold-source-refill-001')

    def test_stale_or_unconfirmed_response_is_rejected_without_acceptance(self):
        initialized = self._initialize()
        request = self._source_refill_request()
        AutoLiveRunOperatorActionWorkbook.activate_source_refill_request(
            initialized['workbook_path'], request
        )
        self._write_response(
            initialized['workbook_path'],
            request,
            {
                'Expected state revision': '4',
                'Requested action': 'refill_same_container',
                'Candidate number': '1',
                'Measured total mass (g)': '14.25',
                'Confirmation': 'REFILL'
            }
        )
        with self.assertRaisesRegex(
                AutoLiveRunOperatorActionWorkbookError,
                'mismatched Expected state revision'):
            AutoLiveRunOperatorActionWorkbook.read_source_refill_response(
                initialized['workbook_path'], request
            )

        self._write_response(
            initialized['workbook_path'],
            request,
            {
                'Requested action': 'refill_same_container',
                'Candidate number': '1',
                'Measured total mass (g)': '14.25',
                'Confirmation': 'NO'
            }
        )
        with self.assertRaisesRegex(
                AutoLiveRunOperatorActionWorkbookError,
                'requires confirmation REFILL'):
            AutoLiveRunOperatorActionWorkbook.read_source_refill_response(
                initialized['workbook_path'], request
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
            [
                'copy', 'math', 'os', 'zipfile',
                'xml.etree.ElementTree', 'auto_live_run_workbook'
            ]
        )
        # Explanatory comments may legitimately state that credentials or
        # Google Drive are not accessed.  Guard the actual dependency surface
        # rather than matching ordinary documentation prose.
        for prohibited_module in (
            'googleapiclient', 'requests', 'google', 'oauth2client'
        ):
            self.assertNotIn(prohibited_module, imported_modules)


class AutoLiveRunOperatorActionControllerPlacementTests(unittest.TestCase):
    '''Confirms C2 feeds only the existing source-refill safe hold.'''

    def test_controller_places_workbook_intake_inside_source_refill_hold(self):
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
        self.assertIn('intake activates only during an', method_source)
        self.assertIn('eligible safe hold', method_source)
        self.assertNotIn('read_operator_response', method_source)

        hold_start = source.index(
            '    def _hold_auto_batch_for_same_container_refill('
        )
        hold_end = source.index(
            '    def _hold_auto_batch_for_complete_tip_rack_replacement(',
            hold_start
        )
        hold_source = source[hold_start:hold_end]
        self.assertIn(
            '_activate_auto_live_source_refill_workbook_request', hold_source
        )
        self.assertIn(
            '_read_auto_live_source_refill_workbook_response', hold_source
        )
        self.assertIn("action == 'workbook'", hold_source)
        self.assertIn('_request_auto_main_source_mass_refresh', hold_source)
        self.assertNotIn('replace_source', hold_source)
