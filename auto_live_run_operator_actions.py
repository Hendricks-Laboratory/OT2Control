'''Local operator-action workbook for one explicitly approved Auto hold.

This module creates one separate workbook beside the controller-written Live
status workbook.  It is deliberately local and dependency-free: the Lab-PC
desktop synchronization client may mirror the normal protocol-output folder,
but this module never accesses Google Drive, credentials, a network, the
original input workbook, a controller, or the robot.

Stage 11C1 creates the action-workbook shell exactly once for a run.  Stage
11C2 may activate one same-container source-refill request and read only a
strictly matching, explicitly confirmed response. It never accesses a
network, changes a robot, or bypasses the controller's existing terminal
recovery path.
'''

import copy
import math
import os
import zipfile
import xml.etree.ElementTree as ElementTree

from auto_live_run_workbook import (
    AutoLiveRunWorkbookError,
    AutoLiveRunWorkbookRenderer
)


class AutoLiveRunOperatorActionWorkbookError(RuntimeError):
    '''Raised when the local operator-action workbook cannot be prepared.'''


class AutoLiveRunOperatorActionWorkbook:
    '''Creates and validates a non-overwriting operator-action workbook.

    The workbook is intentionally distinct from ``<run>_LIVE.xlsx``.  The
    controller may regenerate the status workbook after every durable event,
    whereas this action workbook is created once and left untouched until a
    a hold-specific request is activated. Only Stage 11C2's constrained
    same-container source-refill request is currently supported.
    '''

    ACTION_WORKBOOK_SUFFIX = '_OPERATOR_ACTIONS.xlsx'
    WORKBOOK_SCHEMA_VERSION = 1
    _MAIN_XML_NAMESPACE = (
        'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    )
    _RELATIONSHIP_XML_NAMESPACE = (
        'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    )
    _PACKAGE_RELATIONSHIP_XML_NAMESPACE = (
        'http://schemas.openxmlformats.org/package/2006/relationships'
    )
    _MAX_WORKBOOK_BYTES = 4 * 1024 * 1024
    _MAX_UNCOMPRESSED_XML_BYTES = 8 * 1024 * 1024

    # These labels deliberately describe the physical action rather than the
    # controller's internal command names.  The reader maps their constrained
    # display values back to the established terminal-equivalent actions.
    RESPONSE_FIELD_NAMES = (
        'Run ID',
        'Request ID',
        'Expected state revision',
        'Action to take',
        'Tube you refilled',
        'New total mass (g)',
        'Confirm action',
        'Operator note'
    )
    LEGACY_RESPONSE_FIELD_NAMES = (
        'Run ID',
        'Request ID',
        'Expected state revision',
        'Requested action',
        'Candidate number',
        'Measured total mass (g)',
        'Confirmation',
        'Operator note'
    )
    ACTION_DISPLAY_VALUES = {
        'Refill this same container': 'refill_same_container',
        'Retry the resource check': 'retry_preflight',
        'End this Auto run': 'end_run'
    }
    CONFIRMATION_DISPLAY_VALUES = {
        'Confirm refill': 'REFILL',
        'Confirm retry': 'RETRY',
        'Confirm end run': 'END'
    }
    NO_REFILL_TUBE_DISPLAY = 'No tube refilled'

    @staticmethod
    def _validate_nonempty_string(value, field_name):
        if not isinstance(value, str) or not value.strip():
            raise AutoLiveRunOperatorActionWorkbookError(
                '{} must be a nonempty string.'.format(field_name)
            )

    @staticmethod
    def _validate_nonnegative_integer(value, field_name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AutoLiveRunOperatorActionWorkbookError(
                '{} must be a nonnegative integer.'.format(field_name)
            )

    @classmethod
    def _workbook_path(cls, run_directory, run_display_name):
        cls._validate_nonempty_string(run_directory, 'run_directory')
        cls._validate_nonempty_string(run_display_name, 'run_display_name')
        live_run_directory = os.path.join(
            os.path.abspath(run_directory),
            AutoLiveRunWorkbookRenderer.LIVE_RUN_DIRECTORYNAME
        )
        workbook_name = '{}{}'.format(
            AutoLiveRunWorkbookRenderer._sanitize_filename_component(
                run_display_name
            ),
            cls.ACTION_WORKBOOK_SUFFIX
        )
        return live_run_directory, os.path.join(live_run_directory, workbook_name)

    @classmethod
    def _inactive_sheet_payloads(cls, run_id, initial_state_revision):
        return [
            (
                'Operator Response',
                [
                    ['__TITLE__', 'Auto Operator Action Workbook'],
                    [
                        '__WARNING__',
                        'NO ACTION REQUIRED YET. Auto reads this workbook '
                        'only when it creates a specific safe recovery hold.'
                    ],
                    [
                        '__NOTE__',
                        'When a hold occurs, this page will identify the '
                        'specific tube and show yellow response cells. Do not '
                        'edit the original input workbook or <run>_LIVE.xlsx.'
                    ],
                    ['__SECTION__', 'Current status', 'Waiting for an Auto hold'],
                    ['__READONLY__', 'Run ID', run_id],
                    ['__READONLY__', 'Request ID', 'not assigned'],
                    [
                        '__READONLY__', 'Expected state revision',
                        'not assigned'
                    ],
                    ['__READONLY__', 'Action to take', ''],
                    ['__READONLY__', 'Tube you refilled', ''],
                    ['__READONLY__', 'New total mass (g)', ''],
                    ['__READONLY__', 'Confirm action', ''],
                    ['__READONLY__', 'Operator note', ''],
                    [
                        '__NOTE__',
                        'Initial state revision: {}.'.format(
                            initial_state_revision
                        )
                    ]
                ],
                [34, 74]
            ),
            (
                'Active Request',
                [
                    ['__TITLE__', 'Active Operator Request'],
                    [
                        '__NOTE__',
                        'No active request exists in Stage 11C1. This sheet '
                        'will be populated only by a future safe-hold stage.'
                    ],
                    ['__HEADER__', 'Field', 'Value'],
                    ['Schema version', cls.WORKBOOK_SCHEMA_VERSION],
                    ['Request status', 'inactive'],
                    ['Run ID', run_id],
                    ['Initial state revision', initial_state_revision],
                    ['Request ID', 'not assigned'],
                    ['Expected state revision', 'not assigned'],
                    ['Blocking resource', 'not assigned'],
                    ['Allowed action', 'not assigned']
                ],
                [30, 96]
            ),
            (
                'Instructions',
                [
                    ['__TITLE__', 'How this workbook works'],
                    [
                        '__NOTE__',
                        'This workbook is a narrow recovery form. It never '
                        'changes a recipe, concentration, source location, '
                        'or the original input workbook.'
                    ],
                    [
                        '__NOTE__',
                        'At an active source-refill hold: refill the same '
                        'physical tube, reweigh it, enter its new total mass '
                        'in the yellow cell, save this same XLSX file, then '
                        'enter workbook in the controller terminal.'
                    ],
                    [
                        '__NOTE__',
                        'If synchronization is unavailable or the response '
                        'is rejected, use the terminal recovery prompts. No '
                        'inventory changes occur until the controller accepts '
                        'one exact, confirmed response.'
                    ]
                ],
                [34, 74]
            )
        ]

    @classmethod
    def _active_sheet_payloads(cls, request):
        '''Builds one source-refill request without retaining any response.'''
        candidates = request['same_container_refill_candidates']
        candidate_display_values = cls._candidate_display_values(candidates)
        candidate_lines = [
            '{}. {}'.format(index, value)
            for index, value in enumerate(candidate_display_values, start=1)
        ]

        return [
            (
                'Operator Response',
                [
                    ['__TITLE__', 'Auto Recovery — Action Required'],
                    [
                        '__ACTION__',
                        'The next batch is paused before liquid handling. '
                        'Choose one action in the yellow cells below.'
                    ],
                    [
                        '__NOTE__',
                        'For a refill: add liquid to the same tube, reweigh '
                        'the complete tube, enter its new total mass in grams, '
                        'then choose Confirm refill. Save this same XLSX file '
                        'and enter workbook in the controller terminal.'
                    ],
                    ['__HEADER__', 'Source needing attention', 'Value'],
                    [
                        'Eligible same-container refill tube(s)',
                        '\n'.join(candidate_lines)
                    ],
                    [
                        'What this form can change',
                        'Only the recorded total mass for one existing tube. '
                        'It cannot change reagent identity, concentration, '
                        'location, recipe, or destination wells.'
                    ],
                    ['__SECTION__', 'Your response', 'Only yellow value cells are editable'],
                    ['__INPUT__', 'Action to take', ''],
                    ['__INPUT__', 'Tube you refilled', ''],
                    ['__INPUT__', 'New total mass (g)', ''],
                    ['__INPUT__', 'Confirm action', ''],
                    ['__INPUT__', 'Operator note', ''],
                    [
                        '__NOTE__',
                        'After saving, wait for the edited file to sync back '
                        'to the Lab PC. Then enter workbook in the terminal. '
                        'Use terminal recovery if sync is unavailable.'
                    ],
                    ['__SECTION__', 'Request details — do not edit', ''],
                    ['__READONLY__', 'Run ID', request['run_id']],
                    ['__READONLY__', 'Request ID', request['request_id']],
                    [
                        '__READONLY__', 'Expected state revision',
                        request['expected_state_revision']
                    ]
                ],
                [38, 78],
                {
                    'data_validations': [
                        {
                            'cell': 'B8',
                            'values': list(cls.ACTION_DISPLAY_VALUES)
                        },
                        {
                            'cell': 'B9',
                            'values': [
                                cls.NO_REFILL_TUBE_DISPLAY
                            ] + candidate_display_values
                        },
                        {
                            'cell': 'B11',
                            'values': list(cls.CONFIRMATION_DISPLAY_VALUES)
                        }
                    ]
                }
            ),
            (
                'Active Request',
                [
                    ['__TITLE__', 'Active Recovery Request — Read Only'],
                    ['__HEADER__', 'Field', 'Value'],
                    ['Schema version', cls.WORKBOOK_SCHEMA_VERSION],
                    ['Request status', 'active'],
                    ['Run ID', request['run_id']],
                    ['Request ID', request['request_id']],
                    ['Expected state revision', request['expected_state_revision']],
                    ['Hold action ID', request['hold_action_id']],
                    ['Hold reason', 'same-container source refill'],
                    ['Batch number', request['batch_number']],
                    ['Permitted recovery actions', 'Refill same container; retry resource check; end this Auto run'],
                    ['Eligible refill tube(s)', '\n'.join(candidate_lines)]
                ],
                [34, 86]
            ),
            (
                'Instructions',
                [
                    ['__TITLE__', 'Recovery Instructions'],
                    ['__NOTE__', 'Use the Operator Response sheet first. It is the only editable response surface.'],
                    ['__NOTE__', 'Do not edit <run>_LIVE.xlsx or the original input workbook. This form permits only an exact same-container mass refresh at a safe pre-batch hold.'],
                    ['__NOTE__', 'The controller rejects stale, incomplete, mismatched, or unsupported responses without changing Pi inventory or releasing the batch.']
                ],
                [34, 86]
            )
        ]

    @staticmethod
    def _candidate_display_values(candidates):
        '''Returns stable, human-readable choices without exposing IDs first.'''
        return [
            '{} | deck {} | {}'.format(
                str(candidate['source_chemical_name']).replace('_', ' '),
                candidate['source_deck_pos'],
                candidate['source_loc']
            )
            for candidate in candidates
        ]

    @classmethod
    def _response_field_names(cls, response_fields):
        '''Identifies the current clear-label form or a prior technical form.

        Existing in-flight workbooks from the first Stage 11C2 implementation
        remain readable. Newly activated requests always use the clearer form.
        '''
        if set(cls.RESPONSE_FIELD_NAMES).issubset(response_fields):
            return cls.RESPONSE_FIELD_NAMES
        if set(cls.LEGACY_RESPONSE_FIELD_NAMES).issubset(response_fields):
            return cls.LEGACY_RESPONSE_FIELD_NAMES
        missing = set(cls.RESPONSE_FIELD_NAMES) - set(response_fields)
        raise AutoLiveRunOperatorActionWorkbookError(
            'Operator-action response is missing field(s): {}.'.format(
                ', '.join(sorted(missing))
            )
        )

    @classmethod
    def _resolved_sheet_payloads(cls, request, response, resolution):
        '''Builds a non-active audit record after a handled request.

        This is intentionally an action-workbook presentation update only.
        The controller writes its durable journal/Pi outcome first, so a
        failure to render this convenience record cannot alter recovery.
        '''
        sheet_payloads = cls._active_sheet_payloads(request)
        response_rows = sheet_payloads[0][1]
        response_rows[1] = [
            '__NOTE__',
            'REQUEST RESOLVED: this response has already been handled. It '
            'cannot be replayed to release another batch.'
        ]
        active_request_rows = sheet_payloads[1][1]
        for row in active_request_rows:
            if row and row[0] == 'Request status':
                row[1] = 'resolved'
        active_request_rows.append(['Resolution', resolution])

        candidate_display_values = cls._candidate_display_values(
            request['same_container_refill_candidates']
        )
        response_values = {
            'Action to take': next(
                display_value
                for display_value, action in cls.ACTION_DISPLAY_VALUES.items()
                if action == response['requested_action']
            ),
            'Tube you refilled': (
                cls.NO_REFILL_TUBE_DISPLAY
                if response['candidate_index'] is None
                else candidate_display_values[response['candidate_index']]
            ),
            'New total mass (g)': (
                '' if response['measured_total_mass_g'] is None
                else response['measured_total_mass_g']
            ),
            'Confirm action': next(
                display_value
                for display_value, confirmation in
                cls.CONFIRMATION_DISPLAY_VALUES.items()
                if confirmation == response['confirmation']
            ),
            'Operator note': response['operator_note']
        }
        for row in response_rows:
            if (
                    len(row) >= 3
                    and row[0] == '__INPUT__'
                    and row[1] in response_values):
                row[2] = response_values[row[1]]
        return sheet_payloads

    @staticmethod
    def _text(value):
        return str(value).strip() if value is not None else ''

    @classmethod
    def _validate_source_refill_request(cls, request):
        required_keys = {
            'run_id',
            'request_id',
            'expected_state_revision',
            'hold_action_id',
            'batch_number',
            'permitted_actions',
            'same_container_refill_candidates'
        }
        if not isinstance(request, dict) or set(request) != required_keys:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action request has an invalid schema.'
            )
        for field_name in ('run_id', 'request_id', 'hold_action_id'):
            cls._validate_nonempty_string(request[field_name], field_name)
        for field_name in ('expected_state_revision', 'batch_number'):
            cls._validate_nonnegative_integer(request[field_name], field_name)
        permitted_actions = request['permitted_actions']
        if (
                not isinstance(permitted_actions, list)
                or permitted_actions != [
                    'refill_same_container', 'retry_preflight', 'end_run'
                ]):
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action request has invalid permitted actions.'
            )
        candidates = request['same_container_refill_candidates']
        if not isinstance(candidates, list) or not candidates:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action request requires at least one refill candidate.'
            )
        required_candidate_keys = {
            'source_chemical_name',
            'source_container_index',
            'source_loc',
            'source_deck_pos'
        }
        for candidate in candidates:
            if (
                    not isinstance(candidate, dict)
                    or set(candidate) != required_candidate_keys):
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Operator-action request has an invalid refill candidate.'
                )
            cls._validate_nonempty_string(
                candidate['source_chemical_name'],
                'candidate source_chemical_name'
            )
            cls._validate_nonempty_string(
                candidate['source_loc'], 'candidate source_loc'
            )
            if (
                    isinstance(candidate['source_container_index'], bool)
                    or not isinstance(candidate['source_container_index'], int)
                    or candidate['source_container_index'] < 0
                    or isinstance(candidate['source_deck_pos'], bool)
                    or not isinstance(candidate['source_deck_pos'], int)
                    or candidate['source_deck_pos'] < 1):
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Operator-action request has an invalid refill identity.'
                )
        return copy.deepcopy(request)

    @staticmethod
    def _worksheet_cell_value(cell, shared_strings):
        namespace = '{%s}' % AutoLiveRunOperatorActionWorkbook._MAIN_XML_NAMESPACE
        cell_type = cell.get('t')
        if cell_type == 'inlineStr':
            return ''.join(
                node.text or '' for node in cell.findall(
                    './/{}t'.format(namespace)
                )
            )
        value_node = cell.find('{}v'.format(namespace))
        if value_node is None:
            return ''
        value = value_node.text or ''
        if cell_type == 's':
            try:
                return shared_strings[int(value)]
            except (ValueError, IndexError):
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Operator-action workbook contains an invalid shared string.'
                )
        return value

    @staticmethod
    def _column_index(cell_reference):
        letters = ''.join(character for character in cell_reference if character.isalpha())
        if not letters:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action workbook contains a cell without a column.'
            )
        value = 0
        for character in letters.upper():
            value = value * 26 + ord(character) - ord('A') + 1
        return value - 1

    @classmethod
    def _read_workbook_sheets(cls, workbook_path):
        if not os.path.isfile(workbook_path):
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action workbook is missing: {}.'.format(workbook_path)
            )
        if os.path.getsize(workbook_path) > cls._MAX_WORKBOOK_BYTES:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action workbook is too large to read safely.'
            )
        main_namespace = '{%s}' % cls._MAIN_XML_NAMESPACE
        rel_namespace = '{%s}' % cls._RELATIONSHIP_XML_NAMESPACE
        package_namespace = '{%s}' % cls._PACKAGE_RELATIONSHIP_XML_NAMESPACE
        try:
            with zipfile.ZipFile(workbook_path, 'r') as archive:
                if sum(item.file_size for item in archive.infolist()) > (
                        cls._MAX_UNCOMPRESSED_XML_BYTES):
                    raise AutoLiveRunOperatorActionWorkbookError(
                        'Operator-action workbook expands beyond the safe limit.'
                    )
                workbook_root = ElementTree.fromstring(
                    archive.read('xl/workbook.xml')
                )
                relationship_root = ElementTree.fromstring(
                    archive.read('xl/_rels/workbook.xml.rels')
                )
                relationships = {
                    node.get('Id'): node.get('Target')
                    for node in relationship_root.findall(
                        '{}Relationship'.format(package_namespace)
                    )
                }
                shared_strings = []
                if 'xl/sharedStrings.xml' in archive.namelist():
                    shared_root = ElementTree.fromstring(
                        archive.read('xl/sharedStrings.xml')
                    )
                    shared_strings = [
                        ''.join(
                            node.text or '' for node in item.findall(
                                './/{}t'.format(main_namespace)
                            )
                        )
                        for item in shared_root.findall(
                            '{}si'.format(main_namespace)
                        )
                    ]
                sheets = {}
                for sheet in workbook_root.findall(
                        './/{}sheet'.format(main_namespace)):
                    sheet_name = sheet.get('name')
                    relationship_id = sheet.get('{}id'.format(rel_namespace))
                    target = relationships.get(relationship_id)
                    if not sheet_name or not target:
                        raise AutoLiveRunOperatorActionWorkbookError(
                            'Operator-action workbook has an invalid sheet map.'
                        )
                    sheet_path = target.lstrip('/')
                    if not sheet_path.startswith('xl/'):
                        sheet_path = 'xl/{}'.format(sheet_path)
                    root = ElementTree.fromstring(archive.read(sheet_path))
                    rows = {}
                    for row in root.findall('.//{}row'.format(main_namespace)):
                        row_number = int(row.get('r'))
                        rows[row_number] = {
                            cls._column_index(cell.get('r')): (
                                cls._worksheet_cell_value(cell, shared_strings)
                            )
                            for cell in row.findall('{}c'.format(main_namespace))
                        }
                    sheets[sheet_name] = rows
        except (
                OSError,
                ValueError,
                zipfile.BadZipFile,
                ElementTree.ParseError,
                KeyError) as exc:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Could not read operator-action workbook safely: {}.'.format(exc)
            )
        return sheets

    @classmethod
    def _sheet_field_values(cls, sheets, sheet_name):
        rows = sheets.get(sheet_name)
        if not isinstance(rows, dict):
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action workbook is missing the {} sheet.'.format(
                    sheet_name
                )
            )
        values = {}
        for row in rows.values():
            label = cls._text(row.get(0))
            if label in values:
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Operator-action workbook has duplicate {} field.'.format(
                        label
                    )
                )
            if label:
                values[label] = cls._text(row.get(1))
        return values

    @classmethod
    def _require_exact_response_fields(cls, values):
        cls._response_field_names(values)

    @classmethod
    def activate_source_refill_request(
        cls,
        workbook_path,
        request,
        replace_active_request=False
    ):
        '''Publishes one request only while the local file is safely inactive.

        An explicit reissue may replace an already-active request only after
        the controller has durably recorded its rejection.  Ordinary status
        updates never call this method, so they cannot overwrite an operator
        response in the separate workbook.
        '''
        request = cls._validate_source_refill_request(request)
        sheets = cls._read_workbook_sheets(workbook_path)
        active_fields = cls._sheet_field_values(sheets, 'Active Request')
        response_fields = cls._sheet_field_values(sheets, 'Operator Response')
        cls._require_exact_response_fields(response_fields)
        if active_fields.get('Run ID') != request['run_id']:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action workbook run ID does not match this run.'
            )
        active_status = active_fields.get('Request status')
        if active_status == 'active' and not replace_active_request:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action workbook already has an active request.'
            )
        if active_status not in ('inactive', 'active', 'resolved'):
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action workbook has an invalid request status.'
            )
        if active_status == 'inactive':
            response_field_names = cls._response_field_names(response_fields)
            editable_values = [
                response_fields[field_name]
                for field_name in response_field_names
                if field_name not in (
                    'Run ID', 'Request ID', 'Expected state revision'
                )
            ]
            if any(editable_values):
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Operator-action workbook has an inactive response that '
                    'must not be overwritten.'
                )
        try:
            AutoLiveRunWorkbookRenderer.write_local_workbook(
                workbook_path,
                cls._active_sheet_payloads(request)
            )
        except AutoLiveRunWorkbookError as exc:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Could not activate operator-action workbook: {}.'.format(exc)
            )
        return copy.deepcopy(request)

    @classmethod
    def resolve_source_refill_request(
        cls,
        workbook_path,
        request,
        response,
        resolution
    ):
        '''Marks a handled request resolved without authorizing any action.'''
        request = cls._validate_source_refill_request(request)
        if not isinstance(response, dict) or set(response) != {
                'requested_action',
                'candidate_index',
                'measured_total_mass_g',
                'confirmation',
                'operator_note'}:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action resolution has an invalid response schema.'
            )
        if response['requested_action'] not in request['permitted_actions']:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator-action resolution has an unsupported action.'
            )
        cls._validate_nonempty_string(resolution, 'resolution')
        sheets = cls._read_workbook_sheets(workbook_path)
        active_fields = cls._sheet_field_values(sheets, 'Active Request')
        required_active = {
            'Request status': 'active',
            'Run ID': request['run_id'],
            'Request ID': request['request_id'],
            'Expected state revision': str(request['expected_state_revision'])
        }
        for field_name, expected_value in required_active.items():
            if active_fields.get(field_name) != expected_value:
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Active operator request has a mismatched {}.'.format(
                        field_name
                    )
                )
        try:
            AutoLiveRunWorkbookRenderer.write_local_workbook(
                workbook_path,
                cls._resolved_sheet_payloads(request, response, resolution)
            )
        except AutoLiveRunWorkbookError as exc:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Could not resolve operator-action workbook: {}.'.format(exc)
            )

    @classmethod
    def read_source_refill_response(cls, workbook_path, request):
        '''Reads and strictly validates one active source-refill response.

        This reader has no controller, Pi, or journal side effect.  It returns
        a normalized response only after every identity and action-specific
        confirmation check succeeds.
        '''
        request = cls._validate_source_refill_request(request)
        sheets = cls._read_workbook_sheets(workbook_path)
        active_fields = cls._sheet_field_values(sheets, 'Active Request')
        response_fields = cls._sheet_field_values(sheets, 'Operator Response')
        cls._require_exact_response_fields(response_fields)
        required_active = {
            'Schema version': str(cls.WORKBOOK_SCHEMA_VERSION),
            'Request status': 'active',
            'Run ID': request['run_id'],
            'Request ID': request['request_id'],
            'Expected state revision': str(request['expected_state_revision'])
        }
        for field_name, expected_value in required_active.items():
            if active_fields.get(field_name) != expected_value:
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Active operator request has a mismatched {}.'.format(
                        field_name
                    )
                )
        for field_name in ('Run ID', 'Request ID', 'Expected state revision'):
            expected_value = required_active[field_name]
            if response_fields[field_name] != expected_value:
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Operator response has a mismatched {}.'.format(
                        field_name
                    )
                )
        response_field_names = cls._response_field_names(response_fields)
        if response_field_names == cls.RESPONSE_FIELD_NAMES:
            action = cls.ACTION_DISPLAY_VALUES.get(
                response_fields['Action to take']
            )
            confirmation = cls.CONFIRMATION_DISPLAY_VALUES.get(
                response_fields['Confirm action']
            )
            selected_tube = response_fields['Tube you refilled']
            candidate_display_values = cls._candidate_display_values(
                request['same_container_refill_candidates']
            )
            try:
                candidate_index = candidate_display_values.index(selected_tube)
            except ValueError:
                candidate_index = -1
            measured_mass_text = response_fields['New total mass (g)']
        else:
            action = response_fields['Requested action'].lower()
            confirmation = response_fields['Confirmation'].upper()
            try:
                candidate_index = int(response_fields['Candidate number']) - 1
            except ValueError:
                candidate_index = -1
            measured_mass_text = response_fields['Measured total mass (g)']

        if action not in request['permitted_actions']:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator response requested an unsupported action.'
            )
        measured_mass_g = None
        if action == 'refill_same_container':
            if confirmation != 'REFILL':
                raise AutoLiveRunOperatorActionWorkbookError(
                    'A same-container refill requires confirmation REFILL.'
                )
            if not 0 <= candidate_index < len(
                    request['same_container_refill_candidates']):
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Operator response selected an invalid refill candidate.'
                )
            try:
                measured_mass_g = float(measured_mass_text)
            except ValueError:
                measured_mass_g = float('nan')
            if not math.isfinite(measured_mass_g) or measured_mass_g < 0.0:
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Operator response has an invalid measured total mass.'
                )
        elif action == 'retry_preflight':
            if confirmation != 'RETRY':
                raise AutoLiveRunOperatorActionWorkbookError(
                    'A preflight retry requires confirmation RETRY.'
                )
            candidate_index = None
        elif action == 'end_run':
            if confirmation != 'END':
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Ending Auto requires confirmation END.'
                )
            candidate_index = None
        operator_note = response_fields['Operator note']
        if len(operator_note) > 1000:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Operator response note exceeds the 1000-character limit.'
            )
        return {
            'channel': 'operator_action_workbook',
            'run_id': request['run_id'],
            'request_id': request['request_id'],
            'expected_state_revision': request['expected_state_revision'],
            'hold_action_id': request['hold_action_id'],
            'requested_action': action,
            'candidate_index': candidate_index,
            'measured_total_mass_g': measured_mass_g,
            'operator_note': operator_note
        }

    @classmethod
    def initialize(
        cls,
        run_directory,
        run_display_name,
        run_id,
        initial_state_revision
    ):
        '''Creates one action workbook shell without overwriting it.

        A second initialization call returns the already-existing path without
        reading, validating, or changing the file. This prevents future
        controller status refreshes from overwriting an operator's local or
        desktop-synchronized edits.
        '''
        cls._validate_nonempty_string(run_id, 'run_id')
        cls._validate_nonnegative_integer(
            initial_state_revision,
            'initial_state_revision'
        )
        live_run_directory, workbook_path = cls._workbook_path(
            run_directory,
            run_display_name
        )
        try:
            if not os.path.isdir(live_run_directory):
                os.makedirs(live_run_directory)
        except OSError as exc:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Could not create Live_Run directory {}: {}.'.format(
                    live_run_directory,
                    exc
                )
            )

        if os.path.exists(workbook_path):
            if not os.path.isfile(workbook_path):
                raise AutoLiveRunOperatorActionWorkbookError(
                    'Operator-action workbook path is not a file: {}.'.format(
                        workbook_path
                    )
                )
            return {
                'workbook_path': workbook_path,
                'created': False,
                'run_id': run_id,
                'initial_state_revision': initial_state_revision
            }

        try:
            AutoLiveRunWorkbookRenderer.write_local_workbook(
                workbook_path,
                cls._inactive_sheet_payloads(run_id, initial_state_revision)
            )
        except AutoLiveRunWorkbookError as exc:
            raise AutoLiveRunOperatorActionWorkbookError(
                'Could not create operator-action workbook: {}.'.format(exc)
            )

        return {
            'workbook_path': workbook_path,
            'created': True,
            'run_id': run_id,
            'initial_state_revision': initial_state_revision
        }
