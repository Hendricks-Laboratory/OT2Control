'''Dependency-free local Live workbook rendering for Auto recovery records.

Stage 2 deliberately renders a small human-facing workbook from the durable
Stage 1 journal rather than editing the original experiment workbook.  The
original worksheet remains an immutable input record; this workbook is a
derived status mirror that can be rebuilt after any interruption.

The lab controller's validated Python environment does not include a general
XLSX authoring package.  This module therefore writes the narrow OOXML subset
needed for static status tables using only the Python standard library.  It
does not evaluate formulas, communicate with Google Drive, or inspect any
credential material.
'''

import datetime
import hashlib
import json
import os
import tempfile
import zipfile
from xml.sax.saxutils import escape

from auto_live_run_state import (
    LIFECYCLE_FAULTED_PARTIAL_BATCH,
    LIFECYCLE_FAULTED_PREPARATION
)


class AutoLiveRunWorkbookError(RuntimeError):
    '''Raised when a local Live workbook cannot be rendered safely.'''


class AutoLiveRunWorkbookRenderer:
    '''Renders a compact, read-only status workbook from journal records.'''

    LIVE_RUN_DIRECTORYNAME = 'Live_Run'

    @staticmethod
    def _utc_now_string():
        return datetime.datetime.now(
            datetime.timezone.utc
        ).replace(microsecond=0).isoformat()

    @staticmethod
    def _sanitize_filename_component(value):
        value = str(value).strip()
        if not value:
            raise AutoLiveRunWorkbookError(
                'Live workbook name must be a nonempty string.'
            )

        return ''.join(
            character if character.isalnum() or character in '._-' else '_'
            for character in value
        )

    @staticmethod
    def _column_name(column_number):
        '''Returns an Excel column name for a zero-based integer.'''
        name = ''
        column_number += 1
        while column_number:
            column_number, remainder = divmod(column_number - 1, 26)
            name = chr(65 + remainder) + name
        return name

    @classmethod
    def _cell_xml(cls, row_number, column_number, value, style_index=0):
        reference = '{}{}'.format(
            cls._column_name(column_number),
            row_number
        )
        style_attribute = ' s="{}"'.format(style_index) if style_index else ''

        if isinstance(value, bool):
            return '<c r="{}"{} t="b"><v>{}</v></c>'.format(
                reference,
                style_attribute,
                '1' if value else '0'
            )

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return '<c r="{}"{}><v>{}</v></c>'.format(
                reference,
                style_attribute,
                value
            )

        return '<c r="{}"{} t="inlineStr"><is><t>{}</t></is></c>'.format(
            reference,
            style_attribute,
            escape(str(value))
        )

    @classmethod
    def _sheet_xml(cls, rows, column_widths):
        column_xml = ''.join(
            '<col min="{0}" max="{0}" width="{1}" customWidth="1"/>'.format(
                index + 1,
                width
            )
            for index, width in enumerate(column_widths)
        )
        row_xml = []
        for row_index, row in enumerate(rows, start=1):
            style_index = 0
            if row and row[0] == '__TITLE__':
                style_index = 1
                values = row[1:]
            elif row and row[0] == '__HEADER__':
                style_index = 2
                values = row[1:]
            elif row and row[0] == '__NOTE__':
                style_index = 3
                values = row[1:]
            elif row and row[0] == '__WARNING__':
                style_index = 4
                values = row[1:]
            else:
                values = row

            cells = ''.join(
                cls._cell_xml(
                    row_index,
                    column_index,
                    value,
                    style_index=style_index
                )
                for column_index, value in enumerate(values)
            )
            row_xml.append('<row r="{}">{}</row>'.format(row_index, cells))

        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <cols>{}</cols>
  <sheetData>{}</sheetData>
</worksheet>'''.format(column_xml, ''.join(row_xml))

    @staticmethod
    def _styles_xml():
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="4">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="14"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFC00000"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="2" fillId="2" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="3" fillId="3" borderId="0" xfId="0"/>
  </cellXfs>
</styleSheet>'''

    @staticmethod
    def _workbook_xml(sheet_names):
        sheets_xml = ''.join(
            '<sheet name="{}" sheetId="{}" r:id="rId{}"/>'.format(
                escape(sheet_name),
                index,
                index
            )
            for index, sheet_name in enumerate(sheet_names, start=1)
        )
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>{}</sheets>
</workbook>'''.format(sheets_xml)

    @staticmethod
    def _workbook_relationships_xml(sheet_count):
        relationships = []
        for index in range(1, sheet_count + 1):
            relationships.append(
                '<Relationship Id="rId{0}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{0}.xml"/>'.format(index)
            )
        relationships.append(
            '<Relationship Id="rId{}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'.format(sheet_count + 1)
        )
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{}</Relationships>'''.format(
            ''.join(relationships)
        )

    @staticmethod
    def _content_types_xml(sheet_count):
        overrides = [
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        ]
        overrides.extend(
            '<Override PartName="/xl/worksheets/sheet{0}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'.format(index)
            for index in range(1, sheet_count + 1)
        )
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {}
</Types>'''.format(''.join(overrides))

    @staticmethod
    def _root_relationships_xml():
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    @staticmethod
    def _json_text(value):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(',', ':')
        )

    @staticmethod
    def _fault_event_payload(current_state, events):
        '''Returns the durable terminal-fault payload matching current state.'''
        fault_id = current_state.get('fault_id')
        for event in reversed(events):
            if (
                event.get('event_type') == 'fault_recorded'
                and event.get('payload', {}).get('fault_id') == fault_id
            ):
                return dict(event.get('payload', {}))
        return {}

    @classmethod
    def _build_sheet_payloads(
        cls,
        run_id,
        current_state,
        events,
        manifest,
        runtime_baseline
    ):
        status_rows = [
            ['__TITLE__', 'Auto Live Run Status'],
            ['__NOTE__', 'Derived local mirror. It does not alter the original input workbook or accept recovery actions.'],
            ['__HEADER__', 'Field', 'Value'],
            ['Run ID', run_id],
            ['Lifecycle state', current_state['lifecycle_state']],
            ['State revision', current_state['revision']],
            ['Active batch number', current_state['active_batch_number']],
            ['Last event sequence', current_state['last_event_sequence']],
            ['Updated (UTC)', cls._utc_now_string()],
            ['Git branch', manifest.get('git_branch', 'unavailable')],
            ['Git commit', manifest.get('git_commit', 'unavailable')]
        ]
        current_state_rows = [
            ['__TITLE__', 'Current Durable State'],
            ['__HEADER__', 'Field', 'Value']
        ] + [[key, value] for key, value in sorted(current_state.items())]
        event_rows = [
            ['__TITLE__', 'Append-Only Event Journal'],
            ['__NOTE__', 'Rows are copied from Run_State/events.jsonl in sequence order.'],
            ['__HEADER__', 'Sequence', 'Timestamp (UTC)', 'Event type', 'State revision', 'Payload JSON']
        ]
        event_rows.extend(
            [
                event['sequence'],
                event['timestamp_utc'],
                event['event_type'],
                event['state_revision'],
                cls._json_text(event['payload'])
            ]
            for event in events
        )
        baseline_rows = [
            ['__TITLE__', 'Immutable Run Baseline'],
            ['__NOTE__', 'This view is derived from the immutable Stage 1 manifest and runtime baseline.'],
            ['__HEADER__', 'Record', 'Canonical JSON'],
            ['Run manifest', cls._json_text(manifest)],
            ['Runtime baseline', cls._json_text(runtime_baseline)]
        ]
        sheets = [
            ('Live Status', status_rows, [28, 88]),
            ('Current State', current_state_rows, [28, 88]),
            ('Event Journal', event_rows, [12, 26, 32, 16, 88]),
            ('Run Baseline', baseline_rows, [24, 120])
        ]
        if current_state['lifecycle_state'] not in (
                LIFECYCLE_FAULTED_PREPARATION,
                LIFECYCLE_FAULTED_PARTIAL_BATCH):
            return sheets

        fault_payload = cls._fault_event_payload(current_state, events)
        fault_rows = [
            ['__TITLE__', 'Terminal Fault Disposition'],
            [
                '__WARNING__',
                'AUTO IS FROZEN — HUMAN REVIEW REQUIRED. DO NOT RESUME OR RE-RUN THIS BATCH.'
            ],
            [
                '__NOTE__',
                'This read-only display is derived from the durable local journal. It cannot issue recovery actions or establish the physical outcome of an interrupted operation.'
            ],
            ['__HEADER__', 'Field', 'Value'],
            ['Fault ID', current_state['fault_id']],
            ['Terminal lifecycle state', current_state['lifecycle_state']],
            ['Fault scope', fault_payload.get('fault_scope', 'not recorded')],
            [
                'Physical outcome certainty',
                fault_payload.get('certainty', 'not recorded')
            ],
            [
                'Last known durable event sequence',
                fault_payload.get('last_known_event_sequence', 'not recorded')
            ],
            [
                'Local fault evidence directory',
                fault_payload.get('evidence_directory', 'not recorded')
            ],
            [
                'Evidence publication status',
                fault_payload.get('evidence_write_error') or 'published'
            ],
            [
                'Required disposition',
                'Inspect the physical system and the immutable evidence package. Start a separately identified run only after human review; no automatic continuation is available.'
            ]
        ]
        sheets.append(('Fault Disposition', fault_rows, [34, 118]))
        return sheets

    @classmethod
    def _atomic_write_workbook(cls, destination_path, sheet_payloads):
        destination_directory = os.path.dirname(destination_path)
        file_descriptor = None
        temporary_path = None
        try:
            file_descriptor, temporary_path = tempfile.mkstemp(
                prefix='.{0}.'.format(os.path.basename(destination_path)),
                suffix='.tmp',
                dir=destination_directory
            )
            os.close(file_descriptor)
            file_descriptor = None
            with zipfile.ZipFile(
                temporary_path,
                'w',
                compression=zipfile.ZIP_DEFLATED
            ) as archive:
                sheet_names = [sheet_name for sheet_name, _, _ in sheet_payloads]
                archive.writestr(
                    '[Content_Types].xml',
                    cls._content_types_xml(len(sheet_names))
                )
                archive.writestr('_rels/.rels', cls._root_relationships_xml())
                archive.writestr('xl/workbook.xml', cls._workbook_xml(sheet_names))
                archive.writestr(
                    'xl/_rels/workbook.xml.rels',
                    cls._workbook_relationships_xml(len(sheet_names))
                )
                archive.writestr('xl/styles.xml', cls._styles_xml())
                for index, (_, rows, column_widths) in enumerate(
                    sheet_payloads,
                    start=1
                ):
                    archive.writestr(
                        'xl/worksheets/sheet{}.xml'.format(index),
                        cls._sheet_xml(rows, column_widths)
                    )
            os.replace(temporary_path, destination_path)
            temporary_path = None
        except (OSError, zipfile.BadZipFile) as exc:
            raise AutoLiveRunWorkbookError(
                'Could not render local Live workbook {}: {}.'.format(
                    destination_path,
                    exc
                )
            )
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            if temporary_path is not None and os.path.exists(temporary_path):
                os.remove(temporary_path)

    @classmethod
    def write_local_workbook(cls, destination_path, sheet_payloads):
        '''Writes one local workbook through the shared atomic OOXML writer.

        The Live status workbook and the separate operator-action workbook use
        the same deliberately small, standard-library-only XLSX writer.  The
        caller remains responsible for its own file identity and overwrite
        policy; this method only supplies an atomic local write.
        '''
        cls._atomic_write_workbook(destination_path, sheet_payloads)

    @classmethod
    def render(
        cls,
        run_directory,
        run_display_name,
        journal,
        manifest,
        runtime_baseline,
        events
    ):
        '''Atomically rebuilds one local Live workbook from durable records.'''
        if journal is None:
            raise AutoLiveRunWorkbookError(
                'A live workbook requires an initialized Auto journal.'
            )
        if not isinstance(events, list):
            raise AutoLiveRunWorkbookError('events must be a list.')

        live_run_directory = os.path.join(
            os.path.abspath(run_directory),
            cls.LIVE_RUN_DIRECTORYNAME
        )
        try:
            if not os.path.isdir(live_run_directory):
                os.makedirs(live_run_directory)
        except OSError as exc:
            raise AutoLiveRunWorkbookError(
                'Could not create Live_Run directory {}: {}.'.format(
                    live_run_directory,
                    exc
                )
            )

        workbook_filename = '{}_LIVE.xlsx'.format(
            cls._sanitize_filename_component(run_display_name)
        )
        workbook_path = os.path.join(live_run_directory, workbook_filename)
        sheet_payloads = cls._build_sheet_payloads(
            journal.current_state['run_id'],
            journal.current_state,
            events,
            manifest,
            runtime_baseline
        )
        cls._atomic_write_workbook(workbook_path, sheet_payloads)
        with open(workbook_path, 'rb') as input_file:
            workbook_sha256 = hashlib.sha256(input_file.read()).hexdigest()

        return {
            'workbook_path': workbook_path,
            'workbook_sha256': workbook_sha256,
            'state_revision': journal.current_state['revision'],
            'run_id': journal.current_state['run_id']
        }
