'''Hardware-free Stage 10A tests for fault evidence contracts and packages.'''

import csv
import json
import os
import tempfile
import unittest
from unittest import mock

from auto_live_run_faults import (
    AutoLiveRunFaultError,
    AutoLiveRunFaultEvidenceWriter,
    FAULT_CERTAINTY_PREPARATION_UNKNOWN_OR_PARTIAL,
    FAULT_CERTAINTY_TRANSFER_UNKNOWN_OR_PARTIAL,
    FAULT_SCOPE_AUTO_PREPARATION,
    FAULT_SCOPE_BATCH,
    build_fault_record,
    classify_fault_lifecycle,
    validate_fault_record
)
from auto_live_run_state import (
    LIFECYCLE_EXECUTING_BATCH,
    LIFECYCLE_FAULTED_PARTIAL_BATCH,
    LIFECYCLE_FAULTED_PREPARATION,
    LIFECYCLE_MEASURING_BATCH,
    LIFECYCLE_PROCESSING_BATCH,
    LIFECYCLE_READY_FOR_BATCH
)


class AutoLiveRunFaultEvidenceTests(unittest.TestCase):
    '''Exercises Stage 10A without importing controller or robot code.'''

    RUN_ID = 'RTG_018_5D-fault-test'

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.run_state_directory = os.path.join(
            self.temporary_directory.name,
            'Run_State'
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _batch_fault_record(self):
        return build_fault_record(
            run_id=self.RUN_ID,
            fault_id='batch-transfer-fault-001',
            fault_scope=FAULT_SCOPE_BATCH,
            certainty=FAULT_CERTAINTY_TRANSFER_UNKNOWN_OR_PARTIAL,
            preceding_lifecycle_state='executing_batch',
            lifecycle_state=LIFECYCLE_FAULTED_PARTIAL_BATCH,
            active_batch_number=2,
            exception_class='ConnectionError',
            exception_message='robot response ended before acknowledgement',
            last_event_sequence=14,
            batch_context={
                'planned_protocol_filename': 'protocol_df_batch_2.csv',
                'physical_well_count': 3
            },
            recorded_at_utc='2026-08-07T12:34:56+00:00'
        )

    def test_batch_fault_package_is_complete_and_immutable(self):
        record = self._batch_fault_record()
        evidence_directory = AutoLiveRunFaultEvidenceWriter.write(
            self.run_state_directory,
            record,
            'Traceback (most recent call last):\nConnectionError',
            planned_protocol_columns=['well', 'silver_nitrate_uL'],
            planned_protocol_rows=[{'well': 'A1', 'silver_nitrate_uL': 10.0}]
        )

        self.assertEqual(
            evidence_directory,
            os.path.join(
                self.run_state_directory,
                'Fault_Evidence',
                record['fault_id']
            )
        )
        self.assertEqual(
            sorted(os.listdir(evidence_directory)),
            sorted([
                'batch_context.json',
                'exception_trace.txt',
                'fault_summary.json',
                'fault_summary.md',
                'planned_protocol_dataframe.csv'
            ])
        )
        with open(
            os.path.join(evidence_directory, 'fault_summary.json'),
            'r',
            encoding='utf-8'
        ) as input_file:
            self.assertEqual(record, json.load(input_file))
        with open(
            os.path.join(evidence_directory, 'planned_protocol_dataframe.csv'),
            'r',
            encoding='utf-8',
            newline=''
        ) as input_file:
            rows = list(csv.DictReader(input_file))
        self.assertEqual('A1', rows[0]['well'])
        self.assertEqual('10.0', rows[0]['silver_nitrate_uL'])

        with self.assertRaisesRegex(AutoLiveRunFaultError, 'Refusing to overwrite'):
            AutoLiveRunFaultEvidenceWriter.write(
                self.run_state_directory,
                record,
                'different trace that must not replace evidence'
            )

    def test_preparation_fault_has_no_active_batch_or_protocol_artifact(self):
        record = build_fault_record(
            run_id=self.RUN_ID,
            fault_id='preparation-fault-001',
            fault_scope=FAULT_SCOPE_AUTO_PREPARATION,
            certainty=FAULT_CERTAINTY_PREPARATION_UNKNOWN_OR_PARTIAL,
            preceding_lifecycle_state='ready_for_batch',
            lifecycle_state=LIFECYCLE_FAULTED_PREPARATION,
            active_batch_number=None,
            exception_class='TimeoutError',
            exception_message='preparation acknowledgement timed out',
            last_event_sequence=5,
            batch_context={'preparation_group': 'borohydride-working'},
            recorded_at_utc='2026-08-07T12:34:56+00:00'
        )
        evidence_directory = AutoLiveRunFaultEvidenceWriter.write(
            self.run_state_directory,
            record,
            'TimeoutError: preparation acknowledgement timed out'
        )

        self.assertNotIn(
            'planned_protocol_dataframe.csv',
            os.listdir(evidence_directory)
        )
        with open(
            os.path.join(evidence_directory, 'fault_summary.md'),
            'r',
            encoding='utf-8'
        ) as input_file:
            self.assertIn('not applicable (preparation fault)', input_file.read())

    def test_publish_failure_never_exposes_a_partial_final_package(self):
        record = self._batch_fault_record()
        final_directory = os.path.join(
            self.run_state_directory,
            'Fault_Evidence',
            record['fault_id']
        )

        with mock.patch(
                'auto_live_run_faults.os.rename',
                side_effect=OSError('simulated publish failure')):
            with self.assertRaisesRegex(AutoLiveRunFaultError, 'Could not publish'):
                AutoLiveRunFaultEvidenceWriter.write(
                    self.run_state_directory,
                    record,
                    'Traceback (most recent call last):\nConnectionError'
                )

        self.assertFalse(os.path.exists(final_directory))
        evidence_root = os.path.dirname(final_directory)
        staging_directories = [
            name for name in os.listdir(evidence_root)
            if name.startswith('.{0}.'.format(record['fault_id']))
        ]
        self.assertEqual(1, len(staging_directories))
        self.assertTrue(os.path.isfile(os.path.join(
            evidence_root,
            staging_directories[0],
            'fault_summary.json'
        )))

    def test_fault_contract_rejects_scope_state_and_batch_mismatches(self):
        record = self._batch_fault_record()
        record['active_batch_number'] = None
        with self.assertRaisesRegex(AutoLiveRunFaultError, 'active_batch_number'):
            validate_fault_record(record)

        record = self._batch_fault_record()
        record['lifecycle_state'] = LIFECYCLE_FAULTED_PREPARATION
        with self.assertRaisesRegex(AutoLiveRunFaultError, 'does not match'):
            validate_fault_record(record)

        record = self._batch_fault_record()
        record['certainty'] = FAULT_CERTAINTY_PREPARATION_UNKNOWN_OR_PARTIAL
        with self.assertRaisesRegex(AutoLiveRunFaultError, 'certainty'):
            validate_fault_record(record)

        record = self._batch_fault_record()
        record['preceding_lifecycle_state'] = 'measuring_batch'
        with self.assertRaisesRegex(
                AutoLiveRunFaultError,
                'preceding_lifecycle_state'):
            validate_fault_record(record)

    def test_classifier_marks_only_potentially_physical_boundaries(self):
        preparation = classify_fault_lifecycle(
            LIFECYCLE_READY_FOR_BATCH,
            preparation_execution_may_have_started=True
        )
        self.assertEqual(FAULT_SCOPE_AUTO_PREPARATION,
                         preparation['fault_scope'])
        self.assertEqual(LIFECYCLE_FAULTED_PREPARATION,
                         preparation['lifecycle_state'])

        executing = classify_fault_lifecycle(LIFECYCLE_EXECUTING_BATCH)
        self.assertEqual(FAULT_SCOPE_BATCH, executing['fault_scope'])
        self.assertEqual(
            FAULT_CERTAINTY_TRANSFER_UNKNOWN_OR_PARTIAL,
            executing['certainty']
        )

        measuring = classify_fault_lifecycle(LIFECYCLE_MEASURING_BATCH)
        self.assertEqual(
            'transfer_complete_measurement_unknown',
            measuring['certainty']
        )

        processing = classify_fault_lifecycle(LIFECYCLE_PROCESSING_BATCH)
        self.assertEqual(
            'measurement_complete_processing_unknown',
            processing['certainty']
        )

        self.assertIsNone(classify_fault_lifecycle(
            LIFECYCLE_READY_FOR_BATCH,
            preparation_execution_may_have_started=False
        ))


if __name__ == '__main__':
    unittest.main()
