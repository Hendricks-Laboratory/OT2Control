'''Stage-12B observer tests with no controller, reader, or robot imports.'''

import csv
import os
import tempfile
import unittest

from auto_stability_observer import (
    ACTIVATION_STATUS_ACTIVE,
    ACTIVATION_STATUS_PENDING,
    AutoStabilityObserver,
    AutoStabilityObserverError,
)


class AutoStabilityObserverTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.timestamps = iter([
            '2026-09-09T10:00:00+00:00',
            '2026-09-09T10:00:01+00:00',
            '2026-09-09T10:00:02+00:00',
            '2026-09-09T10:00:03+00:00',
            '2026-09-09T10:00:04+00:00',
            '2026-09-09T10:00:05+00:00',
            '2026-09-09T10:00:06+00:00',
            '2026-09-09T10:00:07+00:00',
        ])
        self.observer = AutoStabilityObserver(
            pr_data_path=self.temporary_directory.name,
            run_id='DEBUG-STABILITY-001',
            trigger_reagent='sodium_borohydride',
            now=lambda: next(self.timestamps)
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _manifest_rows(self):
        with open(
                self.observer.manifest_path,
                newline='',
                encoding='utf-8') as manifest_file:
            return list(csv.DictReader(manifest_file))

    def test_dispatch_is_pending_until_existing_completion_barrier(self):
        event = self.observer.record_trigger_transfer_dispatched(
            batch_number=0,
            wellname='autowell0C1.0',
            transfer_volume_uL=20.0,
            trigger_command_id=17
        )

        self.assertEqual(
            event['activation_status'], ACTIVATION_STATUS_PENDING
        )
        self.assertEqual(self.observer.get_active_wells(), [])

        completion = self.observer.confirm_trigger_transfer_completed(
            'autowell0C1.0'
        )
        self.assertEqual(
            completion['activation_status'], ACTIVATION_STATUS_ACTIVE
        )
        self.assertEqual(
            completion['trigger_completion_time_basis'],
            'controller_observed_save_ftp_barrier'
        )
        self.assertEqual(
            [entry['wellname'] for entry in self.observer.get_active_wells()],
            ['autowell0C1.0']
        )

    def test_raw_scan_reservations_are_unique_and_create_no_scan_file(self):
        self.observer.record_trigger_transfer_dispatched(
            2, 'autowell2C1.0', 15.0, 31
        )
        self.observer.confirm_trigger_transfer_completed('autowell2C1.0')

        first = self.observer.reserve_raw_scan(2, 'autowell2C1.0')
        second = self.observer.reserve_raw_scan(2, 'autowell2C1.0')

        self.assertNotEqual(
            first['raw_scan_relative_path'], second['raw_scan_relative_path']
        )
        self.assertTrue(
            first['raw_scan_relative_path'].startswith(
                os.path.join('stability', 'raw_scans')
            )
        )
        self.assertFalse(
            os.path.exists(os.path.join(
                self.temporary_directory.name,
                first['raw_scan_relative_path']
            ))
        )

        rows = self._manifest_rows()
        self.assertEqual(rows[0]['event_type'], 'observer_initialized')
        self.assertEqual(rows[-1]['event_type'], 'raw_scan_reserved')
        self.assertEqual(rows[-1]['raw_scan_id'], '0002')

    def test_invalid_transition_cannot_create_a_false_active_well(self):
        with self.assertRaises(AutoStabilityObserverError):
            self.observer.confirm_trigger_transfer_completed('missing-well')
        with self.assertRaises(AutoStabilityObserverError):
            self.observer.reserve_raw_scan(0, 'missing-well')

        self.observer.record_trigger_transfer_dispatched(
            0, 'autowell0C2.0', 10.0, 18
        )
        with self.assertRaises(AutoStabilityObserverError):
            self.observer.reserve_raw_scan(0, 'autowell0C2.0')
        self.assertEqual(self.observer.get_active_wells(), [])


if __name__ == '__main__':
    unittest.main()
