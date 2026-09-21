'''Pure tests for the Stage-12F synthetic-evidence dry-debug harness.

The fixture deliberately operates only after manifest-linked reader spectra
have been parsed and blank-corrected. These tests neither import the
controller nor communicate with a reader or robot.
'''

import unittest

from auto_stability_debug import (
    AutoStabilityDebugEvidenceError,
    SYNTHETIC_COMPANION_EVIDENCE_SOURCE,
    SYNTHETIC_REFERENCE_WAVELENGTH_NM,
    build_synthetic_companion_evidence,
)
from auto_stability_reporting import build_stability_reporting_records


def _event(event_type, **values):
    row = {'event_type': event_type}
    row.update(values)
    return row


def _observed_spectrum(value):
    return {
        'spectrum_by_wavelength_nm': {
            624.0: float(value) - 0.01,
            625.0: float(value),
            626.0: float(value) - 0.01,
        },
    }


class AutoStabilityDebugEvidenceTests(unittest.TestCase):
    def _fixture(self):
        wells = ['well_a', 'well_b', 'well_c', 'well_d']
        manifest = []
        for wellname in wells:
            manifest.append(_event(
                'trigger_transfer_completed',
                wellname=wellname,
                trigger_transfer_completion_observed_at_utc=(
                    '2026-09-20T10:00:00+00:00'
                ),
            ))
        for raw_scan_id, started, completed in (
                ('scan_001', '2026-09-20T10:00:10+00:00',
                 '2026-09-20T10:00:20+00:00'),
                ('scan_002', '2026-09-20T10:01:10+00:00',
                 '2026-09-20T10:01:20+00:00')):
            manifest.append(_event(
                'raw_scan_completed',
                raw_scan_id=raw_scan_id,
                active_wellnames=';'.join(wells),
                observation_reason='cadenced_active_set',
                scan_started_at_utc=started,
                scan_completed_at_utc=completed,
            ))
        observed = {
            raw_scan_id: {
                wellname: _observed_spectrum(0.15)
                for wellname in wells
            }
            for raw_scan_id in ('scan_001', 'scan_002')
        }
        conditions = [
            {
                'origin_run_directory': 'DEBUG_STABILITY',
                'batch_number': 0,
                'reaction_number': 0,
                'replicate_sample_names': '["well_a", "well_b"]',
            },
            {
                'origin_run_directory': 'DEBUG_STABILITY',
                'batch_number': 0,
                'reaction_number': 1,
                'replicate_sample_names': '["well_c", "well_d"]',
            },
        ]
        return manifest, observed, conditions

    def test_synthetic_evidence_requires_verified_raw_pairs_and_preserves_replicates(self):
        manifest, observed, conditions = self._fixture()
        synthetic, evidence_rows = build_synthetic_companion_evidence(
            manifest, observed, conditions
        )

        self.assertEqual(len(evidence_rows), 8)
        self.assertTrue(all(row['debug_only'] for row in evidence_rows))
        self.assertTrue(all(
            row['stability_evidence_source'] ==
            SYNTHETIC_COMPANION_EVIDENCE_SOURCE
            for row in evidence_rows
        ))
        self.assertTrue(all(
            row['raw_reader_spectrum_verified'] for row in evidence_rows
        ))

        first = synthetic['scan_001']
        second = synthetic['scan_002']
        self.assertEqual(
            first['well_a']['spectrum_by_wavelength_nm'],
            first['well_b']['spectrum_by_wavelength_nm']
        )
        self.assertNotEqual(
            first['well_a']['spectrum_by_wavelength_nm'],
            first['well_c']['spectrum_by_wavelength_nm']
        )
        self.assertGreater(
            first['well_a']['spectrum_by_wavelength_nm'][
                SYNTHETIC_REFERENCE_WAVELENGTH_NM
            ],
            second['well_a']['spectrum_by_wavelength_nm'][
                SYNTHETIC_REFERENCE_WAVELENGTH_NM
            ]
        )

    def test_synthetic_evidence_drives_complete_debug_only_condition_metrics(self):
        manifest, observed, conditions = self._fixture()
        synthetic, _evidence_rows = build_synthetic_companion_evidence(
            manifest, observed, conditions
        )
        records = build_stability_reporting_records(
            manifest_rows=manifest,
            spectra_by_raw_scan=synthetic,
            condition_rows=conditions,
            observation_window_s=180,
            scan_interval_s=60,
            min_peak_absorbance=0.10,
        )

        self.assertEqual(
            [row['condition_stability_status']
             for row in records['condition_summaries']],
            ['complete', 'complete']
        )
        self.assertEqual(
            [row['condition_signal_status']
             for row in records['condition_summaries']],
            ['complete', 'complete']
        )
        self.assertTrue(all(
            row['loss_rate_absorbance_per_s'] > 0.0
            for row in records['well_metrics']
        ))

    def test_missing_or_nonfinite_reader_pair_fails_closed(self):
        manifest, observed, conditions = self._fixture()
        del observed['scan_002']['well_d']

        with self.assertRaisesRegex(
                AutoStabilityDebugEvidenceError,
                'refuses to bypass missing or non-finite reader data'):
            build_synthetic_companion_evidence(manifest, observed, conditions)


if __name__ == '__main__':
    unittest.main()
