'''Hardware-free tests for Stage-14F2 observed spectral evidence.''' 

import copy
import ast
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

from auto_stability_plotting import write_stability_spectral_evidence_plot_suite
from auto_stability_spectral_evidence import (
    build_raw_spectral_evidence_records,
    extract_observed_spectral_features,
)


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / 'controller.py'


def _manifest():
    return [
        {
            'event_type': 'raw_scan_completed',
            'raw_scan_id': '001',
            'raw_scan_relative_path': 'stability/raw_scans/scan_001.csv',
            'active_wellnames': 'well_a;well_b',
            'active_well_locations': 'well_a=A1;well_b=B1',
            'scan_started_at_utc': '2026-11-11T00:00:10+00:00',
        },
        {
            'event_type': 'raw_scan_completed',
            'raw_scan_id': '002',
            'raw_scan_relative_path': 'stability/raw_scans/scan_002.csv',
            'active_wellnames': 'well_a;well_b',
            'active_well_locations': 'well_a=A1;well_b=B1',
            'scan_started_at_utc': '2026-11-11T00:01:10+00:00',
        },
    ]


def _observed_spectra():
    return {
        '001': {
            'well_a': {'spectrum_by_wavelength_nm': {
                500.0: 0.0, 510.0: 0.5, 520.0: 1.0,
                530.0: 0.5, 540.0: 0.0,
            }},
            'well_b': {'spectrum_by_wavelength_nm': {
                500.0: 0.0, 510.0: 0.8, 520.0: 0.3,
                530.0: 0.9, 540.0: 0.0,
            }},
        },
        '002': {
            'well_a': {'spectrum_by_wavelength_nm': {
                500.0: 0.0, 510.0: 0.4, 520.0: 0.8,
                530.0: 0.4, 540.0: 0.0,
            }},
            'well_b': {'spectrum_by_wavelength_nm': {
                500.0: 0.0, 510.0: 0.7, 520.0: 0.2,
                530.0: 0.9, 540.0: 0.0,
            }},
        },
    }


def _trajectory_rows():
    return [
        {
            'raw_scan_id': scan_id,
            'wellname': wellname,
            'condition_id': 'condition_{}'.format(wellname),
            'trigger_completed_at_utc': '2026-11-11T00:00:00+00:00',
            'scan_started_at_utc': (
                '2026-11-11T00:00:10+00:00'
                if scan_id == '001' else '2026-11-11T00:01:10+00:00'
            ),
            'scan_completed_at_utc': '',
            'elapsed_s_from_trigger_to_scan_start': (
                10.0 if scan_id == '001' else 70.0
            ),
            'window_interval_status': 'fully_within_window',
            'metric_eligible_interval': True,
        }
        for scan_id in ('001', '002')
        for wellname in ('well_a', 'well_b')
    ]


class AutoStabilitySpectralEvidenceTests(unittest.TestCase):
    def test_defined_fwhm_uses_only_an_observed_half_height_bracket(self):
        features = extract_observed_spectral_features([
            (500.0, 0.0), (510.0, 0.5), (520.0, 1.0),
            (530.0, 0.5), (540.0, 0.0),
        ])
        self.assertEqual('defined', features['fwhm_status'])
        self.assertEqual(520.0, features['lambda_max_nm'])
        self.assertAlmostEqual(20.0, features['fwhm_nm'])

    def test_ambiguous_multiple_peaks_never_receive_a_fwhm(self):
        features = extract_observed_spectral_features([
            (500.0, 0.0), (510.0, 0.8), (520.0, 0.3),
            (530.0, 0.9), (540.0, 0.0),
        ])
        self.assertEqual('ambiguous_or_multiple_local_peaks',
                         features['fwhm_status'])
        self.assertIsNone(features['fwhm_nm'])

    def test_builder_preserves_real_observed_spectra_during_debug_context(self):
        manifest = _manifest()
        spectra = _observed_spectra()
        trajectories = _trajectory_rows()
        original = copy.deepcopy((manifest, spectra, trajectories))
        evidence = build_raw_spectral_evidence_records(
            manifest, spectra, trajectories,
            well_metrics=[
                {'wellname': 'well_a', 'batch_number': 0},
                {'wellname': 'well_b', 'batch_number': 1},
            ],
        )
        self.assertEqual((manifest, spectra, trajectories), original)
        self.assertEqual(4, len(evidence['evidence_rows']))
        first = evidence['evidence_rows'][0]
        self.assertEqual('raw_reader', first['stability_evidence_source'])
        self.assertFalse(first['debug_only'])
        self.assertEqual('A1', first['reader_location'])
        self.assertEqual(1.0, first['peak_absorbance'])
        self.assertAlmostEqual(20.0, first['fwhm_nm'])
        self.assertEqual(
            [(500.0, 0.0), (510.0, 0.5), (520.0, 1.0),
             (530.0, 0.5), (540.0, 0.0)],
            evidence['spectra_by_pair'][('001', 'well_a')],
        )

    def test_writer_creates_organized_raw_normalized_and_feature_outputs(self):
        evidence = build_raw_spectral_evidence_records(
            _manifest(), _observed_spectra(), _trajectory_rows(),
        )
        records = {
            'raw_spectral_evidence': evidence,
            'spectral_evidence_decision_horizon_s': 60.0,
            'spectral_evidence_maximum_window_s': 120.0,
            # Mimics the run-level DEBUG flag without replacing the observed
            # source data above. The plot footer must identify this correctly.
            'debug_evidence_rows': [{'debug_only': True}],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = write_stability_spectral_evidence_plot_suite(
                records, directory
            )
            relative_paths = {
                os.path.relpath(path, directory) for path in paths
            }
            self.assertEqual(5, len(paths))
            self.assertIn(os.path.join(
                'spectral_evidence', 'raw_amplitude',
                'stability_raw_spectra_page_01_final.png',
            ), relative_paths)
            self.assertIn(os.path.join(
                'spectral_evidence', 'normalized_shape',
                'stability_normalized_spectra_page_01_final.png',
            ), relative_paths)
            self.assertIn(os.path.join(
                'spectral_evidence', 'features',
                'stability_lambda_max_page_01_final.png',
            ), relative_paths)
            self.assertIn(os.path.join(
                'spectral_evidence', 'features',
                'stability_peak_height_page_01_final.png',
            ), relative_paths)
            self.assertIn(os.path.join(
                'spectral_evidence', 'features',
                'stability_fwhm_page_01_final.png',
            ), relative_paths)
            self.assertTrue(all(
                os.path.isfile(path) and os.path.getsize(path) > 0
                for path in paths
            ))

    def test_controller_keeps_observed_spectra_before_debug_substitution(self):
        source = CONTROLLER_PATH.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(CONTROLLER_PATH))
        auto_contr = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
        )
        methods = {
            node.name: node for node in auto_contr.body
            if isinstance(node, ast.FunctionDef)
        }
        reporting_source = ast.get_source_segment(
            source, methods['_build_auto_stability_reporting_records']
        )
        plot_source = ast.get_source_segment(
            source, methods['_plot_auto_stability_reporting_artifacts']
        )
        self.assertIn('observed_spectra_by_raw_scan', reporting_source)
        self.assertLess(
            reporting_source.index('observed_spectra_by_raw_scan'),
            reporting_source.index('build_synthetic_companion_evidence('),
        )
        self.assertLess(
            reporting_source.index('build_synthetic_companion_evidence('),
            reporting_source.index('build_raw_spectral_evidence_records('),
        )
        self.assertIn('write_stability_spectral_evidence_plot_suite(', plot_source)
        self.assertNotIn('portal.', plot_source)
        self.assertNotIn('pr.', plot_source)


if __name__ == '__main__':
    unittest.main()
