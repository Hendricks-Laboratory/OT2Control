'''Hardware-free tests for the Stage-14F3 stability dashboard.'''

import ast
import copy
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

from auto_stability_dashboard import build_stability_dashboard_records
from auto_stability_dashboard_plotting import write_stability_dashboard_plot_suite


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / 'controller.py'


def _records():
    return {
        'well_metrics': [
            {
                'wellname': 'well_a', 'condition_id': 'condition_0',
                'condition_type': 'seed', 'batch_number': 0,
                'reaction_number': 0, 'status': 'eligible',
                'loss_rate_absorbance_per_s': 0.001,
                'peak_absorbance': 0.60, 'reference_wavelength_nm': 620,
                'metric_input_scan_count': 3, 'raw_trajectory_scan_count': 4,
            },
            {
                'wellname': 'well_b', 'condition_id': 'condition_0',
                'condition_type': 'seed', 'batch_number': 0,
                'reaction_number': 0, 'status': 'low_signal',
                'loss_rate_absorbance_per_s': None,
                'peak_absorbance': 0.03, 'reference_wavelength_nm': 620,
                'metric_input_scan_count': 2, 'raw_trajectory_scan_count': 3,
            },
            {
                'wellname': 'well_c', 'condition_id': 'condition_1',
                'condition_type': 'optimizer_selected', 'batch_number': 1,
                'reaction_number': 0, 'status': 'missing_stability_trajectory',
                'loss_rate_absorbance_per_s': None,
                'peak_absorbance': None, 'reference_wavelength_nm': None,
                'metric_input_scan_count': 0, 'raw_trajectory_scan_count': 0,
            },
        ],
        'condition_summaries': [
            {
                'condition_id': 'condition_0', 'batch_number': 0,
                'reaction_number': 0, 'eligible_well_count': 1,
                'total_well_count': 2, 'condition_stability_status': 'partial',
                'reference_peak_eligible_well_count': 2,
                'condition_signal_status': 'complete',
                'condition_loss_rate_mean_absorbance_per_s': 0.001,
                'condition_loss_rate_sample_sd_absorbance_per_s': None,
                'condition_reference_peak_absorbance_mean': 0.315,
                'condition_reference_peak_absorbance_sample_sd': 0.40,
            },
            {
                'condition_id': 'condition_1', 'batch_number': 1,
                'reaction_number': 0, 'eligible_well_count': 0,
                'total_well_count': 1,
                'condition_stability_status': 'no_eligible_wells',
                'reference_peak_eligible_well_count': 0,
                'condition_signal_status': 'no_eligible_signal_wells',
            },
        ],
        'qc_rows': [
            {
                'wellname': 'well_b', 'raw_scan_id': '', 'qc_status': 'low_signal',
                'qc_reason': 'Peak reference absorbance is below the minimum.',
            },
            {
                'wellname': 'well_c', 'raw_scan_id': '',
                'qc_status': 'missing_stability_trajectory',
                'qc_reason': 'No manifest-linked raw stability trajectory.',
            },
        ],
        'trajectory_rows': [
            {
                'raw_scan_id': 'scan_001', 'wellname': 'well_a',
                'metric_eligible_interval': True,
            },
            {
                'raw_scan_id': 'scan_001', 'wellname': 'well_b',
                'metric_eligible_interval': True,
            },
            {
                'raw_scan_id': 'scan_002', 'wellname': 'well_a',
                'metric_eligible_interval': False,
            },
        ],
        'timing_rows': [
            {
                'raw_scan_id': 'scan_001', 'wellname': 'well_a',
                'observation_reason': 'each_completion',
                'trigger_completed_at_utc': '2026-11-12T00:00:00+00:00',
                'scan_started_at_utc': '2026-11-12T00:00:10+00:00',
                'scan_completed_at_utc': '2026-11-12T00:00:18+00:00',
                'cadence_status': 'not_cadenced',
            },
            {
                'raw_scan_id': 'scan_001', 'wellname': 'well_b',
                'observation_reason': 'each_completion',
                'trigger_completed_at_utc': '2026-11-12T00:00:03+00:00',
                'scan_started_at_utc': '2026-11-12T00:00:10+00:00',
                'scan_completed_at_utc': '2026-11-12T00:00:18+00:00',
                'cadence_status': 'not_cadenced',
            },
            {
                'raw_scan_id': 'scan_002', 'wellname': 'well_a',
                'observation_reason': 'cadenced_active_set',
                'trigger_completed_at_utc': '2026-11-12T00:00:00+00:00',
                'scan_started_at_utc': '2026-11-12T00:01:00+00:00',
                'scan_completed_at_utc': '2026-11-12T00:01:09+00:00',
                'observed_cadence_delay_s': 12.0,
                'cadence_status': 'deferred_by_serial_reader',
            },
        ],
        'debug_evidence_rows': [{'debug_only': True}],
    }


class AutoStabilityDashboardTests(unittest.TestCase):
    def test_replicate_completeness_preserves_partial_and_missing_wells(self):
        records = _records()
        original = copy.deepcopy(records)
        dashboard = build_stability_dashboard_records(records)
        self.assertEqual(records, original)
        rows = dashboard['replicate_rows']
        self.assertEqual(['well_a', 'well_b', 'well_c'], [
            row['wellname'] for row in rows
        ])
        low_signal = next(row for row in rows if row['wellname'] == 'well_b')
        self.assertFalse(low_signal['loss_rate_qc_included'])
        self.assertEqual('low_signal', low_signal['metric_qc_status'])
        self.assertIn('below the minimum', low_signal['metric_qc_reason'])
        missing = next(row for row in rows if row['wellname'] == 'well_c')
        self.assertEqual(1, missing['condition_total_well_count'])
        self.assertEqual(0, missing['condition_eligible_well_count'])

    def test_throughput_reduces_a_multiwell_scan_exactly_once(self):
        dashboard = build_stability_dashboard_records(_records())
        rows = dashboard['throughput_rows']
        self.assertEqual(['scan_001', 'scan_002'], [
            row['raw_scan_id'] for row in rows
        ])
        first, second = rows
        self.assertEqual(2, first['active_well_count'])
        self.assertEqual(2, first['decision_eligible_well_count'])
        self.assertEqual(8.0, first['scan_duration_s'])
        self.assertEqual(2, first['first_observation_well_count'])
        self.assertAlmostEqual(8.5, first['trigger_to_first_scan_delay_mean_s'])
        self.assertEqual(1, second['active_well_count'])
        self.assertEqual(1, second['audit_only_well_count'])
        self.assertEqual(1, second['cadence_deferred_well_count'])
        self.assertEqual(12.0, second['cadence_delay_mean_s'])

    def test_writer_creates_four_organized_dashboard_figures(self):
        records = _records()
        records['dashboard_records'] = build_stability_dashboard_records(records)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_stability_dashboard_plot_suite(records, directory)
            relative_paths = {
                os.path.relpath(path, directory) for path in paths
            }
            self.assertEqual(4, len(paths))
            self.assertIn(os.path.join(
                'replicate_completeness',
                'stability_loss_rate_replicates_final.png',
            ), relative_paths)
            self.assertIn(os.path.join(
                'replicate_completeness',
                'stability_reference_peak_replicates_final.png',
            ), relative_paths)
            self.assertIn(os.path.join(
                'replicate_completeness',
                'stability_well_completeness_qc_final.png',
            ), relative_paths)
            self.assertIn(os.path.join(
                'reader_throughput',
                'stability_reader_throughput_final.png',
            ), relative_paths)
            self.assertTrue(all(
                os.path.isfile(path) and os.path.getsize(path) > 0
                for path in paths
            ))

    def test_controller_routes_finalized_records_only_to_dashboard(self):
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
        plotting_source = ast.get_source_segment(
            source, methods['_plot_auto_stability_reporting_artifacts']
        )
        records_source = ast.get_source_segment(
            source, methods['_build_auto_stability_reporting_records']
        )
        self.assertIn('write_stability_dashboard_plot_suite(', plotting_source)
        self.assertIn("summary['records']", plotting_source)
        self.assertNotIn('portal.', plotting_source)
        self.assertNotIn('pr.', plotting_source)
        self.assertIn('build_stability_dashboard_records(records)', records_source)


if __name__ == '__main__':
    unittest.main()
