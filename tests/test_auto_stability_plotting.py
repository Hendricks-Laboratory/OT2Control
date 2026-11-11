'''Hardware-free tests for Stage-14F1 final stability figures.'''

import copy
import ast
import os
import tempfile
import unittest
from pathlib import Path

from auto_stability_plotting import (
    build_stability_plot_data,
    write_stability_plot_suite,
)


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / 'controller.py'


def _records():
    return {
        'trajectory_rows': [
            {
                'raw_scan_id': 'scan_001',
                'wellname': 'autowell0C1.0',
                'elapsed_s_from_trigger_to_scan_start': 10,
                'reference_absorbance': 0.50,
                'metric_eligible_interval': True,
                'scan_started_at_utc': '2026-11-11T00:00:10+00:00',
            },
            {
                'raw_scan_id': 'scan_002',
                'wellname': 'autowell0C1.0',
                'elapsed_s_from_trigger_to_scan_start': 320,
                'reference_absorbance': 0.40,
                'metric_eligible_interval': False,
                'scan_started_at_utc': '2026-11-11T00:05:20+00:00',
            },
            {
                'raw_scan_id': 'scan_002',
                'wellname': 'autowell1C1.0',
                'elapsed_s_from_trigger_to_scan_start': 10,
                'reference_absorbance': 0.55,
                'metric_eligible_interval': True,
                'scan_started_at_utc': '2026-11-11T00:05:20+00:00',
            },
        ],
        'timing_rows': [
            {
                'raw_scan_id': 'scan_001',
                'wellname': 'autowell0C1.0',
                'observation_reason': 'each_completion',
                'scan_started_at_utc': '2026-11-11T00:00:10+00:00',
            },
            {
                'raw_scan_id': 'scan_002',
                'wellname': 'autowell0C1.0',
                'observation_reason': 'each_completion',
                'scan_started_at_utc': '2026-11-11T00:05:20+00:00',
            },
            {
                'raw_scan_id': 'scan_002',
                'wellname': 'autowell1C1.0',
                'observation_reason': 'each_completion',
                'scan_started_at_utc': '2026-11-11T00:05:20+00:00',
            },
        ],
        'lifecycle_rows': [
            {
                'event_type': 'trigger_transfer_completed',
                'wellname': 'autowell0C1.0',
                'batch_number': '0',
                'recorded_at_utc': '2026-11-11T00:00:00+00:00',
            },
            {
                'event_type': 'targeted_pipette_mix_acknowledged',
                'wellname': 'autowell0C1.0',
                'batch_number': '0',
                'recorded_at_utc': '2026-11-11T00:00:05+00:00',
            },
            {
                'event_type': 'decision_horizon_reached',
                'wellname': 'autowell0C1.0',
                'batch_number': '0',
                'recorded_at_utc': '2026-11-11T00:05:00+00:00',
            },
            {
                'event_type': 'trigger_transfer_completed',
                'wellname': 'autowell1C1.0',
                'batch_number': '1',
                'recorded_at_utc': '2026-11-11T00:05:10+00:00',
            },
            {
                'event_type': 'targeted_pipette_mix_acknowledged',
                'wellname': 'autowell1C1.0',
                'batch_number': '1',
                'recorded_at_utc': '2026-11-11T00:05:15+00:00',
            },
        ],
        'well_metrics': [
            {
                'wellname': 'autowell0C1.0',
                'batch_number': 0,
                'condition_id': 'condition_0',
                'condition_type': 'seed',
                'status': 'eligible',
            },
            {
                'wellname': 'autowell1C1.0',
                'batch_number': 1,
                'condition_id': 'condition_1',
                'condition_type': 'optimizer_selected',
                'status': 'eligible',
            },
        ],
        'condition_summaries': [
            {
                'batch_number': 0,
                'reaction_number': 0,
                'condition_loss_rate_mean_absorbance_per_s': 0.001,
                'condition_loss_rate_sample_sd_absorbance_per_s': 0.0001,
                'eligible_well_count': 1,
                'total_well_count': 1,
            },
            {
                'batch_number': 1,
                'reaction_number': 0,
                'condition_loss_rate_mean_absorbance_per_s': 0.002,
                'condition_loss_rate_sample_sd_absorbance_per_s': 0.0002,
                'eligible_well_count': 1,
                'total_well_count': 1,
            },
        ],
        'debug_evidence_rows': [{}],
        'stability_evidence_source': 'synthetic_companion_debug_only',
    }


class AutoStabilityPlottingTests(unittest.TestCase):
    def test_plot_data_preserves_cross_batch_membership_without_mutating_rows(self):
        records = _records()
        original = copy.deepcopy(records)
        plot_data = build_stability_plot_data(records)

        self.assertEqual(records, original)
        self.assertTrue(plot_data['debug_only'])
        self.assertEqual(
            ['autowell0C1.0', 'autowell1C1.0'],
            plot_data['wellnames'],
        )
        self.assertEqual(
            [['autowell0C1.0'], ['autowell0C1.0', 'autowell1C1.0']],
            [scan['wellnames'] for scan in plot_data['scans']],
        )
        self.assertEqual(
            1, plot_data['well_metadata']['autowell1C1.0']['batch_number']
        )

    def test_writer_creates_the_full_final_suite_in_organized_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_stability_plot_suite(
                _records(), directory,
                decision_horizon_s=300,
                maximum_window_s=1800,
            )
            relative_paths = {
                os.path.relpath(path, directory) for path in paths
            }
            self.assertEqual(7, len(paths))
            self.assertIn(
                'stability_per_well_trajectories_final.png', relative_paths
            )
            self.assertIn(
                os.path.join(
                    'well_trajectories',
                    'stability_well_trajectories_page_01_final.png',
                ),
                relative_paths,
            )
            self.assertIn(
                os.path.join(
                    'condition_summaries',
                    'stability_condition_loss_rate_final.png',
                ),
                relative_paths,
            )
            self.assertIn(
                os.path.join(
                    'lifecycle_timelines',
                    'stability_well_lifecycle_timeline_final.png',
                ),
                relative_paths,
            )
            self.assertIn(
                os.path.join(
                    'scan_membership',
                    'stability_raw_scan_membership_final.png',
                ),
                relative_paths,
            )
            self.assertIn(
                os.path.join(
                    'evidence_boundaries',
                    'stability_decision_vs_audit_final.png',
                ),
                relative_paths,
            )
            self.assertTrue(all(
                os.path.isfile(path) and os.path.getsize(path) > 0
                for path in paths
            ))

    def test_writer_rejects_an_impossible_evidence_window(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_stability_plot_suite(
                    _records(), directory,
                    decision_horizon_s=1801,
                    maximum_window_s=1800,
                )

    def test_controller_routes_only_completed_records_and_both_window_bounds(self):
        source = CONTROLLER_PATH.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(CONTROLLER_PATH))
        auto_contr = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
        )
        plot_method = next(
            node for node in auto_contr.body
            if isinstance(node, ast.FunctionDef) and
            node.name == '_plot_auto_stability_reporting_artifacts'
        )
        method_source = ast.get_source_segment(source, plot_method)
        self.assertIn('write_stability_plot_suite(', method_source)
        self.assertIn("summary['records']", method_source)
        self.assertIn("'auto_stability_decision_horizon_s'", method_source)
        self.assertIn("'auto_stability_max_observation_window_s'", method_source)
        self.assertNotIn('portal.', method_source)
        self.assertNotIn('pr.', method_source)


if __name__ == '__main__':
    unittest.main()
