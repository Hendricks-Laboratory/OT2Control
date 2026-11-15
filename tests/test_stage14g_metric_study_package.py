'''Hardware-free tests for the Stage 14G2 offline study package.'''

import ast
import csv
import json
import pathlib
import tempfile
import unittest

from auto_stability import METRIC_STATUS_ELIGIBLE
from auto_stability_metric_study import (
    compare_stage14g2_metrics,
)
from auto_stability_metric_study_package import (
    AutoStabilityMetricStudyPackageError,
    EVIDENCE_SOURCE_WATER_TIMING,
    build_synthetic_scenarios,
    evaluate_trajectory_records,
    read_saved_trajectory_export,
    summarize_known_rankings,
    write_study_package,
)


class Stage14GMetricStudyPackageTests(unittest.TestCase):
    '''Verify deterministic offline study behavior and execution isolation.'''

    def test_normalized_retention_is_amplitude_invariant(self):
        low = [
            {'timestamp_s': 0, 'reference_absorbance': 0.25},
            {'timestamp_s': 120, 'reference_absorbance': 0.20},
            {'timestamp_s': 240, 'reference_absorbance': 0.16},
        ]
        high = [
            {'timestamp_s': 0, 'reference_absorbance': 0.80},
            {'timestamp_s': 120, 'reference_absorbance': 0.64},
            {'timestamp_s': 240, 'reference_absorbance': 0.512},
        ]
        low_results = compare_stage14g2_metrics(low, 0.1, 120, 1)
        high_results = compare_stage14g2_metrics(high, 0.1, 120, 1)
        for metric_name in ('normalized_retention', 'log_loss_rate'):
            self.assertEqual(
                METRIC_STATUS_ELIGIBLE, low_results[metric_name]['status']
            )
            self.assertEqual(
                METRIC_STATUS_ELIGIBLE, high_results[metric_name]['status']
            )
        self.assertAlmostEqual(
            low_results['normalized_retention']['retention_fraction'],
            high_results['normalized_retention']['retention_fraction'],
        )
        self.assertAlmostEqual(
            low_results['log_loss_rate']['log_loss_rate_per_s'],
            high_results['log_loss_rate']['log_loss_rate_per_s'],
        )
        self.assertLess(
            low_results['fixed_horizon_raw_loss_rate'][
                'loss_rate_absorbance_per_s'
            ],
            high_results['fixed_horizon_raw_loss_rate'][
                'loss_rate_absorbance_per_s'
            ],
        )

    def test_synthetic_scenarios_are_deterministic_and_rankable(self):
        first = build_synthetic_scenarios(seed=31)
        second = build_synthetic_scenarios(seed=31)
        self.assertEqual(first, second)
        rows = evaluate_trajectory_records(first, 0.1, 120, 60)
        ranking_rows = summarize_known_rankings(rows)
        self.assertEqual(5, len(ranking_rows))
        self.assertTrue(all(
            row['known_truth_pair_count'] > 0 for row in ranking_rows
        ))

    def test_saved_debug_export_is_labeled_water_timing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = pathlib.Path(temporary_directory) / 'trajectories.csv'
            path.write_text(
                'wellname,condition_id,elapsed_s_from_trigger_to_scan_start,'
                'reference_absorbance,debug_only\n'
                'wellA,conditionA,70,0.80,True\n'
                'wellA,conditionA,180,0.70,True\n'
                'wellA,conditionA,290,0.60,True\n',
                encoding='utf-8',
            )
            records = read_saved_trajectory_export(path)
        self.assertEqual(1, len(records))
        self.assertEqual(EVIDENCE_SOURCE_WATER_TIMING,
                         records[0]['evidence_source'])
        self.assertEqual(3, len(records[0]['observations']))

    def test_non_debug_export_requires_explicit_chemistry_evidence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = pathlib.Path(temporary_directory) / 'trajectories.csv'
            path.write_text(
                'wellname,condition_id,elapsed_s_from_trigger_to_scan_start,'
                'reference_absorbance,debug_only\n'
                'wellA,conditionA,70,0.80,False\n'
                'wellA,conditionA,180,0.70,False\n',
                encoding='utf-8',
            )
            with self.assertRaises(AutoStabilityMetricStudyPackageError):
                read_saved_trajectory_export(path)
            records = read_saved_trajectory_export(
                path, evidence_source='chemistry'
            )
        self.assertEqual('chemistry', records[0]['evidence_source'])

    def test_missing_assessment_endpoint_has_no_numerical_score(self):
        records = [record for record in build_synthetic_scenarios(seed=9)
                   if record['base_scenario_id'] == 'missing_endpoint']
        rows = evaluate_trajectory_records(records, 0.1, 120, 1)
        endpoint_rows = [
            row for row in rows
            if row['metric_name'] in (
                'fixed_horizon_raw_loss_rate', 'normalized_retention',
                'log_loss_rate'
            )
        ]
        self.assertTrue(endpoint_rows)
        self.assertTrue(all(
            row['status'] == 'assessment_window_unobserved'
            and row['comparison_value'] is None
            for row in endpoint_rows
        ))

    def test_generated_tabular_artifacts_are_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            first_dir = root / 'first'
            second_dir = root / 'second'
            first_metadata = write_study_package(first_dir, seed=102)
            second_metadata = write_study_package(second_dir, seed=102)
            self.assertEqual(
                first_metadata['metric_rows_sha256'],
                second_metadata['metric_rows_sha256'],
            )
            for filename in (
                    'scenario_metric_results.csv',
                    'metric_availability_summary.csv',
                    'known_truth_ranking_summary.csv',
                    'study_metadata.json'):
                self.assertEqual(
                    (first_dir / filename).read_bytes(),
                    (second_dir / filename).read_bytes(),
                )
            for filename in (
                    'stage14g2_metric_study_report.md',
                    'amplitude_confounding.png',
                    'scan_density_sensitivity.png',
                    'missing_endpoint_behavior.png',
                    'estimator_availability.png'):
                self.assertTrue((first_dir / filename).is_file(), filename)

    def test_generated_rows_contain_explicit_evidence_labels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = pathlib.Path(temporary_directory) / 'study'
            write_study_package(output_dir, seed=411)
            with (output_dir / 'scenario_metric_results.csv').open(
                    newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertEqual({'synthetic'}, {row['evidence_source'] for row in rows})

    def test_package_remains_statically_isolated_from_execution_modules(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        source = (root / 'auto_stability_metric_study_package.py').read_text(
            encoding='utf-8'
        )
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        forbidden = {
            'controller', 'optimizers', 'ot2_robot', 'Armchair', 'gspread',
            'openpyxl', 'subprocess', 'socket', 'serial',
        }
        self.assertFalse(forbidden.intersection(imported_modules))
        for filename in ('controller.py', 'optimizers.py', 'ot2_robot.py'):
            self.assertNotIn(
                'auto_stability_metric_study_package',
                (root / filename).read_text(encoding='utf-8'),
            )

    def test_report_metadata_is_valid_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = pathlib.Path(temporary_directory) / 'study'
            write_study_package(output_dir, seed=87)
            metadata = json.loads(
                (output_dir / 'study_metadata.json').read_text(encoding='utf-8')
            )
        self.assertEqual('stage14g2-v1', metadata['study_schema_version'])


if __name__ == '__main__':
    unittest.main()
