'''Hardware-free regression tests for the Stage 14G metric-study harness.'''

import ast
import pathlib
import unittest

from auto_stability import (
    METRIC_STATUS_ELIGIBLE,
    METRIC_STATUS_LOW_SIGNAL,
    compute_stability_metrics,
)
from auto_stability_metric_study import (
    STUDY_STATUS_ASSESSMENT_WINDOW_UNOBSERVED,
    STUDY_STATUS_INSUFFICIENT_POST_PEAK_OBSERVATIONS,
    compare_stability_metric_estimators,
    compute_fixed_horizon_observed_loss_rate,
    compute_robust_post_peak_slope,
    simulate_linear_decay_observations,
)


class Stage14GMetricRobustnessTests(unittest.TestCase):
    '''Prove comparator behavior without involving controller or hardware code.'''

    def test_production_tail_minimum_is_returned_without_alteration(self):
        observations = simulate_linear_decay_observations(
            [0, 60, 120, 180], 0.8, 0.001
        )
        expected = compute_stability_metrics(
            observations, 0.1, trigger_timestamp_s=0, observation_window_s=300
        )
        compared = compare_stability_metric_estimators(
            observations, 0.1, 120, 30,
            trigger_timestamp_s=0, observation_window_s=300,
        )
        self.assertEqual(expected, compared['production_tail_minimum'])
        self.assertEqual(METRIC_STATUS_ELIGIBLE, expected['status'])

    def test_fixed_horizon_uses_first_observed_endpoint_without_interpolation(self):
        observations = [
            {'timestamp_s': 0, 'reference_absorbance': 0.8},
            {'timestamp_s': 70, 'reference_absorbance': 0.73},
            {'timestamp_s': 130, 'reference_absorbance': 0.67},
            {'timestamp_s': 200, 'reference_absorbance': 0.60},
        ]
        result = compute_fixed_horizon_observed_loss_rate(
            observations, 0.1, 120, 20, trigger_timestamp_s=0
        )
        self.assertEqual(METRIC_STATUS_ELIGIBLE, result['status'])
        self.assertEqual(130, result['endpoint_time_s'])
        self.assertAlmostEqual(0.001, result['loss_rate_absorbance_per_s'])

    def test_fixed_horizon_marks_an_unobserved_assessment_window(self):
        observations = simulate_linear_decay_observations(
            [0, 60, 260], 0.8, 0.001
        )
        result = compute_fixed_horizon_observed_loss_rate(
            observations, 0.1, 120, 20, trigger_timestamp_s=0
        )
        self.assertEqual(
            STUDY_STATUS_ASSESSMENT_WINDOW_UNOBSERVED, result['status']
        )
        self.assertIsNone(result['loss_rate_absorbance_per_s'])

    def test_robust_slope_requires_peak_plus_two_later_observations(self):
        observations = simulate_linear_decay_observations([0, 60], 0.8, 0.001)
        result = compute_robust_post_peak_slope(observations, 0.1)
        self.assertEqual(
            STUDY_STATUS_INSUFFICIENT_POST_PEAK_OBSERVATIONS,
            result['status'],
        )

    def test_dense_and_sparse_linear_trajectories_recover_known_rate(self):
        for timestamps in ([0, 60, 120, 180, 240], [0, 120, 240]):
            observations = simulate_linear_decay_observations(
                timestamps, 0.8, 0.001
            )
            compared = compare_stability_metric_estimators(
                observations, 0.1, 120, 60, trigger_timestamp_s=0
            )
            for result in compared.values():
                self.assertEqual(METRIC_STATUS_ELIGIBLE, result['status'])
                self.assertAlmostEqual(
                    0.001, result['loss_rate_absorbance_per_s']
                )

    def test_low_signal_remains_noneligible_in_study_comparators(self):
        observations = simulate_linear_decay_observations(
            [0, 60, 120], 0.05, 0.001
        )
        compared = compare_stability_metric_estimators(
            observations, 0.1, 60, 30, trigger_timestamp_s=0
        )
        for result in compared.values():
            self.assertEqual(METRIC_STATUS_LOW_SIGNAL, result['status'])

    def test_study_module_is_statically_isolated_from_execution_modules(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        source = (root / 'auto_stability_metric_study.py').read_text(
            encoding='utf-8'
        )
        parsed = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        forbidden = {
            'controller', 'optimizers', 'ot2_robot', 'Armchair', 'os',
            'subprocess', 'pandas', 'openpyxl',
        }
        self.assertFalse(forbidden.intersection(imported_modules))
        self.assertNotIn('auto_stability_metric_study', (root / 'controller.py').read_text())
        self.assertNotIn('auto_stability_metric_study', (root / 'optimizers.py').read_text())


if __name__ == '__main__':
    unittest.main()
