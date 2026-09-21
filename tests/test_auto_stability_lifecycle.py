'''Stage-13A pure lifecycle tests with no controller, reader, or robot imports.'''

import ast
import os
import unittest

from auto_stability_lifecycle import (
    AutoStabilityLifecycleError,
    RETIREMENT_REASON_MAX_WINDOW_EXPIRED,
    RETIREMENT_REASON_PLATEAU_CONFIRMED,
    RETIREMENT_REASON_TERMINAL_LOW_SIGNAL,
    WELL_MONITORING_STATE_ACTIVE,
    WELL_MONITORING_STATE_DECISION_READY,
    WELL_MONITORING_STATE_MAX_WINDOW_EXPIRED,
    WELL_MONITORING_STATE_PENDING_TRIGGER,
    WELL_MONITORING_STATE_PLATEAU_CONFIRMED,
    WELL_MONITORING_STATE_TERMINAL_QC_EXCLUDED,
    build_stability_monitoring_policy,
    evaluate_condition_decision_readiness,
    evaluate_stability_plateau,
    evaluate_well_monitoring_lifecycle,
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIFECYCLE_PATH = os.path.join(REPOSITORY_ROOT, 'auto_stability_lifecycle.py')


def _policy(**overrides):
    values = {
        'decision_horizon_s': 30.0,
        'max_observation_window_s': 90.0,
        'adaptive_retirement_enabled': True,
        'plateau_min_tail_observation_count': 3,
        'plateau_consecutive_interval_count': 2,
        'plateau_max_absorbance_slope_per_s': 0.002,
    }
    values.update(overrides)
    return build_stability_monitoring_policy(**values)


def _plateau_observations():
    return [
        {'timestamp_s': 0.0, 'reference_absorbance': 0.10},
        {'timestamp_s': 10.0, 'reference_absorbance': 0.80},
        {'timestamp_s': 20.0, 'reference_absorbance': 0.50},
        {'timestamp_s': 30.0, 'reference_absorbance': 0.49},
        {'timestamp_s': 40.0, 'reference_absorbance': 0.48},
    ]


class AutoStabilityLifecycleTests(unittest.TestCase):
    def test_lifecycle_module_imports_only_pure_stability_dependencies(self):
        with open(LIFECYCLE_PATH, encoding='utf-8') as source_file:
            source = source_file.read()
        tree = ast.parse(source, filename=LIFECYCLE_PATH)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.add(node.module)
        self.assertEqual(
            imported_modules,
            {'__future__', 'math', 'auto_stability'}
        )

    def test_policy_rejects_a_maximum_window_shorter_than_decision_horizon(self):
        with self.assertRaisesRegex(
                AutoStabilityLifecycleError, 'greater than or equal'):
            _policy(decision_horizon_s=91.0)

    def test_policy_requires_tail_observations_to_cover_confirmations(self):
        with self.assertRaisesRegex(
                AutoStabilityLifecycleError, 'at least one more'):
            _policy(
                plateau_min_tail_observation_count=2,
                plateau_consecutive_interval_count=2
            )
        with self.assertRaisesRegex(AutoStabilityLifecycleError, 'boolean'):
            _policy(adaptive_retirement_enabled='off')
        with self.assertRaisesRegex(AutoStabilityLifecycleError, 'missing'):
            evaluate_stability_plateau(
                _plateau_observations(), 0.0, 0.20, {}
            )

    def test_plateau_requires_post_peak_decline_and_sustained_low_slope(self):
        plateau = evaluate_stability_plateau(
            _plateau_observations(), 0.0, 0.20, _policy()
        )
        self.assertTrue(plateau['plateau_confirmed'])
        self.assertEqual(
            plateau['reason'], RETIREMENT_REASON_PLATEAU_CONFIRMED
        )
        self.assertEqual(len(plateau['recent_interval_slopes_absorbance_per_s']), 2)

        no_decline = evaluate_stability_plateau([
            {'timestamp_s': 0.0, 'reference_absorbance': 0.20},
            {'timestamp_s': 10.0, 'reference_absorbance': 0.60},
            {'timestamp_s': 20.0, 'reference_absorbance': 0.60},
            {'timestamp_s': 30.0, 'reference_absorbance': 0.60},
        ], 0.0, 0.20, _policy())
        self.assertFalse(no_decline['plateau_confirmed'])
        self.assertEqual(no_decline['reason'], 'no_post_peak_decline')

    def test_lifecycle_preserves_decision_horizon_before_plateau_retirement(self):
        before = evaluate_well_monitoring_lifecycle(
            0.0, 29.0, _plateau_observations(), 0.20, _policy()
        )
        self.assertEqual(before['state'], WELL_MONITORING_STATE_ACTIVE)
        self.assertFalse(before['decision_ready'])
        self.assertFalse(before['retired'])

        after = evaluate_well_monitoring_lifecycle(
            0.0, 40.0, _plateau_observations(), 0.20, _policy()
        )
        self.assertEqual(after['state'], WELL_MONITORING_STATE_PLATEAU_CONFIRMED)
        self.assertTrue(after['decision_ready'])
        self.assertTrue(after['retired'])
        self.assertEqual(
            after['retirement_reason'], RETIREMENT_REASON_PLATEAU_CONFIRMED
        )

    def test_lifecycle_never_uses_a_future_observation_for_current_state(self):
        result = evaluate_well_monitoring_lifecycle(
            0.0, 30.0, _plateau_observations(), 0.20, _policy()
        )
        self.assertEqual(result['state'], WELL_MONITORING_STATE_DECISION_READY)
        self.assertFalse(result['retired'])

    def test_fixed_window_policy_does_not_retire_an_early_plateau(self):
        result = evaluate_well_monitoring_lifecycle(
            0.0,
            40.0,
            _plateau_observations(),
            0.20,
            _policy(adaptive_retirement_enabled=False)
        )
        self.assertEqual(result['state'], WELL_MONITORING_STATE_DECISION_READY)
        self.assertTrue(result['decision_ready'])
        self.assertFalse(result['retired'])

    def test_maximum_window_and_low_signal_are_terminal_only_after_horizon(self):
        expired = evaluate_well_monitoring_lifecycle(
            0.0, 90.0, _plateau_observations(), 0.20, _policy(
                adaptive_retirement_enabled=False
            )
        )
        self.assertEqual(expired['state'], WELL_MONITORING_STATE_MAX_WINDOW_EXPIRED)
        self.assertEqual(
            expired['retirement_reason'], RETIREMENT_REASON_MAX_WINDOW_EXPIRED
        )

        low_signal_before = evaluate_well_monitoring_lifecycle(
            0.0, 20.0,
            [
                {'timestamp_s': 0.0, 'reference_absorbance': 0.02},
                {'timestamp_s': 10.0, 'reference_absorbance': 0.01},
            ],
            0.20,
            _policy(adaptive_retirement_enabled=False)
        )
        self.assertEqual(low_signal_before['state'], WELL_MONITORING_STATE_ACTIVE)

        low_signal_after = evaluate_well_monitoring_lifecycle(
            0.0, 30.0,
            [
                {'timestamp_s': 0.0, 'reference_absorbance': 0.02},
                {'timestamp_s': 10.0, 'reference_absorbance': 0.01},
            ],
            0.20,
            _policy(adaptive_retirement_enabled=False)
        )
        self.assertEqual(
            low_signal_after['state'], WELL_MONITORING_STATE_TERMINAL_QC_EXCLUDED)
        self.assertEqual(
            low_signal_after['retirement_reason'], RETIREMENT_REASON_TERMINAL_LOW_SIGNAL)

    def test_pending_and_condition_decision_readiness_are_unambiguous(self):
        pending = evaluate_well_monitoring_lifecycle(
            0.0, 0.0, [], 0.20, _policy(), trigger_completed=False
        )
        self.assertEqual(pending['state'], WELL_MONITORING_STATE_PENDING_TRIGGER)

        readiness = evaluate_condition_decision_readiness([
            {'wellname': 'A1', 'decision_ready': True},
            {'wellname': 'A2', 'decision_ready': False},
        ])
        self.assertFalse(readiness['decision_ready'])
        self.assertEqual(readiness['pending_wellnames'], ['A2'])

        with self.assertRaisesRegex(AutoStabilityLifecycleError, 'duplicate'):
            evaluate_condition_decision_readiness([
                {'wellname': 'A1', 'decision_ready': True},
                {'wellname': 'A1', 'decision_ready': True},
            ])


if __name__ == '__main__':
    unittest.main()
