'''Source-isolated Stage-14B adaptive-retirement controller tests.

These tests do not import the hardware-dependent controller. They verify that
the live scheduler can use only immutable raw-reader reporting evidence,
converts reporting-relative time onto the observer's monotonic clock, and
fails closed when that evidence cannot be loaded.
'''

import ast
import math
import os
import unittest

from auto_stability_lifecycle import evaluate_well_monitoring_lifecycle


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


def _load_methods(method_names):
    with open(CONTROLLER_PATH, encoding='utf-8') as source_file:
        tree = ast.parse(source_file.read())
    auto_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
    )
    methods = {
        node.name: node for node in auto_class.body
        if isinstance(node, ast.FunctionDef)
    }
    extracted_class = ast.ClassDef(
        name='AutoContr', bases=[], keywords=[],
        body=[methods[name] for name in method_names], decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(
        body=[extracted_class], type_ignores=[]
    ))
    namespace = {
        'math': math,
        'evaluate_well_monitoring_lifecycle': (
            evaluate_well_monitoring_lifecycle
        ),
    }
    exec(compile(module, CONTROLLER_PATH, 'exec'), namespace)
    return namespace['AutoContr']


class _ObserverStub(object):
    def __init__(self, active_records):
        self.active_records = list(active_records)
        self.retirements = []

    def get_active_wells(self):
        return [
            dict(record) for record in self.active_records
            if record.get('activation_status', 'active') == 'active'
        ]

    def record_adaptive_lifecycle_retirements(self, records, policy):
        self.retirements.append((records, policy))
        retired_wellnames = set(records)
        for record in self.active_records:
            if record['wellname'] in retired_wellnames:
                record['activation_status'] = 'window_complete'


class Stage14BAdaptiveRetirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AutoController = _load_methods([
            '_get_auto_stability_raw_lifecycle_observations',
            '_record_auto_stability_adaptive_retirements',
        ])

    def _controller(self):
        controller = self.AutoController()
        controller.robo_params = {
            'auto_stability_min_peak_absorbance': 0.10,
        }
        controller.auto_stability_observer = _ObserverStub([
            {
                'wellname': 'autowell0C1.0',
                'activation_monotonic_s': 100.0,
            },
        ])
        return controller

    def test_raw_lifecycle_observations_disable_debug_substitution(self):
        controller = self._controller()
        calls = []
        controller._build_auto_stability_reporting_records = (
            lambda **kwargs: calls.append(kwargs) or {
                'trajectory_rows': [
                    {
                        'wellname': 'autowell0C1.0',
                        'metric_eligible_interval': True,
                        'elapsed_s_from_trigger_to_scan_start': 12.5,
                        'reference_absorbance': 0.42,
                        'lambda_max_nm': 625.0,
                    },
                    {
                        'wellname': 'other',
                        'metric_eligible_interval': True,
                        'elapsed_s_from_trigger_to_scan_start': 12.5,
                        'reference_absorbance': 0.42,
                    },
                ],
            }
        )
        observations = controller._get_auto_stability_raw_lifecycle_observations(
            controller.auto_stability_observer.get_active_wells()
        )
        self.assertEqual([{'use_debug_evidence': False}], calls)
        self.assertEqual(112.5, observations['autowell0C1.0'][0]['timestamp_s'])
        self.assertEqual(0.42, observations['autowell0C1.0'][0][
            'reference_absorbance'
        ])

    def test_raw_lifecycle_load_failure_fails_closed(self):
        controller = self._controller()

        def _fail(**unused_kwargs):
            raise RuntimeError('malformed raw reader file')

        controller._build_auto_stability_reporting_records = _fail
        self.assertEqual(
            {},
            controller._get_auto_stability_raw_lifecycle_observations(
                controller.auto_stability_observer.get_active_wells()
            )
        )

    def test_adaptive_retirement_sends_only_eligible_terminal_result(self):
        controller = self._controller()
        controller.auto_stability_observer.active_records.append({
            'wellname': 'autowell1C1.0',
            'activation_monotonic_s': 130.0,
        })
        controller._get_auto_stability_raw_lifecycle_observations = (
            lambda records: {
                'autowell0C1.0': [
                    {'timestamp_s': 100.0, 'reference_absorbance': 0.02},
                    {'timestamp_s': 110.0, 'reference_absorbance': 0.03},
                    {'timestamp_s': 120.0, 'reference_absorbance': 0.02},
                    {'timestamp_s': 130.0, 'reference_absorbance': 0.03},
                    {'timestamp_s': 140.0, 'reference_absorbance': 0.02},
                ],
                'autowell1C1.0': [
                    {'timestamp_s': 130.0, 'reference_absorbance': 0.50},
                    {'timestamp_s': 140.0, 'reference_absorbance': 0.49},
                ],
            }
        )
        policy = {
            'adaptive_retirement_enabled': True,
            'decision_horizon_s': 30.0,
            'max_observation_window_s': 90.0,
            'plateau_min_tail_observation_count': 3,
            'plateau_consecutive_interval_count': 2,
            'plateau_max_absorbance_slope_per_s': 0.002,
        }
        remaining_observations = (
            controller._record_auto_stability_adaptive_retirements(
                policy, 140.0
            )
        )
        retirements = controller.auto_stability_observer.retirements
        self.assertEqual(1, len(retirements))
        result = retirements[0][0]['autowell0C1.0']
        self.assertEqual('terminal_low_signal', result['retirement_reason'])
        self.assertEqual(
            ['autowell1C1.0'], sorted(remaining_observations)
        )

    def test_fixed_policy_never_loads_or_retires_early(self):
        controller = self._controller()
        controller._get_auto_stability_raw_lifecycle_observations = (
            lambda records: self.fail('fixed policy must not load raw evidence')
        )
        self.assertEqual(
            {},
            controller._record_auto_stability_adaptive_retirements(
                {
                    'adaptive_retirement_enabled': False,
                    'decision_horizon_s': 30.0,
                    'max_observation_window_s': 90.0,
                    'plateau_min_tail_observation_count': 3,
                    'plateau_consecutive_interval_count': 2,
                    'plateau_max_absorbance_slope_per_s': 0.002,
                },
                140.0
            )
        )
        self.assertEqual([], controller.auto_stability_observer.retirements)


if __name__ == '__main__':
    unittest.main()
