'''Hardware-free Stage-12F lexicographic selection tests.'''

import ast
import copy
import math
import os
import types
import unittest

import numpy as np
from scipy.optimize import minimize


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIMIZERS_PATH = os.path.join(REPOSITORY_ROOT, 'optimizers.py')


def _load_selection_methods():
    '''Load only pure Stage-12F methods without GPyOpt/hardware imports.'''
    with open(OPTIMIZERS_PATH, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=OPTIMIZERS_PATH)
    optimizer_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'OptimizationModel'
    )
    methods = {
        node.name: node for node in optimizer_class.body
        if isinstance(node, ast.FunctionDef)
    }
    method_names = (
        '_generate_reagent_masks',
        '_get_reagent_masks_for_current_settings',
        '_get_target_then_stability_candidate_diagnostics',
        '_optimize_single_mask_target_then_stability',
        '_record_target_then_stability_lambda_fallback',
        'getNextTargetThenStabilityReaction',
    )
    extracted_class = ast.ClassDef(
        name='OptimizationModel', bases=[], keywords=[],
        body=[methods[name] for name in method_names], decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(
        body=[extracted_class], type_ignores=[]
    ))
    namespace = {
        'copy': copy,
        'math': math,
        'minimize': minimize,
        'np': np,
    }
    exec(compile(module, OPTIMIZERS_PATH, 'exec'), namespace)
    return namespace['OptimizationModel']


OptimizationModel = _load_selection_methods()


class _FittedStabilityModel(object):
    status = 'fitted'
    X = np.asarray([[0.2], [0.8]], dtype=float)

    @staticmethod
    def predict_log10_loss_rate_distribution(recipes):
        values = np.asarray(recipes, dtype=float).reshape(-1, 1)
        # Lower normalized concentration is deliberately more stable.
        return values[:, 0], np.full(values.shape[0], 0.05)


class TargetThenStabilitySelectionTests(unittest.TestCase):
    def test_required_trigger_mask_filters_only_true_zero_masks(self):
        model = OptimizationModel.__new__(OptimizationModel)
        model.allow_true_zero = True
        model.true_zero_reagent_indices = [0, 1]
        model._get_dimension = types.MethodType(lambda self: 2, model)

        masks = model._get_reagent_masks_for_current_settings(
            required_on_reagent_indices=[1]
        )

        self.assertTrue(masks)
        self.assertTrue(all(mask[1] == 1 for mask in masks))
        self.assertTrue(any(mask[0] == 0 for mask in masks))

    def test_constrained_selection_is_target_first_then_lower_stability_rate(self):
        model = OptimizationModel.__new__(OptimizationModel)
        model.acquisition_mode = 'exploit'
        model.allow_true_zero = False
        model.target_value = 625.0
        model.terminal_verbosity = 'essential'
        model._get_dimension = types.MethodType(lambda self: 1, model)
        model._get_masked_bounds = types.MethodType(
            lambda self, mask: [(0.0, 1.0)], model
        )
        model._generate_feasible_masked_starting_points = types.MethodType(
            lambda self, mask, count: [np.asarray([0.2]), np.asarray([0.8])],
            model
        )
        model._optimize_single_mask = types.MethodType(
            lambda self, mask, n_restarts: {'x_active': np.asarray([0.5])},
            model
        )
        model._expand_masked_candidate_to_full_recipe = types.MethodType(
            lambda self, active, mask: np.asarray(active, dtype=float), model
        )
        model._get_candidate_volume_balance = types.MethodType(
            lambda self, recipe: {
                'volume_feasible': True,
                'water_volume': 10.0,
            }, model
        )
        model.predict_lambda_distribution_nm = types.MethodType(
            # Every point in [0.4, 0.6] is target compatible at tolerance 1.
            lambda self, recipe: (620.0 + 10.0 * float(recipe[0]), 2.0),
            model
        )

        proposed = model.getNextTargetThenStabilityReaction(
            stability_model=_FittedStabilityModel(),
            target_tolerance_nm=1.0,
        )

        self.assertEqual(len(proposed), 1)
        self.assertGreaterEqual(float(proposed[0][0]), 0.4 - 1e-6)
        self.assertLessEqual(float(proposed[0][0]), 0.6 + 1e-6)
        self.assertEqual(
            model.last_target_then_stability_selection_metadata[
                'stability_selection_status'
            ],
            'stability_ranked_within_target'
        )
        self.assertTrue(
            model.last_optimizer_volume_balance['volume_feasible']
        )

    def test_insufficient_stability_model_uses_lambda_bootstrap_with_trigger_on(self):
        model = OptimizationModel.__new__(OptimizationModel)
        model.acquisition_mode = 'exploit'
        model.target_value = 625.0
        model._get_dimension = types.MethodType(lambda self: 2, model)
        observed = {}

        def _fallback(self, n_restarts_per_mask, required_on_reagent_indices):
            observed['indices'] = list(required_on_reagent_indices)
            return np.asarray([0.2, 0.4])

        model._optimize_acquisition_with_masks = types.MethodType(
            _fallback, model
        )
        model.predict_lambda_distribution_nm = types.MethodType(
            lambda self, recipe: (625.0, 1.0), model
        )
        model._calculate_acquisition_score = types.MethodType(
            lambda self, **kwargs: 0.0, model
        )
        insufficient = type('Insufficient', (), {
            'status': 'insufficient_observations',
            'X': np.asarray([[0.1, 0.2]]),
        })()

        proposed = model.getNextTargetThenStabilityReaction(
            stability_model=insufficient,
            target_tolerance_nm=10.0,
            trigger_reagent_index=1,
        )

        self.assertTrue(np.array_equal(proposed[0], [0.2, 0.4]))
        self.assertEqual(observed['indices'], [1])
        self.assertEqual(
            model.last_target_then_stability_selection_metadata[
                'stability_selection_status'
            ],
            'lambda_bootstrap_stability_model_insufficient'
        )

    def test_no_target_compatible_result_falls_back_to_lambda_not_stability(self):
        model = OptimizationModel.__new__(OptimizationModel)
        model.acquisition_mode = 'exploit'
        model.target_value = 625.0
        model._get_dimension = types.MethodType(lambda self: 1, model)
        model._get_reagent_masks_for_current_settings = types.MethodType(
            lambda self, required_on_reagent_indices=None: [np.asarray([1])],
            model
        )
        model._optimize_single_mask_target_then_stability = types.MethodType(
            lambda self, **kwargs: {
                'mask': np.asarray([1]), 'x_full': None,
                'target_compatible': False, 'volume_balance': None,
            }, model
        )
        model._optimize_acquisition_with_masks = types.MethodType(
            lambda self, **kwargs: np.asarray([0.75]), model
        )
        model.predict_lambda_distribution_nm = types.MethodType(
            lambda self, recipe: (625.0, 1.0), model
        )
        model._calculate_acquisition_score = types.MethodType(
            lambda self, **kwargs: 0.0, model
        )

        proposed = model.getNextTargetThenStabilityReaction(
            stability_model=_FittedStabilityModel(),
            target_tolerance_nm=0.0,
        )

        self.assertTrue(np.array_equal(proposed[0], [0.75]))
        self.assertEqual(
            model.last_target_then_stability_selection_metadata[
                'stability_selection_status'
            ],
            'lambda_fallback_no_target_compatible_optimizer_result'
        )


if __name__ == '__main__':
    unittest.main()
