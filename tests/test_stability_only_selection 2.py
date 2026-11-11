'''Hardware-free Stage-12F-E target-free stability selection tests.'''

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
    '''Load only target-free selector methods without normal optimizer imports.'''
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
        '_get_stability_only_candidate_diagnostics',
        '_optimize_single_mask_stability_only',
        'getNextStabilityOnlyReaction',
    )
    extracted_class = ast.ClassDef(
        name='OptimizationModel', bases=[], keywords=[],
        body=[methods[name] for name in method_names], decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(
        body=[extracted_class], type_ignores=[]
    ))
    namespace = {'copy': copy, 'math': math, 'minimize': minimize, 'np': np}
    exec(compile(module, OPTIMIZERS_PATH, 'exec'), namespace)
    return namespace['OptimizationModel']


OptimizationModel = _load_selection_methods()


class _FittedLossModel(object):
    status = 'fitted'
    X = np.asarray([[0.2], [0.8]], dtype=float)

    @staticmethod
    def predict_log10_loss_rate_distribution(recipes):
        values = np.asarray(recipes, dtype=float).reshape(-1, 1)
        return values[:, 0], np.full(values.shape[0], 0.04)


class _ConstantSignalModel(object):
    status = 'fitted'
    X = np.asarray([[0.2], [0.8]], dtype=float)

    @staticmethod
    def predict_reference_peak_absorbance_distribution(recipes):
        values = np.asarray(recipes, dtype=float).reshape(-1, 1)
        return np.full(values.shape[0], 0.50), np.full(values.shape[0], 0.05)


class _ConcentrationSignalModel(object):
    status = 'fitted'
    X = np.asarray([[0.2], [0.8]], dtype=float)

    @staticmethod
    def predict_reference_peak_absorbance_distribution(recipes):
        values = np.asarray(recipes, dtype=float).reshape(-1, 1)
        return values[:, 0], np.full(values.shape[0], 0.05)


class _IncompatibleSignalModel(object):
    status = 'fitted'
    X = np.asarray([[0.2], [0.8]], dtype=float)

    @staticmethod
    def predict_reference_peak_absorbance_distribution(recipes):
        values = np.asarray(recipes, dtype=float).reshape(-1, 1)
        return np.full(values.shape[0], 0.10), np.full(values.shape[0], 0.01)


def _model(volume_feasible=None):
    model = OptimizationModel.__new__(OptimizationModel)
    model.acquisition_mode = 'exploit'
    model.allow_true_zero = False
    model.terminal_verbosity = 'essential'
    model._get_dimension = types.MethodType(lambda self: 1, model)
    model._get_masked_bounds = types.MethodType(
        lambda self, mask: [(0.0, 1.0)], model
    )
    model._generate_feasible_masked_starting_points = types.MethodType(
        lambda self, mask, count: [np.asarray([0.2]), np.asarray([0.8])],
        model
    )
    model._expand_masked_candidate_to_full_recipe = types.MethodType(
        lambda self, active, mask: np.asarray(active, dtype=float), model
    )
    if volume_feasible is None:
        volume_feasible = lambda recipe: True
    model._get_candidate_volume_balance = types.MethodType(
        lambda self, recipe: {
            'volume_feasible': bool(volume_feasible(np.asarray(recipe))),
            'water_volume': 10.0,
        }, model
    )
    return model


class StabilityOnlySelectionTests(unittest.TestCase):
    def test_selection_uses_no_lambda_prediction_and_minimizes_loss_rate(self):
        model = _model()
        proposed = model.getNextStabilityOnlyReaction(
            stability_model=_FittedLossModel(),
            signal_model=_ConstantSignalModel(),
            min_peak_absorbance=0.35,
            max_peak_absorbance=0.65,
            signal_confidence_z=1.96,
        )
        self.assertEqual(len(proposed), 1)
        self.assertLess(float(proposed[0][0]), 0.05)
        metadata = model.last_stability_only_selection_metadata
        self.assertEqual(
            metadata['stability_selection_status'],
            'stability_ranked_within_signal'
        )
        self.assertIsNone(model.last_optimizer_predicted_lambda_max)
        self.assertGreaterEqual(
            metadata['predicted_reference_peak_absorbance_interval_lower'],
            0.35 - 1e-9
        )
        self.assertLessEqual(
            metadata['predicted_reference_peak_absorbance_interval_upper'],
            0.65 + 1e-9
        )

    def test_signal_interval_is_a_hard_constraint_not_a_weighted_score(self):
        model = _model()
        proposed = model.getNextStabilityOnlyReaction(
            stability_model=_FittedLossModel(),
            signal_model=_ConcentrationSignalModel(),
            min_peak_absorbance=0.40,
            max_peak_absorbance=0.60,
            signal_confidence_z=1.0,
        )
        # Loss rate favors zero concentration, but the entire signal interval
        # requires 0.45 <= x <= 0.55.
        self.assertGreaterEqual(float(proposed[0][0]), 0.45 - 1e-6)
        self.assertLessEqual(float(proposed[0][0]), 0.55 + 1e-6)

    def test_existing_physical_feasibility_remains_a_hard_gate(self):
        model = _model(volume_feasible=lambda recipe: recipe[0] >= 0.50)
        proposed = model.getNextStabilityOnlyReaction(
            stability_model=_FittedLossModel(),
            signal_model=_ConstantSignalModel(),
            min_peak_absorbance=0.35,
            max_peak_absorbance=0.65,
            signal_confidence_z=1.0,
        )
        self.assertGreaterEqual(float(proposed[0][0]), 0.50 - 1e-6)
        self.assertTrue(model.last_optimizer_volume_balance['volume_feasible'])

    def test_variable_trigger_is_held_on_through_existing_mask_pathway(self):
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

    def test_missing_model_or_no_signal_compatible_recipe_fails_closed(self):
        model = _model()
        with self.assertRaisesRegex(RuntimeError, 'fitted current-run'):
            model.getNextStabilityOnlyReaction(
                stability_model=type('NoFit', (), {'status': 'insufficient'})(),
                signal_model=_ConstantSignalModel(),
                min_peak_absorbance=0.35,
                max_peak_absorbance=0.65,
                signal_confidence_z=1.0,
            )

        prior_recipe = np.asarray([0.77])
        model.last_optimizer_selected_normalized_recipe = prior_recipe.copy()
        with self.assertRaisesRegex(RuntimeError, 'no physically executable'):
            model.getNextStabilityOnlyReaction(
                stability_model=_FittedLossModel(),
                signal_model=_IncompatibleSignalModel(),
                min_peak_absorbance=0.35,
                max_peak_absorbance=0.65,
                signal_confidence_z=1.0,
            )
        self.assertTrue(np.array_equal(
            model.last_optimizer_selected_normalized_recipe, prior_recipe
        ))
        self.assertEqual(
            model.last_stability_only_selection_metadata[
                'stability_selection_status'
            ],
            'no_signal_compatible_candidate'
        )

    def test_invalid_bounds_or_non_exploit_modes_fail_clearly(self):
        model = _model()
        with self.assertRaisesRegex(ValueError, 'strictly greater'):
            model.getNextStabilityOnlyReaction(
                _FittedLossModel(), _ConstantSignalModel(), .5, .5, 1.0
            )
        model.acquisition_mode = 'explore'
        with self.assertRaisesRegex(ValueError, 'requires acquisition_mode'):
            model.getNextStabilityOnlyReaction(
                _FittedLossModel(), _ConstantSignalModel(), .35, .65, 1.0
            )

    def test_source_contract_has_no_lambda_prediction_or_fallback(self):
        with open(OPTIMIZERS_PATH, encoding='utf-8') as handle:
            source = handle.read()
        tree = ast.parse(source, filename=OPTIMIZERS_PATH)
        optimizer_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'OptimizationModel'
        )
        method = next(
            node for node in optimizer_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'getNextStabilityOnlyReaction'
        )
        method_source = ast.get_source_segment(source, method)
        self.assertNotIn('predict_lambda_distribution_nm', method_source)
        self.assertNotIn('getNextReaction(', method_source)
        self.assertNotIn('_optimize_acquisition_with_masks(', method_source)


if __name__ == '__main__':
    unittest.main()
