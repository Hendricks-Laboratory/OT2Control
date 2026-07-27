import ast
import copy
from collections import defaultdict
from contextlib import redirect_stdout
import datetime
import io
import json
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from auto_model_checkpoint import (
    ModelCheckpointError,
    read_model_checkpoint,
    write_model_checkpoint_import_provenance,
    write_model_checkpoint
)

try:
    import GPy
except ImportError:
    # The broader source-extraction suite intentionally remains usable in a
    # minimal review environment. The dedicated classifier checks below run
    # whenever the production GPy dependency is available.
    GPy = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZERS_PATH = REPOSITORY_ROOT / 'optimizers.py'
CONTROLLER_PATH = REPOSITORY_ROOT / 'controller.py'
OT2_ROBOT_PATH = REPOSITORY_ROOT / 'ot2_robot.py'


def _load_optimization_model_methods(
    method_names,
    extra_namespace=None
):
    '''
    Loads selected OptimizationModel methods without importing hardware or
    legacy scientific dependencies.

    The production Auto environment provides GPy/GPyOpt and the remaining
    scientific stack. These tests deliberately extract only pure or
    stub-compatible methods from the exact repository source so they can run
    safely with Python 3.9 and NumPy, without importing robot, plate-reader,
    credential, GPy, or GPyOpt integrations.
    '''
    tree = ast.parse(
        OPTIMIZERS_PATH.read_text(),
        filename=str(OPTIMIZERS_PATH)
    )

    model_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'OptimizationModel'
    )

    acquisition_constant_names = {
        '_SUPPORTED_ACQUISITION_MODES',
        'IMPLEMENTED_ACQUISITION_MODES'
    }

    class_constants = [
        node for node in model_class.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in acquisition_constant_names
            for target in node.targets
        )
    ]

    methods = {
        node.name: node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef)
    }

    extracted_class = ast.ClassDef(
        name='OptimizationModel',
        bases=[],
        keywords=[],
        body=(
            class_constants
            + [methods[method_name] for method_name in method_names]
        ),
        decorator_list=[]
    )

    module = ast.fix_missing_locations(
        ast.Module(body=[extracted_class], type_ignores=[])
    )

    namespace = {
        'math': math,
        'np': np
    }

    if extra_namespace is not None:
        namespace.update(extra_namespace)

    exec(
        compile(module, str(OPTIMIZERS_PATH), 'exec'),
        namespace
    )

    return namespace['OptimizationModel']


def _get_production_method_node(method_name):
    tree = ast.parse(
        OPTIMIZERS_PATH.read_text(),
        filename=str(OPTIMIZERS_PATH)
    )

    model_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'OptimizationModel'
    )

    return next(
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )


def _load_auto_controller_methods(method_names, extra_namespace=None):
    '''Loads pure AutoContr methods without importing hardware dependencies.'''
    tree = ast.parse(
        CONTROLLER_PATH.read_text(),
        filename=str(CONTROLLER_PATH)
    )

    controller_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'AutoContr'
    )
    methods = {
        node.name: node
        for node in controller_class.body
        if isinstance(node, ast.FunctionDef)
    }
    extracted_class = ast.ClassDef(
        name='AutoContr',
        bases=[],
        keywords=[],
        body=[methods[method_name] for method_name in method_names],
        decorator_list=[]
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[extracted_class], type_ignores=[])
    )
    namespace = {
        'copy': copy,
        'defaultdict': defaultdict,
        'json': json,
        'math': math,
        'np': np,
        'os': os,
        'pd': pd,
        're': re
    }

    if extra_namespace is not None:
        namespace.update(extra_namespace)

    exec(
        compile(module, str(CONTROLLER_PATH), 'exec'),
        namespace
    )

    return namespace['AutoContr']


def _load_base_controller_methods(method_names):
    '''Loads pure Controller methods without importing hardware dependencies.'''
    tree = ast.parse(
        CONTROLLER_PATH.read_text(),
        filename=str(CONTROLLER_PATH)
    )
    controller_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'Controller'
    )
    methods = {
        node.name: node
        for node in controller_class.body
        if isinstance(node, ast.FunctionDef)
    }
    extracted_class = ast.ClassDef(
        name='Controller',
        bases=[],
        keywords=[],
        body=[methods[method_name] for method_name in method_names],
        decorator_list=[]
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[extracted_class], type_ignores=[])
    )
    namespace = {
        'math': math,
        'np': np,
        'pd': pd
    }

    exec(
        compile(module, str(CONTROLLER_PATH), 'exec'),
        namespace
    )

    return namespace['Controller']


def _load_robot_methods(method_names, extra_namespace=None):
    '''Loads pure OT2Robot methods without importing Opentrons dependencies.'''
    tree = ast.parse(
        OT2_ROBOT_PATH.read_text(),
        filename=str(OT2_ROBOT_PATH)
    )
    robot_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'OT2Robot'
    )
    methods = {
        node.name: node
        for node in robot_class.body
        if isinstance(node, ast.FunctionDef)
    }
    extracted_class = ast.ClassDef(
        name='OT2Robot',
        bases=[],
        keywords=[],
        body=[methods[method_name] for method_name in method_names],
        decorator_list=[]
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[extracted_class], type_ignores=[])
    )
    namespace = {'math': math}

    if extra_namespace is not None:
        namespace.update(extra_namespace)

    exec(
        compile(module, str(OT2_ROBOT_PATH), 'exec'),
        namespace
    )

    return namespace['OT2Robot']


def _get_auto_controller_method_node(method_name):
    '''Returns one production AutoContr method as an AST node.'''
    tree = ast.parse(
        CONTROLLER_PATH.read_text(),
        filename=str(CONTROLLER_PATH)
    )
    controller_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'AutoContr'
    )

    return next(
        node for node in controller_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )


def _deterministic_minimize(fun, x0, bounds, method):
    '''Small scipy-compatible minimizer stub for deterministic source tests.'''
    x = np.asarray(x0, dtype=float)
    return SimpleNamespace(
        fun=float(fun(x)),
        x=x.copy(),
        success=True,
        message='deterministic hardware-free minimizer'
    )


class AcquisitionScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ScoreModel = _load_optimization_model_methods([
            '_validate_predictive_standard_deviation_nm',
            'set_incumbent_target_error_nm',
            '_standard_normal_pdf',
            '_standard_normal_cdf',
            '_calculate_target_error_expected_improvement_nm',
            '_calculate_acquisition_score'
        ])

    def _build_score_model(self, acquisition_mode='exploit'):
        model = self.ScoreModel()
        model.target_value = 625.0
        model.acquisition_mode = acquisition_mode
        model.balanced_exploration_weight = 1.0
        model.incumbent_target_error_nm = None
        return model

    def test_exploit_matches_legacy_squared_target_distance(self):
        model = self._build_score_model()

        cases = (
            (625.0, 0.0),
            (624.5, 0.25),
            (630.0, 25.0),
            (600.0, 625.0),
            (700.0, 5625.0)
        )

        for predicted_mean_nm, expected_score in cases:
            with self.subTest(predicted_mean_nm=predicted_mean_nm):
                score = model._calculate_acquisition_score(
                    predicted_lambda_mean_nm=predicted_mean_nm,
                    predicted_lambda_std_nm=123.0,
                    incumbent_target_error_nm=4.0
                )

                self.assertEqual(score, expected_score)

    def test_target_ei_requires_qc_approved_incumbent(self):
        model = self._build_score_model('target_ei')

        with self.assertRaisesRegex(
            ValueError,
            "QC-approved condition-level"
        ):
            model._calculate_acquisition_score(625.0, 2.0)

    def test_target_ei_matches_independent_numerical_integration(self):
        model = self._build_score_model('target_ei')
        predicted_mean_nm = 630.0
        predicted_std_nm = 4.0
        incumbent_error_nm = 10.0

        analytic_ei_nm = (
            model._calculate_target_error_expected_improvement_nm(
                predicted_lambda_mean_nm=predicted_mean_nm,
                predicted_lambda_std_nm=predicted_std_nm,
                incumbent_target_error_nm=incumbent_error_nm
            )
        )

        # Independent midpoint integration of:
        # (d_best - |y - target|) * Normal(y; mean, std)
        # over the only interval where improvement is positive.
        integration_steps = 20000
        lower_nm = model.target_value - incumbent_error_nm
        interval_width_nm = 2.0 * incumbent_error_nm
        step_width_nm = interval_width_nm / integration_steps
        numerical_ei_nm = 0.0

        for step_index in range(integration_steps):
            lambda_nm = (
                lower_nm
                + (step_index + 0.5) * step_width_nm
            )
            improvement_nm = (
                incumbent_error_nm
                - abs(lambda_nm - model.target_value)
            )
            normal_density = (
                math.exp(
                    -0.5 * (
                        (lambda_nm - predicted_mean_nm)
                        / predicted_std_nm
                    ) ** 2
                )
                / (
                    predicted_std_nm
                    * math.sqrt(2.0 * math.pi)
                )
            )
            numerical_ei_nm += (
                improvement_nm
                * normal_density
                * step_width_nm
            )

        self.assertAlmostEqual(
            analytic_ei_nm,
            numerical_ei_nm,
            places=7
        )

    def test_target_ei_is_symmetric_about_requested_target(self):
        model = self._build_score_model('target_ei')

        below_target_ei = (
            model._calculate_target_error_expected_improvement_nm(
                620.0,
                4.0,
                10.0
            )
        )
        above_target_ei = (
            model._calculate_target_error_expected_improvement_nm(
                630.0,
                4.0,
                10.0
            )
        )

        self.assertAlmostEqual(
            below_target_ei,
            above_target_ei,
            places=12
        )

    def test_target_ei_uses_exact_zero_uncertainty_limit(self):
        model = self._build_score_model('target_ei')

        cases = (
            (625.0, 10.0),
            (629.0, 6.0),
            (640.0, 0.0)
        )

        for predicted_mean_nm, expected_ei_nm in cases:
            with self.subTest(predicted_mean_nm=predicted_mean_nm):
                result = (
                    model._calculate_target_error_expected_improvement_nm(
                        predicted_mean_nm,
                        0.0,
                        10.0
                    )
                )
                self.assertEqual(result, expected_ei_nm)

    def test_target_ei_score_selects_largest_expected_improvement(self):
        model = self._build_score_model('target_ei')
        model.set_incumbent_target_error_nm(10.0)
        candidates = (
            (625.0, 2.0),
            (630.0, 4.0),
            (640.0, 1.0)
        )

        scores = [
            model._calculate_acquisition_score(
                predicted_lambda_mean_nm=mean_nm,
                predicted_lambda_std_nm=std_nm
            )
            for mean_nm, std_nm in candidates
        ]

        self.assertEqual(scores.index(min(scores)), 0)
        self.assertTrue(all(score <= 0.0 for score in scores))

    def test_target_ei_is_finite_and_bounded_by_incumbent(self):
        model = self._build_score_model('target_ei')
        incumbent_error_nm = 10.0

        for mean_nm in (500.0, 625.0, 750.0):
            for std_nm in (0.0, 1e-9, 2.0, 100.0):
                with self.subTest(mean_nm=mean_nm, std_nm=std_nm):
                    expected_improvement_nm = (
                        model._calculate_target_error_expected_improvement_nm(
                            mean_nm,
                            std_nm,
                            incumbent_error_nm
                        )
                    )
                    self.assertTrue(math.isfinite(expected_improvement_nm))
                    self.assertGreaterEqual(expected_improvement_nm, 0.0)
                    self.assertLessEqual(
                        expected_improvement_nm,
                        incumbent_error_nm
                    )

    def test_target_ei_incumbent_setter_rejects_invalid_values(self):
        model = self._build_score_model('target_ei')

        for invalid_incumbent in (
            None,
            -0.01,
            float('nan'),
            float('inf'),
            'not-a-number'
        ):
            with self.subTest(invalid_incumbent=invalid_incumbent):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite, nonnegative"
                ):
                    model.set_incumbent_target_error_nm(
                        invalid_incumbent
                    )

    def test_balanced_trades_target_proximity_against_uncertainty(self):
        model = self._build_score_model('balanced')
        candidates = (
            # Closest mean, but nearly no uncertainty.
            (626.0, 0.0),
            # Farther mean, but sufficiently uncertain to be preferred.
            (630.0, 10.0),
            # Far from target without enough uncertainty to compensate.
            (650.0, 3.0)
        )

        scores = [
            model._calculate_acquisition_score(
                predicted_lambda_mean_nm=mean_nm,
                predicted_lambda_std_nm=std_nm
            )
            for mean_nm, std_nm in candidates
        ]

        self.assertEqual(scores, [1.0, -5.0, 22.0])
        self.assertEqual(scores.index(min(scores)), 1)

    def test_balanced_weight_has_documented_one_for_one_scale(self):
        model = self._build_score_model('balanced')
        scores_by_weight = {}

        for weight in (0.0, 1.0, 2.0):
            model.balanced_exploration_weight = weight
            scores_by_weight[weight] = model._calculate_acquisition_score(
                predicted_lambda_mean_nm=630.0,
                predicted_lambda_std_nm=4.0
            )

        self.assertEqual(
            scores_by_weight,
            {
                0.0: 5.0,
                1.0: 1.0,
                2.0: -3.0
            }
        )

    def test_balanced_requires_finite_mean_and_valid_uncertainty(self):
        model = self._build_score_model('balanced')

        for invalid_mean in (None, float('nan'), float('inf')):
            with self.subTest(invalid_mean=invalid_mean):
                with self.assertRaisesRegex(ValueError, "finite"):
                    model._calculate_acquisition_score(
                        predicted_lambda_mean_nm=invalid_mean,
                        predicted_lambda_std_nm=2.0
                    )

        for invalid_std in (None, -0.01, float('nan'), float('inf')):
            with self.subTest(invalid_std=invalid_std):
                with self.assertRaisesRegex(
                    ValueError,
                    "standard deviation"
                ):
                    model._calculate_acquisition_score(
                        predicted_lambda_mean_nm=625.0,
                        predicted_lambda_std_nm=invalid_std
                    )

        for invalid_target in (None, float('nan'), float('inf')):
            with self.subTest(invalid_target=invalid_target):
                model.target_value = invalid_target

                with self.assertRaisesRegex(ValueError, "finite target"):
                    model._calculate_acquisition_score(
                        predicted_lambda_mean_nm=625.0,
                        predicted_lambda_std_nm=2.0
                    )

    def test_explore_score_selects_maximum_uncertainty(self):
        model = self._build_score_model('explore')
        candidate_standard_deviations = (1.0, 12.5, 4.0)

        scores = [
            model._calculate_acquisition_score(
                predicted_lambda_mean_nm=predicted_mean_nm,
                predicted_lambda_std_nm=predicted_std_nm
            )
            for predicted_mean_nm, predicted_std_nm in zip(
                (625.0, 700.0, 500.0),
                candidate_standard_deviations
            )
        ]

        selected_index = min(
            range(len(scores)),
            key=lambda index: scores[index]
        )

        self.assertEqual(scores, [-1.0, -12.5, -4.0])
        self.assertEqual(selected_index, 1)

    def test_explore_requires_valid_standard_deviation(self):
        model = self._build_score_model('explore')

        for invalid_std in (None, -0.01, float('nan'), float('inf')):
            with self.subTest(invalid_std=invalid_std):
                with self.assertRaisesRegex(
                    ValueError,
                    "standard deviation"
                ):
                    model._calculate_acquisition_score(
                        predicted_lambda_mean_nm=625.0,
                        predicted_lambda_std_nm=invalid_std
                    )

    def test_invalid_mode_fails_validation(self):
        model = self._build_score_model('ordinary_ei')

        with self.assertRaisesRegex(ValueError, "'ordinary_ei'"):
            model._calculate_acquisition_score(625.0, 2.0)


class MaskedAcquisitionObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.MaskedModel = _load_optimization_model_methods([
            '_validate_predictive_standard_deviation_nm',
            '_standard_normal_pdf',
            '_standard_normal_cdf',
            '_calculate_target_error_expected_improvement_nm',
            '_calculate_acquisition_score',
            '_masked_acquisition_objective',
            '_masked_target_distance_objective'
        ])

    def _build_model(self, volume_balance, acquisition_mode='exploit'):
        model = self.MaskedModel()
        model.target_value = 625.0
        model.acquisition_mode = acquisition_mode
        model.balanced_exploration_weight = 1.0
        model.incumbent_target_error_nm = 10.0
        model._expand_masked_candidate_to_full_recipe = (
            lambda x_active, mask: ('full-recipe', x_active, mask)
        )
        model._get_candidate_volume_balance = (
            lambda full_x: volume_balance
        )
        return model

    def test_feasible_candidate_uses_central_score(self):
        volume_balance = {
            'volume_feasible': True,
            'water_volume': 20.0,
            'volume_does_not_overflow': True,
            'water_transfer_executable': True
        }
        model = self._build_model(volume_balance)
        predictions = []
        model._predict_lambda_max_nm = (
            lambda full_x: predictions.append(full_x) or 627.0
        )

        score = model._masked_acquisition_objective([0.5], [1])

        self.assertEqual(score, 4.0)
        self.assertEqual(len(predictions), 1)

    def test_explore_prefers_most_uncertain_feasible_candidate(self):
        volume_balance = {
            'volume_feasible': True,
            'water_volume': 20.0,
            'volume_does_not_overflow': True,
            'water_transfer_executable': True
        }
        model = self._build_model(
            volume_balance,
            acquisition_mode='explore'
        )
        distributions = {
            0.1: (625.0, 2.0),
            0.2: (700.0, 15.0),
            0.3: (500.0, 6.0)
        }
        model.predict_lambda_distribution_nm = (
            lambda full_x: distributions[full_x[1][0]]
        )

        scores = {
            candidate: model._masked_acquisition_objective(
                [candidate],
                [1]
            )
            for candidate in distributions
        }

        selected_candidate = min(scores, key=scores.get)

        self.assertEqual(selected_candidate, 0.2)
        self.assertEqual(scores[0.2], -15.0)

    def test_balanced_uses_mean_and_uncertainty_for_feasible_candidates(self):
        volume_balance = {
            'volume_feasible': True,
            'water_volume': 20.0,
            'volume_does_not_overflow': True,
            'water_transfer_executable': True
        }
        model = self._build_model(
            volume_balance,
            acquisition_mode='balanced'
        )
        distributions = {
            0.1: (626.0, 0.0),
            0.2: (630.0, 10.0),
            0.3: (650.0, 3.0)
        }
        model.predict_lambda_distribution_nm = (
            lambda full_x: distributions[full_x[1][0]]
        )

        scores = {
            candidate: model._masked_acquisition_objective(
                [candidate],
                [1]
            )
            for candidate in distributions
        }

        selected_candidate = min(scores, key=scores.get)

        self.assertEqual(selected_candidate, 0.2)
        self.assertEqual(scores[0.2], -5.0)

    def test_target_ei_uses_same_feasible_masked_recipe_path(self):
        volume_balance = {
            'volume_feasible': True,
            'water_volume': 20.0,
            'volume_does_not_overflow': True,
            'water_transfer_executable': True
        }
        model = self._build_model(
            volume_balance,
            acquisition_mode='target_ei'
        )
        distributions = {
            0.1: (625.0, 2.0),
            0.2: (630.0, 4.0),
            0.3: (640.0, 1.0)
        }
        model.predict_lambda_distribution_nm = (
            lambda full_x: distributions[full_x[1][0]]
        )

        scores = {
            candidate: model._masked_acquisition_objective(
                [candidate],
                [1]
            )
            for candidate in distributions
        }

        selected_candidate = min(scores, key=scores.get)

        self.assertEqual(selected_candidate, 0.1)
        self.assertLess(scores[0.1], scores[0.2])

    def test_overflow_penalty_is_unchanged_and_skips_gp(self):
        volume_balance = {
            'volume_feasible': False,
            'water_volume': -7.0,
            'volume_does_not_overflow': False,
            'water_transfer_executable': False
        }
        for mode in ('exploit', 'explore', 'balanced', 'target_ei'):
            with self.subTest(mode=mode):
                model = self._build_model(
                    volume_balance,
                    acquisition_mode=mode
                )
                model._predict_lambda_max_nm = (
                    lambda full_x: self.fail(
                        'GP prediction must not run'
                    )
                )
                model.predict_lambda_distribution_nm = (
                    lambda full_x: self.fail(
                        'GP prediction must not run'
                    )
                )

                score = model._masked_acquisition_objective(
                    [0.5],
                    [1]
                )

                self.assertEqual(score, 1e12 + 49.0)

    def test_bad_water_penalty_is_unchanged_and_skips_gp(self):
        volume_balance = {
            'volume_feasible': False,
            'water_volume': 3.0,
            'volume_does_not_overflow': True,
            'water_transfer_executable': False
        }
        for mode in ('exploit', 'explore', 'balanced', 'target_ei'):
            with self.subTest(mode=mode):
                model = self._build_model(
                    volume_balance,
                    acquisition_mode=mode
                )
                model._predict_lambda_max_nm = (
                    lambda full_x: self.fail(
                        'GP prediction must not run'
                    )
                )
                model.predict_lambda_distribution_nm = (
                    lambda full_x: self.fail(
                        'GP prediction must not run'
                    )
                )

                score = model._masked_acquisition_objective(
                    [0.5],
                    [1]
                )

                self.assertEqual(score, 1e12 + 1.0)

    def test_previous_masked_objective_name_remains_compatible(self):
        volume_balance = {
            'volume_feasible': True,
            'water_volume': 20.0,
            'volume_does_not_overflow': True,
            'water_transfer_executable': True
        }
        model = self._build_model(volume_balance)
        model._predict_lambda_max_nm = lambda full_x: 620.0

        self.assertEqual(
            model._masked_target_distance_objective([0.5], [1]),
            25.0
        )


class MaskAndExecutableBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.MaskModel = _load_optimization_model_methods(
            [
                '_get_dimension',
                '_generate_reagent_masks',
                '_get_reagent_masks_for_current_settings',
                '_get_active_mask_indices',
                '_expand_masked_candidate_to_full_recipe',
                '_get_masked_bounds'
            ],
            extra_namespace={'np': np}
        )

    def _build_model(self, allow_true_zero):
        model = self.MaskModel()
        model.variable_reagents = ['A', 'B', 'C']
        model.allow_true_zero = allow_true_zero
        model.min_conc = [0.0, 0.0, 0.0]
        model.max_conc = [1.0, 1.0, 1.0]
        model.total_volume = 100.0
        model._get_variable_reagent_stock_conc = lambda reagent_name: 1.0
        return model

    def test_true_zero_masks_exclude_all_off_and_expand_off_to_exact_zero(self):
        model = self._build_model(allow_true_zero=True)
        masks = [
            mask.tolist()
            for mask in model._get_reagent_masks_for_current_settings()
        ]

        self.assertEqual(len(masks), 7)
        self.assertNotIn([0, 0, 0], masks)
        self.assertIn([1, 0, 1], masks)

        expanded = model._expand_masked_candidate_to_full_recipe(
            [0.25, 0.75],
            [1, 0, 1]
        )

        self.assertEqual(expanded.tolist(), [0.25, 0.0, 0.75])
        self.assertEqual(expanded[1], 0.0)

    def test_true_zero_disabled_uses_only_all_on_mask(self):
        model = self._build_model(allow_true_zero=False)
        masks = model._get_reagent_masks_for_current_settings()

        self.assertEqual([mask.tolist() for mask in masks], [[1, 1, 1]])

    def test_selective_true_zero_masks_keep_required_reagents_on(self):
        model = self._build_model(allow_true_zero=True)
        model.true_zero_reagent_indices = [0, 2]

        masks = [
            mask.tolist()
            for mask in model._get_reagent_masks_for_current_settings()
        ]

        self.assertEqual(
            masks,
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 1],
                [1, 1, 1]
            ]
        )

    def test_active_bounds_start_at_five_ul_executable_transfer(self):
        model = self._build_model(allow_true_zero=True)
        bounds = model._get_masked_bounds([1, 0, 1])

        # stock concentration 1.0 * 5 uL / 100 uL total gives a normalized
        # lower concentration bound of 0.05 for each active reagent.
        self.assertEqual(bounds, [(0.05, 1.0), (0.05, 1.0)])


class GprFeasibilityOverlayTests(unittest.TestCase):
    '''Verifies physical-executability masks used by 2D GP overlay plots.'''

    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_base_controller_methods([
            '_get_2d_gpr_feasibility_overlay_data'
        ])

    def _build_controller_and_model(self, allow_true_zero=True):
        controller = self.Controller()
        controller.variable_reagents = ['reagent_a', 'reagent_b']

        model = SimpleNamespace(
            total_volume=200.0,
            fixed_reagent_volumes={'fixed': 20.0},
            allow_true_zero=allow_true_zero
        )
        model._get_variable_reagent_stock_conc = (
            lambda reagent_name: 1.0
        )

        return controller, model

    def test_overlay_marks_non_executable_transfers_water_band_and_overflow(self):
        controller, model = self._build_controller_and_model()
        overlay = controller._get_2d_gpr_feasibility_overlay_data(
            model=model,
            x_values=[0.0, 0.02, 0.88, 0.90],
            y_values=[0.0, 0.02, 0.10]
        )

        self.assertEqual(overlay['infeasible'].shape, (3, 4))
        self.assertFalse(overlay['infeasible'][0, 0])
        self.assertTrue(overlay['variable_transfer_infeasible'][0, 1])
        self.assertTrue(overlay['variable_transfer_infeasible'][1, 0])
        self.assertTrue(overlay['water_transfer_infeasible'][0, 2])
        self.assertTrue(overlay['overflow'][2, 2])
        self.assertFalse(overlay['water_transfer_infeasible'][0, 3])
        self.assertTrue(overlay['water_is_exact_zero'][0, 3])

    def test_overlay_marks_zero_reagent_transfer_infeasible_without_true_zero(self):
        controller, model = self._build_controller_and_model(
            allow_true_zero=False
        )
        overlay = controller._get_2d_gpr_feasibility_overlay_data(
            model=model,
            x_values=[0.0, 0.10],
            y_values=[0.0, 0.10]
        )

        self.assertTrue(overlay['variable_transfer_infeasible'][0, 0])

    def test_overlay_respects_selected_true_zero_reagents(self):
        controller, model = self._build_controller_and_model(
            allow_true_zero=True
        )
        model.true_zero_reagents = ['reagent_a']

        overlay = controller._get_2d_gpr_feasibility_overlay_data(
            model=model,
            x_values=[0.0, 0.1],
            y_values=[0.0, 0.1]
        )

        # reagent_a may be exactly zero, but reagent_b remains required ON.
        self.assertTrue(overlay['variable_transfer_infeasible'][0, 0])
        self.assertFalse(overlay['variable_transfer_infeasible'][1, 0])


class AcquisitionRoutingTests(unittest.TestCase):
    def _called_method_names(self, method_name):
        method = _get_production_method_node(method_name)

        return {
            node.func.attr
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }

    def test_single_mask_optimizer_uses_generalized_objective(self):
        called_methods = self._called_method_names(
            '_optimize_single_mask'
        )

        self.assertIn(
            '_masked_acquisition_objective',
            called_methods
        )
        self.assertNotIn(
            '_masked_target_distance_objective',
            called_methods
        )

    def test_get_next_reaction_uses_generalized_mask_optimizer(self):
        called_methods = self._called_method_names('getNextReaction')

        self.assertIn(
            '_optimize_acquisition_with_masks',
            called_methods
        )
        self.assertNotIn(
            '_optimize_target_distance_with_masks',
            called_methods
        )

    def test_controller_guard_uses_optimizer_implemented_modes(self):
        tree = ast.parse(
            CONTROLLER_PATH.read_text(),
            filename=str(CONTROLLER_PATH)
        )
        launch_auto = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'launch_auto'
        )
        implemented_mode_guards = [
            node for node in ast.walk(launch_auto)
            if isinstance(node, ast.Compare)
            and any(isinstance(operator, ast.NotIn) for operator in node.ops)
            and any(
                isinstance(child, ast.Attribute)
                and child.attr == 'IMPLEMENTED_ACQUISITION_MODES'
                for child in ast.walk(node)
            )
        ]

        self.assertEqual(len(implemented_mode_guards), 1)

    def test_controller_passes_default_balanced_weight(self):
        tree = ast.parse(
            CONTROLLER_PATH.read_text(),
            filename=str(CONTROLLER_PATH)
        )
        launch_auto = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'launch_auto'
        )
        constructor_call = next(
            node for node in ast.walk(launch_auto)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'OptimizationModel'
        )
        keywords = {
            keyword.arg: keyword.value
            for keyword in constructor_call.keywords
        }
        weight_expression = keywords['balanced_exploration_weight']

        self.assertIsInstance(weight_expression, ast.Call)
        self.assertEqual(
            [argument.value for argument in weight_expression.args],
            ['balanced_exploration_weight', 1.0]
        )

    def test_controller_synchronizes_target_ei_after_successful_gp_changes(
        self
    ):
        tree = ast.parse(
            CONTROLLER_PATH.read_text(),
            filename=str(CONTROLLER_PATH)
        )
        controller_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == 'AutoContr'
        )
        run_method = next(
            node for node in controller_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == '_run'
        )
        model_lifecycle_calls = sorted(
            (
                node.lineno,
                node.func.attr
            )
            for node in ast.walk(run_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {
                'initialize_optimizer',
                '_synchronize_target_ei_incumbent_from_performance'
            }
        )

        initialize_lines = [
            line_number
            for line_number, method_name in model_lifecycle_calls
            if method_name == 'initialize_optimizer'
        ]
        synchronization_lines = [
            line_number
            for line_number, method_name in model_lifecycle_calls
            if method_name
            == '_synchronize_target_ei_incumbent_from_performance'
        ]

        self.assertEqual(len(initialize_lines), 1)
        self.assertEqual(len(synchronization_lines), 1)
        self.assertLess(initialize_lines[0], synchronization_lines[0])

        batch_method = _get_auto_controller_method_node(
            '_run_auto_optimizer_batches'
        )
        batch_lifecycle_calls = sorted(
            (
                node.lineno,
                node.func.attr
            )
            for node in ast.walk(batch_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {
                'update_experiment_data',
                '_synchronize_target_ei_incumbent_from_performance'
            }
        )
        update_lines = [
            line_number
            for line_number, method_name in batch_lifecycle_calls
            if method_name == 'update_experiment_data'
        ]
        batch_synchronization_lines = [
            line_number
            for line_number, method_name in batch_lifecycle_calls
            if method_name == '_synchronize_target_ei_incumbent_from_performance'
        ]

        self.assertEqual(len(update_lines), 1)
        self.assertEqual(len(batch_synchronization_lines), 1)
        self.assertLess(update_lines[0], batch_synchronization_lines[0])

    def test_controller_captures_complete_selection_metadata_before_run(self):
        metadata_method = _get_auto_controller_method_node(
            '_build_auto_optimizer_selection_metadata'
        )
        metadata_dict = next(
            node.value
            for node in ast.walk(metadata_method)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Dict)
        )
        metadata_keys = {
            key.value
            for key in metadata_dict.keys
            if isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        }

        self.assertTrue({
            'acquisition_mode',
            'acquisition_score',
            'predicted_target_error_nm',
            'predicted_lambda_mean_nm',
            'predicted_lambda_std_nm',
            'incumbent_target_error_nm',
            'selected_mask',
            'optimizer_method',
            'optimizer_success',
            'optimizer_status',
            'optimizer_message',
            'balanced_exploration_weight',
            'selected_normalized_recipe',
            'executed_normalized_recipe',
            'selected_physical_recipe',
            'executed_physical_recipe',
            'optimizer_recipe_repaired',
            'optimizer_volume_balance',
            'selected_controller_volume_balances',
            'executed_controller_volume_balances',
            'mask_results'
        }.issubset(metadata_keys))

    def test_performance_log_row_contains_complete_acquisition_audit(self):
        append_method = _get_auto_controller_method_node(
            '_append_auto_model_performance_rows'
        )
        row_dict = next(
            node.value
            for node in ast.walk(append_method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == 'row'
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
        )
        row_keys = {
            key.value
            for key in row_dict.keys
            if isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        }

        self.assertTrue({
            'acquisition_mode',
            'acquisition_score',
            'predicted_target_error_nm',
            'predicted_lambda_mean_nm',
            'predicted_lambda_std_nm',
            'incumbent_target_error_nm',
            'selected_mask',
            'optimizer_method',
            'optimizer_success',
            'optimizer_status',
            'optimizer_message',
            'balanced_exploration_weight',
            'selected_normalized_recipe',
            'executed_normalized_recipe',
            'selected_physical_recipe',
            'executed_physical_recipe',
            'optimizer_recipe_repaired',
            'optimizer_volume_balance',
            'selected_controller_volume_balance',
            'executed_controller_volume_balance',
            'mask_results'
        }.issubset(row_keys))

    def test_run_report_includes_acquisition_settings_and_audit_fields(self):
        report_method = _get_auto_controller_method_node(
            '_write_auto_run_report'
        )
        responsive_section_method = _get_auto_controller_method_node(
            '_build_auto_report_responsive_sections'
        )
        full_audit_appendix_method = _get_auto_controller_method_node(
            '_build_auto_report_full_audit_appendix'
        )
        report_strings = {
            node.value
            for method_node in [
                report_method,
                responsive_section_method,
                full_audit_appendix_method
            ]
            for node in ast.walk(method_node)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }

        self.assertIn('## Acquisition Audit Trail', report_strings)
        self.assertIn('## Full Audit Appendix', report_strings)
        self.assertIn('acquisition_mode', report_strings)
        self.assertIn('acquisition_score', report_strings)
        self.assertIn('predicted_target_error_nm', report_strings)
        self.assertIn('predicted_lambda_mean_nm', report_strings)
        self.assertIn('predicted_lambda_std_nm', report_strings)
        self.assertIn('incumbent_target_error_nm', report_strings)
        self.assertIn('selected_mask', report_strings)
        self.assertIn('balanced_exploration_weight', report_strings)
        self.assertIn('optimizer_recipe_repaired', report_strings)
        self.assertIn('## Condition-Level Results', report_strings)
        self.assertIn('## Model Checkpoint Lineage', report_strings)
        self.assertIn('imported_auto_model_checkpoint', report_strings)
        self.assertIn('provenance_path', report_strings)


class ExperimentDataExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_auto_controller_methods([
            '_build_labeled_auto_experiment_data_export'
        ])

    def test_raw_well_export_labels_physical_concentrations_and_lambda_max(
        self
    ):
        controller = self.Controller()
        controller.variable_reagents = [
            'silver_nitrate',
            'potassium_bromide'
        ]
        controller.experiment_data = pd.DataFrame({
            'silver_nitrate': [0.10],
            'potassium_bromide': [0.002],
            'Experiment_result': [625.0]
        })

        export_dataframe = (
            controller._build_labeled_auto_experiment_data_export()
        )

        self.assertEqual(
            list(export_dataframe.columns),
            [
                'silver_nitrate_final_reaction_concentration_mM',
                'potassium_bromide_final_reaction_concentration_mM',
                'lambda_max_nm'
            ]
        )
        self.assertEqual(
            export_dataframe.iloc[0].tolist(),
            [0.10, 0.002, 625.0]
        )
        self.assertEqual(
            list(controller.experiment_data.columns),
            ['silver_nitrate', 'potassium_bromide', 'Experiment_result']
        )


class ImportedContinuationNumberingTests(unittest.TestCase):
    '''Keeps an imported continuation from reusing source-run numbering.

    A resumed run inherits condition history from its source package.  If new
    conditions restart at zero they collide with those rows, which silently
    overlays two different chemistries on one reaction number and drops
    conditions from batch-filtered plots.
    '''

    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_auto_controller_methods([
            '_get_next_auto_number_after_rows',
            '_summarize_auto_report_physical_well_span',
            '_get_96_well_plate_order'
        ])

    @staticmethod
    def _imported_rows():
        '''One seed plus a three-mode portfolio batch, as a source run logs.'''
        wells = [
            ['A1', 'B1', 'C1'],
            ['D1', 'E1', 'F1'],
            ['G1', 'H1', 'A2'],
            ['B2', 'C2', 'D2']
        ]
        rows = []
        for reaction_number, well_locations in enumerate(wells):
            rows.append({
                'reaction_number': reaction_number,
                'batch_number': 0 if reaction_number == 0 else 1,
                'condition_type': (
                    'seed' if reaction_number == 0 else 'optimizer_selected'
                ),
                'executed_in_current_run': False,
                'replicate_well_locations': json.dumps(well_locations)
            })
        return rows

    def test_numbering_resumes_above_imported_history(self):
        rows = self._imported_rows()

        self.assertEqual(
            self.Controller._get_next_auto_number_after_rows(
                rows,
                'reaction_number'
            ),
            4
        )
        self.assertEqual(
            self.Controller._get_next_auto_number_after_rows(
                rows,
                'batch_number'
            ),
            2
        )

    def test_condition_number_key_is_not_used_for_numbering(self):
        '''Guards the exact key-name defect that reset the counter to zero.'''
        legacy_rows = [
            {'condition_number': 7, 'actual_lambda_mean_qc_nm': 625.0}
        ]

        self.assertEqual(
            self.Controller._get_next_auto_number_after_rows(
                legacy_rows,
                'reaction_number'
            ),
            0
        )

    def test_continuation_batch_never_reuses_imported_batch_numbers(self):
        rows = self._imported_rows()
        batch_start = self.Controller._get_next_auto_number_after_rows(
            rows,
            'batch_number'
        )
        imported_batches = {row['batch_number'] for row in rows}

        self.assertNotIn(batch_start, imported_batches)
        self.assertGreater(batch_start, max(imported_batches))

    def test_no_duplicate_reaction_or_batch_keys_after_continuation(self):
        rows = self._imported_rows()
        reaction_number = self.Controller._get_next_auto_number_after_rows(
            rows,
            'reaction_number'
        )
        batch_number = self.Controller._get_next_auto_number_after_rows(
            rows,
            'batch_number'
        )

        for well_locations in [
            ['A1', 'B1', 'C1'],
            ['D1', 'E1', 'F1'],
            ['G1', 'H1', 'A2']
        ]:
            rows.append({
                'reaction_number': reaction_number,
                'batch_number': batch_number,
                'condition_type': 'optimizer_selected',
                'executed_in_current_run': True,
                'replicate_well_locations': json.dumps(well_locations)
            })
            reaction_number += 1

        keys = [
            (row['batch_number'], row['reaction_number'])
            for row in rows
        ]
        self.assertEqual(len(keys), len(set(keys)))

        # Every condition must survive the final `batch_number <= N` plot
        # window, which previously dropped the imported optimizer batch.
        retained = [
            row for row in rows
            if row['batch_number'] <= batch_number
        ]
        self.assertEqual(len(retained), len(rows))

    def test_well_span_excludes_imported_history(self):
        '''An imported continuation always starts a fresh plate.'''
        rows = self._imported_rows()
        for reaction_number, well_locations in enumerate(
            [
                ['A1', 'B1', 'C1'],
                ['D1', 'E1', 'F1'],
                ['G1', 'H1', 'A2']
            ],
            start=4
        ):
            rows.append({
                'reaction_number': reaction_number,
                'batch_number': 2,
                'condition_type': 'optimizer_selected',
                'executed_in_current_run': True,
                'replicate_well_locations': json.dumps(well_locations)
            })

        controller = self.Controller()
        span = controller._summarize_auto_report_physical_well_span(
            pd.DataFrame(rows)
        )

        self.assertEqual(span['unique_well_count'], 9)
        self.assertEqual(span['first_well'], 'A1')
        self.assertEqual(span['last_well'], 'A2')
        self.assertEqual(span['next_well'], 'B2')

    def test_well_span_unchanged_for_runs_without_the_marker(self):
        '''Legacy performance rows must report exactly as before.'''
        rows = [
            {
                'reaction_number': reaction_number,
                'batch_number': 0 if reaction_number == 0 else 1,
                'replicate_well_locations': json.dumps(well_locations)
            }
            for reaction_number, well_locations in enumerate([
                ['A1', 'B1', 'C1'],
                ['D1', 'E1', 'F1'],
                ['G1', 'H1', 'A2'],
                ['B2', 'C2', 'D2']
            ])
        ]

        controller = self.Controller()
        span = controller._summarize_auto_report_physical_well_span(
            pd.DataFrame(rows)
        )

        self.assertEqual(span['unique_well_count'], 12)
        self.assertEqual(span['last_well'], 'D2')
        self.assertEqual(span['next_well'], 'E2')

    def test_continuation_entry_point_sets_batch_from_imported_start(self):
        '''Source-level check that the continuation route advances the batch.'''
        method_node = _get_auto_controller_method_node(
            '_run_imported_auto_continuation'
        )
        assigned_targets = {
            target.attr
            for node in ast.walk(method_node)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
        }
        referenced_names = {
            node.value
            for node in ast.walk(method_node)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }

        self.assertIn('batch_num', assigned_targets)
        self.assertIn('imported_auto_batch_start', referenced_names)


class InheritedSeedDesignLabelTests(unittest.TestCase):
    '''Keeps an imported run from presenting another run's seed as its own.

    An imported continuation never performs a seed design, so its seed rows
    always belong to an earlier run.  Because a checkpoint can itself have been
    imported, the originating run must be read from the row's own stamp rather
    than from this run's immediate import source.
    '''

    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_auto_controller_methods([
            '_get_auto_run_directory_name',
            '_get_auto_inherited_seed_origin_run',
            '_get_auto_seed_design_label',
            '_apply_auto_seed_design_label'
        ])

    def _controller(self, rows, out_path='/runs/RTG_current', imported=None):
        controller = self.Controller()
        controller.out_path = out_path
        controller.auto_model_performance_rows = rows
        if imported is not None:
            controller.imported_auto_model_checkpoint = imported
        return controller

    def test_local_seed_run_keeps_the_maximin_title(self):
        controller = self._controller([
            {
                'condition_type': 'seed',
                'executed_in_current_run': True,
                'origin_run_directory': 'RTG_current'
            }
        ])

        self.assertIsNone(controller._get_auto_inherited_seed_origin_run())
        self.assertEqual(
            controller._apply_auto_seed_design_label(
                'Initial Maximin Seed Design: Pairwise Projections'
            ),
            'Initial Maximin Seed Design: Pairwise Projections'
        )

    def test_legacy_rows_without_the_marker_keep_the_maximin_title(self):
        '''Runs predating the provenance fields must report unchanged.'''
        controller = self._controller([
            {'condition_type': 'seed', 'actual_lambda_mean_nm': 625.0}
        ])

        self.assertIsNone(controller._get_auto_inherited_seed_origin_run())
        self.assertEqual(
            controller._get_auto_seed_design_label(),
            'Initial Maximin Seed Design'
        )

    def test_single_generation_import_names_the_source_run(self):
        controller = self._controller([
            {
                'condition_type': 'seed',
                'executed_in_current_run': False,
                'origin_run_directory': 'RTG_debuggingsave2'
            },
            {
                'condition_type': 'optimizer_selected',
                'executed_in_current_run': True,
                'origin_run_directory': 'RTG_current'
            }
        ])

        self.assertEqual(
            controller._get_auto_inherited_seed_origin_run(),
            'RTG_debuggingsave2'
        )
        self.assertEqual(
            controller._apply_auto_seed_design_label(
                'Initial Maximin Seed Design: Pairwise Projections'
            ),
            'Inherited Seed Design (from RTG_debuggingsave2): '
            'Pairwise Projections'
        )

    def test_multi_generation_import_names_the_originating_run(self):
        '''A imported into B, B imported into C: the seed belongs to A.'''
        controller = self._controller(
            [
                {
                    'condition_type': 'seed',
                    'executed_in_current_run': False,
                    'origin_run_directory': 'RTG_generation_a'
                },
                {
                    'condition_type': 'optimizer_selected',
                    'executed_in_current_run': False,
                    'origin_run_directory': 'RTG_generation_b'
                }
            ],
            out_path='/runs/RTG_generation_c',
            imported={
                'source_run_id': 'shared_sheet_name',
                'source_metadata': {
                    'source_run_folder': 'RTG_generation_b'
                }
            }
        )

        # The immediate source is generation B, but the seed came from A.
        self.assertEqual(
            controller._get_auto_inherited_seed_origin_run(),
            'RTG_generation_a'
        )
        self.assertIn(
            'RTG_generation_a',
            controller._get_auto_seed_design_label()
        )
        self.assertNotIn(
            'RTG_generation_b',
            controller._get_auto_seed_design_label()
        )

    def test_unstamped_imported_seed_falls_back_to_immediate_source(self):
        '''A pre-provenance checkpoint still reports its lineage.'''
        controller = self._controller(
            [
                {
                    'condition_type': 'seed',
                    'executed_in_current_run': False,
                    'origin_run_directory': ''
                }
            ],
            imported={
                'source_run_id': 'DEBUGRTG_XXX_4variable',
                'source_metadata': {
                    'source_run_folder': 'RTG_debuggingsave2'
                }
            }
        )

        self.assertEqual(
            controller._get_auto_inherited_seed_origin_run(),
            'RTG_debuggingsave2'
        )

    def test_unstamped_manual_inbox_import_falls_back_to_run_id(self):
        '''The manual route records no source folder, only the run id.'''
        controller = self._controller(
            [
                {
                    'condition_type': 'seed',
                    'executed_in_current_run': False
                }
            ],
            imported={
                'source_run_id': 'DEBUGRTG_XXX_4variable',
                'source_metadata': {'import_inbox_path': '/runs/x/Import_Here'}
            }
        )

        self.assertEqual(
            controller._get_auto_inherited_seed_origin_run(),
            'DEBUGRTG_XXX_4variable'
        )

    def test_run_directory_name_comes_from_the_output_path(self):
        controller = self._controller([], out_path='/runs/RTG_debuggingimport/')

        self.assertEqual(
            controller._get_auto_run_directory_name(),
            'RTG_debuggingimport'
        )

    def test_missing_output_path_does_not_raise(self):
        controller = self.Controller()
        controller.auto_model_performance_rows = []

        self.assertEqual(controller._get_auto_run_directory_name(), '')

    def test_only_seed_conditions_drive_the_label(self):
        '''Inherited optimizer rows alone must not retitle a local seed.'''
        controller = self._controller([
            {
                'condition_type': 'seed',
                'executed_in_current_run': True,
                'origin_run_directory': 'RTG_current'
            },
            {
                'condition_type': 'optimizer_selected',
                'executed_in_current_run': False,
                'origin_run_directory': 'RTG_other'
            }
        ])

        self.assertIsNone(controller._get_auto_inherited_seed_origin_run())

    def test_report_headings_route_through_the_seed_label(self):
        '''Source-level check that report embedding relabels seed figures.'''
        method_node = _get_auto_controller_method_node(
            '_auto_report_plot_markdown_if_exists'
        )
        called_attributes = {
            node.func.attr
            for node in ast.walk(method_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        seed_prefixes = {
            node.value
            for node in ast.walk(method_node)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }

        self.assertIn('_apply_auto_seed_design_label', called_attributes)
        self.assertIn('initial_maximin_seed_design_', seed_prefixes)

    def test_seed_figure_titles_route_through_the_seed_label(self):
        '''Source-level check that the figure titles are relabelled too.'''
        method_node = _get_auto_controller_method_node(
            '_plot_initial_training_designs_after_run'
        )
        called_attributes = {
            node.func.attr
            for node in ast.walk(method_node)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }

        self.assertIn('_apply_auto_seed_design_label', called_attributes)


class AutoLambdaMaximaExtractionTests(unittest.TestCase):
    '''Prevents seed-only helper scope from breaking optimizer batches.'''

    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_auto_controller_methods([
            '_extract_auto_lambda_maxima'
        ])

    def test_shared_lambda_extractor_handles_optimizer_style_scan_data(self):
        controller = self.Controller()
        controller._summarize_auto_scan_quality = (
            lambda wavelengths, absorbances: [
                {'wavelength_nm': wavelength, 'absorbance': absorbance}
                for wavelength, absorbance in zip(wavelengths, absorbances)
            ]
        )
        scan_data = pd.DataFrame({
            'optimizer_well': np.zeros(701, dtype=float)
        })

        wavelengths, quality = controller._extract_auto_lambda_maxima(
            scan_data
        )

        self.assertEqual(len(wavelengths), 1)
        self.assertEqual(len(quality), 1)
        self.assertIn('wavelength_nm', quality[0])

    def test_seed_and_optimizer_paths_use_the_shared_class_method(self):
        for method_name in ('_run', '_run_auto_optimizer_batches'):
            method_node = _get_auto_controller_method_node(method_name)
            called_attributes = {
                node.func.attr
                for node in ast.walk(method_node)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            }
            self.assertIn('_extract_auto_lambda_maxima', called_attributes)


class AutoPlotLayoutRegressionTests(unittest.TestCase):
    '''Keeps shared legends and standalone slices out of the data panel.'''

    def test_pairwise_plots_reserve_a_dedicated_legend_header(self):
        method_node = _get_auto_controller_method_node(
            '_plot_initial_training_design_pairwise'
        )
        method_source = ast.get_source_segment(
            CONTROLLER_PATH.read_text(),
            method_node
        )

        self.assertIn('figure_header_height = 1.55', method_source)
        self.assertIn('bbox_to_anchor=(0.5, 0.915)', method_source)
        self.assertIn('top_margin = 0.785', method_source)

    def test_individual_slices_center_visible_axis_and_colorbar_content(self):
        method_node = _get_auto_controller_method_node(
            'plot_higher_dimensional_GPR_conditional_slices'
        )
        method_source = ast.get_source_segment(
            CONTROLLER_PATH.read_text(),
            method_node
        )

        self.assertIn('def _center_heatmap_axes_and_colorbar(', method_source)
        self.assertIn('axis.get_tightbbox(renderer)', method_source)
        self.assertIn('include_decorations=True', method_source)


class TargetEiIncumbentControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_auto_controller_methods([
            '_get_best_qc_approved_target_error_nm',
            '_synchronize_target_ei_incumbent_from_performance'
        ])

    def _build_controller(self, rows):
        controller = self.Controller()
        controller.auto_model_performance_rows = rows
        return controller

    def test_incumbent_uses_best_approved_condition_level_error(self):
        controller = self._build_controller([
            {
                'use_for_model_training': True,
                'eligible_for_target_incumbent': True,
                'target_error_nm': 8.0,
                # A lucky well is intentionally irrelevant to the incumbent.
                'actual_lambda_rep_1_nm': 625.0
            },
            {
                'use_for_model_training': False,
                'eligible_for_target_incumbent': True,
                'target_error_nm': 0.1
            },
            {
                'use_for_model_training': True,
                'eligible_for_target_incumbent': True,
                'target_error_nm': 3.0
            },
            {
                'use_for_model_training': True,
                'eligible_for_target_incumbent': True,
                'target_error_nm': None
            },
            {
                'use_for_model_training': True,
                'eligible_for_target_incumbent': True,
                'target_error_nm': float('nan')
            },
            {
                'use_for_model_training': True,
                'eligible_for_target_incumbent': True,
                'target_error_nm': -1.0
            }
        ])

        self.assertEqual(
            controller._get_best_qc_approved_target_error_nm(),
            3.0
        )

    def test_target_ei_synchronization_stores_condition_incumbent(self):
        controller = self._build_controller([
            {
                'use_for_model_training': True,
                'eligible_for_target_incumbent': True,
                'target_error_nm': 4.5
            }
        ])
        stored_values = []
        model = SimpleNamespace(
            acquisition_mode='target_ei',
            set_incumbent_target_error_nm=stored_values.append
        )

        with redirect_stdout(io.StringIO()):
            result = (
                controller
                ._synchronize_target_ei_incumbent_from_performance(model)
            )

        self.assertEqual(result, 4.5)
        self.assertEqual(stored_values, [4.5])

    def test_later_trainable_but_unreliable_target_match_cannot_replace_incumbent(self):
        controller = self._build_controller([
            {
                'use_for_model_training': True,
                'eligible_for_target_incumbent': True,
                'target_error_nm': 5.0
            },
            {
                'use_for_model_training': True,
                'eligible_for_target_incumbent': False,
                'target_error_nm': 0.0
            }
        ])

        self.assertEqual(
            controller._get_best_qc_approved_target_error_nm(),
            5.0
        )

    def test_target_ei_synchronization_requires_approved_condition(self):
        controller = self._build_controller([
            {
                'use_for_model_training': False,
                'eligible_for_target_incumbent': False,
                'target_error_nm': 0.5
            }
        ])
        model = SimpleNamespace(acquisition_mode='target_ei')

        with self.assertRaisesRegex(
            ValueError,
            "no replicate-validated condition-level"
        ):
            controller._synchronize_target_ei_incumbent_from_performance(
                model
            )

    def test_other_modes_do_not_require_or_store_incumbent(self):
        controller = self.Controller()
        model = SimpleNamespace(acquisition_mode='balanced')

        self.assertIsNone(
            controller._synchronize_target_ei_incumbent_from_performance(
                model
            )
        )


class TargetDecisionEligibilityRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_auto_controller_methods([
            '_safe_float_or_none',
            '_serialize_auto_audit_value',
            '_format_mask_for_report',
            '_get_active_variable_reagents_from_mask',
            '_summarize_duplicate_lambda_values',
            '_get_auto_replicate_outlier_threshold_nm',
            '_get_auto_replicate_sd_tolerance_nm',
            '_get_auto_target_tolerance_nm',
            '_classify_auto_spectral_observations',
            '_run_lambda_replicate_qc',
            '_get_auto_model_training_decision_from_replicate_qc',
            '_get_auto_target_eligibility_decision',
            '_append_auto_model_performance_rows',
            '_get_auto_run_directory_name',
            '_update_auto_model_performance_closest_so_far',
            '_get_best_qc_approved_target_error_nm',
            '_update_auto_quit_from_condition_level_performance'
        ])

    def _build_controller(self, num_duplicates):
        controller = self.Controller()
        controller.num_duplicates = num_duplicates
        controller.variable_reagents = ['reagent_a']
        controller.robo_params = {
            'target': 625.0,
            'target_tolerance_nm': 10.0,
            'replicate_sd_tolerance_nm': 25.0,
            'replicate_outlier_threshold_nm': 50.0
        }
        controller.rxn_sheet_name = 'hardware_free_test'
        controller.auto_model_performance_rows = []
        controller.auto_condition_counter = 0
        controller.getModelInfo = lambda: controller.robo_params
        controller._get_auto_recipe_volume_balance = lambda recipe: {
            'total_volume': 100.0,
            'fixed_transfer_volumes': {'fixed': 10.0},
            'fixed_volume_total': 10.0,
            'variable_transfer_volumes': {'reagent_a': 10.0},
            'variable_transfer_executable_by_reagent': {
                'reagent_a': True
            },
            'variable_transfers_executable': True,
            'variable_volume_total': 10.0,
            'volume_before_water': 20.0,
            'water_volume': 80.0,
            'volume_does_not_overflow': True,
            'water_transfer_executable': True,
            'volume_feasible': True
        }
        return controller

    def _append_condition(self, lambda_values):
        controller = self._build_controller(len(lambda_values))
        controller._append_auto_model_performance_rows(
            unique_recipes=np.array([[0.1]], dtype=float),
            lambda_max_values=lambda_values,
            condition_type='seed',
            batch_number=0
        )
        return controller, controller.auto_model_performance_rows[0]

    def test_one_favorable_replicate_can_train_but_cannot_set_target_decision(self):
        controller, row = self._append_condition([625.0, None, None])

        self.assertTrue(row['use_for_model_training'])
        self.assertFalse(row['eligible_for_target_incumbent'])
        self.assertFalse(row['eligible_for_target_stop'])
        self.assertEqual(
            row['target_eligibility_status'],
            'ineligible_fewer_than_2_qc_replicates'
        )
        self.assertIsNone(
            controller._get_best_qc_approved_target_error_nm()
        )

        model = SimpleNamespace(curr_iter=0, max_iters=4, quit=False)
        with redirect_stdout(io.StringIO()):
            controller._update_auto_quit_from_condition_level_performance(
                model,
                0
            )
        self.assertFalse(model.quit)

    def test_noisy_target_mean_can_train_but_cannot_be_incumbent_or_stop(self):
        controller, row = self._append_condition([600.0, 650.0])

        self.assertAlmostEqual(
            row['actual_lambda_sd_nm'],
            35.35533905932738
        )
        self.assertTrue(row['use_for_model_training'])
        self.assertFalse(row['eligible_for_target_incumbent'])
        self.assertFalse(row['eligible_for_target_stop'])
        self.assertIsNone(
            controller._get_best_qc_approved_target_error_nm()
        )

        model = SimpleNamespace(curr_iter=0, max_iters=4, quit=False)
        with redirect_stdout(io.StringIO()):
            controller._update_auto_quit_from_condition_level_performance(
                model,
                0
            )
        self.assertFalse(model.quit)

    def test_stricter_configured_sd_threshold_rejects_target_mean_stop(self):
        # This mirrors the RTG_014 pattern: the condition mean is inside the
        # target window, but the individual replicate spread is too broad for
        # a 15 nm target-decision agreement requirement. The observations can
        # still be useful GP training data; only incumbent/stopping authority
        # is denied.
        controller = self._build_controller(3)
        controller.robo_params['replicate_sd_tolerance_nm'] = 15.0
        controller._append_auto_model_performance_rows(
            unique_recipes=np.array([[0.1]], dtype=float),
            lambda_max_values=[642.0, 614.0, 600.0],
            condition_type='seed',
            batch_number=0
        )
        row = controller.auto_model_performance_rows[0]

        self.assertAlmostEqual(row['actual_lambda_mean_nm'], 618.6666666667)
        self.assertAlmostEqual(row['target_error_nm'], 6.3333333333)
        self.assertTrue(row['use_for_model_training'])
        self.assertFalse(row['eligible_for_target_incumbent'])
        self.assertFalse(row['eligible_for_target_stop'])
        self.assertEqual(
            row['target_eligibility_status'],
            'ineligible_replicate_sd_above_tolerance'
        )

        model = SimpleNamespace(curr_iter=0, max_iters=4, quit=False)
        with redirect_stdout(io.StringIO()):
            controller._update_auto_quit_from_condition_level_performance(
                model,
                0
            )
        self.assertFalse(model.quit)

    def test_consistent_replicates_can_set_incumbent_and_stop(self):
        controller, row = self._append_condition([624.0, 626.0])

        self.assertTrue(row['eligible_for_target_incumbent'])
        self.assertTrue(row['eligible_for_target_stop'])
        self.assertEqual(
            controller._get_best_qc_approved_target_error_nm(),
            0.0
        )

        model = SimpleNamespace(curr_iter=0, max_iters=4, quit=False)
        with redirect_stdout(io.StringIO()):
            controller._update_auto_quit_from_condition_level_performance(
                model,
                0
            )
        self.assertTrue(model.quit)

    def test_qc_excluded_outlier_leaves_tight_pair_target_eligible(self):
        _, row = self._append_condition([624.0, 626.0, 800.0])

        self.assertEqual(row['n_replicates_excluded'], 1)
        self.assertEqual(row['actual_lambda_values_qc_nm'], [624.0, 626.0])
        self.assertTrue(row['eligible_for_target_incumbent'])

    def test_nonfinite_replicate_is_preserved_raw_but_not_treated_as_valid(self):
        _, row = self._append_condition([625.0, float('inf')])

        self.assertEqual(row['actual_lambda_values_raw_nm'][0], 625.0)
        self.assertTrue(math.isinf(row['actual_lambda_values_raw_nm'][1]))
        self.assertEqual(row['n_replicates_valid'], 1)
        self.assertEqual(row['n_replicates_used'], 1)
        self.assertFalse(row['eligible_for_target_incumbent'])

    def test_invalid_replicate_sd_tolerance_fails_closed(self):
        for invalid_tolerance in (-1.0, float('nan'), float('inf')):
            controller = self._build_controller(2)
            controller.robo_params['replicate_sd_tolerance_nm'] = (
                invalid_tolerance
            )

            with self.subTest(invalid_tolerance=invalid_tolerance):
                with self.assertRaisesRegex(
                    ValueError,
                    'finite, nonnegative'
                ):
                    controller._get_auto_replicate_sd_tolerance_nm()


class AcquisitionHeaderCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_base_controller_methods([
            '_init_robo_header_params',
            '_get_pi_compatible_reagent_payload'
        ])

    def _base_header(self):
        return [
            ['setting', 'value'],
            ['using_temp_ctrl', 'no'],
            ['temp', ''],
            ['dilution_cont', 'water'],
            ['dilution_vol', '100'],
            ['target', '625'],
            ['max_iterations', '4'],
            ['initial_data', '2']
        ]

    def _parse_header(
        self,
        acquisition_mode=None,
        num_duplicates=None,
        acquisition_modes=None,
        portfolio_min_distance=None,
        target_tolerance_nm=None,
        replicate_sd_tolerance_nm=None,
        auto_terminal_verbosity=None,
        auto_source_volume_check=None,
        auto_source_reserve_volume_uL=None,
        pi_legacy_tare_offset_g=None,
        true_zero_reagents=None,
        auto_spectral_response_policy=None,
        auto_model_checkpoint_mode=None
    ):
        controller = self.Controller()
        controller.robo_params = {}
        controller.DilutionParams = lambda container, volume: (
            container,
            volume
        )
        header = self._base_header()

        if acquisition_mode is not None:
            header.append(['acquisition_mode', acquisition_mode])

        if num_duplicates is not None:
            header.append(['num_duplicates', str(num_duplicates)])

        if acquisition_modes is not None:
            header.append(['acquisition_modes', acquisition_modes])

        if portfolio_min_distance is not None:
            header.append([
                'portfolio_min_distance',
                str(portfolio_min_distance)
            ])

        if target_tolerance_nm is not None:
            header.append([
                'target_tolerance_nm',
                str(target_tolerance_nm)
            ])

        if replicate_sd_tolerance_nm is not None:
            header.append([
                'replicate_sd_tolerance_nm',
                str(replicate_sd_tolerance_nm)
            ])

        if auto_terminal_verbosity is not None:
            header.append([
                'auto_terminal_verbosity',
                str(auto_terminal_verbosity)
            ])

        if auto_source_volume_check is not None:
            header.append([
                'auto_source_volume_check',
                str(auto_source_volume_check)
            ])

        if auto_source_reserve_volume_uL is not None:
            header.append([
                'auto_source_reserve_volume_uL',
                str(auto_source_reserve_volume_uL)
            ])

        if pi_legacy_tare_offset_g is not None:
            header.append([
                'pi_legacy_tare_offset_g',
                str(pi_legacy_tare_offset_g)
            ])

        if true_zero_reagents is not None:
            header.append([
                'true_zero_reagents',
                str(true_zero_reagents)
            ])

        if auto_spectral_response_policy is not None:
            header.append([
                'auto_spectral_response_policy',
                str(auto_spectral_response_policy)
            ])

        if auto_model_checkpoint_mode is not None:
            header.append([
                'auto_model_checkpoint_mode',
                str(auto_model_checkpoint_mode)
            ])

        with redirect_stdout(io.StringIO()):
            controller._init_robo_header_params(header)

        return controller.robo_params

    def test_legacy_header_defaults_to_exact_exploit_compatibility(self):
        parsed = self._parse_header()

        self.assertEqual(parsed['acquisition_mode'], 'exploit')
        self.assertEqual(parsed['auto_plot_profile'], 'standard')
        self.assertEqual(parsed['num_duplicates'], 3)
        self.assertFalse(parsed['allow_true_zero'])
        self.assertEqual(parsed['true_zero_reagents_requested'], '')
        self.assertEqual(parsed['target_tolerance_nm'], 10.0)
        self.assertEqual(parsed['replicate_sd_tolerance_nm'], 25.0)
        self.assertEqual(parsed['auto_terminal_verbosity'], 'standard')
        self.assertEqual(parsed['auto_source_volume_check'], 'off')
        self.assertEqual(parsed['pi_legacy_tare_offset_g'], 0.0)
        self.assertEqual(
            parsed['auto_spectral_response_policy'],
            'audit_only'
        )
        self.assertEqual(parsed['auto_model_checkpoint_mode'], 'off')
        self.assertEqual(parsed['acquisition_modes'], ['exploit'])
        self.assertFalse(parsed['using_acquisition_portfolio'])

    def test_header_checkpoint_save_mode_is_optional_and_normalized(self):
        self.assertEqual(
            self._parse_header(
                auto_model_checkpoint_mode='SAVE'
            )['auto_model_checkpoint_mode'],
            'save'
        )
        self.assertEqual(
            self._parse_header(
                auto_model_checkpoint_mode='on'
            )['auto_model_checkpoint_mode'],
            'save'
        )

        self.assertEqual(
            self._parse_header(
                auto_model_checkpoint_mode='resume'
            )['auto_model_checkpoint_mode'],
            'import'
        )

        with self.assertRaisesRegex(ValueError, 'off, save, or import'):
            self._parse_header(auto_model_checkpoint_mode='unsupported')

    def test_header_spectral_response_policy_is_optional_and_normalized(self):
        self.assertEqual(
            self._parse_header(
                auto_spectral_response_policy='Boundary Aware'
            )['auto_spectral_response_policy'],
            'boundary_aware'
        )
        self.assertEqual(
            self._parse_header(
                auto_spectral_response_policy='audit'
            )['auto_spectral_response_policy'],
            'audit_only'
        )

        with self.assertRaisesRegex(
            ValueError,
            'audit_only or boundary_aware'
        ):
            self._parse_header(
                auto_spectral_response_policy='low_signal'
            )

    def test_header_preserves_selective_true_zero_request_for_auto_resolution(self):
        parsed = self._parse_header(
            true_zero_reagents='Silver_Nitrate; PVP'
        )

        self.assertEqual(
            parsed['true_zero_reagents_requested'],
            'Silver_Nitrate; PVP'
        )

    def test_header_source_volume_protection_is_explicit_and_fail_closed(self):
        parsed = self._parse_header(
            auto_source_volume_check='ON',
            auto_source_reserve_volume_uL=50
        )

        self.assertEqual(parsed['auto_source_volume_check'], 'required')
        self.assertEqual(parsed['auto_source_reserve_volume_uL'], 50.0)

        self.assertEqual(
            self._parse_header(
                auto_source_volume_check='required'
            )['auto_source_reserve_volume_uL'],
            0.0
        )

        with self.assertRaisesRegex(ValueError, 'off or required'):
            self._parse_header(auto_source_volume_check='maybe')

        with self.assertRaisesRegex(ValueError, 'reserve_volume'):
            self._parse_header(
                auto_source_volume_check='required',
                auto_source_reserve_volume_uL='-1'
            )

    def test_pi_legacy_tare_offset_is_optional_and_payload_only(self):
        parsed = self._parse_header(pi_legacy_tare_offset_g='0.3')
        self.assertEqual(parsed['pi_legacy_tare_offset_g'], 0.3)

        with self.assertRaisesRegex(ValueError, 'finite, nonnegative mass'):
            self._parse_header(pi_legacy_tare_offset_g='-0.3')

        controller = self.Controller()
        controller.robo_params = {
            'pi_legacy_tare_offset_g': 0.3,
            'reagent_df': pd.DataFrame(
                {
                    'mass': [7.5731, 14.2950],
                    'conc': [1.0, 1.0]
                },
                index=['reagent_aC1.0', 'reagent_bC1.0']
            )
        }

        payload = controller._get_pi_compatible_reagent_payload()

        self.assertEqual(
            list(payload['mass'].values()),
            [7.2731, 13.9950]
        )
        self.assertEqual(
            controller.robo_params['reagent_df'].loc[
                'reagent_aC1.0',
                'mass'
            ],
            7.5731
        )

    def test_header_target_tolerance_is_optional_and_validated(self):
        parsed = self._parse_header(target_tolerance_nm=5.0)
        self.assertEqual(parsed['target_tolerance_nm'], 5.0)

        for invalid_tolerance in ('-1', 'nan', 'inf', 'not_a_number'):
            with self.subTest(invalid_tolerance=invalid_tolerance):
                with self.assertRaisesRegex(
                    ValueError,
                    'finite, nonnegative'
                ):
                    self._parse_header(
                        target_tolerance_nm=invalid_tolerance
                    )

    def test_header_replicate_sd_tolerance_is_optional_and_validated(self):
        parsed = self._parse_header(replicate_sd_tolerance_nm=15.0)
        self.assertEqual(parsed['replicate_sd_tolerance_nm'], 15.0)

        for invalid_tolerance in ('', '-1', 'nan', 'inf', 'not_a_number'):
            with self.subTest(invalid_tolerance=invalid_tolerance):
                with self.assertRaisesRegex(
                    ValueError,
                    'finite, nonnegative'
                ):
                    self._parse_header(
                        replicate_sd_tolerance_nm=invalid_tolerance
                    )

    def test_header_terminal_verbosity_normalizes_and_fails_clearly(self):
        cases = {
            'essential': 'essential',
            'OFF': 'essential',
            'limited': 'standard',
            'on': 'standard',
            'debug': 'diagnostic',
            'ALL': 'diagnostic'
        }

        for workbook_value, expected_value in cases.items():
            with self.subTest(workbook_value=workbook_value):
                parsed = self._parse_header(
                    auto_terminal_verbosity=workbook_value
                )
                self.assertEqual(
                    parsed['auto_terminal_verbosity'],
                    expected_value
                )

        with self.assertRaisesRegex(
            ValueError,
            'auto_terminal_verbosity'
        ):
            self._parse_header(auto_terminal_verbosity='chatty')

    def test_canonical_modes_and_documented_aliases_normalize(self):
        cases = {
            'exploit': 'exploit',
            'exploration': 'explore',
            'straddle': 'balanced',
            'expected improvement': 'target_ei'
        }

        for workbook_value, expected_mode in cases.items():
            with self.subTest(workbook_value=workbook_value):
                parsed = self._parse_header(workbook_value)
                self.assertEqual(
                    parsed['acquisition_mode'],
                    expected_mode
                )

    def test_invalid_acquisition_mode_fails_before_protocol_execution(self):
        with self.assertRaisesRegex(
            ValueError,
            "exploit, explore, balanced, or target_ei"
        ):
            self._parse_header('ordinary_ei')

    def test_target_ei_requires_at_least_two_replicates(self):
        for workbook_value in ('target_ei', 'ei'):
            with self.subTest(workbook_value=workbook_value):
                with self.assertRaisesRegex(
                    ValueError,
                    "target_ei requires num_duplicates to be at least 2"
                ):
                    self._parse_header(
                        workbook_value,
                        num_duplicates=1
                    )

        parsed = self._parse_header('exploit', num_duplicates=1)
        self.assertEqual(parsed['num_duplicates'], 1)

    def test_core3_portfolio_requires_explicit_singular_off(self):
        parsed = self._parse_header(
            acquisition_mode='off',
            acquisition_modes='core3',
            portfolio_min_distance=0.10
        )

        self.assertEqual(
            parsed['acquisition_modes'],
            ['exploit', 'explore', 'balanced']
        )
        self.assertTrue(parsed['using_acquisition_portfolio'])
        self.assertEqual(parsed['portfolio_min_distance'], 0.10)

        with self.assertRaisesRegex(ValueError, 'both active'):
            self._parse_header(
                acquisition_mode='exploit',
                acquisition_modes='core3'
            )

        with self.assertRaisesRegex(ValueError, 'both off'):
            self._parse_header(
                acquisition_mode='off',
                acquisition_modes='off'
            )

    def test_portfolio_distance_accepts_readable_aliases_and_numbers(self):
        cases = {
            'NONE': 0.00,
            'modest': 0.05,
            'Strong': 0.10,
            'very strong': 0.15,
            '0.075': 0.075
        }

        for workbook_value, expected_distance in cases.items():
            with self.subTest(workbook_value=workbook_value):
                parsed = self._parse_header(
                    acquisition_mode='off',
                    acquisition_modes='core3',
                    portfolio_min_distance=workbook_value
                )
                self.assertEqual(
                    parsed['portfolio_min_distance'],
                    expected_distance
                )

    def test_portfolio_rejects_duplicate_modes_and_target_ei_singletons(self):
        with self.assertRaisesRegex(ValueError, 'must not repeat'):
            self._parse_header(
                acquisition_mode='off',
                acquisition_modes='exploit;exploit'
            )

        with self.assertRaisesRegex(ValueError, 'target_ei requires'):
            self._parse_header(
                acquisition_mode='off',
                acquisition_modes='exploit;target_ei',
                num_duplicates=1
            )


class VariableReagentStockConcentrationTests(unittest.TestCase):
    '''Regression coverage for same-stock backup source containers.'''

    @classmethod
    def setUpClass(cls):
        cls.AutoController = _load_auto_controller_methods([
            '_get_variable_reagent_stock_conc'
        ])

    def _build_controller(self, reagent_rows, reagent_index):
        controller = self.AutoController()
        controller.robo_params = {
            'reagent_df': pd.DataFrame(
                reagent_rows,
                index=reagent_index
            )
        }
        return controller

    def test_same_stock_backup_tubes_return_shared_concentration(self):
        controller = self._build_controller(
            {
                'conc': [6.25, 6.25],
                'deck_pos': [3, 3],
                'loc': ['A1', 'A2']
            },
            ['sodium_borohydride', 'sodium_borohydride']
        )

        self.assertEqual(
            controller._get_variable_reagent_stock_conc(
                'sodium_borohydride'
            ),
            6.25
        )

    def test_different_stock_backup_tubes_fail_before_recipe_calculation(self):
        controller = self._build_controller(
            {
                'conc': [6.25, 3.125],
                'deck_pos': [3, 3],
                'loc': ['A1', 'A2']
            },
            ['sodium_borohydride', 'sodium_borohydride']
        )

        with self.assertRaisesRegex(
            ValueError,
            'different stock concentrations'
        ):
            controller._get_variable_reagent_stock_conc(
                'sodium_borohydride'
            )

    def test_concentration_marked_container_name_remains_supported(self):
        controller = self._build_controller(
            {
                'conc': [0.375],
                'deck_pos': [2],
                'loc': ['A1']
            },
            ['silver_nitrateC0.375']
        )

        self.assertEqual(
            controller._get_variable_reagent_stock_conc('silver_nitrate'),
            0.375
        )


class SelectiveTrueZeroControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AutoController = _load_auto_controller_methods([
            '_resolve_true_zero_reagents',
            '_apply_true_zero_transfer_rule_to_volume',
            '_apply_true_zero_transfer_rule_to_recipes'
        ])
        cls.BaseController = _load_base_controller_methods([
            'get_min_conc'
        ])

    def _build_controller(self, requested_value, allow_true_zero=True):
        controller = self.AutoController()
        controller.variable_reagents = [
            'silver_nitrate',
            'potassium_bromide',
            'PVP'
        ]
        controller.robo_params = {
            'allow_true_zero': allow_true_zero,
            'true_zero_reagents_requested': requested_value
        }
        controller.template_meta = {'tot_vol': 100.0}
        controller.get_variable_reagents = lambda: controller.variable_reagents
        controller._get_variable_reagent_stock_conc = (
            lambda reagent_name: 1.0
        )
        return controller

    def _get_min_concentrations(self, resolved_robo_params):
        controller = self.BaseController()
        controller.variable_reagents = [
            'silver_nitrate',
            'potassium_bromide',
            'PVP'
        ]
        controller.robo_params = resolved_robo_params
        controller.template_meta = {'tot_vol': 100.0}
        controller.get_variable_reagents = lambda: controller.variable_reagents
        controller._get_variable_reagent_stock_conc = (
            lambda reagent_name: 1.0
        )

        with redirect_stdout(io.StringIO()):
            return controller.get_min_conc()

    def test_selected_reagents_are_case_insensitive_and_only_they_get_zero_bounds(
        self
    ):
        controller = self._build_controller('SILVER_NITRATE; pvp')

        with redirect_stdout(io.StringIO()):
            controller._resolve_true_zero_reagents()

        min_concentrations = self._get_min_concentrations(
            controller.robo_params
        )

        self.assertEqual(
            controller.robo_params['true_zero_reagents'],
            ['silver_nitrate', 'PVP']
        )
        self.assertEqual(
            controller.robo_params['true_zero_reagent_indices'],
            [0, 2]
        )
        self.assertEqual(min_concentrations['silver_nitrate'], 0.0)
        self.assertEqual(min_concentrations['PVP'], 0.0)
        self.assertEqual(min_concentrations['potassium_bromide'], 0.05)

    def test_legacy_blank_selection_preserves_all_or_none_behavior(self):
        enabled_controller = self._build_controller('')
        disabled_controller = self._build_controller('', allow_true_zero=False)

        with redirect_stdout(io.StringIO()):
            enabled_controller._resolve_true_zero_reagents()
            disabled_controller._resolve_true_zero_reagents()

        self.assertEqual(
            enabled_controller.robo_params['true_zero_reagents'],
            enabled_controller.variable_reagents
        )
        self.assertEqual(
            disabled_controller.robo_params['true_zero_reagents'],
            []
        )

    def test_controller_rejects_zero_for_required_on_reagent(self):
        controller = self._build_controller('silver_nitrate')

        with redirect_stdout(io.StringIO()):
            controller._resolve_true_zero_reagents()

        with self.assertRaisesRegex(ValueError, 'required-ON'):
            controller._apply_true_zero_transfer_rule_to_recipes(
                np.array([[0.0, 0.0, 0.1]])
            )

    def test_invalid_selection_fails_clearly_before_run(self):
        controller = self._build_controller('unknown_reagent')

        with self.assertRaisesRegex(ValueError, 'not variable transfer'):
            with redirect_stdout(io.StringIO()):
                controller._resolve_true_zero_reagents()

        controller = self._build_controller('silver_nitrate', False)

        with self.assertRaisesRegex(ValueError, 'allow_true_zero'):
            with redirect_stdout(io.StringIO()):
                controller._resolve_true_zero_reagents()


class AutoSourceVolumePreflightTests(unittest.TestCase):
    '''Exercises source-volume protection without importing hardware code.'''

    @classmethod
    def setUpClass(cls):
        class FakeConversionError(Exception):
            pass

        cls.FakeConversionError = FakeConversionError
        cls.Controller = _load_auto_controller_methods([
            '_get_auto_batch_source_volume_requirements',
            '_export_auto_source_volume_audit',
            '_preflight_auto_source_volumes',
            '_create_samples'
        ], extra_namespace={'ConversionError': FakeConversionError})

    def _build_controller(self):
        controller = self.Controller()
        controller.batch_num = 0
        controller.robo_params = {
            'auto_source_volume_check': 'required',
            'auto_source_reserve_volume_uL': 5.0,
            'auto_terminal_verbosity': 'essential'
        }
        controller._products = ['autowell0C1.0', 'autowell1C1.0']
        controller._round_transfer_volume = lambda volume: float(volume)
        controller._cached_reader_locs = {
            'reagent_aC1.0': SimpleNamespace(
                loc='A1',
                deck_pos=1,
                vol=100.0,
                aspirable_vol=80.0
            ),
            'WaterC1.0': SimpleNamespace(
                loc='A2',
                deck_pos=1,
                vol=300.0,
                aspirable_vol=280.0
            )
        }
        return controller

    def _build_protocol_dataframe(self):
        return pd.DataFrame([
            {
                'op': 'transfer',
                'chemical_name': 'reagent_aC1.0',
                'autowell0C1.0': 20.0,
                'autowell1C1.0': 20.0
            },
            {
                'op': 'transfer',
                'chemical_name': 'WaterC1.0',
                'autowell0C1.0': 70.0,
                'autowell1C1.0': 70.0
            }
        ])

    def test_preflight_uses_current_cached_aggregate_inventory(self):
        controller = self._build_controller()
        protocol_df = self._build_protocol_dataframe()

        with redirect_stdout(io.StringIO()):
            audit_rows = controller._preflight_auto_source_volumes(
                protocol_df,
                'batch 0'
            )

        audit_by_source = {row['source_chemical_name']: row for row in audit_rows}
        self.assertEqual(
            audit_by_source['reagent_aC1.0']['planned_usage_uL'],
            40.0
        )
        self.assertEqual(audit_by_source['reagent_aC1.0']['source_loc'], 'A1')
        self.assertEqual(
            audit_by_source['reagent_aC1.0'][
                'remaining_aspirable_volume_uL'
            ],
            40.0
        )
        self.assertEqual(
            audit_by_source['WaterC1.0']['required_aspirable_volume_uL'],
            145.0
        )
        self.assertEqual(
            audit_by_source['reagent_aC1.0']['preflight_scope'],
            'aggregate_source_group'
        )

    def test_preflight_rejects_insufficient_aggregate_aspirable_volume(self):
        controller = self._build_controller()
        controller._cached_reader_locs['reagent_aC1.0'].aspirable_vol = 44.0

        with self.assertRaisesRegex(
            ValueError,
            'planned use 40.0000 uL plus reserve 5.0000 uL'
        ):
            with redirect_stdout(io.StringIO()):
                controller._preflight_auto_source_volumes(
                    self._build_protocol_dataframe(),
                    'batch 0'
                )

    def test_legacy_source_volume_setting_preserves_no_preflight_behavior(self):
        controller = self._build_controller()
        controller.robo_params['auto_source_volume_check'] = 'off'

        self.assertIsNone(
            controller._preflight_auto_source_volumes(
                self._build_protocol_dataframe(),
                'legacy batch'
            )
        )

    def test_required_preflight_blocks_unaccounted_conversion_recovery(self):
        controller = self._build_controller()
        controller._validate_auto_recipe_volume_feasibility = (
            lambda recipes, context_label: None
        )
        controller._preflight_auto_source_volumes = lambda rxn_df, context_label: []
        controller.template_meta = {
            'labware': 'test_labware',
            'cont': 'A1',
            'tot_vol': 100.0
        }
        controller.portal = SimpleNamespace(send_pack=lambda *args: None)
        controller._clean_meta = lambda wellnames: None
        controller._build_rxn_df = lambda wellnames, recipes: (
            (_ for _ in ()).throw(self.FakeConversionError())
        )
        controller._insert_tot_vol_transfer = lambda: None
        controller.tot_vols = {}
        recovery_calls = []
        controller._handle_conversion_err = lambda error: recovery_calls.append(error)

        with self.assertRaisesRegex(ValueError, 'unplanned dilution/recovery'):
            controller._create_samples(
                ['autowell0C1.0'],
                np.array([[0.1]], dtype=float)
            )

        self.assertEqual(recovery_calls, [])

    def test_preflight_rejects_missing_cached_source_inventory(self):
        controller = self._build_controller()
        del controller._cached_reader_locs['WaterC1.0']

        with self.assertRaisesRegex(
            ValueError,
            'WaterC1.0: no current source-volume record'
        ):
            with redirect_stdout(io.StringIO()):
                controller._preflight_auto_source_volumes(
                    self._build_protocol_dataframe(),
                    'batch 0'
                )


class PredictiveUncertaintyUnitTests(unittest.TestCase):
    class FakeArray:
        def __init__(self, value):
            self.value = value

        def reshape(self, *shape):
            return self

        def flatten(self):
            return self

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self.value

    @classmethod
    def setUpClass(cls):
        fake_numpy = SimpleNamespace(
            asarray=lambda value, dtype=None: cls.FakeArray(value)
        )
        cls.PredictionModel = _load_optimization_model_methods(
            ['predict_lambda_distribution_nm'],
            extra_namespace={'np': fake_numpy}
        )

    def test_predictive_standard_deviation_converts_directly_to_nm(self):
        model = self.PredictionModel()
        model._get_dimension = lambda: 1
        model.gp_model = SimpleNamespace(
            predict=lambda x: (
                self.FakeArray(0.5),
                self.FakeArray(0.02)
            )
        )

        predicted_mean_nm, predicted_std_nm = (
            model.predict_lambda_distribution_nm([0.25])
        )

        self.assertEqual(predicted_mean_nm, 600.0)
        self.assertEqual(predicted_std_nm, 12.0)

    def test_tiny_negative_predictive_sd_roundoff_clamps_to_zero(self):
        model = self.PredictionModel()
        model._get_dimension = lambda: 1
        model.gp_model = SimpleNamespace(
            predict=lambda x: (
                self.FakeArray(0.5),
                self.FakeArray(-1e-13)
            )
        )

        predicted_mean_nm, predicted_std_nm = (
            model.predict_lambda_distribution_nm([0.25])
        )

        self.assertEqual(predicted_mean_nm, 600.0)
        self.assertEqual(predicted_std_nm, 0.0)

    def test_materially_negative_predictive_sd_fails_closed(self):
        model = self.PredictionModel()
        model._get_dimension = lambda: 1
        model.gp_model = SimpleNamespace(
            predict=lambda x: (
                self.FakeArray(0.5),
                self.FakeArray(-0.01)
            )
        )

        with self.assertRaisesRegex(
            ValueError,
            'materially negative normalized predictive standard deviation'
        ):
            model.predict_lambda_distribution_nm([0.25])


class MaskComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_numpy = SimpleNamespace(
            isfinite=math.isfinite,
            asarray=lambda value, dtype=None: value
        )
        cls.MaskModel = _load_optimization_model_methods(
            ['_optimize_acquisition_with_masks'],
            extra_namespace={'np': fake_numpy}
        )

    def test_equal_scores_select_first_allowed_mask_deterministically(self):
        model = self.MaskModel()
        masks = ('first-mask', 'second-mask')
        model._get_reagent_masks_for_current_settings = lambda: masks

        results = {
            'first-mask': {
                'mask': 'first-mask',
                'success': True,
                'message': 'ok',
                'objective': -10.0,
                'x_full': [0.1],
                'volume_balance': {'volume_feasible': True},
                'predicted_lambda_max': 620.0
            },
            'second-mask': {
                'mask': 'second-mask',
                'success': True,
                'message': 'ok',
                'objective': -10.0,
                'x_full': [0.9],
                'volume_balance': {'volume_feasible': True},
                'predicted_lambda_max': 700.0
            }
        }
        model._optimize_single_mask = (
            lambda mask, n_restarts: results[mask]
        )

        selected = model._optimize_acquisition_with_masks(
            n_restarts_per_mask=1
        )

        self.assertEqual(selected, [0.1])
        self.assertEqual(model.last_selected_mask, 'first-mask')


class GetNextReactionCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.SelectionModel = _load_optimization_model_methods([
            'getNextReaction'
        ])

    def test_exploit_preserves_controller_return_shape_and_predictions(self):
        model = self.SelectionModel()
        model.acquisition_mode = 'exploit'
        model.target_value = 625.0
        model._calculate_acquisition_score = (
            lambda predicted_lambda_mean_nm,
            predicted_lambda_std_nm,
            incumbent_target_error_nm: 0.25
        )
        optimization_calls = []
        model._optimize_acquisition_with_masks = (
            lambda: optimization_calls.append(True) or [0.25]
        )
        model.predict_lambda_distribution_nm = (
            lambda x: (624.5, 1.25)
        )

        output = io.StringIO()

        with redirect_stdout(output):
            result = model.getNextReaction()

        self.assertEqual(optimization_calls, [True])
        self.assertEqual(result, [[0.25]])
        self.assertEqual(
            model.last_optimizer_predicted_lambda_mean_nm,
            624.5
        )
        self.assertEqual(
            model.last_optimizer_predicted_lambda_std_nm,
            1.25
        )
        self.assertEqual(
            model.last_optimizer_predicted_target_error_nm,
            0.5
        )
        self.assertEqual(model.last_optimizer_acquisition_mode, 'exploit')
        self.assertEqual(model.last_optimizer_acquisition_score, 0.25)
        self.assertIsNone(model.last_optimizer_incumbent_target_error_nm)
        self.assertIn('acquisition audit:', output.getvalue())
        self.assertIn('mode=exploit', output.getvalue())
        self.assertIn('predicted_target_error=0.5000 nm', output.getvalue())

    def test_unknown_mode_stops_before_optimization(self):
        model = self.SelectionModel()
        model.acquisition_mode = 'ordinary_ei'
        optimization_calls = []
        model._optimize_acquisition_with_masks = (
            lambda: optimization_calls.append(True)
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "'ordinary_ei'"
        ):
            model.getNextReaction()

        self.assertEqual(optimization_calls, [])

    def test_explore_reaches_optimizer_and_preserves_return_shape(self):
        model = self.SelectionModel()
        model.acquisition_mode = 'explore'
        model.target_value = 625.0
        model._calculate_acquisition_score = (
            lambda predicted_lambda_mean_nm,
            predicted_lambda_std_nm,
            incumbent_target_error_nm: -15.0
        )
        optimization_calls = []
        model._optimize_acquisition_with_masks = (
            lambda: optimization_calls.append(True) or [0.75]
        )
        model.predict_lambda_distribution_nm = (
            lambda x: (700.0, 15.0)
        )

        with redirect_stdout(io.StringIO()):
            result = model.getNextReaction()

        self.assertEqual(optimization_calls, [True])
        self.assertEqual(result, [[0.75]])

    def test_balanced_reaches_optimizer_and_preserves_return_shape(self):
        model = self.SelectionModel()
        model.acquisition_mode = 'balanced'
        model.target_value = 625.0
        model._calculate_acquisition_score = (
            lambda predicted_lambda_mean_nm,
            predicted_lambda_std_nm,
            incumbent_target_error_nm: -3.0
        )
        optimization_calls = []
        model._optimize_acquisition_with_masks = (
            lambda: optimization_calls.append(True) or [0.5]
        )
        model.predict_lambda_distribution_nm = (
            lambda x: (626.0, 4.0)
        )

        with redirect_stdout(io.StringIO()):
            result = model.getNextReaction()

        self.assertEqual(optimization_calls, [True])
        self.assertEqual(result, [[0.5]])

    def test_target_ei_reaches_optimizer_and_records_incumbent(self):
        model = self.SelectionModel()
        model.acquisition_mode = 'target_ei'
        model.target_value = 625.0
        model.incumbent_target_error_nm = 7.5
        model._calculate_acquisition_score = (
            lambda predicted_lambda_mean_nm,
            predicted_lambda_std_nm,
            incumbent_target_error_nm: -4.25
        )
        optimization_calls = []
        model._optimize_acquisition_with_masks = (
            lambda: optimization_calls.append(True) or [0.6]
        )
        model.predict_lambda_distribution_nm = (
            lambda x: (627.0, 3.0)
        )

        with redirect_stdout(io.StringIO()):
            result = model.getNextReaction()

        self.assertEqual(optimization_calls, [True])
        self.assertEqual(result, [[0.6]])
        self.assertEqual(
            model.last_optimizer_incumbent_target_error_nm,
            7.5
        )
        self.assertEqual(model.last_optimizer_acquisition_score, -4.25)
        self.assertEqual(
            model.last_optimizer_predicted_target_error_nm,
            2.0
        )


class OptimizerRecipeHandoffSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_auto_controller_methods([
            '_serialize_auto_audit_value',
            'Normalize_Denormalize_Recipes',
            '_build_auto_optimizer_selection_metadata',
            '_prepare_auto_optimizer_recipe_for_execution',
            '_apply_true_zero_transfer_rule_to_volume',
            '_apply_true_zero_transfer_rule_to_recipes',
            '_get_variable_transfer_volumes_for_recipe',
            '_get_auto_recipe_volume_balance',
            '_validate_auto_recipe_volume_feasibility',
            '_export_auto_batch_recipe_design'
        ])

    def _build_controller_and_model(self):
        controller = self.Controller()
        controller.variable_reagents = ['reagent_a']
        controller.min_conc = [0.0]
        controller.max_conc = [1.0]
        controller.template_meta = {'tot_vol': 100.0}
        controller._get_variable_reagent_stock_conc = lambda name: 1.0
        controller._get_fixed_reagent_volumes = lambda: {'fixed': 10.0}
        exports = []
        controller._export_auto_batch_recipe_design = (
            lambda **kwargs: exports.append(kwargs)
        )

        model = SimpleNamespace(
            acquisition_mode='exploit',
            balanced_exploration_weight=1.0,
            last_optimizer_acquisition_mode='exploit',
            last_optimizer_acquisition_score=1.0,
            last_selected_mask=np.array([1], dtype=int),
            last_optimizer_predicted_target_error_nm=1.0,
            last_optimizer_predicted_lambda_mean_nm=626.0,
            last_optimizer_predicted_lambda_std_nm=2.0,
            last_optimizer_incumbent_target_error_nm=None,
            last_optimizer_volume_balance={
                'volume_feasible': True,
                'water_volume': 80.0
            },
            last_mask_results=[]
        )
        return controller, model, exports

    def test_unchanged_executable_proposal_preserves_selection_provenance(self):
        controller, model, exports = self._build_controller_and_model()
        controller.robo_params = {
            'auto_terminal_verbosity': 'diagnostic'
        }
        proposal = np.array([[0.1]], dtype=float)
        terminal_output = io.StringIO()

        with redirect_stdout(terminal_output):
            executed, metadata = (
                controller._prepare_auto_optimizer_recipe_for_execution(
                    model,
                    proposal,
                    'batch_1'
                )
            )

        np.testing.assert_array_equal(proposal, np.array([[0.1]]))
        np.testing.assert_allclose(executed, [[0.1]])
        self.assertEqual(metadata['selected_normalized_recipe'], [[0.1]])
        self.assertEqual(metadata['executed_normalized_recipe'], [[0.1]])
        self.assertFalse(metadata['optimizer_recipe_repaired'])
        self.assertEqual(
            metadata['optimizer_recipe_repair_max_transfer_delta_uL'],
            0.0
        )
        self.assertEqual(len(exports), 1)
        self.assertIs(
            exports[0]['selection_metadata'],
            metadata
        )

        provenance_line = next(
            line for line in terminal_output.getvalue().splitlines()
            if line.startswith(
                '<<controller diagnostic>> optimizer selection/execution '
                'provenance: '
            )
        )
        terminal_metadata = json.loads(
            provenance_line.split(': ', 1)[1]
        )
        self.assertEqual(
            terminal_metadata['selected_physical_recipe'],
            [[0.1]]
        )
        self.assertEqual(
            terminal_metadata['executed_physical_recipe'],
            [[0.1]]
        )
        self.assertEqual(
            terminal_metadata[
                'selected_controller_volume_balances'
            ][0]['water_volume'],
            80.0
        )
        self.assertFalse(
            terminal_metadata['optimizer_recipe_repaired']
        )

    def test_standard_recipe_handoff_keeps_terminal_concise(self):
        controller, model, _ = self._build_controller_and_model()
        controller.robo_params = {
            'auto_terminal_verbosity': 'standard'
        }
        terminal_output = io.StringIO()

        with redirect_stdout(terminal_output):
            controller._prepare_auto_optimizer_recipe_for_execution(
                model,
                np.array([[0.1]], dtype=float),
                'standard_batch'
            )

        output_text = terminal_output.getvalue()
        self.assertIn(
            'optimizer/controller recipe invariant passed',
            output_text
        )
        self.assertNotIn(
            'optimizer selection/execution provenance:',
            output_text
        )

    def test_selection_metadata_is_deeply_immutable_after_capture(self):
        controller, model, exports = self._build_controller_and_model()
        model.last_optimizer_volume_balance['nested'] = {'value': 1}
        model.last_mask_results = [
            {
                'mask': np.array([1], dtype=int),
                'x_full': np.array([0.1], dtype=float),
                'objective': 1.0,
                'volume_balance': {
                    'volume_feasible': True,
                    'water_volume': 80.0,
                    'nested': {'value': 1}
                }
            }
        ]

        with redirect_stdout(io.StringIO()):
            _, metadata = (
                controller._prepare_auto_optimizer_recipe_for_execution(
                    model,
                    np.array([[0.1]], dtype=float),
                    'immutable_batch'
                )
            )

        model.last_optimizer_volume_balance['nested']['value'] = 99
        model.last_mask_results[0]['mask'][0] = 0
        model.last_mask_results[0]['volume_balance']['nested']['value'] = 99

        self.assertEqual(
            metadata['optimizer_volume_balance']['nested']['value'],
            1
        )
        self.assertEqual(metadata['mask_results'][0]['mask'].tolist(), [1])
        self.assertEqual(
            metadata['mask_results'][0]['volume_balance']['nested']['value'],
            1
        )

        # The model retains its own independent snapshot for post-failure
        # inspection; callers cannot mutate it through the returned metadata.
        metadata['selected_normalized_recipe'][0][0] = 0.9
        self.assertEqual(
            model.last_controller_selection_metadata[
                'selected_normalized_recipe'
            ],
            [[0.1]]
        )

    def test_controller_repair_blocks_optimizer_proposal_after_audit_export(self):
        controller, model, exports = self._build_controller_and_model()

        # 0.03 concentration at 1.0 stock in a 100 uL reaction is a 3 uL
        # transfer, which the controller rounds to 5 uL. Optimizer proposals
        # must never rely on that repair.
        with self.assertRaisesRegex(
            RuntimeError,
            'stopped before wells or robot commands were created'
        ):
            with redirect_stdout(io.StringIO()):
                controller._prepare_auto_optimizer_recipe_for_execution(
                    model,
                    np.array([[0.03]], dtype=float),
                    'batch_2'
                )

        self.assertEqual(len(exports), 1)
        failure_metadata = exports[0]['selection_metadata']
        self.assertTrue(failure_metadata['optimizer_recipe_repaired'])
        self.assertAlmostEqual(
            failure_metadata[
                'optimizer_recipe_repair_max_transfer_delta_uL'
            ],
            2.0
        )

    def test_failed_repair_persists_exact_recipe_design_audit_values(self):
        controller, model, _ = self._build_controller_and_model()

        # Remove the instance test stub so this regression executes the exact
        # production CSV exporter loaded from controller.py.
        del controller.__dict__['_export_auto_batch_recipe_design']
        controller.batch_num = 2

        with TemporaryDirectory() as temp_directory:
            controller.debug_path = temp_directory

            with self.assertRaises(RuntimeError):
                with redirect_stdout(io.StringIO()):
                    controller._prepare_auto_optimizer_recipe_for_execution(
                        model,
                        np.array([[0.03]], dtype=float),
                        'failed_batch'
                    )

            export_path = Path(temp_directory) / (
                'auto_recipe_design/auto_recipe_design_failed_batch.csv'
            )
            persisted = pd.read_csv(export_path)

        row = persisted.iloc[0]
        self.assertEqual(row['acquisition_mode'], 'exploit')
        self.assertEqual(row['acquisition_score'], 1.0)
        self.assertEqual(json.loads(row['selected_mask']), [1])
        self.assertTrue(bool(row['optimizer_recipe_repaired']))
        self.assertEqual(
            row['optimizer_recipe_repair_max_transfer_delta_uL'],
            2.0
        )
        self.assertEqual(row['reagent_a_original_concentration'], 0.03)
        self.assertEqual(row['reagent_a_repaired_concentration'], 0.05)
        self.assertEqual(row['reagent_a_original_normalized'], 0.03)
        self.assertEqual(row['reagent_a_repaired_normalized'], 0.05)
        self.assertEqual(row['reagent_a_original_transfer_uL'], 3.0)
        self.assertEqual(row['reagent_a_repaired_transfer_uL'], 5.0)
        self.assertTrue(bool(row['reagent_a_true_zero_adjusted']))
        self.assertTrue(bool(row['reagent_a_true_zero_valid']))
        self.assertEqual(row['water_volume_uL'], 85.0)
        self.assertTrue(bool(row['variable_transfers_executable']))
        self.assertTrue(bool(row['volume_feasible']))
        self.assertEqual(
            json.loads(row['optimizer_volume_balance'])['water_volume'],
            80.0
        )
        self.assertEqual(
            json.loads(
                row['selected_controller_volume_balances']
            )[0]['water_volume'],
            87.0
        )
        self.assertEqual(
            json.loads(
                row['executed_controller_volume_balances']
            )[0]['water_volume'],
            85.0
        )
        self.assertEqual(json.loads(row['mask_results']), [])

    def test_invalid_optimizer_proposal_shape_and_bounds_fail_before_export(self):
        invalid_proposals = (
            np.empty((0, 1), dtype=float),
            np.array([[0.1], [0.2]], dtype=float),
            np.array([[1.1]], dtype=float),
            np.array([[float('nan')]], dtype=float)
        )

        for proposal in invalid_proposals:
            controller, model, exports = self._build_controller_and_model()

            with self.subTest(proposal=proposal.tolist()):
                with self.assertRaises(ValueError):
                    controller._prepare_auto_optimizer_recipe_for_execution(
                        model,
                        proposal,
                        'invalid_batch'
                    )

                self.assertEqual(exports, [])

    def test_run_prepares_and_validates_proposal_before_any_robot_sample_work(self):
        run_method = _get_auto_controller_method_node(
            '_run_auto_optimizer_batches'
        )
        call_lines = {}

        for node in ast.walk(run_method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {
                    '_prepare_auto_optimizer_recipe_for_execution',
                    'duplicate_list_elements',
                    '_generate_wellname',
                    '_create_samples'
                }
            ):
                call_lines.setdefault(node.func.attr, []).append(node.lineno)

        iterative_prepare_line = call_lines[
            '_prepare_auto_optimizer_recipe_for_execution'
        ][0]
        iterative_duplicate_line = call_lines['duplicate_list_elements'][0]
        iterative_well_line = call_lines['_generate_wellname'][0]
        iterative_create_line = call_lines['_create_samples'][0]

        self.assertLess(iterative_prepare_line, iterative_duplicate_line)
        self.assertLess(iterative_prepare_line, iterative_well_line)
        self.assertLess(iterative_prepare_line, iterative_create_line)

        helper_method = _get_auto_controller_method_node(
            '_prepare_auto_optimizer_recipe_for_execution'
        )
        export_line = next(
            node.lineno
            for node in ast.walk(helper_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_export_auto_batch_recipe_design'
        )
        raise_line = next(
            node.lineno
            for node in ast.walk(helper_method)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == 'RuntimeError'
        )
        self.assertLess(export_line, raise_line)


class CombinedAcquisitionWorkflowTests(unittest.TestCase):
    '''Exercises every mode through feasibility, selection, and audit output.'''

    class FakeMask(list):
        def tolist(self):
            return list(self)

    @classmethod
    def setUpClass(cls):
        cls.WorkflowModel = _load_optimization_model_methods([
            '_validate_predictive_standard_deviation_nm',
            '_standard_normal_pdf',
            '_standard_normal_cdf',
            '_calculate_target_error_expected_improvement_nm',
            '_calculate_acquisition_score',
            '_masked_acquisition_objective',
            'getNextReaction'
        ])

    def _build_workflow_model(self, acquisition_mode):
        model = self.WorkflowModel()
        model.target_value = 625.0
        model.acquisition_mode = acquisition_mode
        model.balanced_exploration_weight = 1.0
        model.incumbent_target_error_nm = (
            10.0
            if acquisition_mode == 'target_ei'
            else None
        )

        # These candidates deliberately make each mode prefer a different
        # scientifically meaningful tradeoff. Candidate 0.5 would dominate
        # uncertainty-driven selection, but is physically infeasible and must
        # be rejected before GP scoring.
        model.synthetic_distributions = {
            0.1: (625.0, 10.0),
            0.2: (630.0, 20.0),
            0.3: (650.0, 30.0),
            0.4: (628.0, 1.0),
            0.5: (625.0, 100.0)
        }

        def candidate_id(candidate):
            if isinstance(candidate, (list, tuple)):
                return float(candidate[0])
            return float(candidate)

        def volume_balance(candidate):
            candidate = candidate_id(candidate)
            is_feasible = candidate != 0.5

            return {
                'volume_feasible': is_feasible,
                'water_volume': 20.0 if is_feasible else -5.0,
                'volume_does_not_overflow': is_feasible,
                'water_transfer_executable': is_feasible,
                'fixed_volume_total': 10.0,
                'variable_volume_total': (
                    70.0 if is_feasible else 95.0
                )
            }

        model._expand_masked_candidate_to_full_recipe = (
            lambda x_active, mask: candidate_id(x_active)
        )
        model._get_candidate_volume_balance = volume_balance
        model._predict_lambda_max_nm = lambda candidate: (
            model.synthetic_distributions[candidate_id(candidate)][0]
        )
        model.predict_lambda_distribution_nm = lambda candidate: (
            model.synthetic_distributions[candidate_id(candidate)]
        )

        def optimize_across_synthetic_candidates():
            candidate_scores = {
                candidate: model._masked_acquisition_objective(
                    [candidate],
                    [1]
                )
                for candidate in model.synthetic_distributions
            }
            selected_candidate = min(
                candidate_scores,
                key=candidate_scores.get
            )

            model.synthetic_candidate_scores = candidate_scores
            model.last_selected_mask = self.FakeMask([1])
            model.last_optimizer_volume_balance = volume_balance(
                selected_candidate
            )

            return [selected_candidate]

        model._optimize_acquisition_with_masks = (
            optimize_across_synthetic_candidates
        )

        return model

    def test_all_modes_select_expected_feasible_candidate_and_audit_it(self):
        expected_selection = {
            'exploit': 0.1,
            'explore': 0.3,
            'balanced': 0.2,
            'target_ei': 0.4
        }
        expected_target_error_nm = {
            'exploit': 0.0,
            'explore': 25.0,
            'balanced': 5.0,
            'target_ei': 3.0
        }

        for acquisition_mode, expected_candidate in (
            expected_selection.items()
        ):
            with self.subTest(acquisition_mode=acquisition_mode):
                model = self._build_workflow_model(acquisition_mode)
                terminal_output = io.StringIO()

                with redirect_stdout(terminal_output):
                    selected = model.getNextReaction()

                self.assertEqual(selected, [[expected_candidate]])
                self.assertEqual(
                    model.last_optimizer_acquisition_mode,
                    acquisition_mode
                )
                self.assertEqual(
                    model.last_optimizer_predicted_target_error_nm,
                    expected_target_error_nm[acquisition_mode]
                )
                self.assertEqual(model.last_selected_mask.tolist(), [1])
                self.assertGreater(
                    model.synthetic_candidate_scores[0.5],
                    1e11
                )
                self.assertIn(
                    f'mode={acquisition_mode}',
                    terminal_output.getvalue()
                )
                self.assertIn(
                    'selected_mask=[1]',
                    terminal_output.getvalue()
                )

                if acquisition_mode == 'target_ei':
                    self.assertEqual(
                        model.last_optimizer_incumbent_target_error_nm,
                        10.0
                    )
                else:
                    self.assertIsNone(
                        model.last_optimizer_incumbent_target_error_nm
                    )


class ExactMaskAndControllerIntegrationTests(unittest.TestCase):
    '''Runs current optimizer and controller source together without hardware.'''

    @classmethod
    def setUpClass(cls):
        cls.Model = _load_optimization_model_methods(
            [
                '_validate_predictive_standard_deviation_nm',
                '_standard_normal_pdf',
                '_standard_normal_cdf',
                '_calculate_target_error_expected_improvement_nm',
                '_calculate_acquisition_score',
                '_get_portfolio_nearest_distance',
                '_masked_acquisition_objective',
                '_optimize_single_mask',
                '_optimize_acquisition_with_masks',
                'getNextReaction',
                'getNextPortfolio'
            ],
            extra_namespace={
                'copy': copy,
                'minimize': _deterministic_minimize
            }
        )
        cls.Controller = _load_auto_controller_methods([
            '_safe_float_or_none',
            '_serialize_auto_audit_value',
            '_format_mask_for_report',
            '_get_active_variable_reagents_from_mask',
            '_summarize_duplicate_lambda_values',
            '_get_auto_replicate_outlier_threshold_nm',
            '_get_auto_replicate_sd_tolerance_nm',
            '_classify_auto_spectral_observations',
            '_run_lambda_replicate_qc',
            '_get_auto_model_training_decision_from_replicate_qc',
            '_get_auto_target_eligibility_decision',
            '_summarize_auto_scan_quality',
            '_build_auto_qc_model_training_data',
            '_append_auto_model_performance_rows',
            '_get_auto_run_directory_name',
            '_update_auto_model_performance_closest_so_far',
            '_export_auto_model_performance_log',
            'Normalize_Denormalize_Recipes',
            '_build_auto_optimizer_selection_metadata',
            '_prepare_auto_optimizer_recipe_for_execution',
            '_prepare_auto_portfolio_recipes_for_execution',
            '_apply_true_zero_transfer_rule_to_volume',
            '_apply_true_zero_transfer_rule_to_recipes',
            '_get_variable_transfer_volumes_for_recipe',
            '_get_auto_recipe_volume_balance',
            '_validate_auto_recipe_volume_feasibility',
            '_export_auto_batch_recipe_design',
            '_safe_auto_report_get',
            '_safe_auto_report_numeric',
            '_format_auto_report_value',
            '_count_auto_report_status',
            '_get_auto_plot_relative_path',
            '_resolve_auto_plot_path',
            '_auto_report_file_line',
            '_auto_report_not_applicable_file_line',
            '_summarize_auto_run_status_for_report',
            '_build_auto_run_status_report_lines',
            '_escape_auto_report_markdown_table_value',
            '_format_auto_report_table_value',
            '_format_auto_report_volume_summary',
            '_format_auto_report_optimizer_status',
            '_format_auto_report_replicate_list_value',
            '_get_96_well_plate_order',
            '_summarize_auto_report_physical_well_span',
            '_build_auto_report_responsive_sections',
            '_build_auto_report_full_audit_appendix',
            '_build_padded_auto_report_markdown_table',
            '_auto_report_plot_markdown_if_exists',
            '_write_auto_run_report'
        ])

    def _build_exact_model(self):
        model = self.Model()
        model.acquisition_mode = 'exploit'
        model.target_value = 625.0
        model.balanced_exploration_weight = 1.0
        model.incumbent_target_error_nm = None
        model.min_conc = [0.0, 0.0]
        model.max_conc = [1.0, 1.0]
        model.variable_reagents = ['reagent_a', 'reagent_b']
        model._get_dimension = lambda: 2

        masks = [
            np.array([1, 0], dtype=int),
            np.array([0, 1], dtype=int),
            np.array([1, 1], dtype=int)
        ]
        model._get_reagent_masks_for_current_settings = lambda: masks
        model._get_active_mask_indices = lambda mask: np.where(mask == 1)[0]
        model._get_masked_bounds = lambda mask: [
            (0.0, 1.0)
            for _ in np.where(mask == 1)[0]
        ]

        starting_points = {
            (1, 0): [np.array([0.1])],
            (0, 1): [np.array([0.2])],
            (1, 1): [np.array([0.3, 0.4])]
        }
        model._generate_feasible_masked_starting_points = (
            lambda mask, n_restarts: starting_points[tuple(mask.tolist())]
        )

        def expand_candidate(x_active, mask):
            full = np.zeros(2, dtype=float)
            full[np.where(mask == 1)[0]] = np.asarray(
                x_active,
                dtype=float
            )
            return full

        model._expand_masked_candidate_to_full_recipe = expand_candidate

        def volume_balance(candidate):
            candidate = np.asarray(candidate, dtype=float).reshape(2)
            variable_transfers = {
                'reagent_a': float(candidate[0] * 100.0),
                'reagent_b': float(candidate[1] * 100.0)
            }
            variable_total = float(sum(variable_transfers.values()))
            water_volume = 100.0 - 10.0 - variable_total
            return {
                'total_volume': 100.0,
                'fixed_transfer_volumes': {'fixed': 10.0},
                'fixed_volume_total': 10.0,
                'variable_transfer_volumes': variable_transfers,
                'variable_transfer_executable_by_reagent': {
                    name: (volume == 0.0 or volume >= 5.0)
                    for name, volume in variable_transfers.items()
                },
                'variable_transfers_executable': True,
                'variable_volume_total': variable_total,
                'volume_before_water': 10.0 + variable_total,
                'water_volume': water_volume,
                'volume_does_not_overflow': water_volume >= 0.0,
                'water_transfer_executable': (
                    water_volume == 0.0 or water_volume >= 5.0
                ),
                'volume_feasible': (
                    water_volume == 0.0 or water_volume >= 5.0
                )
            }

        distributions = {
            (0.1, 0.0): (630.0, 3.0),
            (0.0, 0.2): (626.0, 2.0),
            (0.3, 0.4): (640.0, 10.0)
        }
        model._get_candidate_volume_balance = volume_balance
        model.predict_lambda_distribution_nm = lambda candidate: (
            distributions[
                tuple(
                    np.round(
                        np.asarray(candidate, dtype=float).reshape(2),
                        12
                    ).tolist()
                )
            ]
        )
        model._predict_lambda_max_nm = lambda candidate: (
            model.predict_lambda_distribution_nm(candidate)[0]
        )
        return model

    def _build_exact_controller(self):
        controller = self.Controller()
        controller.variable_reagents = ['reagent_a', 'reagent_b']
        controller.min_conc = [0.0, 0.0]
        controller.max_conc = [1.0, 1.0]
        controller.template_meta = {'tot_vol': 100.0}
        controller._get_variable_reagent_stock_conc = lambda name: 1.0
        controller._get_fixed_reagent_volumes = lambda: {'fixed': 10.0}
        controller._export_auto_batch_recipe_design = lambda **kwargs: None
        controller.num_duplicates = 2
        controller.robo_params = {
            'target': 625.0,
            'replicate_outlier_threshold_nm': 50.0,
            'replicate_sd_tolerance_nm': 25.0,
            'initial_data': 2,
            'max_iterations': 4,
            'num_duplicates': 2,
            'allow_true_zero': True,
            'acquisition_mode': 'exploit',
            'balanced_exploration_weight': 1.0
        }
        controller.getModelInfo = lambda: controller.robo_params
        controller.rxn_sheet_name = 'combined_hardware_free_test'
        controller.auto_model_performance_rows = []
        controller.auto_condition_counter = 0
        return controller

    def test_exploit_matches_frozen_stable_oracle_across_complete_masks(self):
        model = self._build_exact_model()

        with redirect_stdout(io.StringIO()):
            selected = model.getNextReaction()

        # Frozen independent oracle from the pre-acquisition exploit behavior:
        # every feasible candidate minimizes (predicted_mean_nm - target_nm)^2.
        expected_scores = {
            (1, 0): (630.0 - 625.0) ** 2,
            (0, 1): (626.0 - 625.0) ** 2,
            (1, 1): (640.0 - 625.0) ** 2
        }

        self.assertEqual(len(model.last_mask_results), 3)
        for result in model.last_mask_results:
            mask_key = tuple(result['mask'].tolist())
            self.assertEqual(result['objective'], expected_scores[mask_key])
            self.assertEqual(
                result['acquisition_score'],
                expected_scores[mask_key]
            )
            self.assertTrue({
                'acquisition_mode',
                'balanced_exploration_weight',
                'incumbent_target_error_nm',
                'normalized_recipe',
                'physical_concentrations',
                'predicted_lambda_mean_nm',
                'predicted_lambda_std_nm',
                'predicted_target_error_nm',
                'volume_balance',
                'is_selected'
            }.issubset(result))
            self.assertIn(
                'fixed_transfer_volumes',
                result['volume_balance']
            )

        self.assertEqual(selected[0].tolist(), [0.0, 0.2])
        self.assertEqual(model.last_selected_mask.tolist(), [0, 1])
        self.assertEqual(
            [result['is_selected'] for result in model.last_mask_results],
            [False, True, False]
        )

    def test_ordered_portfolio_selects_distinct_immutable_candidates(self):
        model = self._build_exact_model()
        model.portfolio_min_distance = 0.10

        with redirect_stdout(io.StringIO()):
            records = model.getNextPortfolio(
                acquisition_modes=['exploit', 'explore', 'balanced'],
                portfolio_min_distance=0.10
            )

        self.assertEqual(
            [record['acquisition_mode'] for record in records],
            ['exploit', 'explore', 'balanced']
        )
        self.assertEqual(
            [record['portfolio_selection_index'] for record in records],
            [0, 1, 2]
        )
        self.assertIsNone(records[0]['portfolio_nearest_distance'])
        self.assertGreaterEqual(
            records[1]['portfolio_nearest_distance'],
            0.10
        )
        self.assertGreaterEqual(
            records[2]['portfolio_nearest_distance'],
            0.10
        )
        self.assertEqual(model.acquisition_mode, 'exploit')
        self.assertEqual(
            [record['normalized_recipe'].tolist() for record in records],
            [[0.0, 0.2], [0.3, 0.4], [0.1, 0.0]]
        )

        # Later mutation of live optimizer state cannot rewrite a captured
        # portfolio selection record.
        model.last_mask_results[0]['objective'] = -999.0
        self.assertNotEqual(
            records[-1]['mask_results'][0]['objective'],
            -999.0
        )

    def test_target_ei_portfolio_fails_without_valid_incumbent(self):
        model = self._build_exact_model()
        model.incumbent_target_error_nm = None

        with self.assertRaisesRegex(
            ValueError,
            'explicitly requested'
        ):
            model.getNextPortfolio(
                acquisition_modes=['exploit', 'target_ei'],
                portfolio_min_distance=0.05
            )

    def test_portfolio_controller_handoff_preserves_condition_groups(self):
        model = self._build_exact_model()
        controller = self._build_exact_controller()

        with redirect_stdout(io.StringIO()):
            records = model.getNextPortfolio(
                acquisition_modes=['exploit', 'explore', 'balanced'],
                portfolio_min_distance=0.10
            )
            prepared_recipes, metadata_records = (
                controller._prepare_auto_portfolio_recipes_for_execution(
                    model=model,
                    selection_records=records,
                    batch_label='portfolio_batch'
                )
            )
            controller._append_auto_model_performance_rows(
                unique_recipes=prepared_recipes,
                lambda_max_values=[624.0, 626.0, 620.0, 630.0, 623.0, 627.0],
                condition_type='optimizer_selected',
                batch_number=1,
                prediction_metadata=metadata_records
            )

        self.assertEqual(prepared_recipes.shape, (3, 2))
        self.assertEqual(
            [row['acquisition_mode'] for row in controller.auto_model_performance_rows],
            ['exploit', 'explore', 'balanced']
        )
        self.assertEqual(
            [row['portfolio_selection_index'] for row in controller.auto_model_performance_rows],
            [0, 1, 2]
        )
        self.assertEqual(
            [row['actual_lambda_values_raw_nm'] for row in controller.auto_model_performance_rows],
            [[624.0, 626.0], [620.0, 630.0], [623.0, 627.0]]
        )

    def test_every_mode_uses_exact_single_mask_and_cross_mask_audit_path(self):
        expected_recipes = {
            'exploit': [0.0, 0.2],
            'explore': [0.3, 0.4],
            'balanced': [0.0, 0.2],
            'target_ei': [0.0, 0.2]
        }

        for acquisition_mode, expected_recipe in expected_recipes.items():
            with self.subTest(acquisition_mode=acquisition_mode):
                model = self._build_exact_model()
                model.acquisition_mode = acquisition_mode

                if acquisition_mode == 'target_ei':
                    model.incumbent_target_error_nm = 10.0

                with redirect_stdout(io.StringIO()):
                    selected = model.getNextReaction()

                self.assertEqual(selected[0].tolist(), expected_recipe)
                selected_results = [
                    result
                    for result in model.last_mask_results
                    if result['is_selected']
                ]
                self.assertEqual(len(selected_results), 1)
                self.assertEqual(
                    selected_results[0]['objective'],
                    min(
                        result['objective']
                        for result in model.last_mask_results
                    )
                )

                for result in model.last_mask_results:
                    self.assertEqual(
                        result['acquisition_mode'],
                        acquisition_mode
                    )
                    self.assertEqual(
                        result['objective'],
                        result['acquisition_score']
                    )
                    self.assertEqual(
                        result['balanced_exploration_weight'],
                        1.0 if acquisition_mode == 'balanced' else None
                    )
                    self.assertEqual(
                        result['incumbent_target_error_nm'],
                        10.0 if acquisition_mode == 'target_ei' else None
                    )
                    self.assertTrue(
                        result['volume_balance']['volume_feasible']
                    )

    def test_infeasible_high_uncertainty_mask_is_not_scored_or_selected(self):
        model = self._build_exact_model()
        model.acquisition_mode = 'explore'
        starting_points = {
            (1, 0): [np.array([0.1])],
            (0, 1): [np.array([0.2])],
            # This candidate overflows by 90 uL. If GP-scored, its synthetic
            # SD would dominate explore mode; feasibility must stop that call.
            (1, 1): [np.array([0.9, 0.9])]
        }
        model._generate_feasible_masked_starting_points = (
            lambda mask, n_restarts: starting_points[tuple(mask.tolist())]
        )

        prediction_calls = []
        distributions = {
            (0.1, 0.0): (630.0, 3.0),
            (0.0, 0.2): (626.0, 2.0),
            (0.9, 0.9): (625.0, 1000.0)
        }

        def predict(candidate):
            candidate_key = tuple(
                np.round(
                    np.asarray(candidate, dtype=float).reshape(2),
                    12
                ).tolist()
            )
            prediction_calls.append(candidate_key)
            return distributions[candidate_key]

        model.predict_lambda_distribution_nm = predict
        model._predict_lambda_max_nm = lambda candidate: predict(candidate)[0]

        with redirect_stdout(io.StringIO()):
            selected = model.getNextReaction()

        self.assertEqual(selected[0].tolist(), [0.1, 0.0])
        self.assertNotIn((0.9, 0.9), prediction_calls)

        infeasible_result = next(
            result for result in model.last_mask_results
            if result['mask'].tolist() == [1, 1]
        )
        self.assertFalse(infeasible_result['is_selected'])
        self.assertFalse(
            infeasible_result['volume_balance']['volume_feasible']
        )
        self.assertGreater(infeasible_result['objective'], 1e12)
        self.assertIsNone(infeasible_result['acquisition_score'])
        self.assertIsNone(infeasible_result['predicted_lambda_mean_nm'])
        self.assertEqual(
            infeasible_result['physical_concentrations'],
            {'reagent_a': 0.9, 'reagent_b': 0.9}
        )

    def test_single_mask_ranks_the_exact_candidate_after_bound_clipping(self):
        model = self._build_exact_model()
        model._get_masked_bounds = lambda mask: [(0.0, 0.8)]
        model._generate_feasible_masked_starting_points = (
            lambda mask, n_restarts: [np.array([0.1])]
        )
        model.predict_lambda_distribution_nm = lambda candidate: (
            625.0
            + 10.0
            * float(np.asarray(candidate, dtype=float).reshape(2)[0]),
            1.0
        )
        model._predict_lambda_max_nm = lambda candidate: (
            model.predict_lambda_distribution_nm(candidate)[0]
        )

        def out_of_bounds_minimize(fun, x0, bounds, method):
            return SimpleNamespace(
                # This stale value belongs to x0 and must not control ranking.
                fun=float(fun(np.asarray(x0, dtype=float))),
                x=np.array([0.9], dtype=float),
                success=True,
                message='synthetic slightly out-of-bounds result'
            )

        method_globals = self.Model._optimize_single_mask.__globals__
        original_minimize = method_globals['minimize']
        method_globals['minimize'] = out_of_bounds_minimize

        try:
            result = model._optimize_single_mask(
                np.array([1, 0], dtype=int),
                n_restarts=1
            )
        finally:
            method_globals['minimize'] = original_minimize

        self.assertEqual(result['normalized_recipe'].tolist(), [0.8, 0.0])
        self.assertEqual(result['predicted_lambda_mean_nm'], 633.0)
        self.assertEqual(result['acquisition_score'], 64.0)
        self.assertEqual(result['objective'], 64.0)
        self.assertTrue(result['volume_balance']['volume_feasible'])

    def test_single_mask_recovers_failed_slsqp_boundary_status(self):
        model = self._build_exact_model()
        model._get_masked_bounds = lambda mask: [(0.0, 0.8)]
        model._generate_feasible_masked_starting_points = (
            lambda mask, n_restarts: [np.array([0.8])]
        )
        model.predict_lambda_distribution_nm = lambda candidate: (
            625.0
            + float(np.asarray(candidate, dtype=float).reshape(2)[0]),
            1.0
        )
        model._predict_lambda_max_nm = lambda candidate: (
            model.predict_lambda_distribution_nm(candidate)[0]
        )

        methods_called = []

        def failed_then_recovered_minimize(fun, x0, bounds, method):
            methods_called.append(method)
            return SimpleNamespace(
                fun=float(fun(np.asarray(x0, dtype=float))),
                x=np.asarray(x0, dtype=float).copy(),
                success=(method == 'L-BFGS-B'),
                status=(0 if method == 'L-BFGS-B' else 4),
                message=(
                    'converged'
                    if method == 'L-BFGS-B'
                    else 'Inequality constraints incompatible'
                )
            )

        method_globals = self.Model._optimize_single_mask.__globals__
        original_minimize = method_globals['minimize']
        method_globals['minimize'] = failed_then_recovered_minimize

        try:
            result = model._optimize_single_mask(
                np.array([1, 0], dtype=int),
                n_restarts=1
            )
        finally:
            method_globals['minimize'] = original_minimize

        self.assertEqual(methods_called, ['SLSQP', 'L-BFGS-B'])
        self.assertTrue(result['success'])
        self.assertEqual(result['optimizer_method'], 'L-BFGS-B recovery')
        self.assertEqual(result['optimizer_status'], 0)
        self.assertEqual(result['normalized_recipe'].tolist(), [0.8, 0.0])
        self.assertTrue(result['volume_balance']['volume_feasible'])

    def test_optimizer_selection_survives_controller_handoff_and_csv_export(self):
        model = self._build_exact_model()
        controller = self._build_exact_controller()

        with redirect_stdout(io.StringIO()):
            selected = model.getNextReaction()
            executed, metadata = (
                controller._prepare_auto_optimizer_recipe_for_execution(
                    model=model,
                    normalized_recipes=selected,
                    batch_label='batch_1'
                )
            )
            controller._append_auto_model_performance_rows(
                unique_recipes=executed,
                lambda_max_values=[624.0, 626.0],
                condition_type='optimizer_selected',
                batch_number=1,
                prediction_metadata=metadata
            )

        row = controller.auto_model_performance_rows[0]
        self.assertEqual(row['acquisition_mode'], 'exploit')
        self.assertEqual(row['acquisition_score'], 1.0)
        self.assertEqual(
            json.loads(row['selected_normalized_recipe']),
            [[0.0, 0.2]]
        )
        self.assertEqual(
            json.loads(row['executed_normalized_recipe']),
            [[0.0, 0.2]]
        )
        self.assertFalse(row['optimizer_recipe_repaired'])
        self.assertEqual(row['mask_result_count'], 3)
        persisted_mask_results = json.loads(row['mask_results'])
        self.assertEqual(len(persisted_mask_results), 3)
        expected_mask_audit = {
            (1, 0): {
                'score': 25.0,
                'mean': 630.0,
                'std': 3.0,
                'target_error': 5.0,
                'selected': False
            },
            (0, 1): {
                'score': 1.0,
                'mean': 626.0,
                'std': 2.0,
                'target_error': 1.0,
                'selected': True
            },
            (1, 1): {
                'score': 225.0,
                'mean': 640.0,
                'std': 10.0,
                'target_error': 15.0,
                'selected': False
            }
        }
        for mask_result in persisted_mask_results:
            expected = expected_mask_audit[tuple(mask_result['mask'])]
            self.assertEqual(mask_result['acquisition_score'], expected['score'])
            self.assertEqual(
                mask_result['predicted_lambda_mean_nm'],
                expected['mean']
            )
            self.assertEqual(
                mask_result['predicted_lambda_std_nm'],
                expected['std']
            )
            self.assertEqual(
                mask_result['predicted_target_error_nm'],
                expected['target_error']
            )
            self.assertEqual(mask_result['is_selected'], expected['selected'])
            self.assertIsNone(
                mask_result['balanced_exploration_weight']
            )
        self.assertTrue(row['eligible_for_target_incumbent'])

        with TemporaryDirectory() as temp_directory:
            controller.out_path = temp_directory
            os.makedirs(os.path.join(temp_directory, 'pr_data'))
            controller.plot_path = os.path.join(temp_directory, 'Plots')
            os.makedirs(controller.plot_path)

            with redirect_stdout(io.StringIO()):
                csv_path = controller._export_auto_model_performance_log()
                report_path = controller._write_auto_run_report()

            persisted = pd.read_csv(csv_path)
            report_text = Path(report_path).read_text()

        self.assertEqual(persisted.loc[0, 'acquisition_mode'], 'exploit')
        self.assertEqual(persisted.loc[0, 'acquisition_score'], 1.0)
        self.assertEqual(
            json.loads(persisted.loc[0, 'selected_normalized_recipe']),
            [[0.0, 0.2]]
        )
        self.assertEqual(
            json.loads(
                persisted.loc[0, 'executed_controller_volume_balance']
            )['water_volume'],
            70.0
        )
        self.assertIn('## Acquisition Audit Trail', report_text)
        self.assertIn('### Condition 0 — Optimizer Batch 1', report_text)
        self.assertIn('## Full Audit Appendix', report_text)
        self.assertGreater(
            report_text.index('## Full Audit Appendix'),
            report_text.index('## Conclusion')
        )
        self.assertIn('<details>', report_text)
        self.assertIn('"selected_normalized_recipe": [', report_text)
        self.assertIn('"mask_results": [', report_text)
        self.assertIn(
            'Full selected/executed recipe arrays remain in '
            '`auto_model_performance_log.csv`',
            report_text
        )
        self.assertIn(
            'fixed=10 uL; variable=20 uL; water=70 uL; total=100 uL; '
            'feasible=True',
            report_text
        )
        self.assertIn('Optimizer result:', report_text)
        self.assertIn(
            'not applicable: 2-variable Auto run',
            report_text
        )
        self.assertIn('False', report_text)
        self.assertIn(
            'validated target hit for incumbent and stopping decisions',
            report_text
        )

    def test_success_recipe_design_csv_persists_exact_selection_provenance(self):
        model = self._build_exact_model()
        controller = self._build_exact_controller()
        del controller.__dict__['_export_auto_batch_recipe_design']
        controller.batch_num = 1

        with TemporaryDirectory() as temp_directory:
            controller.debug_path = temp_directory

            with redirect_stdout(io.StringIO()):
                selected = model.getNextReaction()
                controller._prepare_auto_optimizer_recipe_for_execution(
                    model=model,
                    normalized_recipes=selected,
                    batch_label='batch_1'
                )

            export_path = Path(temp_directory) / (
                'auto_recipe_design/auto_recipe_design_batch_1.csv'
            )
            persisted = pd.read_csv(export_path)

        row = persisted.iloc[0]
        self.assertEqual(row['acquisition_mode'], 'exploit')
        self.assertEqual(row['acquisition_score'], 1.0)
        self.assertEqual(row['predicted_lambda_mean_nm'], 626.0)
        self.assertEqual(row['predicted_lambda_std_nm'], 2.0)
        self.assertEqual(row['predicted_target_error_nm'], 1.0)
        self.assertEqual(json.loads(row['selected_mask']), [0, 1])
        self.assertFalse(bool(row['optimizer_recipe_repaired']))
        self.assertEqual(row['mask_result_count'], 3)
        self.assertEqual(row['feasible_mask_result_count'], 3)
        self.assertEqual(row['reagent_a_original_concentration'], 0.0)
        self.assertEqual(row['reagent_a_repaired_concentration'], 0.0)
        self.assertEqual(row['reagent_b_original_concentration'], 0.2)
        self.assertEqual(row['reagent_b_repaired_concentration'], 0.2)
        self.assertEqual(row['reagent_a_repaired_transfer_uL'], 0.0)
        self.assertEqual(row['reagent_b_repaired_transfer_uL'], 20.0)
        self.assertEqual(row['water_volume_uL'], 70.0)
        self.assertTrue(bool(row['variable_transfers_executable']))
        self.assertTrue(bool(row['volume_feasible']))
        self.assertEqual(
            json.loads(row['optimizer_volume_balance'])['water_volume'],
            70.0
        )
        self.assertEqual(
            json.loads(
                row['selected_controller_volume_balances']
            )[0]['water_volume'],
            70.0
        )
        self.assertEqual(
            json.loads(
                row['executed_controller_volume_balances']
            )[0]['water_volume'],
            70.0
        )

        mask_results = json.loads(row['mask_results'])
        self.assertEqual(
            [result['mask'] for result in mask_results],
            [[1, 0], [0, 1], [1, 1]]
        )
        self.assertEqual(
            [result['is_selected'] for result in mask_results],
            [False, True, False]
        )
        self.assertEqual(
            [result['predicted_lambda_mean_nm'] for result in mask_results],
            [630.0, 626.0, 640.0]
        )
        self.assertEqual(
            [result['predicted_lambda_std_nm'] for result in mask_results],
            [3.0, 2.0, 10.0]
        )

    def test_balanced_mode_weight_and_score_survive_controller_persistence(self):
        model = self._build_exact_model()
        model.acquisition_mode = 'balanced'
        controller = self._build_exact_controller()
        controller.robo_params['acquisition_mode'] = 'balanced'

        with redirect_stdout(io.StringIO()):
            selected = model.getNextReaction()
            executed, metadata = (
                controller._prepare_auto_optimizer_recipe_for_execution(
                    model=model,
                    normalized_recipes=selected,
                    batch_label='balanced_batch'
                )
            )
            controller._append_auto_model_performance_rows(
                unique_recipes=executed,
                lambda_max_values=[624.0, 626.0],
                condition_type='optimizer_selected',
                batch_number=1,
                prediction_metadata=metadata
            )

        row = controller.auto_model_performance_rows[0]
        self.assertEqual(row['acquisition_mode'], 'balanced')
        self.assertEqual(row['acquisition_score'], -1.0)
        self.assertEqual(row['balanced_exploration_weight'], 1.0)
        self.assertEqual(row['predicted_lambda_mean_nm'], 626.0)
        self.assertEqual(row['predicted_lambda_std_nm'], 2.0)
        self.assertEqual(row['predicted_target_error_nm'], 1.0)
        self.assertEqual(row['selected_mask'], '[0, 1]')

    def test_scan_quality_metadata_warns_without_changing_qc_training(self):
        controller = self._build_exact_controller()
        scan_quality = controller._summarize_auto_scan_quality(
            [300.0, 625.0],
            [0.01, 0.42]
        )

        self.assertEqual(
            scan_quality,
            [
                {
                    'peak_wavelength_nm': 300.0,
                    'blank_corrected_peak_absorbance': 0.01,
                    'peak_at_scan_boundary': True
                },
                {
                    'peak_wavelength_nm': 625.0,
                    'blank_corrected_peak_absorbance': 0.42,
                    'peak_at_scan_boundary': False
                }
            ]
        )

        with redirect_stdout(io.StringIO()):
            controller._append_auto_model_performance_rows(
                unique_recipes=np.array([[0.2, 0.2]], dtype=float),
                lambda_max_values=[300.0, 625.0],
                condition_type='seed',
                batch_number=0,
                scan_quality_by_replicate=scan_quality
            )

        row = controller.auto_model_performance_rows[0]
        self.assertEqual(row['spectral_quality_status'], 'edge_peak_warning')
        self.assertEqual(row['n_spectral_edge_peaks'], 1)
        self.assertEqual(row['spectral_edge_peak_replicate_indices'], [0])
        self.assertTrue(row['use_for_model_training'])
        self.assertEqual(row['auto_spectral_response_policy'], 'audit_only')
        self.assertEqual(
            row['spectral_observation_types'],
            ['lower_scan_censored', 'interior_peak']
        )
        self.assertEqual(row['n_lambda_model_eligible_replicates'], 2)
        self.assertEqual(
            row['spectral_lambda_excluded_replicate_indices'],
            []
        )
        self.assertEqual(
            row['actual_lambda_rep_1_blank_corrected_peak_absorbance'],
            0.01
        )
        self.assertTrue(row['actual_lambda_rep_1_peak_at_scan_boundary'])
        self.assertFalse(row['actual_lambda_rep_2_peak_at_scan_boundary'])

    def test_boundary_aware_policy_censors_endpoints_from_lambda_qc_training(self):
        controller = self._build_exact_controller()
        controller.num_duplicates = 3
        controller.robo_params['num_duplicates'] = 3
        controller.robo_params['auto_spectral_response_policy'] = (
            'boundary_aware'
        )
        scan_quality = controller._summarize_auto_scan_quality(
            [300.0, 624.0, 626.0],
            [0.01, 0.42, 0.41]
        )

        model_recipes, model_values = (
            controller._build_auto_qc_model_training_data(
                np.array([[0.2, 0.2]], dtype=float),
                [300.0, 624.0, 626.0],
                scan_quality_by_replicate=scan_quality
            )
        )

        self.assertEqual(model_recipes.shape, (2, 2))
        np.testing.assert_allclose(model_values, [624.0, 626.0])

        with redirect_stdout(io.StringIO()):
            controller._append_auto_model_performance_rows(
                unique_recipes=np.array([[0.2, 0.2]], dtype=float),
                lambda_max_values=[300.0, 624.0, 626.0],
                condition_type='seed',
                batch_number=0,
                scan_quality_by_replicate=scan_quality
            )

        row = controller.auto_model_performance_rows[0]
        self.assertEqual(row['actual_lambda_values_qc_nm'], [624.0, 626.0])
        self.assertEqual(row['actual_lambda_mean_qc_nm'], 625.0)
        self.assertEqual(row['n_replicates_spectrally_excluded'], 1)
        self.assertEqual(
            row['usable_spectrum_fraction_assessed'],
            2.0 / 3.0
        )
        self.assertEqual(
            row['spectral_lambda_excluded_replicate_indices'],
            [0]
        )
        self.assertTrue(row['use_for_model_training'])
        self.assertTrue(row['eligible_for_target_incumbent'])
        self.assertTrue(row['eligible_for_target_stop'])
        self.assertFalse(
            row['actual_lambda_rep_1_lambda_model_eligible']
        )
        self.assertEqual(
            row['actual_lambda_rep_1_spectral_observation_type'],
            'lower_scan_censored'
        )

    def test_boundary_aware_all_censored_condition_returns_no_lambda_rows(self):
        controller = self._build_exact_controller()
        controller.num_duplicates = 3
        controller.robo_params['num_duplicates'] = 3
        controller.robo_params['auto_spectral_response_policy'] = (
            'boundary_aware'
        )
        scan_quality = controller._summarize_auto_scan_quality(
            [300.0, 1000.0, 300.0],
            [0.01, 0.01, 0.02]
        )

        model_recipes, model_values = (
            controller._build_auto_qc_model_training_data(
                np.array([[0.2, 0.2]], dtype=float),
                [300.0, 1000.0, 300.0],
                scan_quality_by_replicate=scan_quality
            )
        )

        self.assertEqual(model_recipes.shape, (0, 2))
        self.assertEqual(model_values.shape, (0,))

        with redirect_stdout(io.StringIO()):
            controller._append_auto_model_performance_rows(
                unique_recipes=np.array([[0.2, 0.2]], dtype=float),
                lambda_max_values=[300.0, 1000.0, 300.0],
                condition_type='seed',
                batch_number=0,
                scan_quality_by_replicate=scan_quality
            )

        row = controller.auto_model_performance_rows[0]
        self.assertIsNone(row['actual_lambda_mean_qc_nm'])
        self.assertFalse(row['use_for_model_training'])
        self.assertFalse(row['eligible_for_target_incumbent'])
        self.assertFalse(row['eligible_for_target_stop'])

    def test_current_controller_completion_marker_updates_report_status(self):
        controller = self._build_exact_controller()

        with TemporaryDirectory() as temp_directory:
            controller.out_path = temp_directory
            controller.plot_path = os.path.join(temp_directory, 'Plots')
            os.makedirs(os.path.join(temp_directory, 'Debug'))
            os.makedirs(controller.plot_path)

            terminal_path = os.path.join(
                temp_directory,
                'Debug',
                'terminal_output.txt'
            )
            Path(terminal_path).write_text(
                'Exit due to max_iters\n'
                '<<controller>> Auto run completed; condition-level results, '
                'recipe audits, and configured output artifacts were exported.\n'
                '<<controller>> shutting down\n'
            )

            summary = controller._summarize_auto_run_status_for_report()

        self.assertEqual(
            summary['completion_status'],
            'Completed controller workflow'
        )
        self.assertEqual(summary['exit_reason'], 'max_iters reached')
        self.assertTrue(summary['controller_completion_marker_found'])
        self.assertTrue(summary['success_marker_found'])
        self.assertFalse(summary['pre_success_traceback_found'])

    def test_target_hit_exit_reason_precedes_completion_marker(self):
        controller = self._build_exact_controller()

        with TemporaryDirectory() as temp_directory:
            controller.out_path = temp_directory
            controller.plot_path = os.path.join(temp_directory, 'Plots')
            os.makedirs(os.path.join(temp_directory, 'Debug'))
            os.makedirs(controller.plot_path)

            terminal_path = os.path.join(
                temp_directory,
                'Debug',
                'terminal_output.txt'
            )
            Path(terminal_path).write_text(
                '<<controller>> Exit due to validated condition-level target '
                'hit\n'
                '<<controller>> Auto run completed; condition-level results, '
                'recipe audits, and configured output artifacts were exported.\n'
            )

            summary = controller._summarize_auto_run_status_for_report()

        self.assertEqual(
            summary['exit_reason'],
            'validated condition-level target hit'
        )

    def test_report_explains_header_tolerances_portfolio_and_early_stop_wells(self):
        controller = self._build_exact_controller()
        controller.num_duplicates = 3
        controller.robo_params.update({
            'num_duplicates': 3,
            'target_tolerance_nm': 4.0,
            'replicate_sd_tolerance_nm': 5.0,
            'acquisition_mode': 'exploit',
            'acquisition_modes': ['exploit', 'explore', 'balanced'],
            'using_acquisition_portfolio': True,
            'portfolio_min_distance': 0.05,
            'auto_source_volume_check': 'required',
            'auto_source_reserve_volume_uL': 100.0
        })
        controller._cached_reader_locs = {
            'sample_a': SimpleNamespace(loc='A6'),
            'sample_b': SimpleNamespace(loc='B6'),
            'sample_c': SimpleNamespace(loc='C6')
        }

        controller._append_auto_model_performance_rows(
            unique_recipes=np.array([[0.2, 0.2]], dtype=float),
            lambda_max_values=[622.0, 624.0, 625.0],
            condition_type='optimizer_selected',
            batch_number=1,
            prediction_metadata={'acquisition_mode': 'explore'},
            replicate_wellnames=['sample_a', 'sample_b', 'sample_c']
        )

        with TemporaryDirectory() as temp_directory:
            controller.out_path = temp_directory
            controller.plot_path = os.path.join(temp_directory, 'Plots')
            os.makedirs(os.path.join(temp_directory, 'Debug'))
            os.makedirs(controller.plot_path)
            Path(
                os.path.join(temp_directory, 'Debug', 'terminal_output.txt')
            ).write_text(
                '<<controller>> Exit due to validated condition-level target '
                'hit\n'
                '<<controller>> Auto run completed; condition-level results, '
                'recipe audits, and configured output artifacts were exported.\n'
            )

            with redirect_stdout(io.StringIO()):
                report_path = controller._write_auto_run_report()

            report_text = Path(report_path).read_text()

        row = controller.auto_model_performance_rows[0]
        self.assertEqual(
            json.loads(row['replicate_well_locations']),
            ['A6', 'B6', 'C6']
        )
        self.assertIn('## Target-Stopping Decision', report_text)
        self.assertIn(
            'stopped early after measurement and QC of optimizer batch 1',
            report_text
        )
        self.assertIn('A6, B6, C6', report_text)
        self.assertIn(
            'Physical plate-well span: A6 through C6 in controller '
            'execution order (3 unique physical wells).',
            report_text
        )
        self.assertIn(
            'Same-plate reuse guidance: 53 sequential unused wells remain '
            'after C6. Set the active plate-reader first usable well to '
            '`D6` for the next experiment, after confirming no other wells '
            'were used.',
            report_text
        )
        self.assertIn('Target tolerance', report_text)
        self.assertIn('4 nm', report_text)
        self.assertIn('Replicate SD tolerance', report_text)
        self.assertIn('5 nm', report_text)
        self.assertIn('## Spectral Observation Handling', report_text)
        self.assertIn('legacy audit-only policy was active', report_text)
        self.assertIn('minimum normalized RMS distance', report_text)
        self.assertIn('Minimizes `−(predicted GP SD)` (nm)', report_text)
        self.assertIn(
            'Each listed mode selected one physically feasible condition',
            report_text
        )

    def test_report_plot_helper_uses_categorized_progress_path(self):
        controller = self._build_exact_controller()

        with TemporaryDirectory() as temp_directory:
            controller.out_path = temp_directory
            controller.plot_path = os.path.join(temp_directory, 'Plots')
            progress_directory = os.path.join(
                controller.plot_path,
                'progress'
            )
            os.makedirs(progress_directory)
            Path(
                os.path.join(progress_directory, 'lambda_progress_final.png')
            ).touch()

            report_lines = controller._auto_report_plot_markdown_if_exists(
                plot_filename='lambda_progress_final.png',
                title='Final λmax Progress Plot'
            )

        self.assertIn(
            '![Final λmax Progress Plot](../Plots/progress/'
            'lambda_progress_final.png)',
            report_lines
        )

    def test_report_generated_file_status_uses_categorized_progress_path(self):
        controller = self._build_exact_controller()
        controller._append_auto_model_performance_rows(
            unique_recipes=np.array([[0.2, 0.2]], dtype=float),
            lambda_max_values=[624.0, 626.0],
            condition_type='seed',
            batch_number=0
        )

        with TemporaryDirectory() as temp_directory:
            controller.out_path = temp_directory
            controller.plot_path = os.path.join(temp_directory, 'Plots')
            progress_directory = os.path.join(
                controller.plot_path,
                'progress'
            )
            os.makedirs(progress_directory)
            Path(
                os.path.join(progress_directory, 'lambda_progress_final.png')
            ).touch()
            Path(
                os.path.join(progress_directory, 'lambda_replicates_final.png')
            ).touch()

            with redirect_stdout(io.StringIO()):
                report_path = controller._write_auto_run_report()

            report_text = Path(report_path).read_text()

        self.assertIn(
            '![Final λmax Progress Plot](../Plots/progress/'
            'lambda_progress_final.png)',
            report_text
        )
        self.assertIn(
            '`Plots/progress/lambda_progress_final.png` (present)',
            report_text
        )
        self.assertIn(
            '`Plots/progress/lambda_replicates_final.png` (present)',
            report_text
        )

    def test_report_does_not_call_single_replicate_target_match_validated(self):
        controller = self._build_exact_controller()
        controller.num_duplicates = 1
        controller.robo_params['num_duplicates'] = 1

        controller._append_auto_model_performance_rows(
            unique_recipes=np.array([[0.1, 0.1]], dtype=float),
            lambda_max_values=[625.0],
            condition_type='seed',
            batch_number=0
        )

        with TemporaryDirectory() as temp_directory:
            controller.out_path = temp_directory
            controller.plot_path = os.path.join(temp_directory, 'Plots')
            os.makedirs(controller.plot_path)

            with redirect_stdout(io.StringIO()):
                report_path = controller._write_auto_run_report()

            report_text = Path(report_path).read_text()

        self.assertIn(
            'did not pass replicate validation and therefore cannot establish',
            report_text
        )
        self.assertNotIn(
            'validated target hit for incumbent and stopping decisions',
            report_text
        )


class CumulativeGpHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.HistoryModel = _load_optimization_model_methods(
            ['update_experiment_data'],
            extra_namespace={'np': np}
        )

    def test_successive_updates_preserve_every_prior_qc_approved_batch(self):
        model = self.HistoryModel()
        gp_update_calls = []
        quit_update_calls = []
        model.gp_model = SimpleNamespace(
            updateModel=lambda **kwargs: gp_update_calls.append(kwargs)
        )
        model.optimizer = SimpleNamespace(
            X=np.array([[0.1], [0.2]], dtype=float),
            Y=np.array([[0.4], [0.5]], dtype=float)
        )
        model.curr_iter = 0
        model._get_dimension = lambda: 1
        model.update_quit = lambda X_new, Y_new: (
            quit_update_calls.append((X_new.copy(), Y_new.copy()))
        )

        first_X_all = np.array([[0.1], [0.2], [0.3]], dtype=float)
        first_Y_all = np.array([[0.4], [0.5], [0.6]], dtype=float)
        second_X_all = np.array(
            [[0.1], [0.2], [0.3], [0.4]],
            dtype=float
        )
        second_Y_all = np.array(
            [[0.4], [0.5], [0.6], [0.7]],
            dtype=float
        )

        with redirect_stdout(io.StringIO()):
            model.update_experiment_data(
                first_X_all,
                first_Y_all,
                first_X_all[-1:],
                first_Y_all[-1:]
            )
            model.update_experiment_data(
                second_X_all,
                second_Y_all,
                second_X_all[-1:],
                second_Y_all[-1:]
            )

        self.assertEqual(len(gp_update_calls), 2)
        self.assertEqual(len(quit_update_calls), 2)
        self.assertEqual(model.curr_iter, 2)
        self.assertEqual(model.optimizer.X.tolist(), second_X_all.tolist())
        self.assertEqual(model.optimizer.Y.tolist(), second_Y_all.tolist())
        self.assertEqual(
            gp_update_calls[1]['X_all'].tolist(),
            second_X_all.tolist()
        )
        self.assertEqual(
            gp_update_calls[1]['Y_all'].tolist(),
            second_Y_all.tolist()
        )
        self.assertEqual(
            gp_update_calls[1]['X_all'][2].tolist(),
            [0.3]
        )

    def test_failed_gp_update_leaves_optimizer_history_and_iteration_atomic(self):
        model = self.HistoryModel()
        original_X = np.array([[0.1], [0.2]], dtype=float)
        original_Y = np.array([[0.4], [0.5]], dtype=float)
        update_quit_calls = []

        def fail_update(**kwargs):
            raise RuntimeError('synthetic GP refit failure')

        model.gp_model = SimpleNamespace(updateModel=fail_update)
        model.optimizer = SimpleNamespace(
            X=original_X.copy(),
            Y=original_Y.copy()
        )
        model.curr_iter = 3
        model.quit = True
        model._get_dimension = lambda: 1
        model.update_quit = lambda X_new, Y_new: update_quit_calls.append(
            (X_new, Y_new)
        )

        X_all = np.array([[0.1], [0.2], [0.3]], dtype=float)
        Y_all = np.array([[0.4], [0.5], [0.6]], dtype=float)

        with self.assertRaisesRegex(
            RuntimeError,
            'synthetic GP refit failure'
        ):
            model.update_experiment_data(
                X_all,
                Y_all,
                X_all[-1:],
                Y_all[-1:]
            )

        np.testing.assert_array_equal(model.optimizer.X, original_X)
        np.testing.assert_array_equal(model.optimizer.Y, original_Y)
        self.assertEqual(model.curr_iter, 3)
        self.assertTrue(model.quit)
        self.assertEqual(update_quit_calls, [])


class ThreeVariableSliceSupportTests(unittest.TestCase):
    '''Hardware-free checks for the read-only 3D GP slice primitives.'''

    @classmethod
    def setUpClass(cls):
        cls.BatchPredictionModel = _load_optimization_model_methods([
            '_get_dimension',
            '_get_variable_transfer_volumes_for_normalized_candidate',
            '_get_candidate_volume_balance',
            'get_candidate_volume_balance_for_plotting',
            'predict_lambda_distribution_nm_batch'
        ])
        cls.SliceController = _load_auto_controller_methods([
            '_get_auto_target_tolerance_nm',
            '_calculate_auto_target_probability',
            '_calculate_auto_joint_target_success_probability'
        ])
        cls.GeneralSliceController = _load_auto_controller_methods([
            '_get_auto_design_bound_value',
            '_get_auto_target_tolerance_nm',
            '_calculate_auto_target_probability',
            '_calculate_auto_joint_target_success_probability',
            '_build_auto_conditional_slice_panel_data'
        ])

    def test_batch_prediction_converts_mean_and_standard_deviation_units(self):
        class FakeGp:
            def predict(self, x_values):
                rows = x_values.shape[0]
                return (
                    np.full((rows, 1), 0.55),
                    np.full((rows, 1), 0.10)
                )

        model = self.BatchPredictionModel()
        model.variable_reagents = ['A', 'B', 'C']
        model.gp_model = FakeGp()

        mean_nm, std_nm = model.predict_lambda_distribution_nm_batch(
            np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]),
            chunk_size=1
        )

        np.testing.assert_allclose(mean_nm, [630.0, 630.0])
        np.testing.assert_allclose(std_nm, [60.0, 60.0])

    def test_target_probability_is_symmetric_and_has_correct_zero_sd_limit(self):
        controller = self.SliceController()
        probability = controller._calculate_auto_target_probability(
            predicted_mean_nm=np.array([620.0, 630.0, 640.0]),
            predicted_std_nm=np.array([0.0, 0.0, 0.0]),
            target_nm=625.0,
            tolerance_nm=5.0
        )

        np.testing.assert_allclose(probability, [1.0, 1.0, 0.0])

        stochastic_probability = controller._calculate_auto_target_probability(
            predicted_mean_nm=np.array([620.0, 630.0]),
            predicted_std_nm=np.array([8.0, 8.0]),
            target_nm=625.0,
            tolerance_nm=10.0
        )
        self.assertAlmostEqual(
            stochastic_probability[0],
            stochastic_probability[1],
            places=12
        )

    def test_joint_target_success_probability_is_bounded_product(self):
        controller = self.SliceController()
        joint_probability = (
            controller._calculate_auto_joint_target_success_probability(
                np.asarray([0.2, 0.75, 1.0]),
                np.asarray([0.5, 0.4, 0.0])
            )
        )
        np.testing.assert_allclose(
            joint_probability,
            np.asarray([0.1, 0.3, 0.0])
        )

    def test_plotting_feasibility_preserves_nonexecutable_true_zero_band(self):
        model = self.BatchPredictionModel()
        model.variable_reagents = ['A']
        model.min_conc = [0.0]
        model.max_conc = [1.0]
        model.total_volume = 100.0
        model.fixed_reagent_volumes = {}
        model._get_variable_reagent_stock_conc = lambda reagent_name: 1.0

        # A 0.03 normalized concentration would transfer 3 uL. It must be
        # displayed as infeasible rather than silently repaired to exact zero.
        balance = model.get_candidate_volume_balance_for_plotting(
            np.array([0.03])
        )

        self.assertFalse(balance['volume_feasible'])
        self.assertFalse(balance['variable_transfers_executable'])

    def test_slice_renderer_keeps_originals_and_adds_2d_style_overlays(self):
        renderer_node = _get_auto_controller_method_node(
            'plot_3D_GPR_orthogonal_slices'
        )
        renderer_source = ast.unparse(renderer_node)

        self.assertIn(
            'gpr_3d_{field_name}_orthogonal_slices_{final_suffix}.png',
            renderer_source
        )
        self.assertIn(
            'gpr_3d_{field_name}_orthogonal_slices_feasibility_',
            renderer_source
        )
        self.assertIn(
            'Water = 5 uL boundary',
            renderer_source
        )
        self.assertIn(
            'Target = {target_nm:.0f} nm',
            renderer_source
        )
        self.assertIn(
            'field_values = value_getter(panel)',
            renderer_source
        )
        self.assertIn('grid_size=100', renderer_source)
        self.assertIn(
            'feasibility_grid_size = max(401, grid_size)',
            renderer_source
        )
        self.assertIn('feasibility_x_physical', renderer_source)
        self.assertNotIn('masked_where', renderer_source)
        self.assertIn(
            'GP probability of meeting the controller stopping criterion',
            renderer_source
        )
        self.assertIn('maximum = {maximum_probability:.3f}', renderer_source)
        self.assertIn('figsize=(18.2, 5.8)', renderer_source)
        self.assertIn('figsize=(18.2, 6.6)', renderer_source)
        self.assertIn('textwrap.wrap(title, width=52)', renderer_source)
        self.assertIn('figsize=(8.4, 7.4 if feasibility_overlay else 6.5)', renderer_source)
        self.assertIn('ncol=1', renderer_source)

    def test_three_d_design_plot_excludes_best_point_from_base_marker(self):
        renderer_node = _get_auto_controller_method_node(
            '_plot_initial_training_design_3d'
        )
        renderer_source = ast.unparse(renderer_node)

        self.assertIn(
            'display_optimizer_mask = optimizer_mask & ~best_mask',
            renderer_source
        )
        self.assertIn(
            'display_seed_mask = seed_mask & ~best_mask',
            renderer_source
        )

    def test_slice_tolerance_uses_the_controller_stop_setting(self):
        controller = self.SliceController()
        controller.robo_params = {'target_tolerance_nm': 7.5}
        self.assertEqual(controller._get_auto_target_tolerance_nm(), 7.5)

    def test_generalized_slice_data_varies_each_pair_and_holds_all_others(self):
        '''Four-variable slices must evaluate full recipes with two holds.'''
        class FakeModel:
            variable_reagents = ['A', 'B', 'C', 'D']

            def predict_lambda_distribution_nm_batch(self, recipes):
                recipes = np.asarray(recipes, dtype=float)
                weights = np.asarray([1.0, 2.0, 4.0, 8.0])
                return (
                    600.0 + 100.0 * np.dot(recipes, weights),
                    np.full(recipes.shape[0], 5.0)
                )

            def get_candidate_volume_balance_for_plotting(self, recipe):
                recipe = np.asarray(recipe, dtype=float)
                transfer_volumes = {
                    reagent_name: float(value * 100.0)
                    for reagent_name, value in zip(
                        self.variable_reagents,
                        recipe
                    )
                }
                water_volume = 200.0 - sum(transfer_volumes.values())
                variable_transfers_executable = all(
                    value == 0.0 or value >= 5.0
                    for value in transfer_volumes.values()
                )
                return {
                    'volume_feasible': (
                        variable_transfers_executable
                        and (water_volume == 0.0 or water_volume >= 5.0)
                    ),
                    'water_volume': water_volume,
                    'variable_transfer_volumes': transfer_volumes
                }

        controller = self.GeneralSliceController()
        controller.variable_reagents = ['A', 'B', 'C', 'D']
        controller.min_conc = [0.0, 0.0, 0.0, 0.0]
        controller.max_conc = [1.0, 1.0, 1.0, 1.0]
        controller.robo_params = {'target_tolerance_nm': 10.0}
        controller.getModelInfo = lambda: {'target': 625.0}
        controller.auto_model_performance_rows = [{
            'A_concentration': 0.1,
            'B_concentration': 0.2,
            'C_concentration': 0.3,
            'D_concentration': 0.4,
            'target_error_nm': 1.0,
            'use_for_model_training': True,
            'eligible_for_target_incumbent': True
        }]

        slice_data = controller._build_auto_conditional_slice_panel_data(
            model=FakeModel(),
            grid_size=21,
            feasibility_grid_size=21
        )

        self.assertEqual(len(slice_data['panel_data']), 6)
        self.assertEqual(
            [(panel['x_index'], panel['y_index'])
             for panel in slice_data['panel_data']],
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        )

        first_panel = slice_data['panel_data'][0]
        self.assertEqual(first_panel['held_indices'], (2, 3))
        self.assertEqual(
            first_panel['held_recipe'],
            {'C': 0.3, 'D': 0.4}
        )
        # The first pixel varies A and B at zero while C and D remain at the
        # selected reference condition: 600 + 100 * (4*.3 + 8*.4).
        self.assertAlmostEqual(first_panel['mean_nm'][0, 0], 1040.0)
        self.assertEqual(first_panel['mean_nm'].shape, (21, 21))
        self.assertEqual(first_panel['feasibility_mask'].shape, (21, 21))
        self.assertTrue(first_panel['feasibility_mask'][0, 0])
        self.assertFalse(first_panel['feasibility_mask'][-1, -1])


class AutoModelCheckpointControllerTests(unittest.TestCase):
    '''Confirms save mode writes model state only for the real Auto model.'''

    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_auto_controller_methods(
            [
                '_auto_model_checkpoint_saving_enabled',
                '_get_auto_model_checkpoint_directory',
                '_get_auto_model_checkpoint_output_root',
                '_list_existing_auto_model_checkpoint_files',
                '_resolve_auto_model_checkpoint_existing_source',
                '_checkpoint_float_values_match',
                '_validate_auto_model_checkpoint_compatibility',
                '_build_auto_model_checkpoint_manifest',
                '_build_auto_model_checkpoint_arrays',
                '_save_auto_model_checkpoint',
                '_get_next_auto_number_after_rows',
                '_restore_imported_auto_model_checkpoint'
            ],
            extra_namespace={
                'datetime': datetime,
                'ModelCheckpointError': ModelCheckpointError,
                'write_model_checkpoint_import_provenance': (
                    write_model_checkpoint_import_provenance
                ),
                'write_model_checkpoint': write_model_checkpoint
            }
        )

    def _controller_and_model(self, checkpoint_directory, eligible=True):
        controller = self.Controller()
        controller.robo_params = {
            'auto_model_checkpoint_mode': 'save',
            'auto_spectral_response_policy': 'audit_only'
        }
        controller.model_checkpoint_path = checkpoint_directory
        controller.out_path = checkpoint_directory
        controller.rxn_sheet_name = 'DEBUGRTG_checkpoint_controller_test'
        controller.variable_reagents = ['reagent_a', 'reagent_b']
        controller.fixed_reagents = ['fixed_reagent']
        # Mirror the keys production actually writes.  An earlier fixture used
        # a 'condition_number' key that no controller path emits, which let a
        # numbering-continuation defect pass review.
        controller.auto_model_performance_rows = [
            {
                'reaction_number': 0,
                'batch_number': 1,
                'executed_in_current_run': True,
                'actual_lambda_mean_qc_nm': 625.0
            }
        ]
        controller.auto_model_checkpoint_paths = []
        controller.batch_num = 1
        controller._get_fixed_reagent_volumes = lambda: {
            'fixed_reagent': 25.0
        }
        controller._get_variable_reagent_stock_conc = lambda _: 10.0

        model = SimpleNamespace(
            _auto_model_checkpoint_eligible=eligible,
            optimizer=SimpleNamespace(
                X=np.asarray([[0.1, 0.2], [0.7, 0.8]]),
                Y=np.asarray([[0.4], [0.6]])
            ),
            usable_spectrum_X=np.asarray([[0.1, 0.2]]),
            usable_spectrum_Y=np.asarray([[1.0]]),
            min_conc=[0.0, 0.0],
            max_conc=[1.0, 1.0],
            total_volume=200.0,
            target_value=625.0,
            acquisition_mode='balanced',
            acquisition_modes=['balanced'],
            portfolio_min_distance=0.05,
            balanced_exploration_weight=1.0,
            curr_iter=1
        )
        return controller, model

    def test_save_mode_exports_rebuildable_seed_batch_and_final_packages(self):
        with TemporaryDirectory() as temporary_directory:
            controller, model = self._controller_and_model(
                temporary_directory
            )

            seed_path = controller._save_auto_model_checkpoint(
                model,
                'after_seed'
            )
            batch_path = controller._save_auto_model_checkpoint(
                model,
                'after_batch'
            )
            final_path = controller._save_auto_model_checkpoint(
                model,
                'final'
            )

            self.assertEqual(len(controller.auto_model_checkpoint_paths), 3)
            self.assertTrue(seed_path.endswith('model_after_seed.zip'))
            self.assertTrue(batch_path.endswith('model_after_batch_001.zip'))
            self.assertTrue(final_path.endswith('model_final.zip'))

            restored = read_model_checkpoint(final_path)
            self.assertEqual(
                restored['manifest']['checkpoint_stage'],
                'final'
            )
            np.testing.assert_allclose(
                restored['model_arrays']['gp_training_X'],
                model.optimizer.X
            )
            np.testing.assert_allclose(
                restored['model_arrays']['gp_training_Y'],
                model.optimizer.Y
            )

    def test_dummy_preflight_model_cannot_create_checkpoint_files(self):
        with TemporaryDirectory() as temporary_directory:
            controller, dummy_model = self._controller_and_model(
                temporary_directory,
                eligible=False
            )

            self.assertIsNone(
                controller._save_auto_model_checkpoint(
                    dummy_model,
                    'after_seed'
                )
            )
            self.assertEqual(controller.auto_model_checkpoint_paths, [])
            self.assertFalse(
                Path(temporary_directory, 'model_after_seed.zip').exists()
            )

    def test_import_rebuilds_fresh_history_and_continues_numbering(self):
        with TemporaryDirectory() as temporary_directory:
            controller, source_model = self._controller_and_model(
                temporary_directory
            )
            source_path = write_model_checkpoint(
                temporary_directory,
                'model_final',
                controller._build_auto_model_checkpoint_manifest(
                    source_model,
                    'final'
                ),
                controller._build_auto_model_checkpoint_arrays(source_model),
                controller.auto_model_performance_rows
            )
            checkpoint = read_model_checkpoint(source_path)
            checkpoint['source_checkpoint_path'] = source_path
            checkpoint['archived_checkpoint_path'] = source_path

            controller.robo_params['auto_model_checkpoint_mode'] = 'import'
            controller._pending_auto_model_checkpoint = checkpoint
            controller.experiment_data = pd.DataFrame({
                'reagent_a': [0.9],
                'reagent_b': [0.9],
                'Experiment_result': [0.9]
            })
            controller._synchronize_target_ei_incumbent_from_performance = (
                lambda model: None
            )

            restored_model = SimpleNamespace(
                min_conc=[0.0, 0.0],
                max_conc=[1.0, 1.0],
                total_volume=200.0,
                acquisition_modes=['balanced'],
                curr_iter=9,
                quit=True
            )

            def initialize_optimizer(X, Y):
                restored_model.optimizer = SimpleNamespace(
                    X=np.array(X, copy=True),
                    Y=np.array(Y, copy=True)
                )

            restored_model.initialize_optimizer = initialize_optimizer
            restored_model.update_usable_spectrum_model = (
                lambda X, Y: setattr(
                    restored_model,
                    'restored_usable_history',
                    (np.array(X, copy=True), np.array(Y, copy=True))
                )
            )

            self.assertTrue(
                controller._restore_imported_auto_model_checkpoint(
                    restored_model
                )
            )
            np.testing.assert_allclose(
                restored_model.optimizer.X,
                source_model.optimizer.X
            )
            np.testing.assert_allclose(
                restored_model.optimizer.Y,
                source_model.optimizer.Y
            )
            self.assertEqual(restored_model.curr_iter, 0)
            self.assertFalse(restored_model.quit)
            self.assertTrue(restored_model._auto_model_checkpoint_imported)
            self.assertTrue(
                controller._auto_model_checkpoint_saving_enabled(
                    restored_model
                )
            )
            # Numbering must continue above the imported history rather than
            # restart, so a continuation cannot emit duplicate reaction/batch
            # keys that overlay this run's conditions on the source run's.
            self.assertEqual(controller.auto_condition_counter, 1)
            self.assertEqual(controller.imported_auto_batch_start, 2)

            # Imported rows describe the source run's plate, never this one's.
            self.assertTrue(
                all(
                    row['executed_in_current_run'] is False
                    for row in controller.auto_model_performance_rows
                )
            )
            self.assertTrue(controller.experiment_data.empty)

    def test_import_rejects_changed_normalized_bounds(self):
        with TemporaryDirectory() as temporary_directory:
            controller, source_model = self._controller_and_model(
                temporary_directory
            )
            checkpoint_path = write_model_checkpoint(
                temporary_directory,
                'model_final',
                controller._build_auto_model_checkpoint_manifest(
                    source_model,
                    'final'
                ),
                controller._build_auto_model_checkpoint_arrays(source_model),
                controller.auto_model_performance_rows
            )
            checkpoint = read_model_checkpoint(checkpoint_path)
            checkpoint['source_checkpoint_path'] = checkpoint_path
            checkpoint['archived_checkpoint_path'] = checkpoint_path

            controller.robo_params['auto_model_checkpoint_mode'] = 'import'
            controller._pending_auto_model_checkpoint = checkpoint
            changed_bounds_model = SimpleNamespace(
                min_conc=[0.0, 0.0],
                max_conc=[0.8, 1.0],
                total_volume=200.0
            )

            with self.assertRaisesRegex(RuntimeError, 'maximum concentration'):
                controller._restore_imported_auto_model_checkpoint(
                    changed_bounds_model
                )

    def test_existing_run_import_selects_only_canonical_direct_checkpoint(self):
        with TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / 'Protocol_Outputs'
            prior_checkpoint_directory = (
                output_root / 'RTG_020' / 'Model_Checkpoints'
            )
            current_run_directory = output_root / 'RTG_021'
            prior_checkpoint_directory.mkdir(parents=True)
            current_run_directory.mkdir()
            source_path = write_model_checkpoint(
                prior_checkpoint_directory,
                'model_final',
                {
                    'checkpoint_stage': 'final',
                    'run_id': 'RTG_020',
                    'variable_reagents': ['reagent_a', 'reagent_b'],
                    'input_coordinate_system': {'name': 'normalized'}
                },
                {
                    'gp_training_X': np.asarray([[0.1, 0.2]]),
                    'gp_training_Y': np.asarray([[0.5]]),
                    'usable_spectrum_X': np.empty((0, 2)),
                    'usable_spectrum_Y': np.empty((0, 1))
                },
                []
            )
            Path(prior_checkpoint_directory, 'not_a_checkpoint.zip').touch()

            controller = self.Controller()
            controller.out_path = str(current_run_directory)
            resolved = controller._resolve_auto_model_checkpoint_existing_source(
                'RTG_020',
                'final'
            )

            self.assertEqual(
                Path(resolved['source_path']).resolve(),
                Path(source_path).resolve()
            )
            self.assertEqual(resolved['import_method'], 'existing_output_run')
            self.assertEqual(
                resolved['available_filenames'],
                ['model_final.zip']
            )
            with self.assertRaisesRegex(ModelCheckpointError, 'folder name'):
                controller._resolve_auto_model_checkpoint_existing_source(
                    '../RTG_020',
                    'final'
                )
            with self.assertRaisesRegex(ModelCheckpointError, 'own output'):
                controller._resolve_auto_model_checkpoint_existing_source(
                    'RTG_021',
                    'final'
                )


class OptimizationModelConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_gpyopt = SimpleNamespace(
            Design_space=lambda bounds: bounds
        )
        cls.ConfigurationModel = _load_optimization_model_methods(
            ['__init__'],
            extra_namespace={'GPyOpt': fake_gpyopt}
        )

    def _required_constructor_arguments(self):
        return {
            'bounds': [],
            'target_value': 625.0,
            'reagent_info': None,
            'fixed_reagents': [],
            'variable_reagents': [],
            'initial_design_numdata': 2,
            'batch_size': 1,
            'max_iters': 2
        }

    def test_balanced_weight_defaults_to_one_for_legacy_callers(self):
        with redirect_stdout(io.StringIO()):
            model = self.ConfigurationModel(
                **self._required_constructor_arguments()
            )

        self.assertEqual(model.balanced_exploration_weight, 1.0)
        self.assertIsNone(model.incumbent_target_error_nm)
        self.assertEqual(model.terminal_verbosity, 'standard')

    def test_terminal_verbosity_rejects_noncanonical_optimizer_values(self):
        with self.assertRaisesRegex(ValueError, 'terminal_verbosity'):
            self.ConfigurationModel(
                **self._required_constructor_arguments(),
                terminal_verbosity='all'
            )

    def test_balanced_weight_is_stored_and_printed(self):
        output = io.StringIO()

        with redirect_stdout(output):
            model = self.ConfigurationModel(
                acquisition_mode='balanced',
                balanced_exploration_weight=1.5,
                **self._required_constructor_arguments()
            )

        self.assertEqual(model.balanced_exploration_weight, 1.5)
        self.assertIn(
            'balanced exploration weight: 1.5000',
            output.getvalue()
        )

    def test_balanced_weight_rejects_invalid_values(self):
        for invalid_weight in (
            None,
            -0.01,
            float('nan'),
            float('inf'),
            'not-a-number'
        ):
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaisesRegex(
                    ValueError,
                    "balanced_exploration_weight"
                ):
                    self.ConfigurationModel(
                        balanced_exploration_weight=invalid_weight,
                        **self._required_constructor_arguments()
                    )


@unittest.skipIf(
    GPy is None,
    'GPy is not available in this review environment'
)
class UsableSpectrumClassifierTests(unittest.TestCase):
    '''Exercises the cumulative, passive Stage 3 classifier in isolation.'''

    @classmethod
    def setUpClass(cls):
        fake_gpyopt = SimpleNamespace(
            Design_space=lambda bounds: bounds
        )
        cls.Model = _load_optimization_model_methods(
            [
                '__init__',
                '_get_dimension',
                'update_usable_spectrum_model',
                'predict_usable_spectrum_probability'
            ],
            extra_namespace={
                'GPy': GPy,
                'GPyOpt': fake_gpyopt
            }
        )

    def _build_model(self):
        with redirect_stdout(io.StringIO()):
            return self.Model(
                bounds=[],
                target_value=625.0,
                reagent_info=None,
                fixed_reagents=[],
                variable_reagents=['A', 'B'],
                initial_design_numdata=2,
                batch_size=1,
                max_iters=2,
                terminal_verbosity='essential'
            )

    def test_classifier_rebuilds_from_complete_cumulative_history(self):
        model = self._build_model()
        first_X = np.asarray([
            [0.0, 0.0],
            [0.2, 0.2],
            [0.8, 0.8],
            [1.0, 1.0]
        ])
        first_Y = np.asarray([[1.0], [1.0], [0.0], [0.0]])
        first_model = model.update_usable_spectrum_model(first_X, first_Y)

        complete_X = np.vstack((first_X, [[0.4, 0.4], [0.6, 0.6]]))
        complete_Y = np.vstack((first_Y, [[1.0], [0.0]]))
        rebuilt_model = model.update_usable_spectrum_model(
            complete_X,
            complete_Y
        )

        self.assertIsNot(first_model, rebuilt_model)
        np.testing.assert_allclose(model.usable_spectrum_X, complete_X)
        np.testing.assert_allclose(model.usable_spectrum_Y, complete_Y)
        probabilities = model.predict_usable_spectrum_probability(
            np.asarray([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        )
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_classifier_rejects_nonbinary_or_out_of_range_history(self):
        model = self._build_model()

        with self.assertRaisesRegex(ValueError, 'binary'):
            model.update_usable_spectrum_model(
                np.asarray([[0.2, 0.2]]),
                np.asarray([[0.5]])
            )

        with self.assertRaisesRegex(ValueError, 'between zero and one'):
            model.update_usable_spectrum_model(
                np.asarray([[1.2, 0.2]]),
                np.asarray([[1.0]])
            )


class UsableSpectrumControllerSynchronizationTests(unittest.TestCase):
    '''Ensures controller audit rows preserve and replay binary outcomes.'''

    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_auto_controller_methods([
            '_safe_float_or_none',
            'Normalize_Denormalize_Recipes',
            '_synchronize_auto_usable_spectrum_model_from_performance'
        ])

    def test_boundary_aware_policy_replays_all_assessed_replicates(self):
        controller = self.Controller()
        controller.variable_reagents = ['A', 'B']
        controller.min_conc = [0.0, 0.0]
        controller.max_conc = [1.0, 1.0]
        controller.num_duplicates = 3
        controller.robo_params = {
            'auto_spectral_response_policy': 'boundary_aware',
            'auto_terminal_verbosity': 'essential'
        }
        controller.auto_model_performance_rows = [{
            'A_concentration': 0.25,
            'B_concentration': 0.75,
            'actual_lambda_rep_1_usable_spectrum_model_eligible': True,
            'actual_lambda_rep_1_usable_spectrum_observed': True,
            'actual_lambda_rep_2_usable_spectrum_model_eligible': True,
            'actual_lambda_rep_2_usable_spectrum_observed': False,
            'actual_lambda_rep_3_usable_spectrum_model_eligible': False,
            'actual_lambda_rep_3_usable_spectrum_observed': None
        }]
        captured = {}
        model = SimpleNamespace(
            update_usable_spectrum_model=lambda X, Y: captured.update(
                X=X.copy(), Y=Y.copy()
            )
        )

        summary = controller._synchronize_auto_usable_spectrum_model_from_performance(
            model
        )

        self.assertTrue(summary['fitted'])
        self.assertEqual(summary['n_assessed_replicates'], 2)
        self.assertEqual(summary['n_usable_replicates'], 1)
        self.assertEqual(summary['n_censored_replicates'], 1)
        np.testing.assert_allclose(
            captured['X'],
            np.asarray([[0.25, 0.75], [0.25, 0.75]])
        )
        np.testing.assert_allclose(captured['Y'], np.asarray([[1.0], [0.0]]))

    def test_audit_only_policy_does_not_create_a_companion_model(self):
        controller = self.Controller()
        controller.variable_reagents = ['A']
        controller.min_conc = [0.0]
        controller.max_conc = [1.0]
        controller.num_duplicates = 1
        controller.robo_params = {
            'auto_spectral_response_policy': 'audit_only'
        }
        controller.auto_model_performance_rows = []
        model = SimpleNamespace()

        summary = controller._synchronize_auto_usable_spectrum_model_from_performance(
            model
        )

        self.assertFalse(summary['enabled'])
        self.assertFalse(summary['fitted'])


if __name__ == '__main__':
    unittest.main()
