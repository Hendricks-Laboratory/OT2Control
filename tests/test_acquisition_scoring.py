import ast
from contextlib import redirect_stdout
import io
import math
import numpy as np
from pathlib import Path
from types import SimpleNamespace
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZERS_PATH = REPOSITORY_ROOT / 'optimizers.py'
CONTROLLER_PATH = REPOSITORY_ROOT / 'controller.py'


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
        'math': math
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


def _load_auto_controller_methods(method_names):
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
    namespace = {'math': math}

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
    namespace = {'math': math}

    exec(
        compile(module, str(CONTROLLER_PATH), 'exec'),
        namespace
    )

    return namespace['Controller']


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

    def test_active_bounds_start_at_five_ul_executable_transfer(self):
        model = self._build_model(allow_true_zero=True)
        bounds = model._get_masked_bounds([1, 0, 1])

        # stock concentration 1.0 * 5 uL / 100 uL total gives a normalized
        # lower concentration bound of 0.05 for each active reagent.
        self.assertEqual(bounds, [(0.05, 1.0), (0.05, 1.0)])


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
                'update_experiment_data',
                '_synchronize_target_ei_incumbent_from_performance'
            }
        )

        initialize_lines = [
            line_number
            for line_number, method_name in model_lifecycle_calls
            if method_name == 'initialize_optimizer'
        ]
        update_lines = [
            line_number
            for line_number, method_name in model_lifecycle_calls
            if method_name == 'update_experiment_data'
        ]
        synchronization_lines = [
            line_number
            for line_number, method_name in model_lifecycle_calls
            if method_name
            == '_synchronize_target_ei_incumbent_from_performance'
        ]

        self.assertEqual(len(initialize_lines), 1)
        self.assertEqual(len(update_lines), 1)
        self.assertEqual(len(synchronization_lines), 2)
        self.assertLess(initialize_lines[0], synchronization_lines[0])
        self.assertLess(update_lines[0], synchronization_lines[1])

    def test_controller_captures_complete_selection_metadata_before_run(self):
        run_method = _get_auto_controller_method_node('_run')
        metadata_dict = next(
            node.value
            for node in ast.walk(run_method)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == 'optimizer_prediction_metadata'
                for target in node.targets
            )
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
            'selected_mask'
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
            'selected_mask'
        }.issubset(row_keys))

    def test_run_report_includes_acquisition_settings_and_audit_fields(self):
        report_method = _get_auto_controller_method_node(
            '_write_auto_run_report'
        )
        report_strings = {
            node.value
            for node in ast.walk(report_method)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }

        self.assertIn('## Acquisition Audit Trail', report_strings)
        self.assertIn('acquisition_mode', report_strings)
        self.assertIn('acquisition_score', report_strings)
        self.assertIn('predicted_target_error_nm', report_strings)
        self.assertIn('predicted_lambda_mean_nm', report_strings)
        self.assertIn('predicted_lambda_std_nm', report_strings)
        self.assertIn('incumbent_target_error_nm', report_strings)
        self.assertIn('selected_mask', report_strings)


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
                'target_error_nm': 8.0,
                # A lucky well is intentionally irrelevant to the incumbent.
                'actual_lambda_rep_1_nm': 625.0
            },
            {
                'use_for_model_training': False,
                'target_error_nm': 0.1
            },
            {
                'use_for_model_training': True,
                'target_error_nm': 3.0
            },
            {
                'use_for_model_training': True,
                'target_error_nm': None
            },
            {
                'use_for_model_training': True,
                'target_error_nm': float('nan')
            },
            {
                'use_for_model_training': True,
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

    def test_target_ei_synchronization_requires_approved_condition(self):
        controller = self._build_controller([
            {
                'use_for_model_training': False,
                'target_error_nm': 0.5
            }
        ])
        model = SimpleNamespace(acquisition_mode='target_ei')

        with self.assertRaisesRegex(
            ValueError,
            "no QC-approved condition-level"
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


class AcquisitionHeaderCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Controller = _load_base_controller_methods([
            '_init_robo_header_params'
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

    def _parse_header(self, acquisition_mode=None):
        controller = self.Controller()
        controller.robo_params = {}
        controller.DilutionParams = lambda container, volume: (
            container,
            volume
        )
        header = self._base_header()

        if acquisition_mode is not None:
            header.append(['acquisition_mode', acquisition_mode])

        with redirect_stdout(io.StringIO()):
            controller._init_robo_header_params(header)

        return controller.robo_params

    def test_legacy_header_defaults_to_exact_exploit_compatibility(self):
        parsed = self._parse_header()

        self.assertEqual(parsed['acquisition_mode'], 'exploit')
        self.assertEqual(parsed['auto_plot_profile'], 'standard')
        self.assertEqual(parsed['num_duplicates'], 3)
        self.assertFalse(parsed['allow_true_zero'])

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


if __name__ == '__main__':
    unittest.main()
