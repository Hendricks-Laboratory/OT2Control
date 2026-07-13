import ast
from contextlib import redirect_stdout
import io
import math
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
    scientific stack. These scoring tests deliberately extract only pure or
    stub-compatible methods from the exact repository source so they can run
    safely with the standard-library Python 3.9 interpreter.
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


class AcquisitionScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ScoreModel = _load_optimization_model_methods([
            '_calculate_acquisition_score'
        ])

    def _build_score_model(self, acquisition_mode='exploit'):
        model = self.ScoreModel()
        model.target_value = 625.0
        model.acquisition_mode = acquisition_mode
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

    def test_unimplemented_modes_fail_clearly(self):
        for mode in ('balanced', 'target_ei'):
            with self.subTest(mode=mode):
                model = self._build_score_model(mode)

                with self.assertRaisesRegex(
                    NotImplementedError,
                    repr(mode)
                ):
                    model._calculate_acquisition_score(625.0, 2.0)

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
            '_calculate_acquisition_score',
            '_masked_acquisition_objective',
            '_masked_target_distance_objective'
        ])

    def _build_model(self, volume_balance, acquisition_mode='exploit'):
        model = self.MaskedModel()
        model.target_value = 625.0
        model.acquisition_mode = acquisition_mode
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

    def test_overflow_penalty_is_unchanged_and_skips_gp(self):
        volume_balance = {
            'volume_feasible': False,
            'water_volume': -7.0,
            'volume_does_not_overflow': False,
            'water_transfer_executable': False
        }
        for mode in ('exploit', 'explore'):
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
        for mode in ('exploit', 'explore'):
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
        optimization_calls = []
        model._optimize_acquisition_with_masks = (
            lambda: optimization_calls.append(True) or [0.25]
        )
        model.predict_lambda_distribution_nm = (
            lambda x: (624.5, 1.25)
        )

        with redirect_stdout(io.StringIO()):
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

    def test_unimplemented_mode_stops_before_optimization(self):
        for mode in ('balanced', 'target_ei'):
            with self.subTest(mode=mode):
                model = self.SelectionModel()
                model.acquisition_mode = mode
                optimization_calls = []
                model._optimize_acquisition_with_masks = (
                    lambda: optimization_calls.append(True)
                )

                with self.assertRaisesRegex(
                    NotImplementedError,
                    repr(mode)
                ):
                    model.getNextReaction()

                self.assertEqual(optimization_calls, [])

    def test_explore_reaches_optimizer_and_preserves_return_shape(self):
        model = self.SelectionModel()
        model.acquisition_mode = 'explore'
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


if __name__ == '__main__':
    unittest.main()
