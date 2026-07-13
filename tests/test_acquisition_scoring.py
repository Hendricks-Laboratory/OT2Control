import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZERS_PATH = REPOSITORY_ROOT / 'optimizers.py'


def _load_optimization_model_methods(method_names):
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

    class_constants = [
        node for node in model_class.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == '_SUPPORTED_ACQUISITION_MODES'
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

    namespace = {}
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
        for mode in ('explore', 'balanced', 'target_ei'):
            with self.subTest(mode=mode):
                model = self._build_score_model(mode)

                with self.assertRaisesRegex(
                    NotImplementedError,
                    repr(mode)
                ):
                    model._calculate_acquisition_score(625.0, 2.0)

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

    def _build_model(self, volume_balance):
        model = self.MaskedModel()
        model.target_value = 625.0
        model.acquisition_mode = 'exploit'
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

    def test_overflow_penalty_is_unchanged_and_skips_gp(self):
        volume_balance = {
            'volume_feasible': False,
            'water_volume': -7.0,
            'volume_does_not_overflow': False,
            'water_transfer_executable': False
        }
        model = self._build_model(volume_balance)
        model._predict_lambda_max_nm = (
            lambda full_x: self.fail('GP prediction must not run')
        )

        score = model._masked_acquisition_objective([0.5], [1])

        self.assertEqual(score, 1e12 + 49.0)

    def test_bad_water_penalty_is_unchanged_and_skips_gp(self):
        volume_balance = {
            'volume_feasible': False,
            'water_volume': 3.0,
            'volume_does_not_overflow': True,
            'water_transfer_executable': False
        }
        model = self._build_model(volume_balance)
        model._predict_lambda_max_nm = (
            lambda full_x: self.fail('GP prediction must not run')
        )

        score = model._masked_acquisition_objective([0.5], [1])

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
        model = self.SelectionModel()
        model.acquisition_mode = 'balanced'
        optimization_calls = []
        model._optimize_acquisition_with_masks = (
            lambda: optimization_calls.append(True)
        )

        with self.assertRaisesRegex(NotImplementedError, "'balanced'"):
            model.getNextReaction()

        self.assertEqual(optimization_calls, [])


if __name__ == '__main__':
    unittest.main()
