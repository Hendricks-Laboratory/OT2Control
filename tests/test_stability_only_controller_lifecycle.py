'''Hardware-free Stage-12F controller lifecycle contracts.

These tests extract only the target-free controller methods from source.  They
never import ``controller.py`` (which imports hardware-facing dependencies) or
create a robot/reader connection.
'''

import ast
import copy
import os
import types
import unittest

import numpy as np
import pandas as pd

from auto_stability import AutoStabilityValidationError, canonical_reagent_name


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


def _controller_methods(*method_names):
    with open(CONTROLLER_PATH, encoding='utf-8') as handle:
        source = handle.read()
    tree = ast.parse(source, filename=CONTROLLER_PATH)
    controller_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
    )
    methods = {
        node.name: node for node in controller_class.body
        if isinstance(node, ast.FunctionDef)
    }
    return source, methods, [methods[name] for name in method_names]


def _extract_function(method_name, namespace):
    _source, _methods, method_nodes = _controller_methods(method_name)
    method = ast.fix_missing_locations(ast.Module(
        body=[method_nodes[0]], type_ignores=[]
    ))
    exec(compile(method, CONTROLLER_PATH, 'exec'), namespace)
    return namespace[method_name]


class StabilityOnlyControllerLifecycleTests(unittest.TestCase):
    def test_target_free_lifecycle_collects_seed_then_selects_without_lambda_gp(self):
        run = _extract_function(
            '_run_auto_stability_only',
            {
                'np': np,
                'copy': copy,
                'AutoStabilityValidationError': AutoStabilityValidationError,
            }
        )

        class FakeModel(object):
            def __init__(self):
                self._auto_model_checkpoint_imported = False
                self.curr_iter = 0
                self.max_iters = 1
                self.quit = False
                self.last_stability_only_selection_metadata = {
                    'stability_selection_mode': 'stability_only',
                    'signal_model_status': 'fitted',
                }
                self.selection_calls = []

            @staticmethod
            def generate_initial_design():
                return np.asarray([[0.2], [0.8]], dtype=float)

            def getNextStabilityOnlyReaction(self, **kwargs):
                self.selection_calls.append(kwargs)
                return [np.asarray([0.4], dtype=float)]

        class FakeController(object):
            def __init__(self):
                self.robo_params = {
                    'auto_model_checkpoint_mode': 'off',
                    'auto_stability_min_peak_absorbance': 0.20,
                    'auto_stability_max_peak_absorbance': 0.80,
                    'auto_stability_signal_confidence_z': 1.96,
                }
                self.batch_num = None
                self.events = []

            def Normalize_Denormalize_Recipes(self, recipes, normalize_flag):
                if normalize_flag:
                    raise AssertionError('seed design must be denormalized')
                return np.asarray(recipes, dtype=float) * 10.0

            @staticmethod
            def _apply_true_zero_transfer_rule_to_recipes(recipes):
                return np.asarray(recipes, dtype=float)

            def _validate_auto_recipe_volume_feasibility(self, recipes, context_label):
                self.events.append(('validated', context_label, recipes.copy()))

            def _export_auto_batch_recipe_design(self, **kwargs):
                self.events.append(('exported', kwargs['batch_label']))

            def _execute_auto_stability_only_batch(self, **kwargs):
                self.events.append((
                    'executed',
                    kwargs['condition_type'],
                    np.asarray(kwargs['unique_recipes']).copy(),
                    dict(kwargs['prediction_metadata']),
                ))

            @staticmethod
            def _get_auto_stability_only_selection_context():
                return 'loss-model', 'signal-model', 0

            @staticmethod
            def _prepare_auto_optimizer_recipe_for_execution(**kwargs):
                selected = np.asarray(kwargs['normalized_recipes'], dtype=float)
                return selected * 10.0, dict(kwargs['additional_selection_metadata'])

            def _finalize_auto_stability_only_run(self, model):
                self.events.append(('finalized', model.quit))

        controller = FakeController()
        model = FakeModel()
        run(controller, model)

        self.assertEqual(model.curr_iter, 1)
        self.assertTrue(model.quit)
        self.assertEqual(len(model.selection_calls), 1)
        self.assertEqual(model.selection_calls[0]['stability_model'], 'loss-model')
        self.assertEqual(model.selection_calls[0]['signal_model'], 'signal-model')
        self.assertEqual(
            [event[1] for event in controller.events if event[0] == 'executed'],
            ['seed', 'optimizer_selected']
        )
        self.assertEqual(controller.events[-1], ('finalized', True))

    def test_context_requires_two_fitted_current_run_companion_models(self):
        context = _extract_function(
            '_get_auto_stability_only_selection_context',
            {
                'AutoStabilityValidationError': AutoStabilityValidationError,
                'STABILITY_MODE_STABILITY_ONLY': 'stability_only',
                'STABILITY_MODEL_STATUS_FIT_FAILED': 'fit_failed',
                'STABILITY_SIGNAL_MODEL_STATUS_FIT_FAILED': 'fit_failed',
                'canonical_reagent_name': canonical_reagent_name,
            }
        )
        controller = types.SimpleNamespace(
            robo_params={
                'auto_stability_mode': 'stability_only',
                'auto_stability_trigger_reagent': 'sodium borohydride',
            },
            variable_reagents=['silver_nitrate', 'sodium_borohydride'],
            auto_stability_model_summary={
                'status': 'fitted', 'accepted_condition_count': 2,
            },
            auto_stability_signal_model_summary={
                'status': 'fitted', 'accepted_condition_count': 2,
            },
            auto_stability_model=types.SimpleNamespace(status='fitted'),
            auto_stability_signal_model=types.SimpleNamespace(status='fitted'),
        )
        loss_model, signal_model, trigger_index = context(controller)
        self.assertEqual(loss_model.status, 'fitted')
        self.assertEqual(signal_model.status, 'fitted')
        self.assertEqual(trigger_index, 1)

        controller.auto_stability_signal_model_summary = {
            'status': 'insufficient', 'accepted_condition_count': 1,
        }
        with self.assertRaisesRegex(AutoStabilityValidationError, 'fitted current-run'):
            context(controller)

    def test_source_contract_keeps_lambda_observations_audit_only(self):
        source, methods, _unused = _controller_methods(
            '_execute_auto_stability_only_batch',
            '_run_auto_stability_only',
            '_append_auto_model_performance_rows',
        )
        execute_source = ast.get_source_segment(
            source, methods['_execute_auto_stability_only_batch']
        )
        run_source = ast.get_source_segment(
            source, methods['_run_auto_stability_only']
        )
        performance_source = ast.get_source_segment(
            source, methods['_append_auto_model_performance_rows']
        )

        self.assertIn("'wavelength_observation_role': 'audit_only'", execute_source)
        self.assertNotIn('initialize_optimizer(', execute_source)
        self.assertNotIn('model.update_experiment_data(', execute_source)
        self.assertNotIn('getNextReaction(', run_source)
        self.assertNotIn('_synchronize_target_ei_incumbent', run_source)
        self.assertNotIn('_update_auto_quit_from_condition_level_performance', run_source)
        self.assertIn("'not_applicable_stability_only'", performance_source)
        self.assertIn("'audit_only' if stability_only_mode", performance_source)

    def test_legacy_dummy_simulation_is_explicitly_blocked_before_execution(self):
        source, methods, _unused = _controller_methods(
            'run_simulation', 'run_protocol'
        )
        simulation_source = ast.get_source_segment(
            source, methods['run_simulation']
        )
        protocol_source = ast.get_source_segment(source, methods['run_protocol'])
        self.assertIn('legacy Auto simulator cannot validate target-free', simulation_source)
        self.assertIn('DummyMLModel cannot', protocol_source)

    def test_raw_wavelength_export_is_labeled_audit_only_for_target_free_runs(self):
        export = _extract_function(
            '_build_labeled_auto_experiment_data_export', {
                'pd': pd,
                'STABILITY_MODE_STABILITY_ONLY': 'stability_only',
            }
        )
        controller = types.SimpleNamespace(
            variable_reagents=['silver_nitrate'],
            robo_params={'auto_stability_mode': 'stability_only'},
            experiment_data=pd.DataFrame({
                'silver_nitrate': [0.1], 'Experiment_result': [625.0],
            }),
        )
        output = export(controller)
        self.assertIn('lambda_max_nm_audit_only', output.columns)
        self.assertNotIn('lambda_max_nm', output.columns)


if __name__ == '__main__':
    unittest.main()
