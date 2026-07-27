'''Hardware-free lifecycle tests for Auto recipe-concentration history plots.'''

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / 'controller.py'


def _load_auto_plot_suite():
    '''Loads the lifecycle coordinator without importing hardware paths.'''
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding='utf-8'))
    auto_contr = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
    )
    coordinator = next(
        node for node in auto_contr.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == '_generate_auto_plot_suite'
        )
    )
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(
            name='RecipeHistoryLifecycle',
            bases=[], keywords=[], body=[coordinator], decorator_list=[]
        )],
        type_ignores=[]
    ))
    namespace = {'os': os}
    exec(compile(module, str(CONTROLLER_PATH), 'exec'), namespace)
    return namespace['RecipeHistoryLifecycle']


class AutoRecipeHistoryLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Lifecycle = _load_auto_plot_suite()

    def setUp(self):
        self.controller = self.Lifecycle()
        self.controller.robo_params = {
            'auto_plot_profile': 'standard',
            'auto_terminal_verbosity': 'off'
        }
        self.controller.batch_num = 2
        self.controller.variable_reagents = ['citrate', 'silver']
        self.calls = []

        def record(name, return_value=None):
            def _method(*args, **kwargs):
                self.calls.append((name, args, kwargs))
                return return_value
            return _method

        self.controller._plot_lambda_progress_after_batch = record(
            'progress', 'progress.png'
        )
        self.controller._plot_lambda_replicate_progress_after_batch = record(
            'replicates', 'replicates.png'
        )
        self.controller._plot_auto_recipe_concentration_history = record(
            'recipe_history', 'recipe_history.png'
        )
        self.controller._plot_initial_training_designs_after_run = record(
            'design_space', ['design.png']
        )
        self.controller._write_auto_run_report = record(
            'report', 'auto_run_report.md'
        )
        self.controller._append_auto_plot_manifest = record('manifest')

    def _recipe_history_calls(self):
        return [
            call for call in self.calls
            if call[0] == 'recipe_history'
        ]

    def test_standard_after_measurement_generates_both_histories(self):
        self.controller._generate_auto_plot_suite(
            stage='after_measurement',
            model=SimpleNamespace(acquisition_modes=[]),
            batch_number=1
        )

        history_calls = self._recipe_history_calls()
        self.assertEqual(len(history_calls), 2)
        self.assertEqual(history_calls[0][1], (1,))
        self.assertFalse(history_calls[0][2]['include_fixed_reagents'])
        self.assertTrue(history_calls[1][2]['include_fixed_reagents'])

    def test_final_generates_named_histories_before_report(self):
        self.controller._generate_auto_plot_suite(
            stage='final',
            model=SimpleNamespace(acquisition_modes=[]),
            batch_number=1
        )

        history_calls = self._recipe_history_calls()
        self.assertEqual(len(history_calls), 2)
        self.assertEqual(
            history_calls[0][2]['plot_filename'],
            'auto_recipe_concentration_history_final.png'
        )
        self.assertEqual(
            history_calls[1][2]['plot_filename'],
            'auto_complete_recipe_concentration_history_final.png'
        )
        self.assertLess(
            next(index for index, call in enumerate(self.calls)
                 if call[0] == 'recipe_history'),
            next(index for index, call in enumerate(self.calls)
                 if call[0] == 'report')
        )

    def test_final_only_suppresses_per_batch_but_keeps_final_histories(self):
        self.controller.robo_params['auto_plot_profile'] = 'final_only'
        # One variable avoids the unrelated final-only 2D GP refresh path;
        # this test isolates plot-profile handling for recipe history.
        self.controller.variable_reagents = ['citrate']

        self.controller._generate_auto_plot_suite(
            stage='after_measurement',
            model=SimpleNamespace(acquisition_modes=[]),
            batch_number=1
        )
        self.assertEqual(self._recipe_history_calls(), [])

        self.controller._generate_auto_plot_suite(
            stage='final',
            model=SimpleNamespace(acquisition_modes=[]),
            batch_number=1
        )
        self.assertEqual(len(self._recipe_history_calls()), 2)


if __name__ == '__main__':
    unittest.main()
