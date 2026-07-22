'''Hardware-free regression tests for organized Auto plot output paths.'''

import ast
import os
from pathlib import Path
import re
import unittest


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / 'controller.py'


def _load_auto_plot_relative_path_method():
    '''Loads the pure path classifier without importing controller hardware.'''
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding='utf-8'))
    method_node = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'AutoContr':
            for candidate in node.body:
                if (
                    isinstance(candidate, ast.FunctionDef)
                    and candidate.name == '_get_auto_plot_relative_path'
                ):
                    method_node = candidate
                    break

    if method_node is None:
        raise AssertionError('Auto plot path classifier was not found.')

    module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'os': os, 're': re}
    exec(compile(module, str(CONTROLLER_PATH), 'exec'), namespace)
    return namespace['_get_auto_plot_relative_path']


class AutoPlotOrganizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.relative_path = _load_auto_plot_relative_path_method()

    def test_classifier_separates_auto_plot_families(self):
        cases = {
            'lambda_progress_final.png': (
                'progress/lambda_progress_final.png'
            ),
            'gpr_predictions_batch_1.png': (
                'gp_surfaces/2d/mean/atlases/gpr_predictions_batch_1.png'
            ),
            'gpr_predictions_feasibility_batch_1.png': (
                'gp_surfaces/2d/mean/feasibility_overlays/atlases/'
                'gpr_predictions_feasibility_batch_1.png'
            ),
            'gpr_uncertainty_feasibility_batch_1.png': (
                'gp_surfaces/2d/uncertainty/feasibility_overlays/atlases/'
                'gpr_uncertainty_feasibility_batch_1.png'
            ),
            'auto_design_space_exploration_3d.png': (
                'design_space/auto_design_space_exploration_3d.png'
            )
        }

        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    self.relative_path(filename),
                    expected
                )

    def test_classifier_separates_3d_atlases_and_conditional_slices(self):
        cases = {
            'gpr_3d_mean_orthogonal_slices_final.png': (
                'gp_surfaces/3d/mean/atlases/'
                'gpr_3d_mean_orthogonal_slices_final.png'
            ),
            'gpr_3d_uncertainty_slice__hold--silver_nitrate--0.2mM_final.png': (
                'gp_surfaces/3d/uncertainty/conditional_slices/'
                'gpr_3d_uncertainty_slice__hold--silver_nitrate--0.2mM_final.png'
            ),
            'gpr_3d_mean_slice_feasibility__hold--citrate--0.1mM_final.png': (
                'gp_surfaces/3d/mean/feasibility_overlays/conditional_slices/'
                'gpr_3d_mean_slice_feasibility__hold--citrate--0.1mM_final.png'
            ),
            'gpr_3d_mean_orthogonal_slices_feasibility_final.png': (
                'gp_surfaces/3d/mean/feasibility_overlays/atlases/'
                'gpr_3d_mean_orthogonal_slices_feasibility_final.png'
            )
        }

        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    self.relative_path(filename),
                    expected
                )

    def test_classifier_separates_higher_dimensional_atlases_and_slices(self):
        cases = {
            'gpr_4d_mean_conditional_slices_page_01_final.png': (
                'gp_surfaces/4d/mean/atlases/'
                'gpr_4d_mean_conditional_slices_page_01_final.png'
            ),
            'gpr_4d_uncertainty_slice__x--A__y--B__reference_recipe_final.png': (
                'gp_surfaces/4d/uncertainty/conditional_slices/'
                'gpr_4d_uncertainty_slice__x--A__y--B__reference_recipe_final.png'
            ),
            'gpr_5d_mean_slice_feasibility__x--A__y--B__reference_recipe_final.png': (
                'gp_surfaces/5d/mean/feasibility_overlays/conditional_slices/'
                'gpr_5d_mean_slice_feasibility__x--A__y--B__reference_recipe_final.png'
            ),
            'gpr_5d_uncertainty_conditional_slices_feasibility_page_02_final.png': (
                'gp_surfaces/5d/uncertainty/feasibility_overlays/atlases/'
                'gpr_5d_uncertainty_conditional_slices_feasibility_page_02_final.png'
            ),
            'gpr_10d_target_probability_slice__x--A__y--B__reference_recipe_final.png': (
                'gp_surfaces/10d/target_probability/conditional_slices/'
                'gpr_10d_target_probability_slice__x--A__y--B__reference_recipe_final.png'
            )
        }

        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(self.relative_path(filename), expected)

    def test_portfolio_trace_labels_observed_and_gp_error_bars(self):
        '''The portfolio figure must not leave its two uncertainty types vague.'''
        source = CONTROLLER_PATH.read_text(encoding='utf-8')

        self.assertIn(
            'observed mean ± replicate SEM',
            source
        )
        self.assertIn(
            'pre-execution GP mean ± posterior SD',
            source
        )
        self.assertIn(
            'filled error bars = replicate SEM; hollow error ',
            source
        )


if __name__ == '__main__':
    unittest.main()
