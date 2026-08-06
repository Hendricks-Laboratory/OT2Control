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
            ),
            'initial_maximin_seed_design_feasibility_2d.png': (
                'design_space/seed_feasibility_overlays/'
                'initial_maximin_seed_design_feasibility_2d.png'
            ),
            'initial_maximin_seed_design_feasibility_'
            'conditional_slices_atlas_page_01.png': (
                'design_space/seed_feasibility_overlays/atlases/'
                'initial_maximin_seed_design_feasibility_'
                'conditional_slices_atlas_page_01.png'
            ),
            'auto_recipe_concentration_history_final.png': (
                'recipe_history/auto_recipe_concentration_history_final.png'
            ),
            'auto_complete_recipe_concentration_history_final.png': (
                'recipe_history/'
                'auto_complete_recipe_concentration_history_final.png'
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

    def test_seed_feasibility_views_use_optimizer_feasibility_authority(self):
        '''Seed overlays must not maintain a second physical-rule model.'''
        source = CONTROLLER_PATH.read_text(encoding='utf-8')

        self.assertIn(
            'def _plot_initial_seed_feasibility_views(',
            source
        )
        self.assertIn(
            'get_candidate_feasibility_batch_for_plotting',
            source
        )
        self.assertIn(
            "'initial_maximin_seed_design_feasibility_'",
            source
        )
        self.assertIn(
            "'final initial seed physical-feasibility plots'",
            source
        )
        self.assertIn(
            "hatches=['///']",
            source
        )
        self.assertIn(
            "hatch='///'",
            source
        )
        self.assertIn(
            'include_reference_seed_marker=False',
            source
        )


if __name__ == '__main__':
    unittest.main()
