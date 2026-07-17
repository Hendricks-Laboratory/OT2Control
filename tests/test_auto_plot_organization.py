'''Hardware-free regression tests for organized Auto plot output paths.'''

import ast
import os
from pathlib import Path
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
    namespace = {'os': os}
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
                'gp_surfaces/2d/mean/gpr_predictions_batch_1.png'
            ),
            'gpr_predictions_feasibility_batch_1.png': (
                'gp_surfaces/2d/mean/feasibility_overlays/'
                'gpr_predictions_feasibility_batch_1.png'
            ),
            'gpr_uncertainty_feasibility_batch_1.png': (
                'gp_surfaces/2d/uncertainty/feasibility_overlays/'
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
            )
        }

        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    self.relative_path(filename),
                    expected
                )


if __name__ == '__main__':
    unittest.main()
