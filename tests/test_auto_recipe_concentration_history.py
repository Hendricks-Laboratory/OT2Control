'''Hardware-free tests for condition-level Auto recipe-concentration plots.'''

import ast
import os
from pathlib import Path
import re
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / 'controller.py'


def _load_recipe_history_renderer():
    '''Loads only plot helpers, without importing controller hardware paths.'''
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding='utf-8'))
    auto_contr = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
    )
    names = {
        '_format_auto_design_axis_label',
        '_get_auto_design_plot_font_sizes',
        '_get_auto_design_concentration_column',
        '_get_auto_recipe_concentration_history_dataframe',
        '_get_auto_plot_relative_path',
        '_get_auto_plot_output_path',
        '_plot_auto_recipe_concentration_history'
    }
    body = [
        node for node in auto_contr.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(
            name='RecipeHistoryRenderer',
            bases=[], keywords=[], body=body, decorator_list=[]
        )],
        type_ignores=[]
    ))
    namespace = {
        'math': __import__('math'),
        'np': np,
        'os': os,
        'pd': pd,
        'plt': plt,
        're': re
    }
    exec(compile(module, str(CONTROLLER_PATH), 'exec'), namespace)
    return namespace['RecipeHistoryRenderer']


class AutoRecipeConcentrationHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Renderer = _load_recipe_history_renderer()

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.renderer = self.Renderer()
        self.renderer.plot_path = self.temp_dir.name
        self.renderer.variable_reagents = ['citrate', 'silver']
        self.renderer.fixed_reagents = ['buffer']
        self.renderer.auto_model_performance_rows = [
            {
                'reaction_number': 2,
                'batch_number': 1,
                'citrate_concentration': 0.10,
                'citrate_executed_concentration': 0.15,
                'silver_concentration': 0.002,
                'silver_executed_concentration': 0.003,
                'buffer_executed_concentration': 0.5,
                'total_volume_uL': 200.0,
                'executed_in_current_run': True
            },
            {
                'reaction_number': 0,
                'batch_number': 0,
                'citrate_concentration': 0.20,
                'silver_concentration': 0.004,
                'buffer_executed_concentration': 0.5,
                'total_volume_uL': 200.0,
                'executed_in_current_run': False
            },
            {
                'reaction_number': 1,
                'batch_number': 0,
                'citrate_concentration': 0.00,
                'silver_concentration': 0.006,
                'buffer_executed_concentration': 0.5,
                'total_volume_uL': 200.0,
                'executed_in_current_run': False
            }
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_history_uses_one_sorted_row_per_condition_and_executed_values(self):
        dataframe, descriptors = (
            self.renderer._get_auto_recipe_concentration_history_dataframe()
        )

        self.assertEqual(
            dataframe['reaction_number'].astype(int).tolist(),
            [0, 1, 2]
        )
        self.assertEqual(
            [descriptor['reagent_name'] for descriptor in descriptors],
            ['citrate', 'silver']
        )
        self.assertEqual(
            dataframe[descriptors[0]['plot_column']].tolist(),
            [0.20, 0.00, 0.15]
        )
        self.assertEqual(
            dataframe[descriptors[1]['plot_column']].tolist(),
            [0.004, 0.006, 0.003]
        )

    def test_complete_history_uses_recorded_fixed_concentrations(self):
        dataframe, descriptors = (
            self.renderer._get_auto_recipe_concentration_history_dataframe(
                include_fixed_reagents=True
            )
        )

        self.assertEqual(
            [descriptor['reagent_name'] for descriptor in descriptors],
            ['citrate', 'silver', 'buffer']
        )
        fixed_values = dataframe[descriptors[-1]['plot_column']].tolist()
        self.assertEqual(fixed_values, [0.5, 0.5, 0.5])
        self.assertTrue(descriptors[-1]['is_fixed'])

    def test_complete_history_rejects_unrecorded_fixed_concentrations(self):
        for row in self.renderer.auto_model_performance_rows:
            row.pop('buffer_executed_concentration')

        with self.assertRaisesRegex(
            ValueError,
            'no recorded executed concentration'
        ):
            self.renderer._get_auto_recipe_concentration_history_dataframe(
                include_fixed_reagents=True
            )

    def test_duplicate_condition_numbers_fail_closed(self):
        self.renderer.auto_model_performance_rows.append({
            'reaction_number': 1,
            'batch_number': 1,
            'citrate_concentration': 0.30,
            'silver_concentration': 0.007,
            'total_volume_uL': 200.0
        })

        with self.assertRaisesRegex(ValueError, 'unique conditions'):
            self.renderer._get_auto_recipe_concentration_history_dataframe()

    def test_grouped_plot_writes_recipe_history_category(self):
        output_path = self.renderer._plot_auto_recipe_concentration_history(
            batch_number=1,
            include_fixed_reagents=True,
            plot_filename='auto_complete_recipe_concentration_history_test.png'
        )

        self.assertTrue(os.path.exists(output_path))
        self.assertIn(
            os.path.join('recipe_history',
                         'auto_complete_recipe_concentration_history_test.png'),
            output_path
        )


if __name__ == '__main__':
    unittest.main()
