'''Source-level Stage 9A checks without importing hardware modules.'''

import ast
import os
import unittest


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


class AutoPreparationControllerContractTests(unittest.TestCase):
    '''Guard Stage 9A's planning-only, fail-closed controller boundary.'''

    @classmethod
    def setUpClass(cls):
        with open(CONTROLLER_PATH, 'r', encoding='utf-8') as source_file:
            cls.source = source_file.read()
        cls.tree = ast.parse(cls.source, filename=CONTROLLER_PATH)
        cls.auto_class = next(
            node for node in cls.tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
        )
        cls.auto_methods = {
            node.name: node
            for node in cls.auto_class.body
            if isinstance(node, ast.FunctionDef)
        }

    def test_planning_and_phase_methods_exist_once(self):
        for method_name in (
            '_initialize_auto_preparation_plan',
            '_execute_auto_preparation_phase'
        ):
            self.assertIn(method_name, self.auto_methods)
            self.assertEqual(
                sum(
                    1 for node in self.auto_class.body
                    if isinstance(node, ast.FunctionDef) and node.name == method_name
                ),
                1
            )

    def test_planning_runs_before_connection_and_seed_generation(self):
        initialization = ast.get_source_segment(
            self.source,
            self.auto_methods['_initialize_auto_preparation_plan']
        )
        run_method = ast.get_source_segment(self.source, self.auto_methods['_run'])
        self.assertIn('build_preparation_manifest(', initialization)
        self.assertLess(
            run_method.index('self.create_connection('),
            run_method.index('self._execute_auto_preparation_phase(')
        )
        self.assertLess(
            run_method.index('self._execute_auto_preparation_phase('),
            run_method.index('model.generate_initial_design(')
        )

    def test_phase_is_planning_only_and_cannot_issue_old_one_tube_execution(self):
        phase_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_execute_auto_preparation_phase']
        )
        self.assertIn('planning-only', phase_source)
        self.assertIn('Auto-main group-reservation protocol', phase_source)
        self.assertNotIn('_execute_auto_preparation_entry(', phase_source)
        self.assertNotIn('_activate_auto_prepared_sources(', phase_source)
        self.assertNotIn('execute_protocol_df(', phase_source)


if __name__ == '__main__':
    unittest.main()
