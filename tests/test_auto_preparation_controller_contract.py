'''Source-level Stage 9B contract checks without importing hardware modules.'''

import ast
import os
import unittest


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


class AutoPreparationControllerContractTests(unittest.TestCase):
    '''Guard the placement and fail-closed boundaries of Stage 9B wiring.'''

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

    def test_required_stage_9b_methods_exist_once(self):
        for method_name in (
            '_initialize_auto_preparation_plan',
            '_execute_auto_preparation_entry',
            '_activate_auto_prepared_sources',
            '_apply_auto_prepared_source_cache_policy',
            '_execute_auto_preparation_phase'
        ):
            self.assertIn(method_name, self.auto_methods)
            self.assertEqual(
                sum(
                    1
                    for node in self.auto_class.body
                    if (
                        isinstance(node, ast.FunctionDef)
                        and node.name == method_name
                    )
                ),
                1
            )

    def test_execution_is_before_seed_generation_and_after_connection(self):
        run_method = self.auto_methods['_run']
        run_source = ast.get_source_segment(self.source, run_method)
        self.assertLess(
            run_source.index('self.create_connection('),
            run_source.index('self._execute_auto_preparation_phase(')
        )
        self.assertLess(
            run_source.index('self._execute_auto_preparation_phase('),
            run_source.index('model.generate_initial_design(')
        )

    def test_working_source_cache_policy_precedes_volume_conversion(self):
        build_method = self.auto_methods['_build_rxn_df']
        build_source = ast.get_source_segment(self.source, build_method)
        self.assertLess(
            build_source.index('self._apply_auto_prepared_source_cache_policy()'),
            build_source.index('self._convert_conc_to_vol(')
        )

    def test_local_preflight_does_not_physically_prepare_or_mutate_bounds(self):
        phase_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_execute_auto_preparation_phase']
        )
        simulated_branch = phase_source.split('if simulate:', 1)[1].split(
            "self._record_auto_live_run_event(", 1
        )[0]
        self.assertIn('deferred during', simulated_branch)
        self.assertNotIn('_execute_auto_preparation_entry', simulated_branch)
        self.assertNotIn('_activate_auto_prepared_sources', simulated_branch)


if __name__ == '__main__':
    unittest.main()
