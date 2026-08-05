'''Source-level Stage 9A/9C checks without importing hardware modules.'''

import ast
import os
import unittest


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


class AutoPreparationControllerContractTests(unittest.TestCase):
    '''Guard the staged, fail-closed grouped-preparation controller boundary.'''

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
            '_execute_auto_preparation_phase',
            '_build_auto_preparation_reservation_request',
            '_validate_auto_preparation_group_reservation',
            '_request_auto_preparation_group_reservation',
            '_build_auto_preparation_execution_request',
            '_validate_auto_preparation_group_execution',
            '_request_auto_preparation_group_execution'
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

    def test_phase_reserves_then_executes_before_source_activation(self):
        phase_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_execute_auto_preparation_phase']
        )
        self.assertIn('_build_auto_preparation_reservation_request(', phase_source)
        self.assertIn('_request_auto_preparation_group_reservation(', phase_source)
        self.assertIn('_build_auto_preparation_execution_request(', phase_source)
        self.assertIn('_request_auto_preparation_group_execution(', phase_source)
        self.assertNotIn('_execute_auto_preparation_entry(', phase_source)
        self.assertIn('_activate_auto_prepared_sources(', phase_source)
        self.assertNotIn('execute_protocol_df(', phase_source)

    def test_execution_contract_is_versioned_and_journaled_before_activation(self):
        phase_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_execute_auto_preparation_phase']
        )
        request_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_build_auto_preparation_reservation_request']
        )
        validation_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_validate_auto_preparation_group_reservation']
        )
        execution_validation = ast.get_source_segment(
            self.source,
            self.auto_methods['_validate_auto_preparation_group_execution']
        )
        self.assertIn("'schema_version': 1", request_source)
        self.assertIn("'expected_source_inventory_revision'", request_source)
        self.assertIn("'auto_preparation_groups_reserved'", validation_source)
        self.assertIn("'stock_uses_temperature_module'", validation_source)
        self.assertIn("'physical_execution_started'", execution_validation)
        self.assertIn("'auto_preparation_groups_executed'", execution_validation)
        self.assertIn('_record_auto_live_run_event(', phase_source)


if __name__ == '__main__':
    unittest.main()
