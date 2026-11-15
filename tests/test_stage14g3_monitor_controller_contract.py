'''Source-contract checks for the isolated Stage 14G3 monitor-only route.'''

from __future__ import print_function

import ast
import os
import unittest


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(__file__))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


def _read_controller():
    with open(CONTROLLER_PATH, encoding='utf-8') as handle:
        return handle.read()


def _class_method_source(source, class_name, method_name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == method_name:
                    return ast.get_source_segment(source, method)
    raise AssertionError('Missing {}.{}.'.format(class_name, method_name))


class Stage14G3MonitorControllerContractTests(unittest.TestCase):
    def test_launcher_uses_dedicated_monitor_controller(self):
        source = _read_controller()
        launcher = _class_method_source(
            source, 'AutoStabilityMonitorContr', '__init__'
        )
        self.assertIn('_stability_monitor_only_execution = True', launcher)
        self.assertIn('super().__init__', launcher)
        self.assertIn('auto_stability_monitor_only', source)
        self.assertIn("elif args.mode == 'stability_monitor'", source)

    def test_monitor_execution_never_constructs_or_updates_an_optimizer(self):
        source = _read_controller()
        runner = _class_method_source(
            source, 'AutoStabilityMonitorContr', '_run_monitor_only'
        )
        self.assertNotIn('OptimizationModel(', runner)
        self.assertNotIn('initialize_optimizer(', runner)
        self.assertNotIn('model.update_', runner)
        self.assertIn("'model_updates_permitted': False", runner)
        self.assertIn("condition_type='monitor_only'", runner)
        self.assertIn('LIFECYCLE_PROCESSING_BATCH', runner)
        self.assertIn("'batch_measurement_completed'", runner)
        self.assertIn('LIFECYCLE_READY_FOR_BATCH', runner)

    def test_normal_auto_is_rejected_before_connection(self):
        source = _read_controller()
        parser_method = _class_method_source(
            source, 'AutoContr',
            '_initialize_auto_stability_monitor_only_configuration'
        )
        self.assertIn('Use -m stability_monitor.', parser_method)
        self.assertIn('auto_preparation_mode=off', parser_method)
        self.assertIn('auto_stability_debug_mode=off', parser_method)

    def test_monitor_values_are_explicitly_audit_only(self):
        source = _read_controller()
        export_method = _class_method_source(
            source, 'AutoContr', '_build_labeled_auto_experiment_data_export'
        )
        self.assertIn("'auto_stability_monitor_only'", export_method)
        self.assertIn("'lambda_max_nm_audit_only'", export_method)

    def test_fixed_conditions_are_loaded_before_connection(self):
        source = _read_controller()
        loader = _class_method_source(
            source, 'AutoStabilityMonitorContr',
            '_initialize_auto_preparation_plan'
        )
        self.assertIn('g2d.download', loader)
        self.assertIn('build_stability_monitor_condition_manifest', loader)
        self.assertIn("'auto_preparation_mode', 'off'", loader)


if __name__ == '__main__':
    unittest.main()
