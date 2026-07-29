"""Hardware-free contract tests for the Auto-main compatibility snapshot."""

import ast
import copy
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROBOT_SOURCE = REPO_ROOT / 'ot2_robot.py'


def _ot2robot_class_node():
    source_tree = ast.parse(ROBOT_SOURCE.read_text())
    for node in source_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'OT2Robot':
            return node
    raise AssertionError('OT2Robot class was not found in ot2_robot.py')


def _class_assignment_value(class_node, name):
    for node in class_node.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError('OT2Robot.{} was not found'.format(name))


def _class_method_node(class_node, name):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError('OT2Robot.{} was not found'.format(name))


class AutoMainStateSnapshotTests(unittest.TestCase):
    """Verify only pure source-level snapshot behavior; do not import hardware."""

    def setUp(self):
        self.class_node = _ot2robot_class_node()

    def test_snapshot_reports_calibrated_tares_and_read_only_metadata(self):
        method = copy.deepcopy(
            _class_method_node(self.class_node, '_build_robot_state_snapshot')
        )
        method.decorator_list = []
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[method], type_ignores=[]
        )), str(ROBOT_SOURCE), 'exec'), namespace)

        tare_map = _class_assignment_value(
            self.class_node, 'TARE_CALIBRATION_G'
        )
        stub = type('RobotStub', (), {
            'AUTO_MAIN_PROTOCOL_VERSION': _class_assignment_value(
                self.class_node, 'AUTO_MAIN_PROTOCOL_VERSION'
            ),
            'TARE_CALIBRATION_ID': _class_assignment_value(
                self.class_node, 'TARE_CALIBRATION_ID'
            ),
            'TARE_CALIBRATION_G': tare_map,
            'simulate': True,
            'containers': {'reagent': object()},
            'pipettes': {'left': object()},
            'temp_module': None
        })()

        snapshot = namespace['_build_robot_state_snapshot'](stub)

        self.assertEqual(1, snapshot['snapshot_schema_version'])
        self.assertEqual('Auto-main', snapshot['runtime_role'])
        self.assertEqual('auto-main-state-v1', snapshot['protocol_version'])
        self.assertEqual('ot2control_tube_tares_2026_07_v1',
                         snapshot['tare_calibration_id'])
        self.assertEqual({
            'tube_2ml': 1.7,
            'tube_15ml': 7.2731,
            'tube_50ml': 13.6950
        }, snapshot['tare_calibration_g'])
        self.assertEqual(['get_robot_state_snapshot'],
                         snapshot['supported_commands'])
        self.assertTrue(snapshot['simulate'])
        self.assertEqual(1, snapshot['container_count'])
        self.assertEqual(1, snapshot['pipette_count'])
        self.assertFalse(snapshot['temperature_module_initialized'])

        snapshot['tare_calibration_g']['tube_2ml'] = -1
        self.assertEqual(1.7, stub.TARE_CALIBRATION_G['tube_2ml'])

    def test_snapshot_command_is_registered_without_automatic_ready_reply(self):
        handler = _class_method_node(
            self.class_node, '_exec_get_robot_state_snapshot'
        )
        self.assertEqual(1, len(handler.decorator_list))
        decorator = handler.decorator_list[0]
        self.assertIsInstance(decorator, ast.Call)
        self.assertEqual('exec_func', decorator.func.id)
        self.assertEqual('get_robot_state_snapshot',
                         ast.literal_eval(decorator.args[0]))
        self.assertFalse(ast.literal_eval(decorator.args[2]))

        send_calls = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'send_pack'
        ]
        self.assertEqual(1, len(send_calls))
        self.assertEqual('robot_state_snapshot',
                         ast.literal_eval(send_calls[0].args[0]))


if __name__ == '__main__':
    unittest.main()
