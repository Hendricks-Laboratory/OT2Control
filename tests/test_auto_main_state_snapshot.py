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


def _load_robot_methods(method_names):
    '''Extracts pure state methods without importing Opentrons dependencies.'''
    class_node = _ot2robot_class_node()
    methods = []
    for method_name in method_names:
        method = copy.deepcopy(_class_method_node(class_node, method_name))
        method.decorator_list = []
        methods.append(method)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(
            name='OT2Robot',
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[]
        )], type_ignores=[]
    )), str(ROBOT_SOURCE), 'exec'), namespace)
    return namespace['OT2Robot']


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
            'temp_module': None,
            'source_inventory_revision': 0
        })()

        snapshot = namespace['_build_robot_state_snapshot'](stub)

        self.assertEqual(1, snapshot['snapshot_schema_version'])
        self.assertEqual('Auto-main', snapshot['runtime_role'])
        self.assertEqual('auto-main-state-v3', snapshot['protocol_version'])
        self.assertEqual('ot2control_tube_tares_2026_07_v1',
                         snapshot['tare_calibration_id'])
        self.assertEqual({
            'tube_2ml': 1.7,
            'tube_15ml': 7.2731,
            'tube_50ml': 13.6950
        }, snapshot['tare_calibration_g'])
        self.assertEqual([
            'get_robot_state_snapshot',
            'preflight_transfer_plan',
            'refresh_source_container_mass'
        ], snapshot['supported_commands'])
        self.assertEqual(0, snapshot['source_inventory_revision'])
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

    def test_snapshot_packet_types_are_ghost_messages(self):
        source_tree = ast.parse(
            (REPO_ROOT / 'Armchair' / 'armchair.py').read_text()
        )
        armchair_class = next(
            node for node in source_tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'Armchair'
        )
        assignments = {
            target.id: node.value
            for node in armchair_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        packet_types = ast.literal_eval(assignments['PACK_TYPES'].args[0])
        ghost_types = ast.literal_eval(assignments['GHOST_TYPES'])

        self.assertEqual(b'\x11', packet_types['get_robot_state_snapshot'])
        self.assertEqual(b'\x12', packet_types['robot_state_snapshot'])
        self.assertEqual(b'\x13', packet_types['preflight_transfer_plan'])
        self.assertEqual(b'\x14', packet_types['transfer_plan_preflight'])
        self.assertEqual(b'\x15', packet_types['refresh_source_container_mass'])
        self.assertEqual(b'\x16', packet_types['source_container_mass_refreshed'])
        self.assertIn('get_robot_state_snapshot', ghost_types)
        self.assertIn('robot_state_snapshot', ghost_types)
        self.assertIn('preflight_transfer_plan', ghost_types)
        self.assertIn('transfer_plan_preflight', ghost_types)
        self.assertIn('refresh_source_container_mass', ghost_types)
        self.assertIn('source_container_mass_refreshed', ghost_types)


class AutoMainSourceMassRefreshTests(unittest.TestCase):
    '''Exercise the same-container inventory update without robot imports.'''

    @classmethod
    def setUpClass(cls):
        cls.Robot = _load_robot_methods([
            '_get_preflight_source_containers',
            '_get_source_container_for_mass_refresh',
            '_get_source_mass_refresh_specification',
            '_validate_source_mass_refresh_request',
            '_build_source_group_mass_refresh_summary',
            '_build_source_container_mass_refresh'
        ])

    def _build_robot(self):
        tube_class = type('Tube2000uL', (), {})
        tube = tube_class()
        tube.name = 'reagent_aC1.0'
        tube.loc = 'A1'
        tube.deck_pos = 3
        tube.vol = 300.0
        tube.mass = 0.29995185
        tube.DEAD_VOL = 250.0
        tube.MAX_VOL = 1500.0
        tube.height_update_calls = 0

        def update_height():
            tube.height_update_calls += 1
        tube._update_height = update_height

        multi_source = type('MultiContainerStub', (), {})()
        multi_source.cont_list = [tube]
        multi_source._cont_i = 0
        multi_source.cont = tube

        robot = self.Robot()
        robot.TARE_CALIBRATION_G = {
            'tube_2ml': 1.7,
            'tube_15ml': 7.2731,
            'tube_50ml': 13.6950
        }
        robot.source_inventory_revision = 4
        robot.containers = {'reagent_aC1.0': multi_source}

        def preflight_number(value, label, minimum=None):
            numeric_value = float(value)
            if minimum is not None and numeric_value < minimum:
                raise ValueError('{} is below its allowed minimum.'.format(label))
            return numeric_value
        robot._preflight_number = preflight_number
        return robot, tube

    @staticmethod
    def _request(revision=4, mass=2.6998395):
        return {
            'schema_version': 1,
            'action_id': 'test-action-1',
            'expected_source_inventory_revision': revision,
            'source_chemical_name': 'reagent_aC1.0',
            'source_container_index': 0,
            'source_loc': 'A1',
            'source_deck_pos': 3,
            'measured_total_mass_g': mass
        }

    def test_same_container_refresh_updates_only_measured_inventory(self):
        robot, tube = self._build_robot()

        result = robot._build_source_container_mass_refresh(self._request())

        self.assertTrue(result['accepted'])
        self.assertEqual(
            'source_container_mass_refreshed',
            result['record_type']
        )
        self.assertEqual(5, result['source_inventory_revision'])
        self.assertAlmostEqual(1000.0, tube.vol, places=6)
        self.assertAlmostEqual(0.9998395, tube.mass, places=7)
        self.assertEqual(1, tube.height_update_calls)
        self.assertEqual('A1', result['source_container']['source_loc'])
        self.assertAlmostEqual(
            750.0,
            result['source_container']['source_group_aspirable_volume_uL'],
            places=6
        )

    def test_stale_or_mismatched_request_rejects_without_mutation(self):
        for request in (
                self._request(revision=3),
                dict(self._request(), source_loc='A2')):
            robot, tube = self._build_robot()
            result = robot._build_source_container_mass_refresh(request)

            self.assertFalse(result['accepted'])
            self.assertEqual(4, result['source_inventory_revision'])
            self.assertEqual(300.0, tube.vol)
            self.assertEqual(0, tube.height_update_calls)

    def test_refresh_command_is_a_structured_ghost_acknowledgement(self):
        handler = _class_method_node(
            _ot2robot_class_node(),
            '_exec_refresh_source_container_mass'
        )
        self.assertEqual(1, len(handler.decorator_list))
        decorator = handler.decorator_list[0]
        self.assertFalse(ast.literal_eval(decorator.args[2]))
        self.assertEqual(
            'refresh_source_container_mass',
            ast.literal_eval(decorator.args[0])
        )
        send_calls = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'send_pack'
        ]
        self.assertEqual(1, len(send_calls))
        self.assertEqual(
            'source_container_mass_refreshed',
            ast.literal_eval(send_calls[0].args[0])
        )


if __name__ == '__main__':
    unittest.main()
