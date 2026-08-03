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


def _labware_class_node():
    source_tree = ast.parse(ROBOT_SOURCE.read_text())
    for node in source_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Labware':
            return node
    raise AssertionError('Labware class was not found in ot2_robot.py')


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
    namespace = {'copy': copy}
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
            'source_inventory_revision': 0,
            'tip_inventory_revision': 0
        })()

        snapshot = namespace['_build_robot_state_snapshot'](stub)

        self.assertEqual(1, snapshot['snapshot_schema_version'])
        self.assertEqual('Auto-main', snapshot['runtime_role'])
        self.assertEqual('auto-main-state-v5', snapshot['protocol_version'])
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
                'refresh_source_container_mass',
                'reset_pipette_tip_racks',
                'register_auto_plate_generation'
            ], snapshot['supported_commands'])
        self.assertEqual(0, snapshot['source_inventory_revision'])
        self.assertEqual(0, snapshot['tip_inventory_revision'])
        self.assertEqual(0, snapshot['plate_mapping_revision'])
        self.assertEqual(0, snapshot['plate_generation'])
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
        self.assertEqual(b'\x17', packet_types['reset_pipette_tip_racks'])
        self.assertEqual(b'\x18', packet_types['pipette_tip_racks_reset'])
        self.assertEqual(b'\x19', packet_types['register_auto_plate_generation'])
        self.assertEqual(b'\x1A', packet_types['auto_plate_generation_registered'])
        self.assertIn('get_robot_state_snapshot', ghost_types)
        self.assertIn('robot_state_snapshot', ghost_types)
        self.assertIn('preflight_transfer_plan', ghost_types)
        self.assertIn('transfer_plan_preflight', ghost_types)
        self.assertIn('refresh_source_container_mass', ghost_types)
        self.assertIn('source_container_mass_refreshed', ghost_types)
        self.assertIn('reset_pipette_tip_racks', ghost_types)
        self.assertIn('pipette_tip_racks_reset', ghost_types)
        self.assertIn('register_auto_plate_generation', ghost_types)
        self.assertIn('auto_plate_generation_registered', ghost_types)


class AutoMainLabwareIdentityTests(unittest.TestCase):
    '''Ensure Auto aliases do not mutate read-only Opentrons labware names.'''

    def test_configured_logical_name_is_wrapper_owned(self):
        robot_class = _ot2robot_class_node()
        add_to_deck = _class_method_node(robot_class, '_add_to_deck')
        assignments = [
            node for node in ast.walk(add_to_deck)
            if isinstance(node, ast.Assign)
        ]

        assigned_attributes = [
            target.attr
            for assignment in assignments
            for target in assignment.targets
            if isinstance(target, ast.Attribute)
        ]
        self.assertIn('_logical_name', assigned_attributes)
        self.assertNotIn('name', assigned_attributes)

    def test_labware_name_prefers_logical_alias_without_mutating_api_object(self):
        labware_class = _labware_class_node()
        init_method = copy.deepcopy(
            next(
                node for node in labware_class.body
                if isinstance(node, ast.FunctionDef) and node.name == '__init__'
            )
        )
        name_property = copy.deepcopy(
            next(
                node for node in labware_class.body
                if isinstance(node, ast.FunctionDef) and node.name == 'name'
            )
        )
        name_property.decorator_list = [
            decorator for decorator in name_property.decorator_list
            if isinstance(decorator, ast.Name) and decorator.id == 'property'
        ]
        module = ast.fix_missing_locations(ast.Module(
            body=[ast.ClassDef(
                name='LabwareIdentity',
                bases=[],
                keywords=[],
                body=[init_method, name_property],
                decorator_list=[]
            )],
            type_ignores=[]
        ))
        namespace = {}
        exec(compile(module, str(ROBOT_SOURCE), 'exec'), namespace)
        api_labware = type('ApiLabware', (), {'name': 'plate_reader_4'})()
        wrapper = namespace['LabwareIdentity'](api_labware, 4)

        self.assertEqual('plate_reader_4', wrapper.name)
        wrapper._logical_name = 'platereader4'
        self.assertEqual('platereader4', wrapper.name)
        self.assertEqual('plate_reader_4', api_labware.name)


class AutoMainPlateGenerationTests(unittest.TestCase):
    '''Verify a plate reset mutates cursors only after full validation.'''

    @classmethod
    def setUpClass(cls):
        cls.Robot = _load_robot_methods([
            '_validate_auto_plate_generation_request',
            '_build_auto_plate_generation_registration'
        ])

    def _robot(self):
        robot = self.Robot()
        robot.plate_mapping_revision = 4
        robot.plate_generation = 2
        robot._get_auto_plate_wrappers = lambda: {
            'platereader4': type('Plate', (), {
                'current_well': 7,
                'full': False,
                'labware': type('Labware', (), {'wells': lambda self: [0] * 96})(),
                'get_well_cursor_index': lambda self, well: 12 if well else 96
            })(),
            'platereader7': type('Plate', (), {
                'current_well': 9,
                'full': False,
                'labware': type('Labware', (), {'wells': lambda self: [0] * 96})(),
                'get_well_cursor_index': lambda self, well: 15 if well else 96
            })()
        }
        return robot

    def _request(self, revision=4, generation=3):
        return {
            'schema_version': 1,
            'action_id': 'plate-3-A1',
            'expected_plate_mapping_revision': revision,
            'plate_generation': generation,
            'start_well': 'A1',
            'plate_cursors': {'platereader4': 'A1', 'platereader7': None}
        }

    def test_registration_changes_only_plate_cursor_revisions(self):
        robot = self._robot()
        response = robot._build_auto_plate_generation_registration(
            self._request()
        )
        self.assertTrue(response['accepted'])
        self.assertEqual(5, robot.plate_mapping_revision)
        self.assertEqual(3, robot.plate_generation)

    def test_stale_or_malformed_registration_is_rejected_without_mutation(self):
        for request in (self._request(revision=3), {'bad': 'request'}):
            robot = self._robot()
            response = robot._build_auto_plate_generation_registration(request)
            self.assertFalse(response['accepted'])
            self.assertEqual(4, robot.plate_mapping_revision)
            self.assertEqual(2, robot.plate_generation)


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


class _TipWellStub:
    def __init__(self, name):
        self.name = name


class _TipRackResetStub:
    def __init__(self):
        self._wells = {
            '{}{}'.format(row, column): _TipWellStub(
                '{}{}'.format(row, column)
            )
            for row in 'ABCDEFGH'
            for column in range(1, 13)
        }

    def well(self, name):
        return self._wells[name]


class _PipetteTipResetStub:
    def __init__(self, rack_count=2):
        self.tip_racks = [_TipRackResetStub() for _ in range(rack_count)]
        self.reset_calls = 0

    def reset_tipracks(self):
        self.reset_calls += 1


class AutoMainCompleteTipRackResetTests(unittest.TestCase):
    '''Verify only the safe, all-racks-per-pipette reset contract.'''

    @classmethod
    def setUpClass(cls):
        cls.Robot = _load_robot_methods([
            '_standard_96_tip_well_order',
            '_preflight_number',
            '_get_pipette_tip_rack_summary',
            '_validate_tip_rack_reset_request',
            '_build_pipette_tip_rack_reset'
        ])

    def _build_robot(self):
        robot = self.Robot()
        robot.tip_inventory_revision = 6
        robot._preflight_number = (
            lambda value, unused_label, minimum=0.0: float(value)
        )
        robot._standard_96_tip_well_order = lambda: [
            '{}{}'.format(row, column)
            for row in 'ABCDEFGH'
            for column in range(1, 13)
        ]
        pipette = _PipetteTipResetStub(rack_count=2)
        robot.pipettes = {
            'left': {
                'size': 300.0,
                'pipette': pipette,
                'configured_tip_wells': [],
                'tip_rack_deck_positions': [8, 11],
                'tip_rack_names': ['tip_rack_300uL', 'tip_rack_300uL']
            }
        }
        return robot, pipette

    @staticmethod
    def _request(revision=6, arm='left'):
        return {
            'schema_version': 1,
            'action_id': 'replace-all-left-racks',
            'expected_tip_inventory_revision': revision,
            'pipette_arm': arm
        }

    def test_reset_replaces_the_complete_registered_rack_set_only(self):
        robot, pipette = self._build_robot()

        result = robot._build_pipette_tip_rack_reset(self._request())

        self.assertTrue(result['accepted'])
        self.assertEqual('pipette_tip_racks_reset', result['record_type'])
        self.assertEqual(7, result['tip_inventory_revision'])
        self.assertEqual(1, pipette.reset_calls)
        self.assertEqual(
            [8, 11], result['tip_rack_summary']['tip_rack_deck_positions']
        )
        self.assertEqual(
            192, len(robot.pipettes['left']['configured_tip_wells'])
        )

    def test_stale_or_partial_reset_request_cannot_mutate_tip_state(self):
        for request in (
                self._request(revision=5),
                dict(self._request(), pipette_arm='right'),
                dict(self._request(), selected_deck_position=8)):
            robot, pipette = self._build_robot()
            before_wells = list(robot.pipettes['left']['configured_tip_wells'])

            result = robot._build_pipette_tip_rack_reset(request)

            self.assertFalse(result['accepted'])
            self.assertEqual(6, result['tip_inventory_revision'])
            self.assertEqual(0, pipette.reset_calls)
            self.assertEqual(
                before_wells, robot.pipettes['left']['configured_tip_wells']
            )

    def test_reset_command_is_a_structured_ghost_acknowledgement(self):
        handler = _class_method_node(
            _ot2robot_class_node(), '_exec_reset_pipette_tip_racks'
        )
        self.assertEqual(1, len(handler.decorator_list))
        decorator = handler.decorator_list[0]
        self.assertFalse(ast.literal_eval(decorator.args[2]))
        self.assertEqual(
            'reset_pipette_tip_racks', ast.literal_eval(decorator.args[0])
        )
        send_calls = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'send_pack'
        ]
        self.assertEqual(1, len(send_calls))
        self.assertEqual(
            'pipette_tip_racks_reset',
            ast.literal_eval(send_calls[0].args[0])
        )


if __name__ == '__main__':
    unittest.main()
