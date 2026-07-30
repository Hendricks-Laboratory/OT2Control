"""Hardware-free tests for the Stage 4 exact Pi transfer preflight."""

import ast
import copy
import math
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROBOT_SOURCE = REPO_ROOT / 'ot2_robot.py'


def _robot_preflight_class():
    source_tree = ast.parse(ROBOT_SOURCE.read_text())
    robot_class = next(
        node for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'OT2Robot'
    )
    method_names = {
        '_preflight_number',
        '_get_preferred_pipette_arm_for_sizes',
        '_count_available_tips',
        '_validate_transfer_plan_preflight_request',
        '_get_preflight_source_containers',
        '_simulate_preflight_source',
        '_simulate_preflight_tips',
        '_build_transfer_plan_preflight'
    }
    methods = []
    for node in robot_class.body:
        if isinstance(node, ast.FunctionDef) and node.name in method_names:
            method = copy.deepcopy(node)
            method.decorator_list = [
                decorator for decorator in method.decorator_list
                if isinstance(decorator, ast.Name)
                and decorator.id == 'staticmethod'
            ]
            methods.append(method)
    if len(methods) != len(method_names):
        raise AssertionError('A required Stage 4 preflight helper is absent.')
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(
            name='OT2Robot',
            bases=[],
            keywords=[],
            body=methods,
            decorator_list=[]
        )],
        type_ignores=[]
    ))
    namespace = {'math': math}
    exec(compile(module, str(ROBOT_SOURCE), 'exec'), namespace)
    return namespace['OT2Robot']


class _WellStub:
    def __init__(self, has_tip=True):
        self.has_tip = has_tip


class _TipRackStub:
    def __init__(self, unused_tip_count):
        self._wells = [_WellStub(True) for _ in range(unused_tip_count)]

    def wells(self):
        return list(self._wells)


class _PipetteStub:
    def __init__(self, has_tip=True, unused_tip_count=8):
        self.has_tip = has_tip
        self.tip_racks = [_TipRackStub(unused_tip_count)]


class _ContainerStub:
    def __init__(self, loc, deck_pos, vol, dead_volume):
        self.loc = loc
        self.deck_pos = deck_pos
        self.vol = vol
        self.DEAD_VOL = dead_volume


class _MultiContainerStub:
    def __init__(self, containers, active_index=0):
        self.cont_list = containers
        self._cont_i = active_index


class AutoMainTransferPreflightTests(unittest.TestCase):
    """Verify plan simulation without importing Opentrons or moving hardware."""

    @classmethod
    def setUpClass(cls):
        cls.Robot = _robot_preflight_class()

    def _build_robot(self, right_tip_count=8):
        robot = self.Robot()
        robot.pipettes = {
            'left': {
                'size': 300.0,
                'last_used': 'clean',
                'pipette': _PipetteStub(unused_tip_count=8)
            },
            'right': {
                'size': 20.0,
                'last_used': 'clean',
                'pipette': _PipetteStub(unused_tip_count=right_tip_count)
            }
        }
        robot.containers = {
            'reagent_aC1.0': _MultiContainerStub([
                _ContainerStub('A1', 3, 258.0, 250.0),
                _ContainerStub('A2', 3, 400.0, 250.0)
            ]),
            'reagent_bC1.0': _ContainerStub('B1', 3, 400.0, 250.0)
        }
        return robot

    @staticmethod
    def _request():
        return {
            'schema_version': 1,
            'batch_number': 4,
            'reserve_volume_uL': 10.0,
            'source_plan': [
                {
                    'source_chemical_name': 'reagent_aC1.0',
                    'transfer_steps': [{
                        'destination_name': 'autowell0C1.0',
                        'volume_uL': 20.0
                    }]
                },
                {
                    'source_chemical_name': 'reagent_bC1.0',
                    'transfer_steps': [{
                        'destination_name': 'autowell0C1.0',
                        'volume_uL': 50.0
                    }]
                }
            ]
        }

    def test_backup_is_selected_without_mutating_source_state(self):
        robot = self._build_robot()
        primary = robot.containers['reagent_aC1.0'].cont_list[0]
        backup = robot.containers['reagent_aC1.0'].cont_list[1]
        before = (primary.vol, backup.vol,
                  robot.containers['reagent_aC1.0']._cont_i)

        result = robot._build_transfer_plan_preflight(self._request())

        self.assertTrue(result['passed'])
        first_allocation = result['allocations'][0]
        self.assertEqual('A2', first_allocation['source_loc'])
        self.assertEqual(1, first_allocation['source_container_index'])
        self.assertEqual(before, (
            primary.vol,
            backup.vol,
            robot.containers['reagent_aC1.0']._cont_i
        ))

    def test_tip_shortage_is_rejected_without_mutating_tip_state(self):
        robot = self._build_robot(right_tip_count=0)
        right_pipette = robot.pipettes['right']['pipette']
        before = (right_pipette.has_tip, robot.pipettes['right']['last_used'])

        result = robot._build_transfer_plan_preflight(self._request())

        self.assertFalse(result['passed'])
        self.assertIn('tip_inventory', [
            deficit['deficit_type'] for deficit in result['deficits']
        ])
        self.assertEqual(before, (
            right_pipette.has_tip,
            robot.pipettes['right']['last_used']
        ))

    def test_reserve_boundary_is_rejected(self):
        robot = self._build_robot()
        request = self._request()
        request['reserve_volume_uL'] = 131.0

        result = robot._build_transfer_plan_preflight(request)

        self.assertFalse(result['passed'])
        self.assertIn('reserve_volume', [
            deficit['deficit_type'] for deficit in result['deficits']
        ])

    def test_invalid_payload_returns_structured_rejection(self):
        robot = self._build_robot()
        result = robot._build_transfer_plan_preflight({'bad': 'payload'})

        self.assertFalse(result['passed'])
        self.assertEqual('invalid_request', result['deficits'][0]['deficit_type'])

    def test_protocol_handler_is_ghost_and_returns_the_preflight_record(self):
        source_tree = ast.parse(ROBOT_SOURCE.read_text())
        robot_class = next(
            node for node in source_tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'OT2Robot'
        )
        handler = next(
            node for node in robot_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == '_exec_preflight_transfer_plan'
        )
        decorator = handler.decorator_list[0]
        self.assertEqual('exec_func', decorator.func.id)
        self.assertEqual('preflight_transfer_plan',
                         ast.literal_eval(decorator.args[0]))
        self.assertFalse(ast.literal_eval(decorator.args[2]))
        response_types = [
            ast.literal_eval(call.args[0])
            for call in ast.walk(handler)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == 'send_pack'
        ]
        self.assertEqual(['transfer_plan_preflight'], response_types)


if __name__ == '__main__':
    unittest.main()
