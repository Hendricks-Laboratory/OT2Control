"""Hardware-free tests for the Stage 4 exact Pi transfer preflight."""

import ast
import copy
import json
import math
import pathlib
import unittest
from types import SimpleNamespace

import numpy as np


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
        '_get_auto_completed_well_mix_p300_arm',
        '_count_available_tips',
        '_validate_transfer_plan_preflight_request',
        '_validate_targeted_mix_preflight_plan',
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
            body=[
                ast.Assign(
                    targets=[ast.Name(
                        id='AUTO_COMPLETED_WELL_MIX_MAX_FRACTION',
                        ctx=ast.Store()
                    )],
                    value=ast.Constant(value=0.50)
                ),
                ast.Assign(
                    targets=[ast.Name(
                        id='AUTO_COMPLETED_WELL_MIX_MIN_VOLUME_UL',
                        ctx=ast.Store()
                    )],
                    value=ast.Constant(value=5.0)
                ),
                ast.Assign(
                    targets=[ast.Name(
                        id='AUTO_COMPLETED_WELL_MIX_MAX_CYCLES',
                        ctx=ast.Store()
                    )],
                    value=ast.Constant(value=10)
                ),
                ast.Assign(
                    targets=[ast.Name(
                        id='AUTO_COMPLETED_WELL_MIX_REQUIRED_PIPETTE_VOLUME_UL',
                        ctx=ast.Store()
                    )],
                    value=ast.Constant(value=300.0)
                )
            ] + methods,
            decorator_list=[]
        )],
        type_ignores=[]
    ))
    namespace = {'math': math}
    exec(compile(module, str(ROBOT_SOURCE), 'exec'), namespace)
    return namespace['OT2Robot']


def _multicontainer_availability_class():
    '''Loads only MultiContainer's availability property without hardware.'''
    source_tree = ast.parse(ROBOT_SOURCE.read_text())
    multicontainer_class = next(
        node for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'MultiContainer'
    )
    availability_property = next(
        node for node in multicontainer_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'aspiratible_vol'
    )
    test_class = ast.ClassDef(
        name='MultiContainerAvailability',
        bases=[],
        keywords=[],
        body=[copy.deepcopy(availability_property)],
        decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(
        body=[test_class],
        type_ignores=[]
    ))
    namespace = {}
    exec(compile(module, str(ROBOT_SOURCE), 'exec'), namespace)
    return namespace['MultiContainerAvailability']


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

    def _build_robot(self, left_tip_count=8, right_tip_count=8):
        robot = self.Robot()
        robot.pipettes = {
            'left': {
                'size': 300.0,
                'last_used': 'clean',
                'pipette': _PipetteStub(unused_tip_count=left_tip_count),
                'tip_rack_deck_positions': [9],
                'tip_rack_names': ['tip_rack_300uL']
            },
            'right': {
                'size': 20.0,
                'last_used': 'clean',
                'pipette': _PipetteStub(unused_tip_count=right_tip_count),
                'tip_rack_deck_positions': [8],
                'tip_rack_names': ['tip_rack_20uL']
            }
        }
        robot.containers = {
            'reagent_aC1.0': _MultiContainerStub([
                _ContainerStub('A1', np.int64(3), 258.0, 250.0),
                _ContainerStub('A2', np.int64(3), 400.0, 250.0)
            ]),
            'reagent_bC1.0': _ContainerStub(
                'B1', np.int64(3), 400.0, 250.0
            )
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

    @classmethod
    def _targeted_mix_request(cls):
        request = cls._request()
        request['schema_version'] = 2
        request['targeted_mix_plan'] = [{
            'wellname': 'autowell0C1.0',
            'trigger_chemical_name': 'reagent_bC1.0',
            'mix_volume_uL': 20.0,
            'cycle_count': 1,
            'expected_well_volume_uL': 200.0
        }]
        return request

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

    def test_legacy_schema_one_response_remains_unchanged(self):
        result = self._build_robot()._build_transfer_plan_preflight(
            self._request()
        )

        self.assertEqual(1, result['schema_version'])
        self.assertNotIn('targeted_mix_requirements', result)

    def test_targeted_mix_tips_are_interleaved_and_audited_without_mutation(self):
        '''A dirty P300 requires a mix tip plus its clean replacement.'''
        robot = self._build_robot()
        left_pipette = robot.pipettes['left']['pipette']
        before = (left_pipette.has_tip, robot.pipettes['left']['last_used'])

        result = robot._build_transfer_plan_preflight(
            self._targeted_mix_request()
        )

        self.assertTrue(result['passed'])
        self.assertEqual(2, result['schema_version'])
        targeted_requirements = result['targeted_mix_requirements']
        self.assertEqual(1, len(targeted_requirements))
        self.assertEqual({
            'wellname': 'autowell0C1.0',
            'trigger_chemical_name': 'reagent_bC1.0',
            'pipette_arm': 'left',
            'mix_volume_uL': 20.0,
            'cycle_count': 1,
            'required_new_tips': 2,
            'post_mix_tip_policy': 'dedicated_discarded_with_clean_replacement'
        }, targeted_requirements[0])
        left_requirement = next(
            requirement for requirement in result['tip_requirements']
            if requirement['pipette_arm'] == 'left'
        )
        self.assertEqual(2, left_requirement['required_new_tips'])
        self.assertEqual(before, (
            left_pipette.has_tip,
            robot.pipettes['left']['last_used']
        ))

    def test_targeted_mix_tip_shortage_blocks_before_transfer(self):
        robot = self._build_robot(left_tip_count=1)

        result = robot._build_transfer_plan_preflight(
            self._targeted_mix_request()
        )

        self.assertFalse(result['passed'])
        left_shortage = next(
            deficit for deficit in result['deficits']
            if (deficit['deficit_type'] == 'tip_inventory'
                and deficit['pipette_arm'] == 'left')
        )
        self.assertEqual(2, left_shortage['required_new_tips'])
        self.assertEqual(1, left_shortage['available_new_tips'])

    def test_targeted_mix_preflight_rejects_non_p300_configuration(self):
        robot = self._build_robot()
        robot.pipettes['left']['size'] = 20.0
        robot.pipettes['right']['size'] = 1000.0

        result = robot._build_transfer_plan_preflight(
            self._targeted_mix_request()
        )

        self.assertFalse(result['passed'])
        self.assertTrue(any(
            deficit['deficit_type'] == 'tip_state'
            and 'requires exactly one configured P300' in deficit['message']
            for deficit in result['deficits']
        ))

    def test_invalid_targeted_mix_plan_returns_versioned_rejection(self):
        '''A stale or mismatched trigger can never be silently preflighted.'''
        robot = self._build_robot()
        request = self._targeted_mix_request()
        request['targeted_mix_plan'][0]['trigger_chemical_name'] = (
            'missing_triggerC1.0'
        )

        result = robot._build_transfer_plan_preflight(request)

        self.assertFalse(result['passed'])
        self.assertEqual(2, result['schema_version'])
        self.assertEqual([], result['targeted_mix_requirements'])
        self.assertEqual('invalid_request', result['deficits'][0]['deficit_type'])

    def test_multicontainer_aggregate_ignores_sub_dead_source_deficits(self):
        """An empty primary cannot reduce a usable backup's inventory."""
        availability_class = _multicontainer_availability_class()
        container = availability_class()
        container._cont_i = 0
        container.cont_list = [
            SimpleNamespace(aspiratible_vol=-4995.0),
            SimpleNamespace(aspiratible_vol=10764.0)
        ]

        self.assertEqual(container.aspiratible_vol, 10764.0)

    def test_preflight_result_is_json_serializable_with_numpy_deck_position(self):
        robot = self._build_robot()

        result = robot._build_transfer_plan_preflight(self._request())

        self.assertTrue(result['passed'])
        self.assertIsInstance(
            result['source_containers'][0]['source_deck_pos'],
            int
        )
        json.dumps(result, sort_keys=True)

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

    def test_tip_preflight_honors_configured_rack_suffix(self):
        '''A rack declared to begin at H12 has no replacement tips after H12.'''
        robot = self._build_robot(right_tip_count=95)
        right_pipette = robot.pipettes['right']['pipette']

        # The one configured H12 tip is already attached after initialization.
        # Other logical rack wells may still say has_tip=True, but were
        # declared physically unavailable by first_usable=H12.
        robot.pipettes['right']['configured_tip_wells'] = [
            _WellStub(has_tip=False)
        ]
        before = (right_pipette.has_tip, robot.pipettes['right']['last_used'])

        result = robot._build_transfer_plan_preflight(self._request())

        right_requirement = next(
            requirement for requirement in result['tip_requirements']
            if requirement['pipette_arm'] == 'right'
        )
        self.assertFalse(result['passed'])
        self.assertEqual(0, right_requirement['available_new_tips'])
        self.assertGreater(right_requirement['required_new_tips'], 0)
        self.assertEqual([8], right_requirement['tip_rack_deck_positions'])
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
