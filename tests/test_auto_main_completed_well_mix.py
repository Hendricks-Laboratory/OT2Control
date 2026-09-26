"""Hardware-free contract tests for the Stage 13B targeted Auto-well mix."""

import ast
import copy
from datetime import datetime, timezone
import math
import pathlib
import re
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROBOT_SOURCE = REPO_ROOT / 'ot2_robot.py'


class _WellBase(object):
    pass


class _Well96(_WellBase):
    pass


class _TargetWell(_Well96):
    def __init__(self, name='autowell7C1.0', deck_pos=4, loc='B3',
                 volume_uL=200.0, trigger='sodium_borohydrideC6.25'):
        self.name = name
        self.deck_pos = deck_pos
        self.loc = loc
        self.vol = volume_uL
        self.asp_height = 1.0
        self.history = [
            ('2026-09-22T10:00:00+00:00', 'WaterC1.0', 120.0),
            ('2026-09-22T10:00:02+00:00', trigger, 80.0)
        ]
        self._well = object()
        self.labware = object()
        self.targeted_mix_calls = []

    def get_well(self):
        return self._well

    def mix_targeted(self, pipette, mix_volume_uL, cycle_count):
        self.targeted_mix_calls.append((pipette, mix_volume_uL, cycle_count))


class _TipWell(object):
    def __init__(self, has_tip=True):
        self.has_tip = has_tip


class _Pipette(object):
    def __init__(self, has_tip=True, min_volume=1.0, max_volume=20.0):
        self.has_tip = has_tip
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.tip_racks = []
        self.raise_on_mix = False
        self.well_bottom_clearance = type(
            'Clearance', (), {'aspirate': None, 'dispense': None}
        )()
        self.calls = []

    def pick_up_tip(self):
        self.calls.append(('pick_up_tip',))
        self.has_tip = True

    def drop_tip(self):
        self.calls.append(('drop_tip',))
        self.has_tip = False

    def mix(self, cycles, volume_uL, well, rate):
        self.calls.append(('mix', cycles, volume_uL, well, rate))
        if self.raise_on_mix:
            raise RuntimeError('synthetic pipette mix fault')


class _PytzStub(object):
    utc = timezone.utc

    @staticmethod
    def timezone(unused_name):
        return timezone.utc


def _robot_mix_class():
    source_tree = ast.parse(ROBOT_SOURCE.read_text())
    robot_class = next(
        node for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'OT2Robot'
    )
    method_names = {
        '_preflight_number',
        '_get_preferred_pipette_arm_for_sizes',
        '_count_available_tips',
        '_validate_auto_completed_well_mix_request',
        '_build_auto_completed_well_mix_plan',
        '_execute_auto_completed_well_mix_plan'
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
        raise AssertionError('A required targeted-mix helper is absent.')
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
    namespace = {
        'datetime': datetime,
        'math': math,
        'pytz': _PytzStub,
        're': re,
        'Well': _WellBase,
        'Well96': _Well96
    }
    exec(compile(module, str(ROBOT_SOURCE), 'exec'), namespace)
    return namespace['OT2Robot']


def _well96_targeted_mix_class():
    source_tree = ast.parse(ROBOT_SOURCE.read_text())
    source_well96 = next(
        node for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'Well96'
    )
    members = []
    for node in source_well96.body:
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith('TARGETED_MIX_')):
            members.append(copy.deepcopy(node))
        elif (isinstance(node, ast.FunctionDef)
              and node.name == 'mix_targeted'):
            members.append(copy.deepcopy(node))
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(
            name='Well96', bases=[], keywords=[], body=members,
            decorator_list=[]
        )],
        type_ignores=[]
    ))
    namespace = {}
    exec(compile(module, str(ROBOT_SOURCE), 'exec'), namespace)
    return namespace['Well96']


class AutoMainCompletedWellMixTests(unittest.TestCase):
    """Exercise only stubs; no Opentrons import or robot movement occurs."""

    @classmethod
    def setUpClass(cls):
        cls.Robot = _robot_mix_class()

    def _robot(self, right_has_tip=True, right_clean=True,
               right_new_tip_count=2):
        robot = self.Robot()
        right_pipette = _Pipette(
            has_tip=right_has_tip, min_volume=20.0, max_volume=300.0
        )
        left_pipette = _Pipette(
            has_tip=True, min_volume=1.0, max_volume=20.0
        )
        robot.pipettes = {
            'left': {
                'size': 20.0,
                'last_used': 'clean',
                'pipette': left_pipette,
                'configured_tip_wells': [_TipWell() for _ in range(4)]
            },
            'right': {
                'size': 300.0,
                'last_used': 'clean' if right_clean else 'silver_nitrateC0.375',
                'pipette': right_pipette,
                'configured_tip_wells': [
                    _TipWell() for _ in range(right_new_tip_count)
                ]
            }
        }
        target = _TargetWell()
        robot.containers = {target.name: target}
        robot.lab_deck = [None] * 8
        robot.lab_deck[4] = type(
            'Plate', (), {'name': 'platereader4', 'labware': target.labware}
        )()
        robot.plate_mapping_revision = 5
        robot.plate_generation = 2
        robot.AUTO_COMPLETED_WELL_MIX_MAX_FRACTION = 0.50
        robot.AUTO_COMPLETED_WELL_MIX_MIN_VOLUME_UL = 5.0
        robot.AUTO_COMPLETED_WELL_MIX_MAX_CYCLES = 10
        robot.protocol = type('Protocol', (), {'_commands': []})()
        return robot, target, right_pipette

    @staticmethod
    def _request(**overrides):
        request = {
            'schema_version': 1,
            'action_id': 'stability-mix-0007',
            'wellname': 'autowell7C1.0',
            'expected_deck_pos': 4,
            'expected_loc': 'B3',
            'expected_plate_mapping_revision': 5,
            'expected_plate_generation': 2,
            'trigger_chemical_name': 'sodium_borohydrideC6.25',
            'mix_volume_uL': 100.0,
            'cycle_count': 3
        }
        request.update(overrides)
        return request

    def test_valid_plan_is_exactly_one_completed_autowell(self):
        robot, target, _ = self._robot()

        plan = robot._build_auto_completed_well_mix_plan(self._request())

        self.assertIs(target, plan['target'])
        self.assertEqual('right', plan['pipette_arm'])
        self.assertTrue(plan['use_existing_clean_tip'])
        self.assertEqual(1, plan['required_new_tips'])
        self.assertEqual(2, plan['available_new_tips'])
        self.assertEqual(100.0, plan['maximum_mix_volume_uL'])
        self.assertEqual(20.0, plan['pipette_min_volume_uL'])
        self.assertEqual(300.0, plan['pipette_max_volume_uL'])
        self.assertEqual(200.0, target.vol)

    def test_targeted_mix_uses_preparation_pipette_policy_not_transfer_policy(self):
        robot, _, _ = self._robot()

        # Existing reagent transfers retain their validated small-volume
        # routing: 20 uL selects the left-mount P20 on Eve.
        self.assertEqual(
            'left',
            robot._get_preferred_pipette_arm_for_sizes(
                20.0, {'left': 20.0, 'right': 300.0}
            )
        )
        # A completed-well mix instead follows the existing preparation
        # policy, which asks the larger P300 to mix even for a smaller valid
        # mix aliquot.
        plan = robot._build_auto_completed_well_mix_plan(
            self._request(mix_volume_uL=20.0)
        )
        self.assertEqual('right', plan['pipette_arm'])

    def test_bad_identity_or_uncompleted_target_is_rejected_before_motion(self):
        cases = [
            self._request(wellname='WaterC1.0'),
            self._request(expected_loc='A1'),
            self._request(expected_plate_mapping_revision=4),
            self._request(mix_volume_uL=101.0),
            self._request(cycle_count=11)
        ]
        for request in cases:
            robot, target, pipette = self._robot()
            with self.subTest(request=request):
                with self.assertRaises(ValueError):
                    robot._build_auto_completed_well_mix_plan(request)
                self.assertEqual([], pipette.calls)
                self.assertEqual(200.0, target.vol)

        robot, target, pipette = self._robot()
        target.history[-1] = (
            target.history[-1][0], 'silver_nitrateC0.375', target.history[-1][2]
        )
        with self.assertRaisesRegex(ValueError, 'final recorded transfer'):
            robot._build_auto_completed_well_mix_plan(self._request())
        self.assertEqual([], pipette.calls)

        robot, target, pipette = self._robot()
        with self.assertRaisesRegex(ValueError, 'below the selected pipette'):
            robot._build_auto_completed_well_mix_plan(
                self._request(mix_volume_uL=19.0)
            )
        self.assertEqual([], pipette.calls)
        self.assertEqual(200.0, target.vol)

        robot, _, pipette = self._robot()
        pipette.max_volume = 200.0
        with self.assertRaisesRegex(ValueError, 'configuration disagrees'):
            robot._build_auto_completed_well_mix_plan(self._request())
        self.assertEqual([], pipette.calls)

    def test_non_96_well_or_mismatched_registered_labware_is_rejected(self):
        robot, target, pipette = self._robot()
        target.__class__ = _WellBase
        with self.assertRaisesRegex(ValueError, '96-well reaction well'):
            robot._build_auto_completed_well_mix_plan(self._request())
        self.assertEqual([], pipette.calls)

        robot, _, pipette = self._robot()
        robot.lab_deck[4].labware = object()
        with self.assertRaisesRegex(ValueError, 'not bound'):
            robot._build_auto_completed_well_mix_plan(self._request())
        self.assertEqual([], pipette.calls)

    def test_tip_shortage_rejects_without_mutating_robot_state(self):
        robot, target, pipette = self._robot(
            right_has_tip=False, right_clean=False, right_new_tip_count=1
        )
        before = (pipette.has_tip, robot.pipettes['right']['last_used'])

        with self.assertRaisesRegex(ValueError, 'needs 2 unused right tip'):
            robot._build_auto_completed_well_mix_plan(self._request())

        self.assertEqual(before, (
            pipette.has_tip, robot.pipettes['right']['last_used']
        ))
        self.assertEqual([], pipette.calls)
        self.assertEqual(200.0, target.vol)

    def test_execution_uses_one_target_and_discards_dedicated_mix_tip(self):
        robot, target, pipette = self._robot()
        plan = robot._build_auto_completed_well_mix_plan(self._request())

        robot._execute_auto_completed_well_mix_plan(plan)

        self.assertEqual([
            ('drop_tip',),
            ('pick_up_tip',)
        ], pipette.calls)
        self.assertEqual([
            (pipette, 100.0, 3)
        ], target.targeted_mix_calls)
        self.assertTrue(pipette.has_tip)
        self.assertEqual('clean', robot.pipettes['right']['last_used'])
        self.assertEqual(200.0, target.vol)
        self.assertEqual(1, len(robot.protocol._commands))

    def test_dirty_tip_requires_two_fresh_tips_and_never_reuses_it_for_mix(self):
        robot, _, pipette = self._robot(
            right_has_tip=True, right_clean=False, right_new_tip_count=2
        )
        plan = robot._build_auto_completed_well_mix_plan(self._request())

        self.assertTrue(plan['requires_fresh_mix_tip'])
        self.assertEqual(2, plan['required_new_tips'])
        robot._execute_auto_completed_well_mix_plan(plan)

        self.assertEqual('drop_tip', pipette.calls[0][0])
        self.assertEqual('pick_up_tip', pipette.calls[1][0])
        self.assertEqual([
            (pipette, 100.0, 3)
        ], robot.containers['autowell7C1.0'].targeted_mix_calls)
        self.assertEqual('drop_tip', pipette.calls[2][0])
        self.assertEqual('pick_up_tip', pipette.calls[3][0])

    def test_well96_targeted_mix_preserves_legacy_one_mm_geometry(self):
        source_tree = ast.parse(ROBOT_SOURCE.read_text())
        well96 = next(
            node for node in source_tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'Well96'
        )
        constants = {
            assignment.targets[0].id: ast.literal_eval(assignment.value)
            for assignment in well96.body
            if isinstance(assignment, ast.Assign)
            and isinstance(assignment.targets[0], ast.Name)
            and isinstance(assignment.value, ast.Constant)
        }
        self.assertEqual(1.0, constants['TARGETED_MIX_ASPIRATE_CLEARANCE_MM'])
        self.assertEqual(1.0, constants['TARGETED_MIX_DISPENSE_CLEARANCE_MM'])
        self.assertEqual(1.0, constants['TARGETED_MIX_RATE'])

        method = next(
            node for node in well96.body
            if isinstance(node, ast.FunctionDef) and node.name == 'mix_targeted'
        )
        mix_calls = [
            node for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'mix'
        ]
        self.assertEqual(1, len(mix_calls))
        self.assertEqual(1, ast.literal_eval(mix_calls[0].args[0]))
        disallowed_motion_calls = [
            node.func.attr for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ('move_to', 'touch_tip', 'blow_out')
        ]
        self.assertEqual([], disallowed_motion_calls)

    def test_well96_targeted_mix_repeats_one_cycle_and_restores_clearances(self):
        well = _well96_targeted_mix_class()()
        target_well = object()
        well.get_well = lambda: target_well
        pipette = _Pipette()
        pipette.well_bottom_clearance.aspirate = 7.0
        pipette.well_bottom_clearance.dispense = 9.0

        well.mix_targeted(pipette, 100.0, 3)

        self.assertEqual([
            ('mix', 1, 100.0, target_well, 1.0),
            ('mix', 1, 100.0, target_well, 1.0),
            ('mix', 1, 100.0, target_well, 1.0)
        ], pipette.calls)
        self.assertEqual(7.0, pipette.well_bottom_clearance.aspirate)
        self.assertEqual(9.0, pipette.well_bottom_clearance.dispense)

    def test_well96_targeted_mix_restores_clearances_when_pipette_faults(self):
        well = _well96_targeted_mix_class()()
        well.get_well = lambda: object()
        pipette = _Pipette()
        pipette.raise_on_mix = True
        pipette.well_bottom_clearance.aspirate = 7.0
        pipette.well_bottom_clearance.dispense = 9.0

        with self.assertRaisesRegex(RuntimeError, 'synthetic pipette mix fault'):
            well.mix_targeted(pipette, 100.0, 3)

        self.assertEqual(7.0, pipette.well_bottom_clearance.aspirate)
        self.assertEqual(9.0, pipette.well_bottom_clearance.dispense)


if __name__ == '__main__':
    unittest.main()
