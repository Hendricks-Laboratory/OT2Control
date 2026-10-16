"""Hardware-free C2d tests for calibration well binding/resource preflight."""

import ast
import copy
import math
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROBOT_SOURCE = REPO_ROOT / 'ot2_robot.py'


class _TipWell(object):
    def __init__(self, has_tip=True):
        self.has_tip = has_tip


class _Pipette(object):
    def __init__(self):
        self.has_tip = True


class _Container(object):
    def __init__(self, loc, deck_pos, volume_uL, dead_volume_uL):
        self.loc = loc
        self.deck_pos = deck_pos
        self.vol = volume_uL
        self.DEAD_VOL = dead_volume_uL


class _Well96(object):
    def __init__(self, name, deck_pos, loc, labware, vol=0, conc=1):
        self.name = name
        self.deck_pos = deck_pos
        self.loc = loc
        self.labware = labware
        self.vol = vol
        self.conc = conc


class _PlateLabware(object):
    def wells_by_name(self):
        return {'B3': object(), 'C3': object()}


class _PlateWrapper(object):
    name = 'platereader4'

    def __init__(self):
        self.labware = _PlateLabware()


def _resource_robot_class():
    """Extract only pure C2d methods; never import robot dependencies."""
    source_tree = ast.parse(ROBOT_SOURCE.read_text())
    robot_class = next(
        node for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'OT2Robot'
    )
    method_names = {
        '_preflight_number',
        '_get_preferred_pipette_arm_for_sizes',
        '_count_available_tips',
        '_get_preflight_source_containers',
        '_simulate_preflight_source',
        '_validate_auto_mix_calibration_binding_request',
        '_auto_mix_calibration_binding_response',
        '_bind_auto_mix_calibration_well',
        '_validate_auto_mix_calibration_resource_request',
        '_auto_mix_calibration_resource_response',
        '_get_auto_mix_calibration_resource_pipette',
        '_build_auto_mix_calibration_resource_preflight'
    }
    members = []
    for node in robot_class.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id.startswith('AUTO_MIX_CALIBRATION_')
                for target in node.targets):
            members.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name in method_names:
            method = copy.deepcopy(node)
            method.decorator_list = [
                decorator for decorator in method.decorator_list
                if isinstance(decorator, ast.Name)
                and decorator.id == 'staticmethod'
            ]
            members.append(method)
    if len([member for member in members if isinstance(member, ast.FunctionDef)]) != len(method_names):
        raise AssertionError('A required C2d helper is absent.')
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(
            name='OT2Robot', bases=[], keywords=[], body=members,
            decorator_list=[]
        )], type_ignores=[]
    ))
    namespace = {'math': math, 'Well96': _Well96}
    exec(compile(module, str(ROBOT_SOURCE), 'exec'), namespace)
    return namespace['OT2Robot']


class AutoMainMixCalibrationResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Robot = _resource_robot_class()

    @staticmethod
    def _binding_request():
        return {
            'schema_version': 1,
            'binding_id': 'calibration-a-binding',
            'calibration_id': 'calibration-a',
            'wellname': 'autowell-calibration-a',
            'reader_well': 'B3',
            'expected_deck_pos': 4,
            'expected_loc': 'B3',
            'expected_plate_mapping_revision': 0,
            'expected_plate_generation': 0,
            'expected_final_well_volume_uL': 200.0
        }

    @staticmethod
    def _resource_request(profile='water_safety_20ul_first'):
        first_volume, second_volume = {
            'water_safety_20ul_first': (20.0, 180.0),
            'qc_two_dye_5ul_red': (5.0, 195.0)
        }[profile]
        return {
            'schema_version': 1,
            'resource_preflight_id': 'calibration-a-resources',
            'binding_id': 'calibration-a-binding',
            'calibration_id': 'calibration-a',
            'assay_profile': profile,
            'wellname': 'autowell-calibration-a',
            'expected_deck_pos': 4,
            'expected_loc': 'B3',
            'expected_plate_mapping_revision': 0,
            'expected_plate_generation': 0,
            'source_requirements': [
                {
                    'slot': 'first',
                    'source_container': 'BaseC1.0',
                    'required_volume_uL': first_volume
                },
                {
                    'slot': 'second',
                    'source_container': 'TriggerC1.0',
                    'required_volume_uL': second_volume
                }
            ],
            'ordinary_transfer_tip_requirements': [
                {'pipette': 'p20', 'required_new_tips': 1},
                {'pipette': 'p300', 'required_new_tips': 1}
            ],
            'dedicated_mix_tip_requirements': [
                {'pipette': 'p300', 'required_new_tips': 11}
            ]
        }

    def _robot(self, p20_tip_count=1, p300_tip_count=12,
               base_volume_uL=500.0, trigger_volume_uL=500.0):
        robot = self.Robot()
        robot.containers = {
            'BaseC1.0': _Container('A1', 3, base_volume_uL, 50.0),
            'TriggerC1.0': _Container('A2', 3, trigger_volume_uL, 50.0)
        }
        robot.pipettes = {
            'left': {
                'size': 20.0,
                'pipette': _Pipette(),
                'configured_tip_wells': [_TipWell() for unused in range(
                    p20_tip_count
                )]
            },
            'right': {
                'size': 300.0,
                'pipette': _Pipette(),
                'configured_tip_wells': [_TipWell() for unused in range(
                    p300_tip_count
                )]
            }
        }
        robot.lab_deck = [None] * 13
        robot.lab_deck[4] = _PlateWrapper()
        robot.plate_mapping_revision = 0
        robot.plate_generation = 0
        return robot

    def test_binding_claims_one_vacant_location_without_resource_mutation(self):
        robot = self._robot()
        before = dict(robot.containers)

        binding = robot._bind_auto_mix_calibration_well(
            self._binding_request()
        )

        self.assertEqual('B3', binding['reader_well'])
        self.assertEqual(set(before).union({'autowell-calibration-a'}),
                         set(robot.containers))
        target = robot.containers['autowell-calibration-a']
        self.assertIsInstance(target, _Well96)
        self.assertEqual(0, target.vol)
        self.assertEqual('B3', target.loc)
        self.assertEqual(4, target.deck_pos)
        self.assertEqual(500.0, robot.containers['BaseC1.0'].vol)
        self.assertEqual(500.0, robot.containers['TriggerC1.0'].vol)

    def test_rejected_acknowledgements_expose_no_partial_resource_evidence(self):
        robot = self._robot()
        binding_ack = robot._auto_mix_calibration_binding_response(
            self._binding_request()
        )
        self.assertEqual({
            'schema_version', 'record_type', 'binding_id', 'calibration_id',
            'accepted', 'message', 'wellname', 'reader_well',
            'expected_deck_pos', 'expected_loc', 'plate_mapping_revision',
            'plate_generation', 'well_was_vacant', 'binding_created'
        }, set(binding_ack))
        resource_ack = robot._auto_mix_calibration_resource_response(
            self._resource_request()
        )
        self.assertFalse(resource_ack['accepted'])
        self.assertEqual([], resource_ack['source_requirements'])
        self.assertEqual([], resource_ack['ordinary_transfer_tip_requirements'])
        self.assertEqual([], resource_ack['dedicated_mix_tip_requirements'])

    def test_binding_rejects_duplicate_or_occupied_location_without_mutation(self):
        robot = self._robot()
        request = self._binding_request()
        robot._bind_auto_mix_calibration_well(request)
        before = dict(robot.containers)
        with self.assertRaisesRegex(ValueError, 'already'):
            robot._bind_auto_mix_calibration_well(request)
        self.assertEqual(before, robot.containers)

        duplicate_binding = dict(request)
        duplicate_binding.update({
            'calibration_id': 'calibration-b',
            'wellname': 'autowell-calibration-b',
            'reader_well': 'C3',
            'expected_loc': 'C3'
        })
        with self.assertRaisesRegex(ValueError, 'already'):
            robot._bind_auto_mix_calibration_well(duplicate_binding)
        self.assertEqual(before, robot.containers)

        occupied = self._robot()
        occupied.containers['autowell-existing'] = _Well96(
            'autowell-existing', 4, 'B3', occupied.lab_deck[4].labware
        )
        before = dict(occupied.containers)
        with self.assertRaisesRegex(ValueError, 'occupied'):
            occupied._bind_auto_mix_calibration_well(request)
        self.assertEqual(before, occupied.containers)

    def test_water_resource_preflight_uses_native_source_state_and_combined_p300_tips(self):
        robot = self._robot()
        robot._bind_auto_mix_calibration_well(self._binding_request())
        base = robot.containers['BaseC1.0']
        trigger = robot.containers['TriggerC1.0']
        before = (base.vol, trigger.vol)

        evidence = robot._build_auto_mix_calibration_resource_preflight(
            self._resource_request('water_safety_20ul_first')
        )

        self.assertEqual([20.0, 180.0], [
            item['required_volume_uL'] for item in evidence['source_requirements']
        ])
        self.assertEqual([450.0, 450.0], [
            item['available_volume_uL'] for item in evidence['source_requirements']
        ])
        self.assertEqual(1, evidence['ordinary_transfer_tip_requirements'][0][
            'available_new_tips'
        ])
        self.assertEqual(12, evidence['ordinary_transfer_tip_requirements'][1][
            'available_new_tips'
        ])
        self.assertEqual(before, (base.vol, trigger.vol))

    def test_two_dye_resource_preflight_uses_reviewed_five_then_195_volumes(self):
        robot = self._robot()
        robot._bind_auto_mix_calibration_well(self._binding_request())

        evidence = robot._build_auto_mix_calibration_resource_preflight(
            self._resource_request('qc_two_dye_5ul_red')
        )

        self.assertEqual([5.0, 195.0], [
            item['required_volume_uL'] for item in evidence['source_requirements']
        ])

    def test_resource_preflight_fails_closed_without_partial_evidence(self):
        robot = self._robot(p300_tip_count=11)
        robot._bind_auto_mix_calibration_well(self._binding_request())
        with self.assertRaisesRegex(ValueError, '12 unused P300'):
            robot._build_auto_mix_calibration_resource_preflight(
                self._resource_request()
            )

        source_short = self._robot(trigger_volume_uL=200.0)
        source_short._bind_auto_mix_calibration_well(self._binding_request())
        trigger = source_short.containers['TriggerC1.0']
        with self.assertRaisesRegex(ValueError, 'No remaining same-name source'):
            source_short._build_auto_mix_calibration_resource_preflight(
                self._resource_request()
            )
        self.assertEqual(200.0, trigger.vol)

    def test_handlers_are_explicitly_no_motion(self):
        source_tree = ast.parse(ROBOT_SOURCE.read_text())
        robot_class = next(
            node for node in source_tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'OT2Robot'
        )
        for name in ('_exec_bind_auto_mix_calibration_well',
                     '_exec_preflight_auto_mix_calibration_resources'):
            handler = next(
                node for node in robot_class.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            called_attributes = {
                node.func.attr for node in ast.walk(handler)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            }
            self.assertNotIn('_exec_transfer', called_attributes)
            self.assertNotIn('_exec_mix', called_attributes)
            self.assertNotIn('home', called_attributes)


if __name__ == '__main__':
    unittest.main()
