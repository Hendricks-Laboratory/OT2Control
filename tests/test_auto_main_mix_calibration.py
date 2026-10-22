'''Hardware-free safety tests for the Stage 13G calibration-only Pi contract.'''

import ast
import copy
from datetime import datetime, timezone
import math
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ROBOT_SOURCE = REPO_ROOT / 'ot2_robot.py'


class _WellBase(object):
    pass


class _Well96(_WellBase):
    pass


class _Source(object):
    def __init__(self, aspiratible_vol=500.0):
        self.aspiratible_vol = aspiratible_vol


class _MultiContainer(object):
    pass


class _TargetWell(_Well96):
    def __init__(self, base_name, trigger_name, first_volume_uL=20.0,
                 second_volume_uL=180.0):
        self.name = 'autowell7C1.0'
        self.deck_pos = 4
        self.loc = 'B3'
        self.labware = object()
        self.vol = 200.0
        self.history = [
            ('time-1', base_name, first_volume_uL),
            ('time-2', trigger_name, second_volume_uL)
        ]
        self.calibration_calls = []
        self.raise_on_mix = False

    def mix_calibration(self, pipette):
        self.calibration_calls.append((pipette,))
        if self.raise_on_mix:
            raise RuntimeError('synthetic calibration mix fault')


class _TipWell(object):
    def __init__(self, has_tip=True):
        self.has_tip = has_tip


class _Pipette(object):
    def __init__(self, has_tip=True):
        self.has_tip = has_tip
        self.min_volume = 20.0
        self.max_volume = 300.0
        self.calls = []

    def pick_up_tip(self):
        self.calls.append(('pick_up_tip',))
        self.has_tip = True

    def drop_tip(self):
        self.calls.append(('drop_tip',))
        self.has_tip = False


class _Portal(object):
    def __init__(self):
        self.sent_packets = []

    def send_pack(self, packet_name, payload):
        self.sent_packets.append((packet_name, payload))


class _PytzStub(object):
    utc = timezone.utc

    @staticmethod
    def timezone(unused_name):
        return timezone.utc


class _PlateLabware(object):
    def wells_by_name(self):
        return {'B3': object()}


class _DeckPlate(object):
    name = 'platereader4'

    def __init__(self):
        self.labware = _PlateLabware()


def _robot_calibration_class():
    source_tree = ast.parse(ROBOT_SOURCE.read_text())
    robot_class = next(
        node for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'OT2Robot'
    )
    method_names = {
        '_preflight_number',
        '_get_auto_completed_well_mix_p300_arm',
        '_count_available_tips',
        '_validate_auto_mix_calibration_request',
        '_get_auto_mix_calibration_p300',
        '_validate_auto_mix_calibration_plate',
        '_validate_auto_mix_calibration_sources',
        '_build_auto_mix_calibration_preflight',
        '_auto_mix_calibration_preflight_response',
        '_exec_preflight_auto_mix_calibration',
        '_validate_auto_mix_calibration_transfer_request',
        '_auto_mix_calibration_transfer_response',
        '_build_auto_mix_calibration_transfer_plan',
        '_prepare_auto_mix_calibration_transfer_pipettes',
        '_exec_transfer_auto_mix_calibration_source',
        '_build_auto_mix_calibration_action_plan',
        '_execute_auto_mix_calibration_action_plan'
    }
    members = []
    for node in robot_class.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and (target.id.startswith('AUTO_MIX_CALIBRATION_')
                     or target.id == 'AUTO_COMPLETED_WELL_MIX_REQUIRED_PIPETTE_VOLUME_UL')
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
        raise AssertionError('A required calibration helper is absent.')
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(
            name='OT2Robot', bases=[], keywords=[], body=members,
            decorator_list=[]
        )], type_ignores=[]
    ))
    namespace = {
        'datetime': datetime,
        'math': math,
        'pytz': _PytzStub,
        'Well96': _Well96,
        'MultiContainer': _MultiContainer,
    }
    exec(compile(module, str(ROBOT_SOURCE), 'exec'), namespace)
    return namespace['OT2Robot']


def _well96_calibration_class():
    source_tree = ast.parse(ROBOT_SOURCE.read_text())
    source_well96 = next(
        node for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'Well96'
    )
    members = []
    for node in source_well96.body:
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith('CALIBRATION_MIX_')):
            members.append(copy.deepcopy(node))
        elif isinstance(node, ast.FunctionDef) and node.name == 'mix_calibration':
            members.append(copy.deepcopy(node))
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ClassDef(
            name='Well96', bases=[], keywords=[], body=members,
            decorator_list=[]
        )], type_ignores=[]
    ))
    namespace = {}
    exec(compile(module, str(ROBOT_SOURCE), 'exec'), namespace)
    return namespace['Well96']


class AutoMainMixCalibrationTests(unittest.TestCase):
    def _request(self, identity_field='preflight_id'):
        request = {
            'schema_version': 2,
            identity_field: 'calibration-{}'.format(identity_field),
            'calibration_id': 'calibration-session-a',
            'calibration_mode': 'water_safety',
            'assay_profile': 'water_safety_20ul_first',
            'wellname': 'autowell7C1.0',
            'expected_deck_pos': 4,
            'expected_loc': 'B3',
            'expected_plate_mapping_revision': 0,
            'expected_plate_generation': 0,
            'base_chemical_name': 'CalibrationBaseWaterC1.0',
            'trigger_chemical_name': 'CalibrationTriggerWaterC1.0',
            'first_transfer_volume_uL': 20.0,
            'second_transfer_volume_uL': 180.0,
            'expected_final_well_volume_uL': 200.0,
            'mix_volume_uL': 150.0,
            'total_cycle_count': 10,
            'well_clearance_mm': 2.0
        }
        if identity_field == 'action_id':
            request['cycle_index'] = 1
            request['preflight_id'] = 'calibration-preflight_id'
        return request

    def _robot(self, allocated_target=False, available_tip_count=11,
               p300_has_clean_tip=False):
        robot_class = _robot_calibration_class()
        base_name = 'CalibrationBaseWaterC1.0'
        trigger_name = 'CalibrationTriggerWaterC1.0'
        target = _TargetWell(base_name, trigger_name)
        pipette = _Pipette(has_tip=p300_has_clean_tip)
        robot = robot_class()
        robot.containers = {
            base_name: _Source(),
            trigger_name: _Source()
        }
        if allocated_target:
            robot.containers[target.name] = target
        robot.pipettes = {
            'left': {'size': 20.0, 'pipette': _Pipette(),
                     'last_used': 'clean', 'configured_tip_wells': []},
            'right': {
                'size': 300.0,
                'pipette': pipette,
                'last_used': 'clean' if p300_has_clean_tip else base_name,
                'configured_tip_wells': [
                    _TipWell() for unused in range(available_tip_count)
                ]
            }
        }
        robot.lab_deck = [None] * 13
        robot.lab_deck[4] = _DeckPlate()
        target.labware = robot.lab_deck[4].labware
        robot.plate_mapping_revision = 0
        robot.plate_generation = 0
        robot.protocol = type('Protocol', (), {'_commands': []})()
        return robot, target, pipette

    @staticmethod
    def _awaiting_first_mix_session(request):
        return {
            'status': 'awaiting_first_mix',
            'next_cycle_index': 1,
            'calibration_mode': request['calibration_mode'],
            'wellname': request['wellname'],
            'expected_deck_pos': request['expected_deck_pos'],
            'expected_loc': request['expected_loc'],
            'expected_plate_mapping_revision': request[
                'expected_plate_mapping_revision'
            ],
            'expected_plate_generation': request[
                'expected_plate_generation'
            ],
            'base_chemical_name': request['base_chemical_name'],
            'trigger_chemical_name': request['trigger_chemical_name'],
            'assay_profile': request['assay_profile'],
            'first_transfer_volume_uL': request[
                'first_transfer_volume_uL'
            ],
            'second_transfer_volume_uL': request[
                'second_transfer_volume_uL'
            ],
        }

    @staticmethod
    def _transfer_request(action_request, transfer_index):
        source_name = (
            action_request['base_chemical_name'] if transfer_index == 1
            else action_request['trigger_chemical_name']
        )
        volume_uL = (
            action_request['first_transfer_volume_uL'] if transfer_index == 1
            else action_request['second_transfer_volume_uL']
        )
        return {
            'schema_version': 1,
            'transfer_id': '{}-transfer-{}'.format(
                action_request['calibration_id'], transfer_index
            ),
            'calibration_id': action_request['calibration_id'],
            'preflight_id': action_request['preflight_id'],
            'binding_id': 'binding-a',
            'resource_preflight_id': 'resources-a',
            'transfer_index': transfer_index,
            'wellname': action_request['wellname'],
            'expected_deck_pos': action_request['expected_deck_pos'],
            'expected_loc': action_request['expected_loc'],
            'expected_plate_mapping_revision': action_request[
                'expected_plate_mapping_revision'
            ],
            'expected_plate_generation': action_request[
                'expected_plate_generation'
            ],
            'assay_profile': action_request['assay_profile'],
            'source_chemical_name': source_name,
            'destination_wellname': action_request['wellname'],
            'volume_uL': volume_uL,
        }

    def _configure_exact_transfer_evidence(self, robot, action_request):
        physical_identity_fields = (
            'wellname', 'expected_deck_pos', 'expected_loc',
            'expected_plate_mapping_revision', 'expected_plate_generation'
        )
        preflight_identity_fields = physical_identity_fields + ('assay_profile',)
        preflight = dict(
            (field_name, action_request[field_name])
            for field_name in (
                'preflight_id', 'calibration_mode', 'base_chemical_name',
                'trigger_chemical_name', 'first_transfer_volume_uL',
                'second_transfer_volume_uL'
            ) + preflight_identity_fields
        )
        binding = dict(
            (field_name, action_request[field_name])
            for field_name in physical_identity_fields
        )
        binding['binding_id'] = 'binding-a'
        resources = dict(
            (field_name, action_request[field_name])
            for field_name in preflight_identity_fields
        )
        resources.update({
            'resource_preflight_id': 'resources-a',
            'source_requirements': [
                {
                    'source_container': action_request['base_chemical_name'],
                    'required_volume_uL': action_request[
                        'first_transfer_volume_uL'
                    ],
                },
                {
                    'source_container': action_request['trigger_chemical_name'],
                    'required_volume_uL': action_request[
                        'second_transfer_volume_uL'
                    ],
                },
            ],
        })
        robot._auto_mix_calibration_preflights = {
            action_request['calibration_id']: preflight
        }
        robot._auto_mix_calibration_bindings = {
            action_request['calibration_id']: binding
        }
        robot._auto_mix_calibration_resource_preflights = {
            action_request['calibration_id']: resources
        }

    def test_preflight_accepts_only_complete_exact_session_before_transfer(self):
        robot, _, _ = self._robot(allocated_target=False)
        plan = robot._build_auto_mix_calibration_preflight(self._request())
        self.assertEqual(11, plan['required_new_tips'])
        self.assertEqual(11, plan['available_new_tips'])
        self.assertEqual('right', plan['pipette_arm'])
        self.assertNotIn('_auto_mix_calibration_sessions', robot.__dict__)

    def test_preflight_handler_persists_full_action_identity(self):
        robot, _, _ = self._robot(allocated_target=False)
        robot.portal = _Portal()
        request = self._request()

        robot._exec_preflight_auto_mix_calibration(request)

        self.assertEqual('auto_mix_calibration_preflight',
                         robot.portal.sent_packets[-1][0])
        self.assertTrue(robot.portal.sent_packets[-1][1]['accepted'])
        stored = robot._auto_mix_calibration_preflights[
            request['calibration_id']
        ]
        for field_name in (
                'preflight_id', 'calibration_mode', 'wellname',
                'expected_deck_pos', 'expected_loc',
                'expected_plate_mapping_revision', 'expected_plate_generation',
                'base_chemical_name', 'trigger_chemical_name', 'assay_profile',
                'first_transfer_volume_uL', 'second_transfer_volume_uL'):
            self.assertEqual(request[field_name], stored[field_name])

    def test_preflight_rejects_allocated_target_insufficient_tips_or_bad_geometry(self):
        cases = [
            (True, 11, 'wellname', 'autowell7C1.0'),
            (False, 10, 'wellname', 'autowell7C1.0'),
            (False, 11, 'mix_volume_uL', 149.0),
            (False, 11, 'well_clearance_mm', 1.0),
        ]
        for allocated_target, tip_count, field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                robot, _, _ = self._robot(
                    allocated_target=allocated_target,
                    available_tip_count=tip_count
                )
                request = self._request()
                request[field_name] = value
                with self.assertRaises(ValueError):
                    robot._build_auto_mix_calibration_preflight(request)

    def test_action_accepts_150_ul_in_200_ul_and_enforces_sequence(self):
        robot, target, pipette = self._robot(allocated_target=True)
        request = self._request('action_id')
        robot._auto_mix_calibration_preflights = {
            request['calibration_id']: {
                'preflight_id': request['preflight_id'],
                'calibration_mode': request['calibration_mode'],
                'wellname': request['wellname'],
                'expected_deck_pos': request['expected_deck_pos'],
                'expected_loc': request['expected_loc'],
                'expected_plate_mapping_revision': 0,
                'expected_plate_generation': 0,
                'base_chemical_name': request['base_chemical_name'],
                'trigger_chemical_name': request['trigger_chemical_name'],
                'assay_profile': request['assay_profile'],
                'first_transfer_volume_uL': request['first_transfer_volume_uL'],
                'second_transfer_volume_uL': request['second_transfer_volume_uL']
            }
        }
        robot._auto_mix_calibration_sessions = {
            request['calibration_id']: self._awaiting_first_mix_session(request)
        }
        plan = robot._build_auto_mix_calibration_action_plan(request)
        self.assertEqual(150.0, plan['mix_volume_uL'])
        self.assertEqual(2, plan['required_new_tips'])
        robot._execute_auto_mix_calibration_action_plan(plan)
        self.assertEqual([(pipette,)], target.calibration_calls)
        self.assertEqual([
            ('pick_up_tip',), ('drop_tip',), ('pick_up_tip',)
        ], pipette.calls)
        self.assertEqual('clean', robot.pipettes['right']['last_used'])

        robot._auto_mix_calibration_action_ids = {request['action_id']}
        with self.assertRaisesRegex(ValueError, 'already used'):
            robot._build_auto_mix_calibration_action_plan(request)

        next_request = self._request('action_id')
        next_request['action_id'] = 'calibration-action-02'
        next_request['cycle_index'] = 2
        robot._auto_mix_calibration_sessions = {
            next_request['calibration_id']: {
                'status': 'awaiting_next_cycle',
                'next_cycle_index': 2,
                'calibration_mode': next_request['calibration_mode'],
                'wellname': next_request['wellname'],
                'expected_deck_pos': next_request['expected_deck_pos'],
                'expected_loc': next_request['expected_loc'],
                'expected_plate_mapping_revision': 0,
                'expected_plate_generation': 0,
                'base_chemical_name': next_request['base_chemical_name'],
                'trigger_chemical_name': next_request['trigger_chemical_name'],
                'assay_profile': next_request['assay_profile'],
                'first_transfer_volume_uL': next_request['first_transfer_volume_uL'],
                'second_transfer_volume_uL': next_request['second_transfer_volume_uL']
            }
        }
        robot._auto_mix_calibration_preflights = {
            next_request['calibration_id']: {
                'preflight_id': next_request['preflight_id'],
                'calibration_mode': next_request['calibration_mode'],
                'wellname': next_request['wellname'],
                'expected_deck_pos': next_request['expected_deck_pos'],
                'expected_loc': next_request['expected_loc'],
                'expected_plate_mapping_revision': 0,
                'expected_plate_generation': 0,
                'base_chemical_name': next_request['base_chemical_name'],
                'trigger_chemical_name': next_request['trigger_chemical_name'],
                'assay_profile': next_request['assay_profile'],
                'first_transfer_volume_uL': next_request['first_transfer_volume_uL'],
                'second_transfer_volume_uL': next_request['second_transfer_volume_uL']
            }
        }
        robot._auto_mix_calibration_action_ids = set()
        self.assertEqual(2, robot._build_auto_mix_calibration_action_plan(
            next_request
        )['cycle_index'])

    def test_exact_source_transfers_are_ordered_no_fallback_and_non_replayable(self):
        robot, target, _ = self._robot(allocated_target=True)
        target.history = []
        target.vol = 0.0
        robot.portal = _Portal()
        action_request = self._request('action_id')
        self._configure_exact_transfer_evidence(robot, action_request)
        self.assertNotIn(
            'assay_profile',
            robot._auto_mix_calibration_bindings[
                action_request['calibration_id']
            ]
        )
        events = []

        robot._prepare_auto_mix_calibration_transfer_pipettes = lambda plan: events.append(
            ('prepare', plan['source_chemical_name'], plan['pipette_arm'])
        )
        robot._get_preffered_pipette = lambda volume: (
            'left' if volume <= 20.0 else 'right'
        )

        def fake_transfer(source, destination, volume, allow_auto_make=True):
            events.append((source, destination, volume, allow_auto_make))
            target.history.append(('synthetic', source, volume))
            target.vol += volume

        robot._transfer_step = fake_transfer
        first = self._transfer_request(action_request, 1)
        robot._exec_transfer_auto_mix_calibration_source(first)
        first_ack = robot.portal.sent_packets[-1][1]
        self.assertTrue(first_ack['accepted'])
        self.assertEqual('awaiting_second_transfer', robot._auto_mix_calibration_sessions[
            action_request['calibration_id']
        ]['status'])

        second = self._transfer_request(action_request, 2)
        robot._exec_transfer_auto_mix_calibration_source(second)
        second_ack = robot.portal.sent_packets[-1][1]
        self.assertTrue(second_ack['accepted'])
        self.assertEqual('awaiting_first_mix', robot._auto_mix_calibration_sessions[
            action_request['calibration_id']
        ]['status'])
        self.assertEqual([
            ('prepare', action_request['base_chemical_name'], 'left'),
            (action_request['base_chemical_name'], action_request['wellname'],
             20.0, False),
            ('prepare', action_request['trigger_chemical_name'], 'right'),
            (action_request['trigger_chemical_name'], action_request['wellname'],
             180.0, False),
        ], events)

        robot._exec_transfer_auto_mix_calibration_source(first)
        replay_ack = robot.portal.sent_packets[-1][1]
        self.assertFalse(replay_ack['accepted'])
        self.assertEqual(4, len(events))

        invalid = dict(second)
        invalid['transfer_index'] = 1.0
        with self.assertRaises(ValueError):
            robot._validate_auto_mix_calibration_transfer_request(invalid)

    def test_second_transfer_does_not_pick_an_unmounted_unused_arm(self):
        robot, _, p300 = self._robot(allocated_target=True,
                                     p300_has_clean_tip=True)
        left_pipette = robot.pipettes['left']['pipette']
        left_pipette.has_tip = False
        robot.pipettes['left']['last_used'] = 'CalibrationBaseWaterC1.0'
        plan = {
            'source_chemical_name': 'CalibrationTriggerWaterC1.0',
            'volume_uL': 180.0,
            'pipette_arm': 'right',
        }
        robot._get_preffered_pipette = lambda volume: 'right'

        robot._prepare_auto_mix_calibration_transfer_pipettes(plan)

        self.assertEqual([], left_pipette.calls)
        self.assertEqual('clean', robot.pipettes['left']['last_used'])
        self.assertEqual([], p300.calls)
        self.assertTrue(p300.has_tip)

    def test_transfer_refuses_dirty_unselected_mounted_tip(self):
        robot, _, _ = self._robot(allocated_target=True,
                                  p300_has_clean_tip=True)
        robot.pipettes['left']['last_used'] = 'OtherChemicalC1.0'
        plan = {
            'source_chemical_name': 'CalibrationTriggerWaterC1.0',
            'volume_uL': 180.0,
            'pipette_arm': 'right',
        }
        robot._get_preffered_pipette = lambda volume: 'right'

        with self.assertRaisesRegex(ValueError, 'unselected left arm'):
            robot._prepare_auto_mix_calibration_transfer_pipettes(plan)

    def test_source_transfer_rejects_profile_mismatch_before_motion(self):
        robot, _, _ = self._robot(allocated_target=True)
        action_request = self._request('action_id')
        self._configure_exact_transfer_evidence(robot, action_request)
        robot._auto_mix_calibration_resource_preflights[
            action_request['calibration_id']
        ]['assay_profile'] = 'wrong_profile'
        with self.assertRaisesRegex(ValueError, 'assay profile changed'):
            robot._build_auto_mix_calibration_transfer_plan(
                self._transfer_request(action_request, 1)
            )

    def test_calibration_well_primitive_is_fixed_150_ul_two_mm_and_restores(self):
        well = _well96_calibration_class()()
        target_well = object()
        well.get_well = lambda: target_well
        pipette = _Pipette()
        pipette.well_bottom_clearance = type(
            'Clearance', (), {'aspirate': 7.0, 'dispense': 9.0}
        )()
        pipette.mix = lambda cycles, volume, location, rate: setattr(
            pipette, 'mix_call', (cycles, volume, location, rate)
        )
        well.mix_calibration(pipette)
        self.assertEqual((1, 150.0, target_well, 1.0), pipette.mix_call)
        self.assertEqual(7.0, pipette.well_bottom_clearance.aspirate)
        self.assertEqual(9.0, pipette.well_bottom_clearance.dispense)

    def test_source_requires_exact_two_transfer_history(self):
        robot, target, _ = self._robot(allocated_target=True)
        target.history.append(('time-3', 'WaterC1.0', 1.0))
        request = self._request('action_id')
        robot._auto_mix_calibration_preflights = {
            request['calibration_id']: {
                'preflight_id': request['preflight_id'],
                'calibration_mode': request['calibration_mode'],
                'wellname': request['wellname'],
                'expected_deck_pos': request['expected_deck_pos'],
                'expected_loc': request['expected_loc'],
                'expected_plate_mapping_revision': 0,
                'expected_plate_generation': 0,
                'base_chemical_name': request['base_chemical_name'],
                'trigger_chemical_name': request['trigger_chemical_name'],
                'assay_profile': request['assay_profile'],
                'first_transfer_volume_uL': request['first_transfer_volume_uL'],
                'second_transfer_volume_uL': request['second_transfer_volume_uL']
            }
        }
        robot._auto_mix_calibration_sessions = {
            request['calibration_id']: self._awaiting_first_mix_session(request)
        }
        with self.assertRaisesRegex(ValueError, 'exactly the base'):
            robot._build_auto_mix_calibration_action_plan(
                request
            )

    def test_action_accepts_each_profile_exactly_as_transferred(self):
        robot, target, _ = self._robot(allocated_target=True)
        target.history = [
            ('time-1', 'CalibrationBaseWaterC1.0', 5.0),
            ('time-2', 'CalibrationTriggerWaterC1.0', 195.0)
        ]
        request = self._request('action_id')
        request.update({
            'calibration_mode': 'two_dye',
            'assay_profile': 'qc_two_dye_5ul_red',
            'first_transfer_volume_uL': 5.0,
            'second_transfer_volume_uL': 195.0,
        })
        robot._auto_mix_calibration_preflights = {
            request['calibration_id']: dict(
                (field_name, request[field_name]) for field_name in (
                    'preflight_id', 'calibration_mode', 'assay_profile',
                    'wellname', 'expected_deck_pos', 'expected_loc',
                    'expected_plate_mapping_revision',
                    'expected_plate_generation', 'base_chemical_name',
                    'trigger_chemical_name', 'first_transfer_volume_uL',
                    'second_transfer_volume_uL'
                )
            )
        }
        robot._auto_mix_calibration_sessions = {
            request['calibration_id']: self._awaiting_first_mix_session(request)
        }
        self.assertEqual(
            1, robot._build_auto_mix_calibration_action_plan(request)[
                'cycle_index'
            ]
        )


if __name__ == '__main__':
    unittest.main()
