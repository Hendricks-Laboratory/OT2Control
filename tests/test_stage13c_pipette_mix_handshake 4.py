'''Source-extracted controller tests for the dormant Stage-13C handshake.'''

import ast
import copy
import os
from collections import namedtuple
from types import SimpleNamespace
import unittest
import uuid

from auto_stability_pipette_mix import (
    AutoStabilityPipetteMixContractError,
    TARGETED_MIX_ACKNOWLEDGEMENT,
    TARGETED_MIX_COMMAND,
    validate_targeted_mix_acknowledgement,
    validate_targeted_mix_request,
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')
ARMCHAIR_PATH = os.path.join(REPOSITORY_ROOT, 'Armchair', 'armchair.py')


def _load_armchair_packet_registry():
    '''Read the live wire map without importing legacy socket dependencies.'''
    with open(ARMCHAIR_PATH, encoding='utf-8') as source_file:
        source = source_file.read()
    tree = ast.parse(source)
    armchair_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'Armchair'
    )
    assignments = {
        node.targets[0].id: node.value
        for node in armchair_class.body
        if (isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name))
    }
    packet_expression = assignments['PACK_TYPES']
    packet_map = ast.literal_eval(packet_expression.args[0])
    ghost_types = ast.literal_eval(assignments['GHOST_TYPES'])
    return type(
        'ArmchairRegistryStub',
        (),
        {'PACK_TYPES': packet_map, 'GHOST_TYPES': ghost_types}
    )


ARMCHAIR_REGISTRY = _load_armchair_packet_registry()


def _load_plate_reader_mapping():
    '''Read the controller's canonical translation table without imports.'''
    with open(CONTROLLER_PATH, encoding='utf-8') as source_file:
        tree = ast.parse(source_file.read())
    controller_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'Controller'
    )
    mapping_assignment = next(
        node for node in controller_class.body
        if (isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == 'PLATEREADER_INDEX_TRANSLATOR')
    )
    return ast.literal_eval(mapping_assignment.value.args[0])


PLATEREADER_INDEX_TRANSLATOR = _load_plate_reader_mapping()


def _load_methods(method_names):
    with open(CONTROLLER_PATH, encoding='utf-8') as source_file:
        tree = ast.parse(source_file.read())
    auto_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
    )
    methods = {
        node.name: node for node in auto_class.body
        if isinstance(node, ast.FunctionDef)
    }
    extracted_class = ast.ClassDef(
        name='AutoContr', bases=[], keywords=[],
        body=[methods[name] for name in method_names], decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(
        body=[extracted_class], type_ignores=[]
    ))
    namespace = {
        'copy': copy,
        'uuid': uuid,
        'LIFECYCLE_EXECUTING_BATCH': 'executing_batch',
        'AutoStabilityPipetteMixContractError': (
            AutoStabilityPipetteMixContractError
        ),
        'TARGETED_MIX_ACKNOWLEDGEMENT': TARGETED_MIX_ACKNOWLEDGEMENT,
        'TARGETED_MIX_COMMAND': TARGETED_MIX_COMMAND,
        'Armchair': ARMCHAIR_REGISTRY,
        'validate_targeted_mix_acknowledgement': (
            validate_targeted_mix_acknowledgement
        ),
        'validate_targeted_mix_request': validate_targeted_mix_request,
    }
    exec(compile(module, CONTROLLER_PATH, 'exec'), namespace)
    return namespace['AutoContr']


def _method_source(method_name, class_name='AutoContr'):
    with open(CONTROLLER_PATH, encoding='utf-8') as source_file:
        source = source_file.read()
    tree = ast.parse(source)
    controller_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in controller_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method)


class Stage13CHandshakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AutoController = _load_methods([
            '_require_auto_main_targeted_mix_capability',
            '_get_auto_stability_robot_location',
            '_build_auto_stability_targeted_mix_request',
            '_record_auto_stability_targeted_mix_rejection',
            '_request_auto_stability_targeted_mix'
        ])

    def _controller(self, response):
        controller = self.AutoController()
        controller.auto_main_robot_state_snapshot = {
            'supported_commands': [TARGETED_MIX_COMMAND],
            'plate_mapping_revision': 3,
            'plate_generation': 1
        }
        controller.auto_plate_generation = 1
        controller.batch_num = 4
        controller._cached_reader_locs = {
            'autowell4C1.0': SimpleNamespace(deck_pos=4, loc='A1')
        }
        controller.PLATEREADER_INDEX_TRANSLATOR = (
            PLATEREADER_INDEX_TRANSLATOR
        )
        controller.auto_live_run_journal = SimpleNamespace(current_state={
            'lifecycle_state': 'executing_batch',
            'active_batch_number': 4
        })
        controller.auto_stability_observer = _ObserverStub()
        controller._auto_stability_pipette_mix_action_ids = set()
        controller.events = []
        controller._record_auto_live_run_event = (
            lambda event_type, payload: controller.events.append(
                (event_type, copy.deepcopy(payload))
            )
        )
        controller.portal = _PortalStub(response)
        return controller

    def test_successful_handshake_records_intent_before_send_and_ack_afterward(self):
        controller = self._controller(None)
        request = controller._build_auto_stability_targeted_mix_request(
            'autowell4C1.0', 'sodium_borohydrideC6.25', 20.0, 1
        )
        self.assertEqual('E1', request['expected_loc'])
        self.assertEqual(
            'A1', controller._cached_reader_locs['autowell4C1.0'].loc
        )
        controller.portal.response = (
            TARGETED_MIX_ACKNOWLEDGEMENT, 0, (_successful_ack(request),)
        )
        result = controller._request_auto_stability_targeted_mix(request)

        self.assertTrue(result['accepted'])
        self.assertEqual(
            [(TARGETED_MIX_COMMAND, request)], controller.portal.sent
        )
        self.assertEqual(
            ['stability_pipette_mix_intent_recorded',
             'stability_pipette_mix_acknowledged'],
            [event[0] for event in controller.events]
        )
        self.assertEqual(
            ['intent', 'acknowledged'], controller.auto_stability_observer.calls
        )

    def test_reader_to_robot_translation_is_bijective_for_every_plate_well(self):
        controller = self._controller(None)
        mappings = PLATEREADER_INDEX_TRANSLATOR
        self.assertEqual(96, len(mappings))
        self.assertEqual(96, len(set(mappings.values())))

        for reader_location, (robot_location, labware_name) in mappings.items():
            deck_pos = int(labware_name.replace('platereader', ''))
            self.assertEqual(
                robot_location,
                controller._get_auto_stability_robot_location(
                    reader_location, deck_pos
                )
            )

    def test_request_uses_the_plate_matched_robot_location_on_deck_seven(self):
        controller = self._controller(None)
        controller._cached_reader_locs['autowell4C1.0'] = SimpleNamespace(
            deck_pos=7, loc='A12'
        )

        request = controller._build_auto_stability_targeted_mix_request(
            'autowell4C1.0', 'sodium_borohydrideC6.25', 20.0, 1
        )

        self.assertEqual(7, request['expected_deck_pos'])
        self.assertEqual('A1', request['expected_loc'])
        self.assertEqual(
            'A12', controller._cached_reader_locs['autowell4C1.0'].loc
        )

    def test_cross_plate_reader_cache_is_rejected_before_a_packet_is_built(self):
        controller = self._controller(None)
        controller._cached_reader_locs['autowell4C1.0'] = SimpleNamespace(
            deck_pos=7, loc='A1'
        )

        with self.assertRaisesRegex(RuntimeError, 'belongs to platereader4'):
            controller._build_auto_stability_targeted_mix_request(
                'autowell4C1.0', 'sodium_borohydrideC6.25', 20.0, 1
            )
        self.assertEqual([], controller.portal.sent)

    def test_ordinary_transfer_packets_do_not_apply_reader_coordinate_mapping(self):
        transfer_source = _method_source(
            '_send_transfer_command', class_name='Controller'
        )
        self.assertNotIn('PLATEREADER_INDEX_TRANSLATOR', transfer_source)
        self.assertIn("self.portal.send_pack('transfer'", transfer_source)

    def test_malformed_acknowledgement_is_terminal_and_is_not_resent(self):
        controller = self._controller(('wrong_packet', 0, ({},)))
        request = controller._build_auto_stability_targeted_mix_request(
            'autowell4C1.0', 'sodium_borohydrideC6.25', 20.0, 1
        )
        with self.assertRaisesRegex(RuntimeError, 'do not retry'):
            controller._request_auto_stability_targeted_mix(request)
        self.assertEqual(1, len(controller.portal.sent))
        self.assertEqual(
            'stability_pipette_mix_acknowledgement_rejected',
            controller.events[-1][0]
        )
        with self.assertRaisesRegex(RuntimeError, 'already attempted'):
            controller._request_auto_stability_targeted_mix(request)
        self.assertEqual(1, len(controller.portal.sent))

    def test_capability_is_optional_until_future_stage_selects_pipette_mixing(self):
        controller = self._controller(None)
        controller.auto_main_robot_state_snapshot['supported_commands'] = []
        with self.assertRaisesRegex(RuntimeError, 'does not advertise'):
            controller._build_auto_stability_targeted_mix_request(
                'autowell4C1.0', 'sodium_borohydrideC6.25', 20.0, 1
            )

    def test_local_packet_registry_has_targeted_mix_wire_codes(self):
        self.assertEqual(
            b'\x1F', ARMCHAIR_REGISTRY.PACK_TYPES[TARGETED_MIX_COMMAND]
        )
        self.assertEqual(
            b'\x20', ARMCHAIR_REGISTRY.PACK_TYPES[
                TARGETED_MIX_ACKNOWLEDGEMENT
            ]
        )
        self.assertIn(TARGETED_MIX_COMMAND, ARMCHAIR_REGISTRY.GHOST_TYPES)
        self.assertIn(
            TARGETED_MIX_ACKNOWLEDGEMENT, ARMCHAIR_REGISTRY.GHOST_TYPES
        )

    def test_missing_local_packet_fails_before_a_mix_request_is_built(self):
        controller = self._controller(None)
        original_packet_types = ARMCHAIR_REGISTRY.PACK_TYPES
        try:
            incomplete_packet_types = original_packet_types.copy()
            del incomplete_packet_types[TARGETED_MIX_COMMAND]
            ARMCHAIR_REGISTRY.PACK_TYPES = incomplete_packet_types
            with self.assertRaisesRegex(
                    RuntimeError, 'local controller Armchair packet'):
                controller._build_auto_stability_targeted_mix_request(
                    'autowell4C1.0', 'sodium_borohydrideC6.25', 20.0, 1
                )
        finally:
            ARMCHAIR_REGISTRY.PACK_TYPES = original_packet_types

    def test_runtime_checks_targeted_packet_capability_before_preparation(self):
        run_source = _method_source('_run')
        self.assertLess(
            run_source.index(
                'self._require_auto_main_targeted_mix_capability()'
            ),
            run_source.index('self._execute_auto_preparation_phase(')
        )

    def test_stale_plate_identity_cannot_send_a_previously_built_request(self):
        controller = self._controller(None)
        request = controller._build_auto_stability_targeted_mix_request(
            'autowell4C1.0', 'sodium_borohydrideC6.25', 20.0, 1
        )
        controller.auto_main_robot_state_snapshot['plate_generation'] = 2
        controller.auto_plate_generation = 2
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            controller._request_auto_stability_targeted_mix(request)
        self.assertEqual([], controller.portal.sent)


class _PortalStub(object):
    def __init__(self, response):
        self.response = response
        self.sent = []

    def send_pack(self, *args):
        self.sent.append(args)

    def recv_pack(self):
        return self.response


class _ObserverStub(object):
    def __init__(self):
        self.calls = []

    def record_targeted_pipette_mix_intent(self, request, batch_number):
        self.calls.append('intent')

    def record_targeted_pipette_mix_acknowledged(self, acknowledgement):
        self.calls.append('acknowledged')


def _successful_ack(request):
    return {
        'schema_version': 1,
        'record_type': 'auto_completed_well_mixed',
        'action_id': request['action_id'],
        'accepted': True,
        'message': 'targeted completed Auto well mixing completed',
        'wellname': request['wellname'],
        'expected_deck_pos': request['expected_deck_pos'],
        'expected_loc': request['expected_loc'],
        'trigger_chemical_name': request['trigger_chemical_name'],
        'plate_mapping_revision': request['expected_plate_mapping_revision'],
        'plate_generation': request['expected_plate_generation'],
        'pipette_arm': 'right',
        'mix_volume_uL': request['mix_volume_uL'],
        'cycle_count': request['cycle_count'],
        'required_new_tips': 1,
        'available_new_tips': 24,
        'tip_policy': 'dedicated_discarded',
        'physical_execution_started': True,
        'started_at_utc': '2026-10-02T12:00:00+00:00',
        'completed_at_utc': '2026-10-02T12:00:02+00:00'
    }


if __name__ == '__main__':
    unittest.main()
