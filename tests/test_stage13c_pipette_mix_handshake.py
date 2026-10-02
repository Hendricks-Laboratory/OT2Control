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
        'validate_targeted_mix_acknowledgement': (
            validate_targeted_mix_acknowledgement
        ),
        'validate_targeted_mix_request': validate_targeted_mix_request,
    }
    exec(compile(module, CONTROLLER_PATH, 'exec'), namespace)
    return namespace['AutoContr']


class Stage13CHandshakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AutoController = _load_methods([
            '_require_auto_main_targeted_mix_capability',
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
