'''Hardware-free tests for the C2eA calibration preflight bridge.'''

import copy
import ast
import json
import os
import tempfile
import unittest

from auto_mix_calibration import (
    MIX_CALIBRATION_ASSAY_PROFILE_HEADER,
    MIX_CALIBRATION_FIRST_REAGENT_HEADER,
    MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED,
    MIX_CALIBRATION_SECOND_REAGENT_HEADER,
    parse_auto_mix_calibration_header_settings,
)
from auto_mix_calibration_plan import build_auto_mix_calibration_run_plan
from auto_mix_calibration_preflight import (
    AutoMixCalibrationPreflightError,
    run_auto_mix_calibration_pi_preflight,
)
from auto_mix_calibration_resources import (
    MIX_CALIBRATION_RESOURCE_PROTOCOL_VERSION,
)


def _plan():
    settings = parse_auto_mix_calibration_header_settings({
        'auto_mix_calibration_mode': 'two_dye',
        MIX_CALIBRATION_ASSAY_PROFILE_HEADER: (
            MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED
        ),
        MIX_CALIBRATION_FIRST_REAGENT_HEADER: 'acid_red_1',
        MIX_CALIBRATION_SECOND_REAGENT_HEADER: 'copper_sulfate_dye',
        'auto_mix_calibration_target_reader_well': 'C4',
    })
    return build_auto_mix_calibration_run_plan(
        settings=settings,
        calibration_id='dye-calibration-001',
        logical_wellname='autowell-calibration-001',
        source_container_names={
            'first': 'AcidRed1C1.44',
            'second': 'CopperSulfateDyeC8.0',
        },
        reader_location_mapping={'C4': ('B3', 'platereader4')},
        plate_mapping_revision=4,
        plate_generation=2,
        deck_reader_well='C4',
    )


def _packet_registry():
    return {
        'preflight_auto_mix_calibration': b'\x21',
        'auto_mix_calibration_preflight': b'\x22',
        'mix_auto_calibration_well': b'\x23',
        'auto_calibration_well_mixed': b'\x24',
        'bind_auto_mix_calibration_well': b'\x25',
        'auto_mix_calibration_well_bound': b'\x26',
        'preflight_auto_mix_calibration_resources': b'\x27',
        'auto_mix_calibration_resources_preflight': b'\x28',
    }


def _snapshot():
    return {
        'auto_mix_calibration_protocol_version': (
            MIX_CALIBRATION_RESOURCE_PROTOCOL_VERSION
        ),
        'supported_commands': [
            'preflight_auto_mix_calibration',
            'mix_auto_calibration_well',
            'bind_auto_mix_calibration_well',
            'preflight_auto_mix_calibration_resources',
        ],
    }


def _v1_ack(request, accepted=True):
    return {
        'schema_version': 1,
        'record_type': 'auto_mix_calibration_preflight',
        'preflight_id': request['preflight_id'],
        'calibration_id': request['calibration_id'],
        'accepted': accepted,
        'message': 'ready' if accepted else 'occupied',
        'wellname': request['wellname'],
        'expected_deck_pos': request['expected_deck_pos'],
        'expected_loc': request['expected_loc'],
        'base_chemical_name': request['base_chemical_name'],
        'trigger_chemical_name': request['trigger_chemical_name'],
        'plate_mapping_revision': request['expected_plate_mapping_revision'],
        'plate_generation': request['expected_plate_generation'],
        'pipette_arm': 'right',
        'mix_volume_uL': 150.0,
        'total_cycle_count': 10,
        'well_clearance_mm': 2.0,
        'required_new_tips': 11 if accepted else 0,
        'available_new_tips': 96,
        'tip_policy': 'dedicated_discarded',
    }


def _binding_ack(request, accepted=True):
    return {
        'schema_version': 1,
        'record_type': 'auto_mix_calibration_well_bound',
        'binding_id': request['binding_id'],
        'calibration_id': request['calibration_id'],
        'accepted': accepted,
        'message': 'bound' if accepted else 'occupied',
        'wellname': request['wellname'],
        'reader_well': request['reader_well'],
        'expected_deck_pos': request['expected_deck_pos'],
        'expected_loc': request['expected_loc'],
        'plate_mapping_revision': request['expected_plate_mapping_revision'],
        'plate_generation': request['expected_plate_generation'],
        'well_was_vacant': accepted,
        'binding_created': accepted,
    }


def _resource_ack(request, accepted=True):
    return {
        'schema_version': 1,
        'record_type': 'auto_mix_calibration_resources_preflight',
        'resource_preflight_id': request['resource_preflight_id'],
        'binding_id': request['binding_id'],
        'calibration_id': request['calibration_id'],
        'assay_profile': request['assay_profile'],
        'wellname': request['wellname'],
        'expected_deck_pos': request['expected_deck_pos'],
        'expected_loc': request['expected_loc'],
        'expected_plate_mapping_revision': (
            request['expected_plate_mapping_revision']
        ),
        'expected_plate_generation': request['expected_plate_generation'],
        'accepted': accepted,
        'message': 'ready' if accepted else 'tip deficit',
        'source_requirements': (
            [dict(row, available_volume_uL=500.0)
             for row in request['source_requirements']] if accepted else []
        ),
        'ordinary_transfer_tip_requirements': (
            [dict(row, available_new_tips=96)
             for row in request['ordinary_transfer_tip_requirements']]
            if accepted else []
        ),
        'dedicated_mix_tip_requirements': (
            [dict(row, available_new_tips=96)
             for row in request['dedicated_mix_tip_requirements']]
            if accepted else []
        ),
    }


class _Transport(object):
    def __init__(self):
        self.calls = []
        self.reject_v1 = False
        self.reject_binding = False
        self.reject_resources = False
        self.snapshot = _snapshot()

    def get_robot_state_snapshot(self):
        self.calls.append(('snapshot',))
        return copy.deepcopy(self.snapshot)

    def request_pi_preflight(self, request):
        self.calls.append(('v1', request['preflight_id']))
        return _v1_ack(request, accepted=not self.reject_v1)

    def request_pi_well_binding(self, request):
        self.calls.append(('binding', request['binding_id']))
        return _binding_ack(request, accepted=not self.reject_binding)

    def request_pi_resource_preflight(self, request):
        self.calls.append(('resources', request['resource_preflight_id']))
        return _resource_ack(request, accepted=not self.reject_resources)


class _Portal(object):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def send_pack(self, command, *arguments):
        self.calls.append((command, arguments))

    def recv_pack(self):
        return self.responses.pop(0)


def _controller_transport_class():
    repository_root = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))
    controller_path = os.path.join(repository_root, 'controller.py')
    with open(controller_path, encoding='utf-8') as source_file:
        tree = ast.parse(source_file.read())
    transport_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == 'AutoMixCalibrationPiPreflightTransport'
    )
    namespace = {
        'MIX_CALIBRATION_PREFLIGHT_COMMAND': (
            'preflight_auto_mix_calibration'
        ),
        'MIX_CALIBRATION_PREFLIGHT_ACKNOWLEDGEMENT': (
            'auto_mix_calibration_preflight'
        ),
        'MIX_CALIBRATION_BIND_WELL_COMMAND': (
            'bind_auto_mix_calibration_well'
        ),
        'MIX_CALIBRATION_BIND_WELL_ACKNOWLEDGEMENT': (
            'auto_mix_calibration_well_bound'
        ),
        'MIX_CALIBRATION_RESOURCE_PREFLIGHT_COMMAND': (
            'preflight_auto_mix_calibration_resources'
        ),
        'MIX_CALIBRATION_RESOURCE_PREFLIGHT_ACKNOWLEDGEMENT': (
            'auto_mix_calibration_resources_preflight'
        ),
    }
    module = ast.Module(body=[transport_class], type_ignores=[])
    exec(compile(module, controller_path, 'exec'), namespace)
    return namespace['AutoMixCalibrationPiPreflightTransport']


class AutoMixCalibrationPreflightTests(unittest.TestCase):
    def test_exact_v1_binding_resource_order_and_durable_record(self):
        transport = _Transport()
        with tempfile.TemporaryDirectory() as temporary_dir:
            record_path = os.path.join(temporary_dir, 'preflight.json')
            record = run_auto_mix_calibration_pi_preflight(
                _plan(), transport, _packet_registry(), 'UV_VIS_STANDARD',
                record_path=record_path
            )
            self.assertTrue(os.path.isfile(record_path))
            with open(record_path, encoding='utf-8') as record_file:
                persisted = json.load(record_file)
        self.assertEqual([
            ('snapshot',), ('v1', 'dye-calibration-001-preflight'),
            ('binding', 'dye-calibration-001-binding'),
            ('resources', 'dye-calibration-001-resources'),
        ], transport.calls)
        self.assertEqual('deferred_until_first_raw_scan',
                         record['reader_wavelength_coverage_status'])
        self.assertFalse(record['physical_execution_started'])
        self.assertFalse(record['automatic_retry_permitted'])
        self.assertEqual(record['calibration_id'], persisted['calibration_id'])

    def test_rejected_v1_blocks_binding_and_resource_preflight(self):
        transport = _Transport()
        transport.reject_v1 = True
        with self.assertRaises(AutoMixCalibrationPreflightError):
            run_auto_mix_calibration_pi_preflight(
                _plan(), transport, _packet_registry(), 'UV_VIS_STANDARD'
            )
        self.assertEqual(['snapshot', 'v1'],
                         [call[0] for call in transport.calls])

    def test_rejected_binding_blocks_resource_preflight(self):
        transport = _Transport()
        transport.reject_binding = True
        with self.assertRaises(AutoMixCalibrationPreflightError):
            run_auto_mix_calibration_pi_preflight(
                _plan(), transport, _packet_registry(), 'UV_VIS_STANDARD'
            )
        self.assertEqual(['snapshot', 'v1', 'binding'],
                         [call[0] for call in transport.calls])

    def test_rejected_resources_create_no_execution_permission(self):
        transport = _Transport()
        transport.reject_resources = True
        with self.assertRaises(AutoMixCalibrationPreflightError):
            run_auto_mix_calibration_pi_preflight(
                _plan(), transport, _packet_registry(), 'UV_VIS_STANDARD'
            )
        self.assertEqual(['snapshot', 'v1', 'binding', 'resources'],
                         [call[0] for call in transport.calls])

    def test_missing_v2_capability_blocks_all_pi_action_requests(self):
        transport = _Transport()
        transport.snapshot['supported_commands'].remove(
            'preflight_auto_mix_calibration_resources'
        )
        with self.assertRaises(AutoMixCalibrationPreflightError):
            run_auto_mix_calibration_pi_preflight(
                _plan(), transport, _packet_registry(), 'UV_VIS_STANDARD'
            )
        self.assertEqual([('snapshot',)], transport.calls)

    def test_missing_reader_protocol_blocks_before_snapshot(self):
        transport = _Transport()
        with self.assertRaises(AutoMixCalibrationPreflightError):
            run_auto_mix_calibration_pi_preflight(
                _plan(), transport, _packet_registry(), ' '
            )
        self.assertEqual([], transport.calls)

    def test_controller_transport_stays_packet_only(self):
        repository_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
        controller_path = os.path.join(repository_root, 'controller.py')
        with open(controller_path, encoding='utf-8') as source_file:
            tree = ast.parse(source_file.read())
        transport_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == 'AutoMixCalibrationPiPreflightTransport'
        )
        method_names = set(
            node.name for node in transport_class.body
            if isinstance(node, ast.FunctionDef)
        )
        self.assertEqual(set((
            '__init__', '_unpack_single_response', 'get_robot_state_snapshot',
            '_request', 'request_pi_preflight', 'request_pi_well_binding',
            'request_pi_resource_preflight'
        )), method_names)
        forbidden_calls = set((
            'transfer', 'mix', 'home', 'run_protocol', 'exec_macro', 'shake',
            'init_robot', 'close_connection'
        ))
        called_attributes = set(
            node.func.attr for node in ast.walk(transport_class)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        )
        self.assertFalse(called_attributes.intersection(forbidden_calls))

    def test_controller_transport_uses_only_exact_c2eA_packet_pairs(self):
        plan = _plan()
        v1_request = plan['preflight_request']
        binding_request = {
            'binding_id': 'dye-calibration-001-binding'
        }
        resource_request = {
            'resource_preflight_id': 'dye-calibration-001-resources'
        }
        portal = _Portal([
            ('robot_state_snapshot', 0, (_snapshot(),)),
            ('auto_mix_calibration_preflight', 0, (_v1_ack(v1_request),)),
            ('auto_mix_calibration_well_bound', 0,
             (_binding_ack(dict(
                 schema_version=1,
                 binding_id=binding_request['binding_id'],
                 calibration_id=plan['calibration_id'], wellname=plan[
                     'logical_wellname'
                 ], reader_well='C4', expected_deck_pos=4,
                 expected_loc='B3', expected_plate_mapping_revision=4,
                 expected_plate_generation=2,
                 expected_final_well_volume_uL=200.0
             )),)),
            ('auto_mix_calibration_resources_preflight', 0,
             (_resource_ack(dict(
                 schema_version=1,
                 resource_preflight_id=resource_request[
                     'resource_preflight_id'
                 ], binding_id=binding_request['binding_id'],
                 calibration_id=plan['calibration_id'], assay_profile=plan[
                     'assay_profile'
                 ], wellname=plan['logical_wellname'], expected_deck_pos=4,
                 expected_loc='B3', expected_plate_mapping_revision=4,
                 expected_plate_generation=2,
                 source_requirements=[], ordinary_transfer_tip_requirements=[],
                 dedicated_mix_tip_requirements=[]
             )),)),
        ])
        transport = _controller_transport_class()(portal)
        self.assertEqual(_snapshot(), transport.get_robot_state_snapshot())
        self.assertEqual(_v1_ack(v1_request),
                         transport.request_pi_preflight(v1_request))
        # The controller transport deliberately forwards arbitrary already
        # validated request dictionaries. Schema validation remains in the
        # pure bridge, so these two calls only prove the packet pairing.
        transport.request_pi_well_binding(binding_request)
        transport.request_pi_resource_preflight(resource_request)
        self.assertEqual([
            'get_robot_state_snapshot', 'preflight_auto_mix_calibration',
            'bind_auto_mix_calibration_well',
            'preflight_auto_mix_calibration_resources'
        ], [call[0] for call in portal.calls])


if __name__ == '__main__':
    unittest.main()
