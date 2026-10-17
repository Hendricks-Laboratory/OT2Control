'''Hardware-free packet/capability tests for supervised mix calibration.'''

import unittest

from auto_mix_calibration_protocol import (
    AutoMixCalibrationProtocolError,
    MIX_CALIBRATION_COMMAND,
    MIX_CALIBRATION_PREFLIGHT_COMMAND,
    MIX_CALIBRATION_PROTOCOL_VERSION,
    validate_mix_calibration_acknowledgement,
    validate_mix_calibration_capability,
    validate_mix_calibration_preflight_acknowledgement,
    validate_mix_calibration_preflight_request,
    validate_mix_calibration_request,
)


def _base_request():
    return {
        'schema_version': 2,
        'calibration_id': 'mix-cal-2026-10-16-a',
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
        'well_clearance_mm': 2.0,
    }


class AutoMixCalibrationProtocolTests(unittest.TestCase):
    def test_preflight_requires_exact_reviewed_conditions(self):
        request = _base_request()
        request['preflight_id'] = 'preflight-a'
        normalized = validate_mix_calibration_preflight_request(request)
        self.assertEqual('B3', normalized['expected_loc'])

        for field_name, value in (
                ('mix_volume_uL', 149.0),
                ('expected_final_well_volume_uL', 199.0),
                ('total_cycle_count', 9),
                ('well_clearance_mm', 1.0),
                ('first_transfer_volume_uL', 180.0),
                ('second_transfer_volume_uL', 20.0),
                ('assay_profile', 'qc_two_dye_5ul_red'),
                ('base_chemical_name', 'calibrationtriggerwaterc1.0')):
            invalid = dict(request)
            invalid[field_name] = value
            with self.subTest(field_name=field_name):
                with self.assertRaises(AutoMixCalibrationProtocolError):
                    validate_mix_calibration_preflight_request(invalid)

    def test_each_mix_is_one_indexed_discrete_action(self):
        request = _base_request()
        request.update({
            'action_id': 'cycle-01', 'preflight_id': 'preflight-a',
            'cycle_index': 1
        })
        normalized = validate_mix_calibration_request(request)
        self.assertEqual(1, normalized['cycle_index'])

        request['cycle_index'] = 11
        with self.assertRaises(AutoMixCalibrationProtocolError):
            validate_mix_calibration_request(request)

    def test_successful_pi_preflight_and_action_acknowledgements_are_exact(self):
        preflight_request = _base_request()
        preflight_request['preflight_id'] = 'preflight-a'
        preflight_ack = {
            'schema_version': 2,
            'record_type': 'auto_mix_calibration_preflight',
            'preflight_id': 'preflight-a',
            'calibration_id': preflight_request['calibration_id'],
            'accepted': True,
            'message': 'passed',
            'wellname': preflight_request['wellname'],
            'expected_deck_pos': 4,
            'expected_loc': 'B3',
            'base_chemical_name': preflight_request['base_chemical_name'],
            'trigger_chemical_name': preflight_request['trigger_chemical_name'],
            'plate_mapping_revision': 0,
            'plate_generation': 0,
            'pipette_arm': 'right',
            'mix_volume_uL': 150.0,
            'total_cycle_count': 10,
            'well_clearance_mm': 2.0,
            'required_new_tips': 11,
            'available_new_tips': 11,
            'tip_policy': 'dedicated_discarded'
        }
        self.assertTrue(validate_mix_calibration_preflight_acknowledgement(
            preflight_ack, preflight_request
        )['accepted'])

        action_request = _base_request()
        action_request.update({
            'action_id': 'cycle-01', 'preflight_id': 'preflight-a',
            'cycle_index': 1
        })
        action_ack = {
            'schema_version': 2,
            'record_type': 'auto_calibration_well_mixed',
            'action_id': 'cycle-01',
            'calibration_id': action_request['calibration_id'],
            'preflight_id': 'preflight-a',
            'accepted': True,
            'message': 'completed',
            'cycle_index': 1,
            'wellname': action_request['wellname'],
            'expected_deck_pos': 4,
            'expected_loc': 'B3',
            'plate_mapping_revision': 0,
            'plate_generation': 0,
            'pipette_arm': 'right',
            'mix_volume_uL': 150.0,
            'well_clearance_mm': 2.0,
            'required_new_tips': 2,
            'available_new_tips': 11,
            'tip_policy': 'dedicated_discarded',
            'physical_execution_started': True,
            'started_at_utc': '2026-10-16T12:00:00+00:00',
            'completed_at_utc': '2026-10-16T12:00:02+00:00'
        }
        self.assertTrue(validate_mix_calibration_acknowledgement(
            action_ack, action_request
        )['accepted'])

    def test_capability_is_opt_in_and_requires_exact_packet_pairing(self):
        snapshot = {
            'auto_mix_calibration_protocol_version': (
                MIX_CALIBRATION_PROTOCOL_VERSION
            ),
            'supported_commands': [
                MIX_CALIBRATION_PREFLIGHT_COMMAND, MIX_CALIBRATION_COMMAND
            ]
        }
        packets = {
            'preflight_auto_mix_calibration': b'\x21',
            'auto_mix_calibration_preflight': b'\x22',
            'mix_auto_calibration_well': b'\x23',
            'auto_calibration_well_mixed': b'\x24',
        }
        self.assertTrue(validate_mix_calibration_capability(snapshot, packets))
        packets['mix_auto_calibration_well'] = b'\x25'
        with self.assertRaises(AutoMixCalibrationProtocolError):
            validate_mix_calibration_capability(snapshot, packets)


if __name__ == '__main__':
    unittest.main()
