'''Hardware-free Stage-13C wire-contract tests.'''

import unittest

from auto_stability_pipette_mix import (
    AutoStabilityPipetteMixContractError,
    validate_targeted_mix_acknowledgement,
    validate_targeted_mix_request,
)


def _request():
    return {
        'schema_version': 1,
        'action_id': 'auto-stability-pipette-mix-001',
        'wellname': 'autowell0C1.0',
        'expected_deck_pos': 4,
        'expected_loc': 'A1',
        'expected_plate_mapping_revision': 3,
        'expected_plate_generation': 1,
        'trigger_chemical_name': 'sodium_borohydrideC6.25',
        'mix_volume_uL': 20.0,
        'cycle_count': 1
    }


def _acknowledgement(request=None):
    request = request or _request()
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


class TargetedMixWireContractTests(unittest.TestCase):
    def test_request_normalizes_location_and_preserves_exact_identity(self):
        request = _request()
        request['expected_loc'] = ' a1 '
        normalized = validate_targeted_mix_request(request)
        self.assertEqual('A1', normalized['expected_loc'])
        self.assertEqual(20.0, normalized['mix_volume_uL'])

    def test_request_rejects_non_auto_or_non_reader_targets(self):
        for field_name, value in (
                ('wellname', 'manualwell0'),
                ('expected_deck_pos', 3),
                ('cycle_count', 0),
                ('cycle_count', 11)):
            request = _request()
            request[field_name] = value
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaises(AutoStabilityPipetteMixContractError):
                    validate_targeted_mix_request(request)

    def test_completed_acknowledgement_requires_exact_request_and_timing(self):
        request = _request()
        accepted = validate_targeted_mix_acknowledgement(
            _acknowledgement(request), request
        )
        self.assertTrue(accepted['accepted'])

        for field_name, value in (
                ('wellname', 'autowell999C1.0'),
                ('plate_generation', 2),
                ('mix_volume_uL', 21.0),
                ('tip_policy', 'shared_tip'),
                ('completed_at_utc', '2026-10-02T11:59:59+00:00')):
            acknowledgement = _acknowledgement(request)
            acknowledgement[field_name] = value
            with self.subTest(field_name=field_name):
                with self.assertRaises(AutoStabilityPipetteMixContractError):
                    validate_targeted_mix_acknowledgement(acknowledgement, request)

    def test_clean_pi_rejection_is_parseable_but_not_successful(self):
        request = _request()
        rejection = _acknowledgement(request)
        rejection.update({
            'accepted': False,
            'message': 'target tip inventory is insufficient',
            'pipette_arm': None,
            'mix_volume_uL': None,
            'cycle_count': None,
            'required_new_tips': None,
            'available_new_tips': None,
            'physical_execution_started': False,
            'started_at_utc': None,
            'completed_at_utc': None
        })
        validated = validate_targeted_mix_acknowledgement(rejection, request)
        self.assertFalse(validated['accepted'])


if __name__ == '__main__':
    unittest.main()
