'''Hardware-free C2c binding/resource-preflight contract tests.'''

import ast
import copy
import os
import unittest

from auto_mix_calibration import (
    MIX_CALIBRATION_ASSAY_PROFILE_HEADER,
    MIX_CALIBRATION_FIRST_REAGENT_HEADER,
    MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED,
    MIX_CALIBRATION_SECOND_REAGENT_HEADER,
    parse_auto_mix_calibration_header_settings,
)
from auto_mix_calibration_plan import build_auto_mix_calibration_run_plan
from auto_mix_calibration_resources import (
    AutoMixCalibrationResourceError,
    MIX_CALIBRATION_BIND_WELL_COMMAND,
    MIX_CALIBRATION_RESOURCE_PREFLIGHT_COMMAND,
    MIX_CALIBRATION_RESOURCE_PROTOCOL_VERSION,
    build_auto_mix_calibration_resource_preflight_request,
    build_auto_mix_calibration_well_binding_request,
    validate_auto_mix_calibration_resource_capability,
    validate_auto_mix_calibration_resource_preflight_acknowledgement,
    validate_auto_mix_calibration_well_binding_acknowledgement,
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
    )


def _binding_ack(request, accepted=True):
    return {
        'schema_version': 1,
        'record_type': 'auto_mix_calibration_well_bound',
        'binding_id': request['binding_id'],
        'calibration_id': request['calibration_id'],
        'accepted': accepted,
        'message': 'bound' if accepted else 'well is occupied',
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
    source_requirements = []
    ordinary_tips = []
    mix_tips = []
    if accepted:
        source_requirements = [dict(row, available_volume_uL=500.0)
                               for row in request['source_requirements']]
        ordinary_tips = [dict(row, available_new_tips=96)
                         for row in request[
                             'ordinary_transfer_tip_requirements'
                         ]]
        mix_tips = [dict(row, available_new_tips=95)
                    for row in request['dedicated_mix_tip_requirements']]
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
        'message': 'all requirements passed' if accepted else 'tip deficit',
        'source_requirements': source_requirements,
        'ordinary_transfer_tip_requirements': ordinary_tips,
        'dedicated_mix_tip_requirements': mix_tips,
    }


class AutoMixCalibrationResourceTests(unittest.TestCase):
    def test_vacant_well_binding_then_exact_resource_preflight(self):
        plan = _plan()
        binding_request = build_auto_mix_calibration_well_binding_request(plan)
        binding_ack = validate_auto_mix_calibration_well_binding_acknowledgement(
            _binding_ack(binding_request), binding_request
        )
        self.assertTrue(binding_ack['binding_created'])
        resource_request = build_auto_mix_calibration_resource_preflight_request(
            plan, binding_request['binding_id']
        )
        self.assertEqual('qc_two_dye_5ul_red',
                         resource_request['assay_profile'])
        self.assertEqual(5.0, resource_request['source_requirements'][0][
            'required_volume_uL'
        ])
        self.assertEqual(195.0, resource_request['source_requirements'][1][
            'required_volume_uL'
        ])
        result = validate_auto_mix_calibration_resource_preflight_acknowledgement(
            _resource_ack(resource_request), resource_request
        )
        self.assertTrue(result['accepted'])

    def test_rejected_binding_cannot_claim_a_created_container(self):
        request = build_auto_mix_calibration_well_binding_request(_plan())
        rejected = _binding_ack(request, accepted=False)
        self.assertFalse(
            validate_auto_mix_calibration_well_binding_acknowledgement(
                rejected, request
            )['accepted']
        )
        invalid = copy.deepcopy(rejected)
        invalid['binding_created'] = True
        with self.assertRaises(AutoMixCalibrationResourceError):
            validate_auto_mix_calibration_well_binding_acknowledgement(
                invalid, request
            )

    def test_water_safety_profile_retains_its_reviewed_20_180_allocation(self):
        settings = parse_auto_mix_calibration_header_settings({
            'auto_mix_calibration_mode': 'water_safety',
            MIX_CALIBRATION_FIRST_REAGENT_HEADER: 'water_first',
            MIX_CALIBRATION_SECOND_REAGENT_HEADER: 'water_second',
            'auto_mix_calibration_target_reader_well': 'C4',
        })
        plan = build_auto_mix_calibration_run_plan(
            settings=settings,
            calibration_id='water-calibration-001',
            logical_wellname='autowell-calibration-002',
            source_container_names={
                'first': 'WaterFirstC1.0', 'second': 'WaterSecondC1.0'
            },
            reader_location_mapping={'C4': ('B3', 'platereader4')},
            plate_mapping_revision=4,
            plate_generation=2,
        )
        request = build_auto_mix_calibration_resource_preflight_request(
            plan, 'water-calibration-001-binding'
        )
        self.assertEqual(20.0, request['source_requirements'][0][
            'required_volume_uL'
        ])
        self.assertEqual(180.0, request['source_requirements'][1][
            'required_volume_uL'
        ])

    def test_insufficient_accepted_source_or_tip_evidence_is_rejected(self):
        request = build_auto_mix_calibration_resource_preflight_request(
            _plan(), 'dye-calibration-001-binding'
        )
        for mutate in (
                lambda ack: ack['source_requirements'][1].update(
                    available_volume_uL=194.0
                ),
                lambda ack: ack['ordinary_transfer_tip_requirements'][0].update(
                    available_new_tips=0
                ),
                lambda ack: ack['dedicated_mix_tip_requirements'][0].update(
                    available_new_tips=10
                ),
        ):
            with self.subTest(mutate=mutate):
                acknowledgement = _resource_ack(request)
                mutate(acknowledgement)
                with self.assertRaises(AutoMixCalibrationResourceError):
                    validate_auto_mix_calibration_resource_preflight_acknowledgement(
                        acknowledgement, request
                    )

    def test_rejection_cannot_smuggle_a_partial_allocation(self):
        request = build_auto_mix_calibration_resource_preflight_request(
            _plan(), 'dye-calibration-001-binding'
        )
        acknowledgement = _resource_ack(request, accepted=False)
        acknowledgement['source_requirements'] = [
            dict(request['source_requirements'][0], available_volume_uL=500.0)
        ]
        with self.assertRaises(AutoMixCalibrationResourceError):
            validate_auto_mix_calibration_resource_preflight_acknowledgement(
                acknowledgement, request
            )

    def test_v2_capability_requires_new_packet_pairs_and_commands(self):
        snapshot = {
            'auto_mix_calibration_protocol_version': (
                MIX_CALIBRATION_RESOURCE_PROTOCOL_VERSION
            ),
            'supported_commands': [
                'preflight_auto_mix_calibration',
                'mix_auto_calibration_well',
                MIX_CALIBRATION_BIND_WELL_COMMAND,
                MIX_CALIBRATION_RESOURCE_PREFLIGHT_COMMAND,
            ],
        }
        packet_registry = {
            'preflight_auto_mix_calibration': b'\x21',
            'auto_mix_calibration_preflight': b'\x22',
            'mix_auto_calibration_well': b'\x23',
            'auto_calibration_well_mixed': b'\x24',
            'bind_auto_mix_calibration_well': b'\x25',
            'auto_mix_calibration_well_bound': b'\x26',
            'preflight_auto_mix_calibration_resources': b'\x27',
            'auto_mix_calibration_resources_preflight': b'\x28',
        }
        self.assertTrue(validate_auto_mix_calibration_resource_capability(
            snapshot, packet_registry
        ))
        incomplete = copy.deepcopy(snapshot)
        incomplete['supported_commands'] = [
            'preflight_auto_mix_calibration',
            'mix_auto_calibration_well',
            MIX_CALIBRATION_BIND_WELL_COMMAND
        ]
        with self.assertRaises(AutoMixCalibrationResourceError):
            validate_auto_mix_calibration_resource_capability(
                incomplete, packet_registry
            )

    def test_local_armchair_registry_contains_exact_v2_ghost_packet_pairs(self):
        repository_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
        source_path = os.path.join(repository_root, 'Armchair', 'armchair.py')
        with open(source_path, encoding='utf-8') as source_file:
            source_text = source_file.read()
        ast.parse(source_text)
        # Parse-free string checks keep this contract test independent of the
        # legacy runtime's optional bidict package.
        for name, code in (
                ('bind_auto_mix_calibration_well', "b'\\x25'"),
                ('auto_mix_calibration_well_bound', "b'\\x26'"),
                ('preflight_auto_mix_calibration_resources', "b'\\x27'"),
                ('auto_mix_calibration_resources_preflight', "b'\\x28'")):
            self.assertIn("'{}':{}".format(name, code), source_text)


if __name__ == '__main__':
    unittest.main()
