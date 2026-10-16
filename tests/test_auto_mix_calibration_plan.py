'''Hardware-free contract tests for the calibration-only execution plan.'''

import ast
import copy
import pathlib
import unittest

from auto_mix_calibration import parse_auto_mix_calibration_header_settings
from auto_mix_calibration_plan import (
    AutoMixCalibrationPlanError,
    MIX_CALIBRATION_EXPECTED_SCAN_COUNT,
    build_auto_mix_calibration_run_plan,
    resolve_auto_mix_calibration_target,
    validate_auto_mix_calibration_run_plan,
)


def _settings():
    return parse_auto_mix_calibration_header_settings({
        'auto_mix_calibration_mode': 'two_dye',
        'auto_mix_calibration_base_reagent': 'Copper EDTA Base',
        'auto_mix_calibration_trigger_reagent': 'Acid Red 1',
        'auto_mix_calibration_target_reader_well': 'C4',
    })


def _mapping():
    # Matches the established controller translator for reader C4.
    return {'C4': ('B3', 'platereader4')}


def _plan():
    return build_auto_mix_calibration_run_plan(
        settings=_settings(),
        calibration_id='dye-calibration-001',
        logical_wellname='autowell-calibration-001',
        source_container_names={
            'base': 'CopperEDTABaseC1.0',
            'trigger': 'AcidRed1C1.0',
        },
        reader_location_mapping=_mapping(),
        plate_mapping_revision=4,
        plate_generation=2,
    )


class AutoMixCalibrationPlanTests(unittest.TestCase):
    def test_plan_module_remains_execution_free(self):
        source_path = pathlib.Path(__file__).resolve().parents[1] / (
            'auto_mix_calibration_plan.py'
        )
        source_tree = ast.parse(source_path.read_text())
        imported_roots = set()
        for node in source_tree.body:
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split('.')[0]
                                      for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split('.')[0])
        self.assertFalse(imported_roots.intersection(
            set(('controller', 'ot2_robot', 'Armchair', 'socket', 'serial'))
        ))

    def test_reader_target_uses_one_user_coordinate_and_derives_pi_identity(self):
        target = resolve_auto_mix_calibration_target('c4', _mapping())
        self.assertEqual('C4', target['reader_well'])
        self.assertEqual('B3', target['expected_loc'])
        self.assertEqual(4, target['expected_deck_pos'])
        self.assertEqual('platereader4', target['expected_labware_name'])

    def test_exact_plan_is_base_trigger_baseline_then_ten_mix_scan_pairs(self):
        plan = _plan()
        self.assertFalse(plan['normal_auto_execution_permitted'])
        self.assertFalse(plan['normal_stability_execution_permitted'])
        self.assertEqual(200.0, plan['expected_final_well_volume_uL'])
        self.assertEqual(11, plan['expected_scan_count'])
        self.assertEqual(MIX_CALIBRATION_EXPECTED_SCAN_COUNT,
                         len(plan['scan_steps']))
        self.assertEqual(24, len(plan['execution_steps']))

        transfers = plan['transfer_steps']
        self.assertEqual('CopperEDTABaseC1.0',
                         transfers[0]['source_chemical_name'])
        self.assertEqual(180.0, transfers[0]['volume_uL'])
        self.assertEqual('AcidRed1C1.0',
                         transfers[1]['source_chemical_name'])
        self.assertEqual(20.0, transfers[1]['volume_uL'])
        self.assertEqual('transfer-base', transfers[1]['requires_step_id'])

        baseline = plan['scan_steps'][0]
        self.assertEqual(0, baseline['scan_index'])
        self.assertEqual(0, baseline['cycle_index'])
        self.assertEqual('baseline_after_trigger_before_mix',
                         baseline['observation_reason'])
        self.assertEqual('transfer-trigger', baseline['requires_step_id'])
        self.assertFalse(baseline['record_in_aggregate'])

        action_ids = set()
        for cycle_index, (mix_step, scan_step) in enumerate(zip(
                plan['mix_steps'], plan['scan_steps'][1:]), start=1):
            self.assertEqual(cycle_index, mix_step['cycle_index'])
            self.assertFalse(mix_step['automatic_retry_permitted'])
            self.assertEqual(cycle_index, scan_step['scan_index'])
            self.assertEqual(cycle_index, scan_step['cycle_index'])
            self.assertEqual('post_mix_cycle_{:02d}'.format(cycle_index),
                             scan_step['observation_reason'])
            self.assertEqual(mix_step['step_id'], scan_step['requires_step_id'])
            action_ids.add(mix_step['request']['action_id'])
        self.assertEqual(10, len(action_ids))
        self.assertEqual('scan-00', plan['mix_steps'][0]['requires_step_id'])
        self.assertEqual('scan-09', plan['mix_steps'][-1]['requires_step_id'])

    def test_tampering_cannot_create_a_different_sequence_or_aggregate_scan(self):
        for mutator in (
                lambda plan: plan['transfer_steps'][0].update(volume_uL=179.0),
                lambda plan: plan['mix_steps'][1]['request'].update(cycle_index=1),
                lambda plan: plan['scan_steps'][1].update(
                    record_in_aggregate=True
                ),
                lambda plan: plan.update(normal_auto_execution_permitted=True),
                lambda plan: plan['execution_steps'].__setitem__(
                    4, copy.deepcopy(plan['execution_steps'][3])
                ),
        ):
            with self.subTest(mutator=mutator):
                altered = copy.deepcopy(_plan())
                mutator(altered)
                with self.assertRaises(AutoMixCalibrationPlanError):
                    validate_auto_mix_calibration_run_plan(altered)

    def test_malformed_mapping_or_ambiguous_source_identity_fails_before_plan(self):
        with self.assertRaises(AutoMixCalibrationPlanError):
            resolve_auto_mix_calibration_target('C4', {'C4': ('B3', 'deck4')})

        with self.assertRaises(AutoMixCalibrationPlanError):
            build_auto_mix_calibration_run_plan(
                settings=_settings(),
                calibration_id='dye-calibration-001',
                logical_wellname='autowell-calibration-001',
                source_container_names={
                    'base': 'SameContainerC1.0',
                    'trigger': 'samecontainerc1.0',
                },
                reader_location_mapping=_mapping(),
                plate_mapping_revision=4,
                plate_generation=2,
            )


if __name__ == '__main__':
    unittest.main()
