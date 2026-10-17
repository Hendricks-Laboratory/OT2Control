'''Synthetic tests for the isolated calibration execution boundary.

No test imports ``controller.py`` or opens a reader/socket.  The adapter below
records calls so the controlled 5/195 -> baseline -> (mix, scan) x10 order can
be proven without any physical side effect.
'''

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
from auto_mix_calibration_execution import (
    AutoMixCalibrationExecutionError,
    execute_auto_mix_calibration_plan,
)
from auto_mix_calibration_plan import build_auto_mix_calibration_run_plan


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


def _preflight_ack(request, accepted=True):
    return {
        'schema_version': 1,
        'record_type': 'auto_mix_calibration_preflight',
        'preflight_id': request['preflight_id'],
        'calibration_id': request['calibration_id'],
        'accepted': accepted,
        'message': 'passed' if accepted else 'declined',
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
        'required_new_tips': 11,
        'available_new_tips': 11,
        'tip_policy': 'dedicated_discarded',
    }


def _mix_ack(request, accepted=True):
    return {
        'schema_version': 1,
        'record_type': 'auto_calibration_well_mixed',
        'action_id': request['action_id'],
        'calibration_id': request['calibration_id'],
        'preflight_id': request['preflight_id'],
        'accepted': accepted,
        'message': 'completed' if accepted else 'declined',
        'cycle_index': request['cycle_index'],
        'wellname': request['wellname'],
        'expected_deck_pos': request['expected_deck_pos'],
        'expected_loc': request['expected_loc'],
        'plate_mapping_revision': request['expected_plate_mapping_revision'],
        'plate_generation': request['expected_plate_generation'],
        'pipette_arm': 'right',
        'mix_volume_uL': 150.0,
        'well_clearance_mm': 2.0,
        'required_new_tips': 1 if accepted else 0,
        'available_new_tips': 11,
        'tip_policy': 'dedicated_discarded',
        'physical_execution_started': accepted,
        'started_at_utc': '2026-10-16T12:00:00+00:00',
        'completed_at_utc': '2026-10-16T12:00:02+00:00',
    }


class _Adapter(object):
    def __init__(self):
        self.calls = []
        self.resource_preflight = {
            'approved': True,
            'message': 'all sources, tips, reader wavelengths, and plate match',
            'source_requirements': [
                {
                    'slot': 'first', 'source_container': 'AcidRed1C1.44',
                    'required_volume_uL': 5.0, 'available_volume_uL': 50.0,
                },
                {
                    'slot': 'second', 'source_container': 'CopperSulfateDyeC8.0',
                    'required_volume_uL': 195.0, 'available_volume_uL': 500.0,
                },
            ],
            'ordinary_transfer_tip_requirements': [
                {'pipette': 'p20', 'required_new_tips': 1,
                 'available_new_tips': 96},
                {'pipette': 'p300', 'required_new_tips': 1,
                 'available_new_tips': 96},
            ],
            'dedicated_mix_tip_requirements': [
                {'pipette': 'p300', 'required_new_tips': 11,
                 'available_new_tips': 95},
            ],
            'reader_wavelengths_nm': [300, 520, 625, 730, 1000],
            'plate_mapping_revision': 4,
            'plate_generation': 2,
        }
        self.reject_preflight = False
        self.reject_mix_cycle = None

    def preflight_resources(self, plan):
        self.calls.append(('resource-preflight', plan['calibration_id']))
        return copy.deepcopy(self.resource_preflight)

    def request_pi_preflight(self, request):
        self.calls.append(('pi-preflight', request['preflight_id']))
        return _preflight_ack(request, accepted=not self.reject_preflight)

    def transfer(self, source, destination, volume_uL):
        self.calls.append(('transfer', source, destination, volume_uL))

    def scan(self, scan_step):
        self.calls.append(('scan', scan_step['scan_index']))
        return {'scan_index': scan_step['scan_index'], 'raw_scan_id': (
            'calibration-scan-{:02d}'.format(scan_step['scan_index'])
        )}

    def request_pi_mix(self, request):
        self.calls.append(('mix', request['cycle_index']))
        accepted = request['cycle_index'] != self.reject_mix_cycle
        return _mix_ack(request, accepted=accepted)


class AutoMixCalibrationExecutionTests(unittest.TestCase):
    def test_two_dye_execution_is_exactly_ordered_without_retry(self):
        adapter = _Adapter()
        record = execute_auto_mix_calibration_plan(_plan(), adapter)
        self.assertEqual(11, len(record['scan_records']))
        self.assertEqual(10, len(record['mix_acknowledgements']))
        self.assertEqual([
            ('resource-preflight', 'dye-calibration-001'),
            ('pi-preflight', 'dye-calibration-001-preflight'),
            ('transfer', 'AcidRed1C1.44', 'autowell-calibration-001', 5.0),
            ('transfer', 'CopperSulfateDyeC8.0', 'autowell-calibration-001',
             195.0),
            ('scan', 0), ('mix', 1), ('scan', 1), ('mix', 2), ('scan', 2),
        ], adapter.calls[:9])
        self.assertEqual(('mix', 10), adapter.calls[-2])
        self.assertEqual(('scan', 10), adapter.calls[-1])

    def test_incomplete_resource_proof_blocks_before_pi_or_transfer(self):
        adapter = _Adapter()
        adapter.resource_preflight['ordinary_transfer_tip_requirements'][0][
            'available_new_tips'
        ] = 0
        with self.assertRaises(AutoMixCalibrationExecutionError):
            execute_auto_mix_calibration_plan(_plan(), adapter)
        self.assertEqual([('resource-preflight', 'dye-calibration-001')],
                         adapter.calls)

    def test_missing_analysis_wavelength_blocks_before_liquid_handling(self):
        adapter = _Adapter()
        adapter.resource_preflight['reader_wavelengths_nm'] = [300, 520, 1000]
        with self.assertRaises(AutoMixCalibrationExecutionError):
            execute_auto_mix_calibration_plan(_plan(), adapter)
        self.assertEqual([('resource-preflight', 'dye-calibration-001')],
                         adapter.calls)

    def test_pi_preflight_rejection_blocks_transfers(self):
        adapter = _Adapter()
        adapter.reject_preflight = True
        with self.assertRaises(AutoMixCalibrationExecutionError):
            execute_auto_mix_calibration_plan(_plan(), adapter)
        self.assertEqual(['resource-preflight', 'pi-preflight'],
                         [call[0] for call in adapter.calls])

    def test_mix_rejection_is_not_retried_or_followed_by_scan(self):
        adapter = _Adapter()
        adapter.reject_mix_cycle = 3
        with self.assertRaises(AutoMixCalibrationExecutionError):
            execute_auto_mix_calibration_plan(_plan(), adapter)
        self.assertEqual(1, adapter.calls.count(('mix', 3)))
        self.assertNotIn(('scan', 3), adapter.calls)
        self.assertNotIn(('mix', 4), adapter.calls)

    def test_normal_auto_constructor_has_a_calibration_only_rejection_gate(self):
        repository_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
        controller_path = os.path.join(repository_root, 'controller.py')
        with open(controller_path, encoding='utf-8') as source_file:
            tree = ast.parse(source_file.read())
        auto_class = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef)
                          and node.name == 'AutoContr')
        method_names = set(node.name for node in auto_class.body
                           if isinstance(node, ast.FunctionDef))
        self.assertIn('_initialize_auto_mix_calibration_configuration',
                      method_names)
        init_method = next(node for node in auto_class.body
                           if isinstance(node, ast.FunctionDef)
                           and node.name == '__init__')
        calls = [node.func.attr for node in ast.walk(init_method)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        self.assertIn('_initialize_auto_mix_calibration_configuration', calls)

    def test_calibration_runner_records_phase_and_traceback_on_failure(self):
        repository_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
        controller_path = os.path.join(repository_root, 'controller.py')
        with open(controller_path, encoding='utf-8') as source_file:
            tree = ast.parse(source_file.read())
        calibration_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == 'AutoMixCalibrationContr'
        )
        run_protocol = next(
            node for node in calibration_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'run_protocol'
        )
        decorator_names = [
            decorator.id for decorator in run_protocol.decorator_list
            if isinstance(decorator, ast.Name)
        ]
        self.assertIn('terminal_output_capture_guard', decorator_names)
        string_values = [
            node.value for node in ast.walk(run_protocol)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        self.assertIn(
            'Pi initialization acknowledgement',
            string_values
        )
        self.assertIn(
            'Auto-main compatibility snapshot',
            string_values
        )
        self.assertIn(
            '<<controller>> mix calibration failed during {}: {}: {}',
            string_values
        )

    def test_execution_boundary_remains_hardware_free(self):
        repository_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
        execution_path = os.path.join(
            repository_root, 'auto_mix_calibration_execution.py'
        )
        with open(execution_path, encoding='utf-8') as source_file:
            tree = ast.parse(source_file.read())
        imported_roots = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split('.')[0]
                                      for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split('.')[0])
        self.assertFalse(imported_roots.intersection(set((
            'controller', 'ot2_robot', 'Armchair', 'socket', 'serial'
        ))))


if __name__ == '__main__':
    unittest.main()
