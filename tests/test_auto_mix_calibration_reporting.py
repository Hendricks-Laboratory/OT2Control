'''Synthetic tests for C2eB execution-after-preflight and C3 exports.

These tests use only contracts, fake adapters, and temporary files. They never
import the controller, contact a reader, open a socket, or initialize hardware.
'''

import ast
import copy
import json
import os
import tempfile
import unittest

import pandas as pd

from auto_mix_calibration import (
    MIX_CALIBRATION_ASSAY_PROFILE_HEADER,
    MIX_CALIBRATION_FIRST_REAGENT_HEADER,
    MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED,
    MIX_CALIBRATION_SECOND_REAGENT_HEADER,
    parse_auto_mix_calibration_header_settings,
)
from auto_mix_calibration_execution import (
    AutoMixCalibrationExecutionError,
    execute_auto_mix_calibration_plan_after_preflight,
)
from auto_mix_calibration_plan import build_auto_mix_calibration_run_plan
from auto_mix_calibration_reporting import (
    AutoMixCalibrationReporter,
    AutoMixCalibrationReportingError,
    reindex_calibration_reader_spectrum_from_raw_file,
)
from auto_mix_calibration_resources import (
    build_auto_mix_calibration_resource_preflight_request,
    build_auto_mix_calibration_well_binding_request,
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
        calibration_id='dye-calibration-002',
        logical_wellname='autowell-calibration-002',
        source_container_names={
            'first': 'acid_red_1C1.44',
            'second': 'copper_sulfate_dyeC8.0',
        },
        reader_location_mapping={'C4': ('B3', 'platereader4')},
        plate_mapping_revision=4,
        plate_generation=2,
        deck_reader_well='C4',
    )


def _pi_preflight_ack(request, accepted=True):
    return {
        'schema_version': 2,
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


def _mix_ack(request):
    return {
        'schema_version': 2,
        'record_type': 'auto_calibration_well_mixed',
        'action_id': request['action_id'],
        'calibration_id': request['calibration_id'],
        'preflight_id': request['preflight_id'],
        'accepted': True,
        'message': 'completed',
        'cycle_index': request['cycle_index'],
        'wellname': request['wellname'],
        'expected_deck_pos': request['expected_deck_pos'],
        'expected_loc': request['expected_loc'],
        'plate_mapping_revision': request['expected_plate_mapping_revision'],
        'plate_generation': request['expected_plate_generation'],
        'pipette_arm': 'right',
        'mix_volume_uL': 150.0,
        'well_clearance_mm': 2.0,
        'required_new_tips': 1,
        'available_new_tips': 11,
        'tip_policy': 'dedicated_discarded',
        'physical_execution_started': True,
        'started_at_utc': '2026-10-16T12:00:00+00:00',
        'completed_at_utc': '2026-10-16T12:00:02+00:00',
    }


def _transfer_ack(request, accepted=True):
    acknowledgement = dict(request)
    acknowledgement.update({
        'record_type': 'auto_mix_calibration_source_transferred',
        'accepted': accepted,
        'message': 'completed' if accepted else 'declined',
        'pipette_arm': (
            'left' if request['transfer_index'] == 1 else 'right'
        ) if accepted else None,
        'physical_execution_started': accepted,
        'started_at_utc': (
            '2026-10-16T12:00:00+00:00' if accepted else None
        ),
        'completed_at_utc': (
            '2026-10-16T12:00:01+00:00' if accepted else None
        ),
    })
    return acknowledgement


def _preflight_record(plan):
    binding_request = build_auto_mix_calibration_well_binding_request(plan)
    resource_request = build_auto_mix_calibration_resource_preflight_request(
        plan, binding_request['binding_id']
    )
    binding_ack = {
        'schema_version': 1,
        'record_type': 'auto_mix_calibration_well_bound',
        'binding_id': binding_request['binding_id'],
        'calibration_id': binding_request['calibration_id'],
        'accepted': True,
        'message': 'bound',
        'wellname': binding_request['wellname'],
        'reader_well': binding_request['reader_well'],
        'expected_deck_pos': binding_request['expected_deck_pos'],
        'expected_loc': binding_request['expected_loc'],
        'plate_mapping_revision': binding_request[
            'expected_plate_mapping_revision'
        ],
        'plate_generation': binding_request['expected_plate_generation'],
        'well_was_vacant': True,
        'binding_created': True,
    }
    resource_ack = dict(resource_request)
    resource_ack.update({
        'record_type': 'auto_mix_calibration_resources_preflight',
        'accepted': True,
        'message': 'all reviewed resources available',
        'source_requirements': [
            dict(record, available_volume_uL=500.0)
            for record in resource_request['source_requirements']
        ],
        'ordinary_transfer_tip_requirements': [
            dict(record, available_new_tips=96)
            for record in resource_request['ordinary_transfer_tip_requirements']
        ],
        'dedicated_mix_tip_requirements': [
            dict(record, available_new_tips=96)
            for record in resource_request['dedicated_mix_tip_requirements']
        ],
    })
    return {
        'record_version': 1,
        'record_type': 'auto_mix_calibration_preflight',
        'created_at_utc': '2026-10-16T12:00:00+00:00',
        'calibration_id': plan['calibration_id'],
        'logical_wellname': plan['logical_wellname'],
        'assay_profile': plan['assay_profile'],
        'target': copy.deepcopy(plan['target']),
        'plate_mapping_revision': plan['plate_mapping_revision'],
        'plate_generation': plan['plate_generation'],
        'reader_scan_protocol': 'UVVis',
        'reader_wavelength_coverage_status': 'deferred_until_first_raw_scan',
        'expected_scan_count': plan['expected_scan_count'],
        'pi_snapshot': {'snapshot': 'synthetic'},
        'pi_preflight': _pi_preflight_ack(plan['preflight_request']),
        'well_binding': binding_ack,
        'resource_preflight': resource_ack,
        'physical_execution_started': False,
        'automatic_retry_permitted': False,
    }


class _Adapter(object):
    def __init__(self, include_730=True):
        self.include_730 = include_730
        self.calls = []

    def request_pi_transfer(self, request):
        self.calls.append((
            'transfer', request['transfer_index'],
            request['source_chemical_name'], request['destination_wellname'],
            request['volume_uL']
        ))
        return _transfer_ack(request)

    def request_pi_mix(self, request):
        self.calls.append(('mix', request['cycle_index']))
        return _mix_ack(request)

    def scan(self, scan_step):
        self.calls.append(('scan', scan_step['scan_index']))
        wavelengths = [300.0, 520.0, 1000.0]
        if self.include_730:
            wavelengths.insert(2, 730.0)
        return {
            'scan_index': scan_step['scan_index'],
            'cycle_index': scan_step['cycle_index'],
            'reader_well': scan_step['reader_well'],
            'logical_wellname': scan_step['wellname'],
            'raw_scan_relative_path': 'mix_calibration/raw_scans/scan_{:02d}.csv'.format(
                scan_step['scan_index']
            ),
            'scan_started_at_utc': '2026-10-16T12:00:00+00:00',
            'scan_completed_at_utc': '2026-10-16T12:00:01+00:00',
            'wavelengths_nm': wavelengths,
        }


class AutoMixCalibrationC2eBC3Tests(unittest.TestCase):
    def test_raw_reader_declaration_reindexes_legacy_positional_spectrum(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = os.path.join(directory, 'scan_00.csv')
            with open(raw_path, 'w', encoding='latin-1') as raw_file:
                raw_file.write('Wavelengths: 300nm...1000nm, 1nm steps\n')
            spectrum = pd.Series(
                [float(index) for index in range(701)],
                index=range(1, 702)
            )
            reindexed = reindex_calibration_reader_spectrum_from_raw_file(
                spectrum, raw_path
            )
            self.assertEqual(300.0, reindexed.index[0])
            self.assertEqual(1000.0, reindexed.index[-1])
            self.assertEqual(220.0, reindexed.loc[520.0])
            self.assertEqual(430.0, reindexed.loc[730.0])

    def test_raw_reader_declaration_rejects_mismatched_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = os.path.join(directory, 'scan_00.csv')
            with open(raw_path, 'w', encoding='latin-1') as raw_file:
                raw_file.write('Wavelengths: 300nm...302nm, 1nm steps\n')
            with self.assertRaisesRegex(
                    AutoMixCalibrationReportingError, 'sample count'):
                reindex_calibration_reader_spectrum_from_raw_file(
                    pd.Series([0.1, 0.2], index=[1, 2]), raw_path
                )

    def test_raw_reader_declaration_rejects_missing_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = os.path.join(directory, 'scan_00.csv')
            with open(raw_path, 'w', encoding='latin-1') as raw_file:
                raw_file.write('No wavelength declaration\n')
            with self.assertRaisesRegex(
                    AutoMixCalibrationReportingError, 'does not declare'):
                reindex_calibration_reader_spectrum_from_raw_file(
                    pd.Series([0.1], index=[1]), raw_path
                )

    def test_execution_consumes_preflight_then_orders_transfer_scan_mix(self):
        plan = _plan()
        adapter = _Adapter()
        record = execute_auto_mix_calibration_plan_after_preflight(
            plan, _preflight_record(plan), adapter
        )
        self.assertEqual(11, len(record['scan_records']))
        self.assertEqual(10, len(record['mix_acknowledgements']))
        self.assertEqual(2, len(record['transfer_acknowledgements']))
        self.assertEqual([
            ('transfer', 1, 'acid_red_1C1.44',
             'autowell-calibration-002', 5.0),
            ('transfer', 2, 'copper_sulfate_dyeC8.0',
             'autowell-calibration-002', 195.0),
            ('scan', 0), ('mix', 1), ('scan', 1), ('mix', 2), ('scan', 2),
        ], adapter.calls[:7])
        self.assertEqual(('mix', 10), adapter.calls[-2])
        self.assertEqual(('scan', 10), adapter.calls[-1])

    def test_missing_baseline_wavelength_stops_before_first_mix(self):
        plan = _plan()
        adapter = _Adapter(include_730=False)
        with self.assertRaises(AutoMixCalibrationExecutionError):
            execute_auto_mix_calibration_plan_after_preflight(
                plan, _preflight_record(plan), adapter
            )
        self.assertEqual(['transfer', 'transfer', 'scan'], [
            call[0] for call in adapter.calls
        ])

    def test_rejected_c2ea_record_blocks_before_transfer(self):
        plan = _plan()
        record = _preflight_record(plan)
        record['resource_preflight']['accepted'] = False
        record['resource_preflight']['source_requirements'] = []
        record['resource_preflight']['ordinary_transfer_tip_requirements'] = []
        record['resource_preflight']['dedicated_mix_tip_requirements'] = []
        adapter = _Adapter()
        with self.assertRaises(AutoMixCalibrationExecutionError):
            execute_auto_mix_calibration_plan_after_preflight(plan, record, adapter)
        self.assertEqual([], adapter.calls)

    def test_reader_protocol_mismatch_blocks_before_transfer(self):
        plan = _plan()
        adapter = _Adapter()
        with self.assertRaises(AutoMixCalibrationExecutionError):
            execute_auto_mix_calibration_plan_after_preflight(
                plan, _preflight_record(plan), adapter,
                expected_reader_scan_protocol='different-reader-protocol'
            )
        self.assertEqual([], adapter.calls)

    def test_reporter_exports_raw_scan_manifest_metrics_and_plots(self):
        plan = _plan()
        with tempfile.TemporaryDirectory() as directory:
            reporter = AutoMixCalibrationReporter(
                directory, plan, _preflight_record(plan)
            )
            baseline = pd.Series(
                [0.01, 0.20, 0.10, 0.03], index=[300, 520, 730, 1000]
            )
            after_mix = pd.Series(
                [0.01, 0.24, 0.12, 0.03], index=[300, 520, 730, 1000]
            )
            reporter.record_scan(
                plan['scan_steps'][0], baseline,
                '2026-10-16T12:00:00+00:00', '2026-10-16T12:00:01+00:00'
            )
            reporter.record_scan(
                plan['scan_steps'][1], after_mix,
                '2026-10-16T12:00:02+00:00', '2026-10-16T12:00:03+00:00'
            )
            paths = reporter.finalize()
            for path in paths.values():
                self.assertTrue(os.path.exists(path), path)
            with open(paths['manifest_path'], encoding='utf-8') as manifest_file:
                manifest = json.load(manifest_file)
            self.assertEqual('completed', manifest['status'])
            self.assertTrue(manifest['human_mix_count_selection_required'])
            self.assertEqual(2, manifest['completed_scan_count'])
            metrics = pd.read_csv(paths['metrics_path'])
            self.assertEqual([0, 1], metrics['scan_index'].tolist())
            self.assertAlmostEqual(0.04, metrics.loc[1, 'delta_520nm_from_previous'])
            self.assertEqual(0.0, metrics.loc[0, 'elapsed_s_from_baseline_scan_start'])
            self.assertEqual(2.0, metrics.loc[1, 'elapsed_s_from_baseline_scan_start'])

    def test_controller_declares_a_distinct_calibration_route(self):
        repository_root = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
        controller_path = os.path.join(repository_root, 'controller.py')
        with open(controller_path, encoding='utf-8') as source_file:
            source = source_file.read()
        tree = ast.parse(source)
        classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }
        calibration_class = classes['AutoMixCalibrationContr']
        self.assertEqual(['Controller'], [
            base.id for base in calibration_class.bases
            if isinstance(base, ast.Name)
        ])
        methods = {
            node.name for node in calibration_class.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn('run_protocol', methods)
        self.assertIn('_run_auto_mix_calibration_raw_scan', methods)

        raw_scan_method = next(
            node for node in calibration_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == '_run_auto_mix_calibration_raw_scan'
        )
        raw_scan_source = ast.get_source_segment(
            source, raw_scan_method
        )
        self.assertIn("record_in_aggregate=False", raw_scan_source)
        self.assertIn("exec_macro('PlateIn')", raw_scan_source)
        self.assertIn("exec_macro('PlateOut')", raw_scan_source)
        self.assertLess(
            raw_scan_source.index('shutil.move(source_path, destination_path)'),
            raw_scan_source.index(
                'reindex_calibration_reader_spectrum_from_raw_file('
            )
        )
        self.assertLess(
            raw_scan_source.index(
                'reindex_calibration_reader_spectrum_from_raw_file('
            ),
            raw_scan_source.index('reporter.record_scan(')
        )


if __name__ == '__main__':
    unittest.main()
