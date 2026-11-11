'''Pure Stage-14E1 decision-snapshot integrity tests; no controller import.'''

import ast
import csv
import hashlib
import json
import os
import tempfile
import unittest
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

from auto_stability_snapshot_audit import (
    build_stability_decision_snapshot_integrity_audit,
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


def _event(sequence, event_type, **values):
    row = {
        'event_sequence': sequence,
        'event_type': event_type,
    }
    row.update(values)
    return row


def _configuration():
    return {
        'auto_stability_mode': 'stability_only',
        'auto_stability_mixing_mode': 'pipette_mix',
        'auto_stability_scan_schedule': 'each_completion',
        'auto_stability_decision_horizon_s': 300,
        'auto_stability_max_observation_window_s': 600,
    }


def _valid_inputs():
    manifest = [
        _event(1, 'trigger_transfer_completed', wellname='well_a', batch_number='0'),
        _event(2, 'decision_horizon_reached', wellname='well_a', batch_number='0',
               decision_horizon_s='300'),
        _event(3, 'trigger_transfer_completed', wellname='well_b', batch_number='1'),
        _event(4, 'decision_horizon_reached', wellname='well_b', batch_number='1',
               decision_horizon_s='300'),
    ]
    conditions = [
        {
            'condition_id': 'run|0|0', 'batch_number': '0',
            'total_well_count': '1', 'snapshot_status': 'frozen',
            'decision_horizon_s': '300',
        },
        {
            'condition_id': 'run|1|1', 'batch_number': '1',
            'total_well_count': '1', 'snapshot_status': 'frozen',
            'decision_horizon_s': '300',
        },
    ]
    wells = [
        {
            'wellname': 'well_a', 'condition_id': 'run|0|0',
            'batch_number': '0', 'snapshot_status': 'frozen',
            'decision_horizon_s': '300',
        },
        {
            'wellname': 'well_b', 'condition_id': 'run|1|1',
            'batch_number': '1', 'snapshot_status': 'frozen',
            'decision_horizon_s': '300',
        },
    ]
    return manifest, conditions, wells


class Stage14ESnapshotIntegrityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONTROLLER_PATH, encoding='utf-8') as source_file:
            tree = ast.parse(source_file.read(), filename=CONTROLLER_PATH)
        auto_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
        )
        method_names = {
            '_get_auto_stability_file_checksum',
            '_write_auto_stability_decision_snapshot_integrity_audit',
        }
        methods = [
            node for node in auto_class.body
            if isinstance(node, ast.FunctionDef) and node.name in method_names
        ]
        module = ast.fix_missing_locations(ast.Module(
            body=[ast.ClassDef(
                name='AuditWriter', bases=[], keywords=[], body=methods,
                decorator_list=[]
            )],
            type_ignores=[]
        ))
        namespace = {
            'os': os,
            'csv': csv,
            'json': json,
            'hashlib': hashlib,
            'NamedTemporaryFile': NamedTemporaryFile,
            'build_stability_decision_snapshot_integrity_audit': (
                build_stability_decision_snapshot_integrity_audit
            ),
        }
        exec(compile(module, CONTROLLER_PATH, 'exec'), namespace)
        cls.AuditWriter = namespace['AuditWriter']

    def test_valid_frozen_snapshots_are_internally_consistent(self):
        manifest, conditions, wells = _valid_inputs()
        summary = build_stability_decision_snapshot_integrity_audit(
            manifest, conditions, wells, configuration=_configuration()
        )
        self.assertEqual('no_detected_snapshot_violation', summary['status'])
        self.assertTrue(summary['applicable'])
        self.assertEqual(2, summary['snapshot_wells']['unique_well_count'])

    def test_duplicate_well_and_missing_trigger_are_detected(self):
        manifest, conditions, wells = _valid_inputs()
        wells.append(dict(wells[0]))
        wells.append({
            'wellname': 'well_missing', 'condition_id': 'run|0|0',
            'batch_number': '0', 'snapshot_status': 'frozen',
            'decision_horizon_s': '300',
        })
        summary = build_stability_decision_snapshot_integrity_audit(
            manifest, conditions, wells, configuration=_configuration()
        )
        self.assertEqual('snapshot_violations_detected', summary['status'])
        codes = set(item['code'] for item in summary['violations'])
        self.assertIn('duplicate_well_snapshot_identity', codes)
        self.assertIn('snapshot_well_missing_trigger_completion', codes)
        self.assertIn('snapshot_well_missing_decision_horizon_event', codes)

    def test_horizon_and_condition_well_batch_mismatches_are_detected(self):
        manifest, conditions, wells = _valid_inputs()
        conditions[0]['decision_horizon_s'] = '301'
        wells[0]['batch_number'] = '4'
        summary = build_stability_decision_snapshot_integrity_audit(
            manifest, conditions, wells, configuration=_configuration()
        )
        codes = set(item['code'] for item in summary['violations'])
        self.assertIn('condition_snapshot_horizon_mismatch', codes)
        self.assertIn('snapshot_well_batch_mismatch', codes)
        self.assertIn('condition_well_snapshot_batch_mismatch', codes)

    def test_non_cross_batch_policy_is_explicitly_not_applicable(self):
        manifest, conditions, wells = _valid_inputs()
        configuration = _configuration()
        configuration['auto_stability_mode'] = 'monitor'
        summary = build_stability_decision_snapshot_integrity_audit(
            manifest, conditions, wells, configuration=configuration
        )
        self.assertEqual('not_applicable', summary['status'])
        self.assertFalse(summary['applicable'])

    def test_equal_horizon_and_maximum_window_is_not_cross_batch(self):
        manifest, conditions, wells = _valid_inputs()
        configuration = _configuration()
        configuration['auto_stability_max_observation_window_s'] = 300
        summary = build_stability_decision_snapshot_integrity_audit(
            manifest, conditions, wells, configuration=configuration
        )
        self.assertEqual('not_applicable', summary['status'])

    def test_controller_writer_exports_atomic_evidence_checksums(self):
        manifest, conditions, wells = _valid_inputs()
        with tempfile.TemporaryDirectory() as directory:
            pr_data_path = os.path.join(directory, 'pr_data')
            stability_path = os.path.join(pr_data_path, 'stability')
            os.makedirs(stability_path)
            manifest_path = os.path.join(
                pr_data_path, 'auto_stability_scan_manifest.csv'
            )

            def write_rows(path, rows):
                fieldnames = []
                for row in rows:
                    for name in row:
                        if name not in fieldnames:
                            fieldnames.append(name)
                with open(path, 'w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

            write_rows(manifest_path, manifest)
            write_rows(os.path.join(
                stability_path, 'stability_decision_snapshot_conditions.csv'
            ), conditions)
            write_rows(os.path.join(
                stability_path, 'stability_decision_snapshot_wells.csv'
            ), wells)
            for filename in (
                    'condition_stability_summary.csv',
                    'stability_well_metrics.csv',
                    'stability_well_trajectories.csv'):
                write_rows(os.path.join(stability_path, filename), [{'status': 'ok'}])

            controller = self.AuditWriter()
            controller.out_path = directory
            controller.robo_params = _configuration()
            controller.auto_stability_observer = SimpleNamespace(
                manifest_path=manifest_path
            )
            output_path, summary = (
                controller._write_auto_stability_decision_snapshot_integrity_audit()
            )
            with open(output_path, encoding='utf-8') as handle:
                persisted = json.load(handle)
            self.assertEqual(summary, persisted)
            self.assertEqual('no_detected_snapshot_violation', summary['status'])
            self.assertEqual(
                6, len(summary['source_artifacts'])
            )
            self.assertTrue(all(
                len(entry['sha256']) == 64
                for entry in summary['source_artifacts'].values()
            ))


if __name__ == '__main__':
    unittest.main()
