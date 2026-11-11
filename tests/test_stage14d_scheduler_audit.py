'''Pure Stage-14D scheduler-audit tests with no controller import.'''

import ast
import csv
import json
import os
import tempfile
import unittest
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

from auto_stability_scheduler_audit import (
    build_stability_scheduler_audit_summary,
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


def _event(sequence, event_type, **values):
    row = {
        'schema_version': '8',
        'event_sequence': sequence,
        'event_type': event_type,
    }
    row.update(values)
    return row


def _valid_manifest():
    return [
        _event(1, 'trigger_transfer_completed', wellname='well_a', batch_number='0'),
        _event(2, 'targeted_pipette_mix_acknowledged', wellname='well_a', batch_number='0'),
        _event(3, 'raw_scan_completed', raw_scan_id='scan_1',
               wellname='__active_set__', batch_number='0',
               active_wellnames='well_a', active_batch_numbers='0',
               active_well_locations='well_a=A1',
               raw_scan_relative_path='stability/raw_scans/scan_1.csv'),
        _event(4, 'decision_horizon_reached', wellname='well_a', batch_number='0'),
        _event(5, 'trigger_transfer_completed', wellname='well_b', batch_number='1'),
        _event(6, 'targeted_pipette_mix_acknowledged', wellname='well_b', batch_number='1'),
        _event(7, 'raw_scan_completed', raw_scan_id='scan_2',
               wellname='__active_set__', batch_number='1',
               active_wellnames='well_a;well_b', active_batch_numbers='0;1',
               active_well_locations='well_a=A1;well_b=B1',
               raw_scan_relative_path='stability/raw_scans/scan_2.csv'),
        _event(8, 'observation_window_completed', wellname='well_a', batch_number='0'),
        _event(9, 'observation_terminal_qc_excluded', wellname='well_b',
               batch_number='1', retirement_reason='terminal_low_signal'),
    ]


class Stage14DSchedulerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONTROLLER_PATH, encoding='utf-8') as source_file:
            tree = ast.parse(source_file.read(), filename=CONTROLLER_PATH)
        auto_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
        )
        writer = next(
            node for node in auto_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == '_write_auto_stability_scheduler_audit_summary'
        )
        module = ast.fix_missing_locations(ast.Module(
            body=[ast.ClassDef(
                name='AuditWriter', bases=[], keywords=[], body=[writer],
                decorator_list=[]
            )],
            type_ignores=[]
        ))
        namespace = {
            'os': os,
            'csv': csv,
            'json': json,
            'NamedTemporaryFile': NamedTemporaryFile,
            'build_stability_scheduler_audit_summary': (
                build_stability_scheduler_audit_summary
            ),
        }
        exec(compile(module, CONTROLLER_PATH, 'exec'), namespace)
        cls.AuditWriter = namespace['AuditWriter']

    def test_valid_cross_batch_manifest_has_one_mix_per_well(self):
        summary = build_stability_scheduler_audit_summary(
            _valid_manifest(),
            configuration={
                'auto_stability_mixing_mode': 'pipette_mix',
                'auto_stability_scan_schedule': 'each_completion',
                'auto_stability_retirement_policy': 'fixed_window',
            },
        )
        self.assertEqual('no_detected_manifest_violation', summary['status'])
        self.assertTrue(summary['cross_batch_union']['observed'])
        self.assertEqual(['scan_2'], summary['cross_batch_union']['raw_scan_ids'])
        self.assertEqual(2, summary['manifest']['acknowledged_targeted_mix_count'])
        self.assertEqual(['well_a', 'well_b'], [
            row['wellname'] for row in summary['wells']
        ])
        self.assertEqual(
            'maximum_observation_window_expired',
            summary['wells'][0]['retirement']['reason']
        )
        self.assertEqual(
            'terminal_low_signal', summary['wells'][1]['retirement']['reason']
        )

    def test_detects_duplicate_mix_and_retired_well_reappearing(self):
        manifest = _valid_manifest()
        manifest.append(_event(
            10, 'targeted_pipette_mix_acknowledged', wellname='well_a',
            batch_number='0'
        ))
        manifest.append(_event(
            11, 'raw_scan_completed', raw_scan_id='scan_3',
            wellname='__active_set__', batch_number='0',
            active_wellnames='well_a', active_batch_numbers='0',
            active_well_locations='well_a=A1',
            raw_scan_relative_path='stability/raw_scans/scan_3.csv'
        ))
        summary = build_stability_scheduler_audit_summary(manifest)
        self.assertEqual('manifest_violations_detected', summary['status'])
        self.assertEqual(
            {
                'multiple_targeted_mix_acknowledgements',
                'retired_well_scanned_again',
            },
            set(item['code'] for item in summary['violations'])
        )

    def test_detects_bad_raw_scan_identity_and_mapping(self):
        summary = build_stability_scheduler_audit_summary([
            _event(1, 'raw_scan_completed', raw_scan_id='',
                   wellname='__active_set__', batch_number='0',
                   active_wellnames='well_a', active_batch_numbers='0;0',
                   active_well_locations='well_a=A1;invalid'),
        ])
        self.assertEqual('manifest_violations_detected', summary['status'])
        self.assertEqual(
            {
                'raw_scan_missing_id',
                'raw_scan_mapping_mismatch',
                'raw_scan_duplicate_active_batch',
                'raw_scan_well_without_trigger_completion',
            },
            set(item['code'] for item in summary['violations'])
        )

    def test_pipette_each_completion_requires_exactly_one_mix_per_trigger(self):
        manifest = [
            _event(1, 'trigger_transfer_completed', wellname='well_a', batch_number='0'),
            _event(2, 'raw_scan_completed', raw_scan_id='scan_1',
                   wellname='__active_set__', batch_number='0',
                   active_wellnames='well_a', active_batch_numbers='0',
                   active_well_locations='well_a=A1'),
        ]
        summary = build_stability_scheduler_audit_summary(
            manifest,
            configuration={
                'auto_stability_mixing_mode': 'pipette_mix',
                'auto_stability_scan_schedule': 'each_completion',
            },
        )
        self.assertIn(
            'targeted_mix_count_not_exactly_one',
            set(item['code'] for item in summary['violations'])
        )

    def test_all_retirement_reasons_are_retained(self):
        manifest = []
        for sequence, (wellname, event_type, reason) in enumerate((
                ('maximum', 'observation_window_completed', ''),
                ('plateau', 'observation_plateau_confirmed', 'plateau_confirmed'),
                ('low_signal', 'observation_terminal_qc_excluded', 'terminal_low_signal'),
        ), 1):
            manifest.append(_event(
                sequence, 'trigger_transfer_completed', wellname=wellname,
                batch_number='0'
            ))
            manifest.append(_event(
                sequence + 10, event_type, wellname=wellname, batch_number='0',
                retirement_reason=reason
            ))
        summary = build_stability_scheduler_audit_summary(manifest)
        reasons = set(
            row['retirement']['reason'] for row in summary['wells']
        )
        self.assertEqual(
            {
                'maximum_observation_window_expired',
                'plateau_confirmed',
                'terminal_low_signal',
            }, reasons
        )

    def test_controller_writer_atomically_exports_the_pure_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            pr_data_path = os.path.join(directory, 'pr_data')
            os.makedirs(pr_data_path)
            manifest_path = os.path.join(
                pr_data_path, 'auto_stability_scan_manifest.csv'
            )
            with open(manifest_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        'schema_version', 'event_sequence', 'event_type',
                        'wellname', 'batch_number'
                    )
                )
                writer.writeheader()
                writer.writerow({
                    'schema_version': 8,
                    'event_sequence': 1,
                    'event_type': 'trigger_transfer_completed',
                    'wellname': 'well_a',
                    'batch_number': 0,
                })
            controller = self.AuditWriter()
            controller.out_path = directory
            controller.robo_params = {
                'auto_stability_mode': 'stability_only',
            }
            controller.auto_stability_observer = SimpleNamespace(
                manifest_path=manifest_path
            )
            output_path, summary = (
                controller._write_auto_stability_scheduler_audit_summary()
            )
            with open(output_path, encoding='utf-8') as handle:
                persisted = json.load(handle)
            self.assertEqual(summary, persisted)
            self.assertEqual(
                1, persisted['manifest']['completed_trigger_count']
            )


if __name__ == '__main__':
    unittest.main()
