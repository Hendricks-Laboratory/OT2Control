'''Pure Stage-12D reporting tests; intentionally no controller import.'''

import unittest

from auto_stability_reporting import (
    TRAJECTORY_WINDOW_FULLY_WITHIN,
    TRAJECTORY_WINDOW_SPANS_END,
    build_stability_reporting_records,
)


def _event(event_type, **values):
    row = {'event_type': event_type}
    row.update(values)
    return row


def _spectra(*values):
    return {float(wavelength): float(absorbance) for wavelength, absorbance in values}


class AutoStabilityReportingTests(unittest.TestCase):
    def _records(self):
        manifest = [
            _event(
                'trigger_transfer_completed',
                wellname='autowell0C1.0',
                trigger_transfer_completion_observed_at_utc='2026-09-10T10:00:00+00:00',
            ),
            _event(
                'raw_scan_completed', raw_scan_id='0001',
                raw_scan_relative_path='stability/raw_scans/scan1.csv',
                active_wellnames='autowell0C1.0',
                observation_reason='trigger_completion',
                scan_started_at_utc='2026-09-10T10:00:10+00:00',
                scan_completed_at_utc='2026-09-10T10:00:20+00:00',
            ),
            _event(
                'raw_scan_completed', raw_scan_id='0002',
                raw_scan_relative_path='stability/raw_scans/scan2.csv',
                active_wellnames='autowell0C1.0',
                observation_reason='cadenced_active_set',
                scan_started_at_utc='2026-09-10T10:00:50+00:00',
                scan_completed_at_utc='2026-09-10T10:01:00+00:00',
            ),
            # This scan began in the 75 s window but completed after it.  It
            # remains present in audit but cannot improve a stability metric.
            _event(
                'raw_scan_completed', raw_scan_id='0003',
                raw_scan_relative_path='stability/raw_scans/scan3.csv',
                active_wellnames='autowell0C1.0',
                observation_reason='cadenced_active_set',
                scan_started_at_utc='2026-09-10T10:01:10+00:00',
                scan_completed_at_utc='2026-09-10T10:01:20+00:00',
            ),
        ]
        spectra = {
            '0001': {'autowell0C1.0': {
                'spectrum_by_wavelength_nm': _spectra((620, 0.8), (621, 0.3))
            }},
            '0002': {'autowell0C1.0': {
                'spectrum_by_wavelength_nm': _spectra((620, 0.4), (621, 0.2))
            }},
            '0003': {'autowell0C1.0': {
                'spectrum_by_wavelength_nm': _spectra((620, 0.1), (621, 0.05))
            }},
        }
        condition_rows = [{
            'origin_run_directory': 'DEBUG',
            'batch_number': 0,
            'reaction_number': 2,
            'condition_type': 'seed',
            'replicate_sample_names': '["autowell0C1.0"]',
        }]
        return build_stability_reporting_records(
            manifest, spectra, condition_rows,
            observation_window_s=75,
            scan_interval_s=30,
            min_peak_absorbance=0.1,
        )

    def test_full_intervals_define_metric_and_spanning_interval_is_audit_only(self):
        records = self._records()
        trajectory = records['trajectory_rows']
        self.assertEqual(
            [row['window_interval_status'] for row in trajectory],
            [
                TRAJECTORY_WINDOW_FULLY_WITHIN,
                TRAJECTORY_WINDOW_FULLY_WITHIN,
                TRAJECTORY_WINDOW_SPANS_END,
            ]
        )
        self.assertEqual(
            [row['metric_eligible_interval'] for row in trajectory],
            [True, True, False]
        )
        metric = records['well_metrics'][0]
        self.assertEqual(metric['status'], 'eligible')
        self.assertEqual(metric['metric_input_scan_count'], 2)
        self.assertAlmostEqual(metric['reference_wavelength_nm'], 620.0)
        self.assertAlmostEqual(metric['loss_rate_absorbance_per_s'], 0.01)

    def test_cadence_audit_records_observed_delay_not_claimed_fixed_cadence(self):
        timing = self._records()['timing_rows']
        self.assertEqual(timing[0]['cadence_status'], 'not_cadenced')
        self.assertEqual(
            timing[1]['requested_cadence_deadline_utc'],
            '2026-09-10T10:00:50+00:00'
        )
        self.assertEqual(timing[1]['cadence_status'], 'met_or_early')
        self.assertAlmostEqual(timing[1]['observed_cadence_delay_s'], 0.0)

    def test_condition_summary_preserves_partial_replicate_coverage(self):
        records = self._records()
        summary = records['condition_summaries'][0]
        self.assertEqual(summary['condition_stability_status'], 'complete')
        self.assertEqual(summary['eligible_well_count'], 1)
        self.assertAlmostEqual(
            summary['condition_loss_rate_mean_absorbance_per_s'], 0.01
        )

        # The durable condition denominator is the planned replicate list,
        # not merely the wells which happened to appear in a raw scan.
        partial = build_stability_reporting_records(
            [_event(
                'trigger_transfer_completed', wellname='well1',
                trigger_transfer_completion_observed_at_utc='2026-09-10T10:00:00+00:00'
            ), _event(
                'raw_scan_completed', raw_scan_id='x', active_wellnames='well1',
                scan_started_at_utc='2026-09-10T10:00:10+00:00',
                scan_completed_at_utc='2026-09-10T10:00:20+00:00',
                observation_reason='trigger_completion'
            ), _event(
                'raw_scan_completed', raw_scan_id='y', active_wellnames='well1',
                scan_started_at_utc='2026-09-10T10:00:40+00:00',
                scan_completed_at_utc='2026-09-10T10:00:50+00:00',
                observation_reason='cadenced_active_set'
            )],
            {'x': {'well1': {'spectrum_by_wavelength_nm': _spectra((620, .8))}},
             'y': {'well1': {'spectrum_by_wavelength_nm': _spectra((620, .4))}}},
            [{'origin_run_directory': 'DEBUG', 'batch_number': 0,
              'reaction_number': 1, 'replicate_sample_names': '["well1", "well2"]'}],
            60, 20, .1
        )
        self.assertEqual(partial['condition_summaries'][0]['total_well_count'], 2)
        self.assertEqual(partial['condition_summaries'][0]['eligible_well_count'], 1)
        self.assertEqual(partial['condition_summaries'][0]['condition_stability_status'], 'partial')

    def test_missing_spectrum_is_retained_as_qc_not_silently_dropped(self):
        records = build_stability_reporting_records(
            [_event(
                'trigger_transfer_completed', wellname='well1',
                trigger_transfer_completion_observed_at_utc='2026-09-10T10:00:00+00:00'
            ), _event(
                'raw_scan_completed', raw_scan_id='x', active_wellnames='well1',
                scan_started_at_utc='2026-09-10T10:00:10+00:00',
                scan_completed_at_utc='2026-09-10T10:00:20+00:00',
                observation_reason='trigger_completion'
            )], {}, [], 60, 30, 0.1
        )
        self.assertEqual(
            records['trajectory_rows'][0]['spectrum_status'],
            'raw_spectrum_unavailable'
        )
        self.assertEqual(records['well_metrics'][0]['status'], 'insufficient_observations')
        self.assertTrue(any(
            row['qc_status'] == 'raw_spectrum_unavailable'
            for row in records['qc_rows']
        ))

    def test_imported_conditions_do_not_become_false_missing_current_run_wells(self):
        records = build_stability_reporting_records(
            [], {}, [{
                'origin_run_directory': 'old-run', 'batch_number': 0,
                'reaction_number': 0, 'executed_in_current_run': False,
                'replicate_sample_names': '["oldwell"]',
            }], 60, 30, .1
        )
        self.assertEqual(records['well_metrics'], [])
        self.assertEqual(records['condition_summaries'], [])
