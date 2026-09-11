'''Stage-12A unit and source-contract tests without hardware imports.'''

import ast
import math
import os
import shutil
import tempfile
import textwrap
import unittest

from auto_stability import (
    AutoStabilityValidationError,
    METRIC_STATUS_ELIGIBLE,
    METRIC_STATUS_INSUFFICIENT_OBSERVATIONS,
    METRIC_STATUS_LOW_SIGNAL,
    METRIC_STATUS_NO_POST_PEAK_DECLINE,
    aggregate_condition_stability_metrics,
    compute_stability_metrics,
    parse_auto_stability_header_settings,
    select_reference_wavelength_nm,
    validate_stability_trigger_reagent,
)


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


def _monitor_header(**overrides):
    settings = {
        'auto_stability_mode': 'monitor',
        'auto_stability_trigger_reagent': 'sodium_borohydride',
        'auto_stability_scan_schedule': 'cadenced_active_set',
        'auto_stability_observation_window_s': '900',
        'auto_stability_scan_interval_s': '60',
        'auto_stability_min_peak_absorbance': '0.10',
        'auto_stability_mixing_mode': 'plate_shake',
    }
    settings.update(overrides)
    return settings


class AutoStabilityConfigurationTests(unittest.TestCase):
    def test_missing_settings_are_inert_and_backward_compatible(self):
        settings = parse_auto_stability_header_settings({})

        self.assertEqual(settings['auto_stability_mode'], 'off')
        self.assertIsNone(settings['auto_stability_trigger_reagent'])
        self.assertIsNone(settings['auto_stability_observation_window_s'])

    def test_monitor_settings_are_canonicalized(self):
        settings = parse_auto_stability_header_settings(_monitor_header(
            auto_stability_trigger_reagent='Sodium Borohydride',
            auto_stability_scan_schedule='cadenced',
            auto_stability_mixing_mode='shake',
        ))

        self.assertEqual(settings['auto_stability_mode'], 'monitor')
        self.assertEqual(
            settings['auto_stability_trigger_reagent'],
            'Sodium_Borohydride'
        )
        self.assertEqual(
            settings['auto_stability_scan_schedule'],
            'cadenced_active_set'
        )
        self.assertEqual(settings['auto_stability_mixing_mode'], 'plate_shake')

    def test_monitor_requires_explicit_scientific_settings(self):
        for missing_key in (
                'auto_stability_trigger_reagent',
                'auto_stability_observation_window_s',
                'auto_stability_scan_interval_s',
                'auto_stability_min_peak_absorbance'):
            header = _monitor_header()
            del header[missing_key]
            with self.assertRaises(AutoStabilityValidationError):
                parse_auto_stability_header_settings(header)

    def test_each_completion_fails_closed_until_a_nonperturbing_policy_exists(self):
        header = _monitor_header(
            auto_stability_scan_schedule='each_completion'
        )
        del header['auto_stability_scan_interval_s']

        with self.assertRaises(AutoStabilityValidationError):
            parse_auto_stability_header_settings(header)

    def test_future_modes_and_unimplemented_mixing_fail_closed(self):
        with self.assertRaises(AutoStabilityValidationError):
            parse_auto_stability_header_settings(_monitor_header(
                auto_stability_mode='target_then_stability'
            ))
        with self.assertRaises(AutoStabilityValidationError):
            parse_auto_stability_header_settings(_monitor_header(
                auto_stability_mixing_mode='pipette_mix'
            ))

    def test_trigger_must_be_final_nonwater_transfer(self):
        transfers = [
            'trisodium_citrate',
            'silver_nitrate',
            'sodium_borohydride',
        ]
        self.assertEqual(
            validate_stability_trigger_reagent(
                'sodium borohydride', transfers
            ),
            'sodium_borohydride'
        )
        with self.assertRaises(AutoStabilityValidationError):
            validate_stability_trigger_reagent('silver_nitrate', transfers)


class AutoStabilityMetricTests(unittest.TestCase):
    def test_peak_envelope_loss_uses_only_observations_after_peak(self):
        metrics = compute_stability_metrics([
            {
                'timestamp_s': 0,
                'reference_absorbance': 0.20,
                'lambda_max_nm': 620,
            },
            {
                'timestamp_s': 10,
                'reference_absorbance': 0.80,
                'lambda_max_nm': 625,
            },
            {
                'timestamp_s': 30,
                'reference_absorbance': 0.50,
                'lambda_max_nm': 622,
            },
            {
                'timestamp_s': 50,
                'reference_absorbance': 0.60,
                'lambda_max_nm': 621,
            },
        ], 0.25)

        self.assertEqual(metrics['status'], METRIC_STATUS_ELIGIBLE)
        self.assertEqual(metrics['peak_index'], 1)
        self.assertEqual(metrics['tail_min_index'], 2)
        self.assertAlmostEqual(metrics['absorbance_loss'], 0.30)
        self.assertAlmostEqual(metrics['loss_rate_absorbance_per_s'], 0.015)
        self.assertEqual(metrics['reference_wavelength_nm'], 620.0)
        self.assertAlmostEqual(metrics['lambda_drift_nm'], 2.0)

    def test_low_signal_and_incomplete_post_peak_data_stay_auditable(self):
        low_signal = compute_stability_metrics([
            {'timestamp_s': 0, 'reference_absorbance': 0.02},
            {'timestamp_s': 10, 'reference_absorbance': 0.01},
        ], 0.10)
        one_scan = compute_stability_metrics([
            {'timestamp_s': 0, 'reference_absorbance': 0.50},
        ], 0.10)
        no_decline = compute_stability_metrics([
            {'timestamp_s': 0, 'reference_absorbance': 0.50},
            {'timestamp_s': 10, 'reference_absorbance': 0.50},
        ], 0.10)

        self.assertEqual(low_signal['status'], METRIC_STATUS_LOW_SIGNAL)
        self.assertEqual(one_scan['status'], METRIC_STATUS_INSUFFICIENT_OBSERVATIONS)
        self.assertEqual(no_decline['status'], METRIC_STATUS_NO_POST_PEAK_DECLINE)
        self.assertIsNone(no_decline['loss_rate_absorbance_per_s'])

    def test_fixed_window_uses_the_actual_trigger_time_not_scan_count(self):
        metrics = compute_stability_metrics([
            {'timestamp_s': 4, 'reference_absorbance': 0.20},
            {'timestamp_s': 6, 'reference_absorbance': 0.80},
            {'timestamp_s': 12, 'reference_absorbance': 0.50},
            # This lower value is outside the 10-second post-trigger window
            # and must not make the recorded loss rate look less stable.
            {'timestamp_s': 16, 'reference_absorbance': 0.10},
        ], 0.10, trigger_timestamp_s=2, observation_window_s=10)

        self.assertEqual(metrics['status'], METRIC_STATUS_ELIGIBLE)
        self.assertEqual(metrics['observation_count'], 4)
        self.assertEqual(metrics['observation_count_within_window'], 3)
        self.assertAlmostEqual(metrics['tail_min_absorbance'], 0.50)
        self.assertAlmostEqual(metrics['loss_rate_absorbance_per_s'], 0.05)

    def test_reference_wavelength_is_first_valid_post_trigger_lambda_max(self):
        reference = select_reference_wavelength_nm([
            {'lambda_max_nm': None},
            {'lambda_max_nm': 'not a wavelength'},
            {'lambda_max_nm': '624.5'},
            {'lambda_max_nm': 630},
        ])

        self.assertEqual(reference, 624.5)

    def test_timestamp_order_and_condition_aggregation_are_deterministic(self):
        with self.assertRaises(AutoStabilityValidationError):
            compute_stability_metrics([
                {'timestamp_s': 1, 'reference_absorbance': 0.4},
                {'timestamp_s': 1, 'reference_absorbance': 0.2},
            ], 0.1)

        summary = aggregate_condition_stability_metrics([
            {'status': METRIC_STATUS_ELIGIBLE,
             'loss_rate_absorbance_per_s': 0.01},
            {'status': METRIC_STATUS_LOW_SIGNAL,
             'loss_rate_absorbance_per_s': None},
            {'status': METRIC_STATUS_ELIGIBLE,
             'loss_rate_absorbance_per_s': 0.03},
        ])
        self.assertEqual(summary['total_well_count'], 3)
        self.assertEqual(summary['eligible_well_count'], 2)
        self.assertAlmostEqual(
            summary['condition_loss_rate_mean_absorbance_per_s'], 0.02
        )
        self.assertAlmostEqual(
            summary['condition_loss_rate_sample_sd_absorbance_per_s'],
            0.0141421356237
        )


class AutoStabilityControllerContractTests(unittest.TestCase):
    '''Ensure Stage 12A cannot accidentally become a physical behavior change.'''

    @classmethod
    def setUpClass(cls):
        with open(CONTROLLER_PATH, 'r', encoding='utf-8') as source_file:
            cls.source = source_file.read()
        tree = ast.parse(cls.source, filename=CONTROLLER_PATH)
        cls.auto_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
        )
        cls.controller_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'Controller'
        )
        cls.plate_reader_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'PlateReader'
        )
        cls.auto_methods = {
            node.name: node
            for node in cls.auto_class.body
            if isinstance(node, ast.FunctionDef)
        }
        cls.controller_methods = {
            node.name: node
            for node in cls.controller_class.body
            if isinstance(node, ast.FunctionDef)
        }

    def test_configuration_method_exists_once_and_runs_before_prechecks(self):
        method_name = '_initialize_auto_stability_configuration'
        self.assertIn(method_name, self.auto_methods)
        self.assertEqual(
            sum(
                1 for node in self.auto_class.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            ),
            1
        )
        init_source = ast.get_source_segment(
            self.source, self.auto_methods['__init__']
        )
        self.assertLess(
            init_source.index('self._initialize_auto_stability_configuration()'),
            init_source.index('self.run_all_checks()')
        )

    def test_configuration_path_does_not_invoke_physical_or_model_methods(self):
        method_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_initialize_auto_stability_configuration']
        )
        for forbidden_text in (
                '_execute_scan(', '_mix(', 'execute_protocol_df(',
                'send_pack(', 'run_protocol(', 'getNextReaction('):
            self.assertNotIn(forbidden_text, method_source)
        self.assertIn('parse_auto_stability_header_settings(', method_source)
        self.assertIn('validate_stability_trigger_reagent(', method_source)

    def test_stability_observer_preserves_legacy_completion_boundary(self):
        for method_name in (
                '_initialize_auto_stability_observer',
                '_record_auto_stability_trigger_dispatch',
                '_confirm_auto_stability_trigger_completion'):
            self.assertIn(method_name, self.auto_methods)

        run_source = ast.get_source_segment(
            self.source, self.auto_methods['_run']
        )
        self.assertLess(
            run_source.index('self._initialize_auto_live_run_journal(model)'),
            run_source.index('self._initialize_auto_stability_observer()')
        )
        self.assertLess(
            run_source.index('self._initialize_auto_stability_observer()'),
            run_source.index('self.create_connection(simulate, no_pr, port)')
        )

        transfer_source = ast.get_source_segment(
            self.source, self.controller_methods['_send_transfer_command']
        )
        self.assertIn(
            'self._record_auto_stability_trigger_dispatch(', transfer_source
        )
        self.assertLess(
            transfer_source.index('self.save()'),
            transfer_source.index(
                'self._confirm_auto_stability_trigger_completion('
            )
        )
        for forbidden_text in ('_execute_scan(', '_mix('):
            self.assertNotIn(forbidden_text, transfer_source)

        # The transfer method belongs to the shared Controller base class;
        # verify manual protocol execution retains an inert extension hook.
        base_dispatch_source = ast.get_source_segment(
            self.source,
            self.controller_methods['_record_auto_stability_trigger_dispatch']
        )
        base_completion_source = ast.get_source_segment(
            self.source,
            self.controller_methods['_confirm_auto_stability_trigger_completion']
        )
        self.assertIn('return []', base_dispatch_source)
        self.assertIn('return None', base_completion_source)

    def test_stage_12c_uses_unmerged_active_set_scans_after_completion(self):
        for method_name in (
                '_run_auto_stability_observation',
                '_complete_auto_stability_observation_window',
                '_get_auto_stability_scan_protocol',
                '_get_auto_stability_reader_locations'):
            self.assertIn(method_name, self.auto_methods)

        confirmation_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_confirm_auto_stability_trigger_completion']
        )
        self.assertIn('self._run_auto_stability_observation(', confirmation_source)
        self.assertIn('shake_before_scan=True', confirmation_source)

        observation_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_run_auto_stability_observation']
        )
        self.assertIn('shake_duration_s = 30.0', observation_source)
        self.assertIn('self.pr.shake(shake_duration_s)', observation_source)
        self.assertIn('self.pr.run_protocol(', observation_source)
        self.assertIn('record_in_aggregate=False', observation_source)
        self.assertIn('shutil.move(source_path, destination_path)', observation_source)
        self.assertNotIn('merge_scans(', observation_source)
        self.assertIn(
            'get_active_wells_within_observation_window(', observation_source
        )
        self.assertLess(
            observation_source.index('get_active_wells_within_observation_window('),
            observation_source.index('observer.reserve_raw_scan(')
        )
        self.assertIn(
            'record_raw_scan_skipped_expired(', observation_source
        )

        plate_reader_methods = {
            node.name: node
            for node in self.plate_reader_class.body
            if isinstance(node, ast.FunctionDef)
        }
        reader_run_source = ast.get_source_segment(
            self.source, plate_reader_methods['run_protocol']
        )
        self.assertIn('if record_in_aggregate:', reader_run_source)
        self.assertIn('self.data.AddToDF(', reader_run_source)
        self.assertIn('self.data.df.to_csv(', reader_run_source)

        transfer_source = ast.get_source_segment(
            self.source, self.controller_methods['_send_transfer_command']
        )
        self.assertIn(
            'self._requires_auto_stability_per_well_completion(',
            transfer_source
        )
        self.assertIn('self.portal.burn_pipe()', transfer_source)
        self.assertIn(
            'self._confirm_auto_stability_trigger_step_completion(',
            transfer_source
        )

        create_samples_source = ast.get_source_segment(
            self.source, self.auto_methods['_create_samples']
        )
        self.assertLess(
            create_samples_source.index('self.execute_protocol_df(model)'),
            create_samples_source.index(
                'self._complete_auto_stability_observation_window()'
            )
        )

    def test_plate_reader_shake_normalizes_whole_seconds_for_dde(self):
        '''The DDE macro rejects decimal strings such as ``'30.0'``.'''
        plate_reader_methods = {
            node.name: node
            for node in self.plate_reader_class.body
            if isinstance(node, ast.FunctionDef)
        }
        shake_source = textwrap.dedent(ast.get_source_segment(
            self.source, plate_reader_methods['shake']
        ))
        namespace = {'math': math}
        exec(shake_source, namespace)
        shake = namespace['shake']

        class FakePlateReader(object):
            def __init__(self):
                self.calls = []

            def exec_macro(self, *args):
                self.calls.append(args)

        fake_reader = FakePlateReader()
        shake(fake_reader, 30.0)
        self.assertEqual(fake_reader.calls, [('Shake', 2, 300, 30)])

        for invalid_time in (0, -1, 30.5, 'not-a-time', float('nan')):
            with self.assertRaises(ValueError):
                shake(fake_reader, invalid_time)

    def test_stage_12c_excludes_expired_wells_at_reader_scan_start(self):
        '''Reader staging must narrow the raw layout before it is reserved.'''
        observation_source = textwrap.dedent(ast.get_source_segment(
            self.source,
            self.auto_methods['_run_auto_stability_observation']
        ))

        class FakeTime(object):
            def __init__(self):
                self.values = iter((50.0, 75.0, 76.0))

            def monotonic(self):
                return next(self.values)

        class FakePortal(object):
            def send_pack(self, command):
                self.last_command = command

            def burn_pipe(self):
                self.burned = True

        class FakeReader(object):
            def __init__(self, data_path):
                self.data_path = data_path
                self.layouts = []

            def exec_macro(self, command):
                self.last_macro = command

            def run_protocol(self, protocol, basename, layout,
                             record_in_aggregate):
                self.layouts.append(list(layout))
                self.record_in_aggregate = record_in_aggregate
                with open(os.path.join(self.data_path, basename + '.csv'),
                          'w', encoding='utf-8') as output_file:
                    output_file.write('synthetic raw scan')

        class FakeObserver(object):
            def __init__(self, data_path):
                self.records = [
                    {'wellname': 'well_a', 'batch_number': 0,
                     'activation_monotonic_s': 10.0},
                    {'wellname': 'well_b', 'batch_number': 0,
                     'activation_monotonic_s': 25.0},
                ]
                self.data_path = data_path
                self.reserved_wellnames = None
                self.completed = None
                self.skips = []

            def get_active_wells(self):
                return [dict(record) for record in self.records]

            def get_active_wells_within_observation_window(
                    self, window_s, now_monotonic_s=None):
                return [
                    dict(record) for record in self.records
                    if now_monotonic_s < (
                        record['activation_monotonic_s'] + window_s
                    )
                ]

            def reserve_raw_scan(
                    self, batch_number, wellnames, reader_locations=None):
                self.reserved_wellnames = list(wellnames)
                self.reserved_reader_locations = list(reader_locations or [])
                return {
                    'event_type': 'raw_scan_reserved',
                    'batch_number': batch_number,
                    'raw_scan_basename': 'synthetic_raw_scan',
                    'raw_scan_relative_path': os.path.join(
                        'stability', 'raw_scans', 'synthetic_raw_scan.csv'
                    ),
                }

            def record_raw_scan_completed(self, **kwargs):
                self.completed = kwargs

            def record_raw_scan_skipped_expired(self, *args):
                self.skips.append(args)

        with tempfile.TemporaryDirectory() as temporary_directory:
            os.makedirs(os.path.join(
                temporary_directory, 'stability', 'raw_scans'
            ))
            fake_time = FakeTime()
            namespace = {
                'os': os,
                'shutil': shutil,
                'time': fake_time,
            }
            exec(observation_source, namespace)
            observation_method = namespace['_run_auto_stability_observation']

            class FakeController(object):
                def __init__(self):
                    self.auto_stability_observer = FakeObserver(
                        temporary_directory
                    )
                    self.robo_params = {
                        'auto_stability_observation_window_s': 60.0,
                    }
                    self.portal = FakePortal()
                    self.pr = FakeReader(temporary_directory)
                    self.reader_wellnames = None

                @staticmethod
                def _auto_stability_utc_now():
                    return '2026-09-11T00:00:00+00:00'

                @staticmethod
                def _get_auto_stability_scan_protocol():
                    return 'NC_synthesis'

                def _get_auto_stability_reader_locations(self, wellnames):
                    self.reader_wellnames = list(wellnames)
                    return ['B01']

            fake_controller = FakeController()
            result = observation_method(
                fake_controller,
                reason='cadenced_active_set',
                shake_before_scan=False
            )

            self.assertEqual(result['raw_scan_basename'], 'synthetic_raw_scan')
            self.assertEqual(
                fake_controller.auto_stability_observer.reserved_wellnames,
                ['well_b']
            )
            self.assertEqual(fake_controller.reader_wellnames, ['well_b'])
            self.assertEqual(
                fake_controller.auto_stability_observer.reserved_reader_locations,
                ['B01']
            )
            self.assertEqual(fake_controller.pr.layouts, [['B01']])
            self.assertFalse(fake_controller.pr.record_in_aggregate)
            self.assertEqual(fake_controller.auto_stability_observer.skips, [])
            self.assertTrue(os.path.exists(os.path.join(
                temporary_directory,
                'stability', 'raw_scans', 'synthetic_raw_scan.csv'
            )))

    def test_stage_12d_is_final_report_only_and_does_not_select_recipes(self):
        for method_name in (
                '_generate_auto_stability_reporting_exports',
                '_load_auto_stability_reporting_spectra',
                '_plot_auto_stability_reporting_artifacts'):
            self.assertIn(method_name, self.auto_methods)
        finalize_source = ast.get_source_segment(
            self.source, self.auto_methods['_finalize_auto_run']
        )
        self.assertLess(
            finalize_source.index('self._export_auto_model_performance_log()'),
            finalize_source.index(
                'self._generate_auto_stability_reporting_exports()'
            )
        )
        reporting_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_generate_auto_stability_reporting_exports']
        )
        for forbidden_text in (
                'update_experiment_data(', 'getNextReaction(',
                'send_pack(', 'run_protocol(', 'execute_protocol_df('):
            self.assertNotIn(forbidden_text, reporting_source)
        self.assertIn('build_stability_reporting_records(', reporting_source)

    def test_stage_12d_uses_saved_reader_locations_not_current_plate_lookup(self):
        load_source = ast.get_source_segment(
            self.source,
            self.auto_methods['_load_auto_stability_reporting_spectra']
        )
        self.assertIn('active_well_locations', load_source)
        self.assertIn('self.pr.load_reader_data(', load_source)
        self.assertIn('self._extract_auto_lambda_maxima(', load_source)
        self.assertNotIn('_get_auto_stability_reader_locations(', load_source)


if __name__ == '__main__':
    unittest.main()
