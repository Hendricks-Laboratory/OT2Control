'''Synthetic Stage-13D-D bounded targeted-mix scheduler tests.

These tests source-extract small ``AutoContr`` methods so they do not import
the hardware-dependent controller module, start a server, or contact a robot
or reader. They validate only the controller's ordering and audit contracts.
'''

import ast
import os
import shutil
import tempfile
import time
import unittest
from collections import namedtuple
from types import SimpleNamespace

from auto_stability_lifecycle import build_fixed_window_monitoring_policy


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


def _load_methods(method_names):
    with open(CONTROLLER_PATH, encoding='utf-8') as source_file:
        tree = ast.parse(source_file.read())
    auto_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'AutoContr'
    )
    methods = {
        node.name: node for node in auto_class.body
        if isinstance(node, ast.FunctionDef)
    }
    extracted_class = ast.ClassDef(
        name='AutoContr', bases=[], keywords=[],
        body=[methods[name] for name in method_names], decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(
        body=[extracted_class], type_ignores=[]
    ))
    namespace = {
        'os': os,
        'shutil': shutil,
        'time': time,
        'STABILITY_MIXING_MODE_PIPETTE_MIX': 'pipette_mix',
        'STABILITY_SCAN_SCHEDULE_EACH_COMPLETION': 'each_completion',
        'TARGETED_MIX_DEFAULT_VOLUME_UL': 20.0,
        'TARGETED_MIX_DEFAULT_CYCLE_COUNT': 1,
        'build_fixed_window_monitoring_policy': (
            build_fixed_window_monitoring_policy
        ),
    }
    exec(compile(module, CONTROLLER_PATH, 'exec'), namespace)
    return namespace['AutoContr']


class _ObserverStub(object):
    def __init__(self):
        self.confirmed = []
        self.completed_windows = []
        self.reservations = []
        self.completed_scans = []
        self._active = [
            {
                'wellname': 'autowell0C1.0',
                'trigger_completion_sequence': 1,
            }
        ]

    def confirm_trigger_transfer_completed(self, wellname, **kwargs):
        self.confirmed.append((wellname, kwargs))

    def get_active_wells(self):
        return list(self._active)

    def complete_expired_observation_windows(self, window, now_monotonic_s):
        self.completed_windows.append((window, now_monotonic_s))
        return []

    def reserve_raw_scan(self, batch_number, wellnames, reader_locations):
        reservation = {
            'raw_scan_basename': 'stability_batch_004_active_set_scan_0001',
            'raw_scan_relative_path': os.path.join(
                'stability', 'raw_scans',
                'stability_batch_004_active_set_scan_0001.csv'
            ),
            'batch_number': batch_number,
            'active_wellnames': ';'.join(wellnames),
            'active_well_locations': ';'.join(reader_locations),
        }
        self.reservations.append(reservation)
        return reservation

    def record_raw_scan_completed(self, **kwargs):
        self.completed_scans.append(kwargs)


class _PortalStub(object):
    def __init__(self):
        self.calls = []

    def send_pack(self, *args):
        self.calls.append(('send_pack', args))

    def burn_pipe(self):
        self.calls.append(('burn_pipe', ()))


class _PlateReaderStub(object):
    def __init__(self, data_path, calls):
        self.data_path = data_path
        self.calls = calls

    def exec_macro(self, name):
        self.calls.append(('macro', name))

    def run_protocol(self, protocol, basename, layout, record_in_aggregate):
        self.calls.append((
            'run_protocol', protocol, basename, list(layout),
            record_in_aggregate
        ))
        with open(
                os.path.join(self.data_path, basename + '.csv'),
                'w', encoding='utf-8') as output_file:
            output_file.write('synthetic raw scan\n')


class Stage13DBoundedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.AutoController = _load_methods([
            '_uses_auto_stability_pipette_mix_schedule',
            '_get_auto_stability_fixed_window_policy',
            '_get_auto_stability_latest_active_wellname',
            '_confirm_auto_stability_trigger_step_completion',
            '_confirm_auto_stability_trigger_completion',
            '_run_auto_stability_pipette_mix_completion',
            '_run_auto_stability_post_mix_active_set_scan',
        ])

    def _controller(self):
        controller = self.AutoController()
        controller.batch_num = 4
        controller.robo_params = {
            'auto_stability_mixing_mode': 'pipette_mix',
            'auto_stability_scan_schedule': 'each_completion',
            'auto_stability_observation_window_s': 90.0,
        }
        controller.auto_stability_observer = _ObserverStub()
        return controller

    def test_completion_acknowledgement_precedes_exactly_one_post_mix_action(self):
        controller = self._controller()
        observed_order = []
        controller._run_auto_stability_pipette_mix_completion = (
            lambda wellname: observed_order.append(wellname)
        )

        self.assertTrue(controller._confirm_auto_stability_trigger_step_completion(
            ['autowell0C1.0']
        ))
        self.assertEqual(
            [('autowell0C1.0', {
                'completion_time_basis': 'controller_observed_transfer_ready'
            })],
            controller.auto_stability_observer.confirmed
        )
        self.assertEqual(['autowell0C1.0'], observed_order)

    def test_completion_finalizer_does_not_add_a_cohort_scan_or_shake(self):
        controller = self._controller()
        controller._run_auto_stability_observation = lambda **kwargs: (
            self.fail('pipette_mix must not invoke the plate-shake observer')
        )

        controller._confirm_auto_stability_trigger_completion(
            [], observe_completed_wells=True
        )
        self.assertEqual([], controller.auto_stability_observer.confirmed)

    def test_targeted_completion_orders_identity_mix_plan_then_scan(self):
        controller = self._controller()
        calls = []
        plan = {'scan_wellnames': ['autowell0C1.0']}
        controller._auto_stability_trigger_chemical_names = {
            'autowell0C1.0': 'sodium_borohydrideC6.25'
        }
        controller._register_auto_stability_targeted_mix_identity = (
            lambda wellname: calls.append(('identity', wellname))
        )
        controller._build_auto_stability_targeted_mix_request = (
            lambda **kwargs: calls.append(('build', kwargs)) or {'request': 1}
        )
        controller._request_auto_stability_targeted_mix = (
            lambda request: calls.append(('acknowledged_mix', request))
        )
        controller._plan_auto_stability_post_mix_active_set = (
            lambda wellname: calls.append(('plan', wellname)) or plan
        )
        controller._run_auto_stability_post_mix_active_set_scan = (
            lambda scan_plan, **kwargs: calls.append(('scan', scan_plan, kwargs))
        )

        controller._run_auto_stability_pipette_mix_completion('autowell0C1.0')

        self.assertEqual('identity', calls[0][0])
        self.assertEqual('build', calls[1][0])
        self.assertEqual('acknowledged_mix', calls[2][0])
        self.assertEqual(('plan', 'autowell0C1.0'), calls[3])
        self.assertEqual('scan', calls[4][0])
        self.assertEqual('each_completion', calls[4][2]['observation_reason'])
        self.assertEqual('pipette_mix', calls[4][2]['mixing_mode'])

    def test_unpaired_runtime_configuration_fails_closed(self):
        controller = self._controller()
        controller.robo_params['auto_stability_scan_schedule'] = (
            'cadenced_active_set'
        )
        with self.assertRaisesRegex(RuntimeError, 'requires the paired'):
            controller._uses_auto_stability_pipette_mix_schedule()

    def test_fixed_window_policy_keeps_adaptive_retirement_disabled(self):
        controller = self._controller()
        policy = controller._get_auto_stability_fixed_window_policy()
        self.assertFalse(policy['adaptive_retirement_enabled'])
        self.assertEqual(90.0, policy['decision_horizon_s'])
        self.assertEqual(90.0, policy['max_observation_window_s'])

    def test_reader_scan_is_unshaken_and_preserves_plan_mapping(self):
        controller = self._controller()
        plan = {
            'scan_wellnames': ['autowell0C1.0'],
            'reader_locations': ['A1'],
            'targeted_mix_acknowledgement_required': True,
            'active_batch_numbers': [4],
            'retired_wellnames': [],
            'triggering_wellname': 'autowell0C1.0',
            'deck_pos': 4,
        }
        controller._plan_auto_stability_post_mix_active_set = (
            lambda wellname: plan
        )
        controller._get_auto_stability_reader_locations = (
            lambda wellnames: ['A1']
        )
        controller._cached_reader_locs = {
            'autowell0C1.0': SimpleNamespace(deck_pos=4)
        }
        controller._get_auto_stability_scan_protocol = lambda: 'UVVis'
        controller._auto_stability_utc_now = lambda: '2026-10-02T12:00:00+00:00'
        controller.portal = _PortalStub()
        with tempfile.TemporaryDirectory() as temporary_directory:
            os.makedirs(os.path.join(
                temporary_directory, 'stability', 'raw_scans'
            ))
            reader_calls = []
            controller.pr = _PlateReaderStub(
                temporary_directory, reader_calls
            )
            reservation = controller._run_auto_stability_post_mix_active_set_scan(
                plan,
                observation_reason='each_completion',
                mixing_mode='pipette_mix'
            )

            self.assertIsNotNone(reservation)
            self.assertEqual(
                [('send_pack', ('home',)), ('burn_pipe', ())],
                controller.portal.calls
            )
            self.assertEqual(
                [('macro', 'PlateIn'),
                 ('run_protocol', 'UVVis',
                  'stability_batch_004_active_set_scan_0001', ['A1'], False),
                 ('macro', 'PlateOut')],
                reader_calls
            )
            self.assertEqual(1, len(
                controller.auto_stability_observer.completed_scans
            ))
            completed = controller.auto_stability_observer.completed_scans[0]
            self.assertEqual('pipette_mix', completed['mixing_mode'])
            self.assertEqual(0.0, completed['shake_duration_s'])
            self.assertEqual('each_completion', completed['observation_reason'])


if __name__ == '__main__':
    unittest.main()
