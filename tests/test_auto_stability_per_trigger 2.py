'''Pure Stage-13D-A cross-batch active-set contract tests.'''

import unittest

from auto_stability_lifecycle import build_stability_monitoring_policy
from auto_stability_per_trigger import (
    AutoStabilityPerTriggerError,
    PER_TRIGGER_ACTIVE_SET_SCHEMA_VERSION,
    build_per_trigger_active_set,
    build_per_trigger_post_mix_active_set,
)


def _policy(**overrides):
    values = {
        'decision_horizon_s': 30.0,
        'max_observation_window_s': 90.0,
        'adaptive_retirement_enabled': True,
        'plateau_min_tail_observation_count': 3,
        'plateau_consecutive_interval_count': 2,
        'plateau_max_absorbance_slope_per_s': 0.002,
    }
    values.update(overrides)
    return build_stability_monitoring_policy(**values)


def _record(wellname, batch_number, activation_time, location, **overrides):
    record = {
        'wellname': wellname,
        'batch_number': batch_number,
        'trigger_completion_sequence': batch_number + 1,
        'activation_status': 'active',
        'activation_monotonic_s': activation_time,
        'plate_generation': 4,
        'plate_mapping_revision': 11,
        'deck_pos': 4,
        'reader_location': location,
    }
    record.update(overrides)
    return record


def _acknowledged_actions(records):
    return {
        record['wellname']: {
            'action_id': 'targeted-mix-{}'.format(record['wellname']),
            'wellname': record['wellname'],
            'batch_number': record['batch_number'],
            'acknowledged': True,
        }
        for record in records
    }


class PerTriggerActiveSetTests(unittest.TestCase):
    def test_post_mix_plan_requires_acknowledgements_for_each_active_well(self):
        records = [
            _record('autowell0C1.0', 0, 0.0, 'A1'),
            _record('autowell1C1.0', 1, 20.0, 'B1'),
        ]
        plan = build_per_trigger_post_mix_active_set(
            records,
            'autowell1C1.0',
            25.0,
            _policy(),
            0.10,
            _acknowledged_actions(records),
        )

        self.assertTrue(plan['targeted_mix_acknowledgement_required'])
        self.assertEqual(
            [
                'targeted-mix-autowell0C1.0',
                'targeted-mix-autowell1C1.0',
            ],
            plan['targeted_mix_action_ids']
        )
        self.assertEqual(
            'targeted-mix-autowell1C1.0',
            plan['triggering_mix_action_id']
        )

    def test_post_mix_plan_rejects_missing_unacknowledged_or_wrong_batch_mix(self):
        records = [
            _record('autowell0C1.0', 0, 0.0, 'A1'),
            _record('autowell1C1.0', 1, 20.0, 'B1'),
        ]
        actions = _acknowledged_actions(records)
        actions.pop('autowell0C1.0')
        with self.assertRaisesRegex(AutoStabilityPerTriggerError, 'match'):
            build_per_trigger_post_mix_active_set(
                records, 'autowell1C1.0', 25.0, _policy(), 0.10, actions
            )

        actions = _acknowledged_actions(records)
        actions['autowell1C1.0']['acknowledged'] = False
        with self.assertRaisesRegex(AutoStabilityPerTriggerError, 'not acknowledged'):
            build_per_trigger_post_mix_active_set(
                records, 'autowell1C1.0', 25.0, _policy(), 0.10, actions
            )

        actions = _acknowledged_actions(records)
        actions['autowell1C1.0']['batch_number'] = 0
        with self.assertRaisesRegex(AutoStabilityPerTriggerError, 'batch mismatch'):
            build_per_trigger_post_mix_active_set(
                records, 'autowell1C1.0', 25.0, _policy(), 0.10, actions
            )

    def test_cross_batch_plan_keeps_prior_active_wells_and_new_trigger(self):
        plan = build_per_trigger_active_set(
            [
                _record('autowell0C1.0', 0, 0.0, 'A1'),
                _record('autowell2C1.0', 2, 40.0, 'B1'),
            ],
            'autowell2C1.0', 45.0, _policy(), 0.10,
        )

        self.assertEqual(PER_TRIGGER_ACTIVE_SET_SCHEMA_VERSION, 1)
        self.assertEqual(plan['scan_reason'], 'each_completion')
        self.assertEqual(
            plan['scan_wellnames'], ['autowell0C1.0', 'autowell2C1.0']
        )
        self.assertEqual(plan['reader_locations'], ['A1', 'B1'])
        self.assertEqual(plan['active_batch_numbers'], [0, 2])
        self.assertEqual(plan['retired_wellnames'], [])

    def test_reader_layout_uses_completion_order_not_dispatch_record_order(self):
        plan = build_per_trigger_active_set(
            [
                _record(
                    'autowell1C1.0', 1, 20.0, 'B1',
                    trigger_completion_sequence=2
                ),
                _record(
                    'autowell0C1.0', 0, 0.0, 'A1',
                    trigger_completion_sequence=1
                ),
            ],
            'autowell1C1.0', 25.0, _policy(), 0.10,
        )
        self.assertEqual(
            plan['scan_wellnames'], ['autowell0C1.0', 'autowell1C1.0']
        )
        self.assertEqual(plan['reader_locations'], ['A1', 'B1'])

    def test_lifecycle_retirement_removes_only_expired_old_well_from_scan(self):
        plan = build_per_trigger_active_set(
            [
                _record('autowell0C1.0', 0, 0.0, 'A1'),
                _record('autowell1C1.0', 1, 100.0, 'B1'),
            ],
            'autowell1C1.0', 100.0, _policy(
                adaptive_retirement_enabled=False
            ), 0.10,
        )

        self.assertEqual(plan['retired_wellnames'], ['autowell0C1.0'])
        self.assertEqual(plan['scan_wellnames'], ['autowell1C1.0'])
        self.assertEqual(
            plan['well_lifecycles'][0]['retirement_reason'],
            'maximum_observation_window_expired'
        )

    def test_decision_ready_well_remains_observable_before_retirement(self):
        plan = build_per_trigger_active_set(
            [
                _record('autowell0C1.0', 0, 0.0, 'A1'),
                _record('autowell1C1.0', 1, 40.0, 'B1'),
            ],
            'autowell1C1.0', 40.0, _policy(), 0.10,
        )
        self.assertEqual(plan['decision_ready_wellnames'], ['autowell0C1.0'])
        self.assertEqual(
            plan['scan_wellnames'], ['autowell0C1.0', 'autowell1C1.0']
        )

    def test_mixed_plate_identity_duplicate_locations_and_unregistered_wells_fail(self):
        records = [
            _record('autowell0C1.0', 0, 0.0, 'A1'),
            _record('autowell1C1.0', 1, 20.0, 'B1', plate_generation=5),
        ]
        with self.assertRaisesRegex(AutoStabilityPerTriggerError, 'multiple'):
            build_per_trigger_active_set(
                records, 'autowell1C1.0', 21.0, _policy(), 0.10
            )

        with self.assertRaisesRegex(AutoStabilityPerTriggerError, 'one reader'):
            build_per_trigger_active_set(
                [
                    _record('autowell0C1.0', 0, 0.0, 'A1'),
                    _record('autowell1C1.0', 1, 20.0, 'A1'),
                ], 'autowell1C1.0', 21.0, _policy(), 0.10
            )

        with self.assertRaisesRegex(AutoStabilityPerTriggerError, 'integer'):
            build_per_trigger_active_set(
                [
                    _record(
                        'autowell0C1.0', 0, 0.0, 'A1',
                        plate_generation=None
                    ),
                ], 'autowell0C1.0', 1.0, _policy(), 0.10
            )

    def test_unknown_observation_source_cannot_silently_affect_retirement(self):
        with self.assertRaisesRegex(
                AutoStabilityPerTriggerError, 'non-active'):
            build_per_trigger_active_set(
                [_record('autowell0C1.0', 0, 0.0, 'A1')],
                'autowell0C1.0', 1.0, _policy(), 0.10,
                observations_by_well={'not-active': []}
            )

    def test_plan_rejects_a_nonlatest_trigger_or_shared_completion_sequence(self):
        records = [
            _record('autowell0C1.0', 0, 0.0, 'A1'),
            _record('autowell1C1.0', 1, 20.0, 'B1'),
        ]
        with self.assertRaisesRegex(AutoStabilityPerTriggerError, 'newest'):
            build_per_trigger_active_set(
                records, 'autowell0C1.0', 21.0, _policy(), 0.10
            )
        with self.assertRaisesRegex(AutoStabilityPerTriggerError, 'share'):
            build_per_trigger_active_set(
                [
                    _record('autowell0C1.0', 0, 0.0, 'A1'),
                    _record(
                        'autowell1C1.0', 1, 20.0, 'B1',
                        trigger_completion_sequence=1
                    ),
                ], 'autowell1C1.0', 21.0, _policy(), 0.10
            )


if __name__ == '__main__':
    unittest.main()
