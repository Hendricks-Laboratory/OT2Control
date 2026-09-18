'''Hardware-free Stage-12F-D reference-peak signal-model tests.'''

import unittest
from unittest import mock

import numpy as np

from auto_stability_signal_model import (
    AutoStabilitySignalModel,
    STABILITY_SIGNAL_MODEL_STATUS_FIT_FAILED,
    STABILITY_SIGNAL_MODEL_STATUS_FITTED,
    STABILITY_SIGNAL_MODEL_STATUS_INSUFFICIENT,
    build_stability_signal_model_training_records,
)


def _condition(run_name, batch_number, reaction_number, **overrides):
    row = {
        'origin_run_directory': run_name,
        'batch_number': batch_number,
        'reaction_number': reaction_number,
        'condition_type': 'seed',
        'executed_in_current_run': True,
        'silver_nitrate_concentration': 0.10,
        'potassium_bromide_concentration': 0.005,
    }
    row.update(overrides)
    return row


def _summary(run_name, batch_number, reaction_number, peak, **overrides):
    row = {
        'condition_id': '{}|{}|{}'.format(
            run_name, batch_number, reaction_number
        ),
        'condition_signal_status': 'complete',
        'reference_peak_eligible_well_count': 2,
        'total_well_count': 2,
        'condition_reference_peak_absorbance_mean': peak,
        'condition_reference_peak_absorbance_sample_sd': 0.01,
    }
    row.update(overrides)
    return row


class StabilitySignalTrainingRecordTests(unittest.TestCase):
    def _records(self, summaries, conditions):
        return build_stability_signal_model_training_records(
            condition_summaries=summaries,
            condition_rows=conditions,
            variable_reagents=['silver_nitrate', 'potassium_bromide'],
            min_concentrations=[0.0, 0.0],
            max_concentrations=[0.20, 0.010],
        )

    def test_complete_signal_maps_to_raw_peak_and_normalized_recipe(self):
        records = self._records(
            [_summary('RUN', 0, 0, 0.42)], [_condition('RUN', 0, 0)]
        )
        self.assertEqual(
            records[0]['stability_signal_model_training_status'], 'accepted'
        )
        self.assertEqual(records[0]['normalized_recipe'], [0.5, 0.5])
        self.assertAlmostEqual(
            records[0]['model_target_reference_peak_absorbance'], 0.42
        )

    def test_no_decline_low_and_high_signal_remain_training_evidence(self):
        records = self._records(
            [
                _summary(
                    'RUN', 0, 0, 0.02,
                    condition_stability_status='no_eligible_wells',
                ),
                _summary(
                    'RUN', 0, 1, 1.50,
                    condition_stability_status='no_eligible_wells',
                ),
            ],
            [
                _condition('RUN', 0, 0),
                _condition('RUN', 0, 1,
                           silver_nitrate_concentration=0.12),
            ],
        )
        self.assertEqual(
            [row['stability_signal_model_training_status'] for row in records],
            ['accepted', 'accepted'],
        )
        self.assertIn('independent', records[0][
            'stability_signal_model_training_reason'
        ])

    def test_incomplete_missing_duplicate_imported_and_orphan_are_audited(self):
        records = self._records(
            [
                _summary('RUN', 0, 0, 0.2,
                         condition_signal_status='partial',
                         reference_peak_eligible_well_count=1),
                _summary('RUN', 0, 1, None),
                _summary('ORPHAN', 1, 2, 0.4),
            ],
            [
                _condition('RUN', 0, 0),
                _condition('RUN', 0, 1),
                _condition('RUN', 0, 1),
                _condition('IMPORTED', 3, 3, executed_in_current_run=False),
                _condition('RUN', 0, 4),
            ],
        )
        statuses = [
            row['stability_signal_model_training_status'] for row in records
        ]
        self.assertEqual(statuses, [
            'rejected_incomplete_signal_evidence',
            'rejected_invalid_reference_peak',
            'rejected_duplicate_condition',
            'rejected_imported_condition',
            'rejected_missing_signal_summary',
            'rejected_missing_condition_provenance',
        ])

    def test_recipe_errors_are_rejected_without_guessing(self):
        records = self._records(
            [_summary('RUN', 0, 0, 0.2), _summary('RUN', 0, 1, 0.3)],
            [
                _condition('RUN', 0, 0, silver_nitrate_concentration=None),
                _condition('RUN', 0, 1, potassium_bromide_concentration=0.02),
            ],
        )
        self.assertEqual(
            records[0]['stability_signal_model_training_status'],
            'rejected_missing_recipe'
        )
        self.assertEqual(
            records[1]['stability_signal_model_training_status'],
            'rejected_recipe_outside_bounds'
        )


class AutoStabilitySignalModelTests(unittest.TestCase):
    @staticmethod
    def _accepted_records(count):
        return [{
            'stability_signal_model_training_status': 'accepted',
            'normalized_recipe': [0.2 + index * 0.1, 0.3 + index * 0.1],
            'model_target_reference_peak_absorbance': 0.2 + index * 0.1,
        } for index in range(count)]

    def test_one_condition_is_retained_without_claiming_a_gp_fit(self):
        model = AutoStabilitySignalModel(
            ['silver_nitrate', 'potassium_bromide']
        )
        result = model.refresh_from_training_records(self._accepted_records(1))
        self.assertEqual(result['status'], STABILITY_SIGNAL_MODEL_STATUS_INSUFFICIENT)
        self.assertFalse(result['fitted'])
        self.assertEqual(model.X.shape, (1, 2))
        self.assertIsNone(model.gp_model)

    def test_cumulative_model_fits_and_predicts_raw_absorbance(self):
        model = AutoStabilitySignalModel(
            ['silver_nitrate', 'potassium_bromide']
        )
        result = model.refresh_from_training_records(self._accepted_records(2))
        self.assertEqual(result['status'], STABILITY_SIGNAL_MODEL_STATUS_FITTED)
        self.assertEqual(model.Y_reference_peak_absorbance.shape, (2, 1))
        mean, standard_deviation = (
            model.predict_reference_peak_absorbance_distribution(
                [[0.3, 0.4], [0.4, 0.5]]
            )
        )
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(standard_deviation.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(standard_deviation >= 0.0))

    def test_failed_fit_or_history_regression_cannot_mutate_fitted_state(self):
        model = AutoStabilitySignalModel(
            ['silver_nitrate', 'potassium_bromide']
        )
        model.refresh_from_training_records(self._accepted_records(2))
        old_model = model.gp_model
        old_X = model.X.copy()
        old_Y = model.Y_reference_peak_absorbance.copy()
        with mock.patch(
                'auto_stability_signal_model.GPy.models.GPRegression',
                side_effect=RuntimeError('synthetic fit failure')):
            result = model.refresh_from_training_records(
                self._accepted_records(3)
            )
        self.assertEqual(result['status'], STABILITY_SIGNAL_MODEL_STATUS_FIT_FAILED)
        self.assertIs(model.gp_model, old_model)
        self.assertTrue(np.array_equal(model.X, old_X))
        self.assertTrue(np.array_equal(model.Y_reference_peak_absorbance, old_Y))
        result = model.refresh_from_training_records(self._accepted_records(1))
        self.assertEqual(result['status'], STABILITY_SIGNAL_MODEL_STATUS_FIT_FAILED)
        self.assertIs(model.gp_model, old_model)
        self.assertTrue(np.array_equal(model.X, old_X))


if __name__ == '__main__':
    unittest.main()
