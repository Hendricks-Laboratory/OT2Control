'''Hardware-free Stage-12E stability-model tests.'''

import math
import unittest
from unittest import mock

import numpy as np

from auto_stability_model import (
    AutoStabilityModel,
    STABILITY_MODEL_STATUS_FIT_FAILED,
    STABILITY_MODEL_STATUS_FITTED,
    STABILITY_MODEL_STATUS_INSUFFICIENT,
    build_stability_model_training_records,
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


def _summary(run_name, batch_number, reaction_number, rate, **overrides):
    row = {
        'condition_id': '{}|{}|{}'.format(
            run_name, batch_number, reaction_number
        ),
        'condition_stability_status': 'complete',
        'eligible_well_count': 2,
        'total_well_count': 2,
        'condition_loss_rate_mean_absorbance_per_s': rate,
    }
    row.update(overrides)
    return row


class StabilityModelTrainingRecordTests(unittest.TestCase):
    def _records(self, summaries, conditions):
        return build_stability_model_training_records(
            condition_summaries=summaries,
            condition_rows=conditions,
            variable_reagents=['silver_nitrate', 'potassium_bromide'],
            min_concentrations=[0.0, 0.0],
            max_concentrations=[0.20, 0.010],
        )

    def test_complete_current_condition_maps_to_log_rate_and_normalized_recipe(self):
        records = self._records(
            [_summary('RUN', 0, 0, 0.001)],
            [_condition('RUN', 0, 0)],
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record['stability_model_training_status'], 'accepted')
        self.assertEqual(record['normalized_recipe'], [0.5, 0.5])
        self.assertAlmostEqual(
            record['model_target_log10_loss_rate'], -3.0
        )
        self.assertAlmostEqual(
            record['loss_rate_absorbance_per_s'], 0.001
        )

    def test_stability_qc_is_independent_and_imported_history_is_excluded(self):
        records = self._records(
            [
                _summary('RUN', 0, 0, 0.001),
                _summary(
                    'RUN', 0, 1, 0.002,
                    condition_stability_status='partial',
                    eligible_well_count=1,
                    total_well_count=2,
                ),
            ],
            [
                _condition('RUN', 0, 0, model_training_status='excluded'),
                _condition('RUN', 0, 1),
                _condition(
                    'IMPORTED', 2, 4,
                    executed_in_current_run=False,
                ),
            ],
        )
        statuses = [
            record['stability_model_training_status'] for record in records
        ]
        self.assertEqual(statuses, [
            'accepted', 'rejected_stability_qc',
            'rejected_imported_condition',
        ])
        self.assertIn(
            'Complete condition-level stability QC',
            records[0]['stability_model_training_reason']
        )

    def test_active_selection_rejects_excessive_log_rate_replicate_spread(self):
        summary = _summary(
            'RUN', 0, 0, 0.001,
            condition_log10_loss_rate_sample_sd=0.31,
        )
        records = build_stability_model_training_records(
            condition_summaries=[summary],
            condition_rows=[_condition('RUN', 0, 0)],
            variable_reagents=['silver_nitrate', 'potassium_bromide'],
            min_concentrations=[0.0, 0.0],
            max_concentrations=[0.20, 0.010],
            replicate_log10_loss_rate_sd_max=0.20,
        )
        self.assertEqual(
            records[0]['stability_model_training_status'],
            'rejected_replicate_log10_loss_rate_sd'
        )
        self.assertIn('exceeds the configured limit',
                      records[0]['stability_model_training_reason'])

    def test_active_selection_requires_log_rate_replicate_spread_evidence(self):
        records = build_stability_model_training_records(
            condition_summaries=[_summary('RUN', 0, 0, 0.001)],
            condition_rows=[_condition('RUN', 0, 0)],
            variable_reagents=['silver_nitrate', 'potassium_bromide'],
            min_concentrations=[0.0, 0.0],
            max_concentrations=[0.20, 0.010],
            replicate_log10_loss_rate_sd_max=0.20,
        )
        self.assertEqual(
            records[0]['stability_model_training_status'],
            'rejected_missing_replicate_log10_loss_rate_sd'
        )

    def test_unknown_summary_and_bad_recipe_are_rejected_without_guessing(self):
        records = self._records(
            [
                _summary('RUN', 0, 0, 0.001),
                _summary('ORPHAN', 1, 2, 0.002),
            ],
            [_condition(
                'RUN', 0, 0,
                silver_nitrate_concentration=None,
            )],
        )
        statuses = {
            record['condition_id']: record['stability_model_training_status']
            for record in records
        }
        self.assertEqual(statuses['RUN|0|0'], 'rejected_missing_recipe')
        self.assertEqual(
            statuses['ORPHAN|1|2'],
            'rejected_missing_condition_provenance'
        )


class AutoStabilityModelTests(unittest.TestCase):
    @staticmethod
    def _accepted_records(count):
        records = []
        for index in range(count):
            rate = 0.001 * (index + 1)
            records.append({
                'stability_model_training_status': 'accepted',
                'normalized_recipe': [0.2 + index * 0.1, 0.3 + index * 0.1],
                'model_target_log10_loss_rate': math.log10(rate),
            })
        return records

    def test_model_retains_one_condition_but_does_not_claim_a_fit(self):
        model = AutoStabilityModel(['silver_nitrate', 'potassium_bromide'])
        summary = model.refresh_from_training_records(self._accepted_records(1))
        self.assertEqual(summary['status'], STABILITY_MODEL_STATUS_INSUFFICIENT)
        self.assertFalse(summary['fitted'])
        self.assertEqual(model.X.shape, (1, 2))
        self.assertIsNone(model.gp_model)

    def test_model_fits_a_fresh_cumulative_gp_from_two_conditions(self):
        model = AutoStabilityModel(['silver_nitrate', 'potassium_bromide'])
        summary = model.refresh_from_training_records(self._accepted_records(2))
        self.assertEqual(summary['status'], STABILITY_MODEL_STATUS_FITTED)
        self.assertTrue(summary['fitted'])
        self.assertEqual(model.X.shape, (2, 2))
        self.assertEqual(model.Y_log10_loss_rate.shape, (2, 1))
        self.assertIsNotNone(model.gp_model)

    def test_fit_failure_keeps_previously_fitted_arrays_and_gp_atomic(self):
        model = AutoStabilityModel(['silver_nitrate', 'potassium_bromide'])
        model.refresh_from_training_records(self._accepted_records(2))
        old_model = model.gp_model
        old_X = model.X.copy()
        old_Y = model.Y_log10_loss_rate.copy()
        with mock.patch(
                'auto_stability_model.GPy.models.GPRegression',
                side_effect=RuntimeError('synthetic fit failure')):
            summary = model.refresh_from_training_records(
                self._accepted_records(3)
            )
        self.assertEqual(summary['status'], STABILITY_MODEL_STATUS_FIT_FAILED)
        self.assertIs(model.gp_model, old_model)
        self.assertTrue(np.array_equal(model.X, old_X))
        self.assertTrue(np.array_equal(model.Y_log10_loss_rate, old_Y))

    def test_candidate_history_regression_cannot_discard_a_fitted_model(self):
        model = AutoStabilityModel(['silver_nitrate', 'potassium_bromide'])
        model.refresh_from_training_records(self._accepted_records(2))
        old_model = model.gp_model
        old_X = model.X.copy()
        summary = model.refresh_from_training_records(self._accepted_records(1))
        self.assertEqual(summary['status'], STABILITY_MODEL_STATUS_FIT_FAILED)
        self.assertIn('regress cumulative', summary['error'])
        self.assertIs(model.gp_model, old_model)
        self.assertTrue(np.array_equal(model.X, old_X))

    def test_prediction_returns_log_rate_mean_and_standard_deviation(self):
        model = AutoStabilityModel(['silver_nitrate', 'potassium_bromide'])
        model.refresh_from_training_records(self._accepted_records(2))
        mean, standard_deviation = model.predict_log10_loss_rate_distribution(
            [[0.3, 0.4], [0.4, 0.5]]
        )
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(standard_deviation.shape, (2,))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(standard_deviation >= 0.0))

    def test_prediction_fails_before_companion_model_is_fitted(self):
        model = AutoStabilityModel(['silver_nitrate', 'potassium_bromide'])
        with self.assertRaisesRegex(ValueError, 'before the companion GP is fitted'):
            model.predict_log10_loss_rate_distribution([[0.3, 0.4]])


if __name__ == '__main__':
    unittest.main()
