'''Pure cumulative GP support for reference-peak absorbance in Auto stability.

This companion model is deliberately distinct from both the established
lambda-max ``OptimizationModel`` and Stage-12E's log-loss-rate GP.  It learns
the condition-level blank-corrected absorbance peak at the fixed reference
wavelength.  Its target remains in raw absorbance units so the explicit
stability-only Header bounds have the same scientific units as its future
predictions.

This module has no controller, reader, robot, filesystem, or recipe-selection
dependency.  Stage 12F-D therefore adds auditable signal evidence without
changing execution or candidate selection.
'''

from __future__ import division

import math

import GPy
import numpy as np


STABILITY_SIGNAL_MODEL_TARGET_TRANSFORM = 'reference_peak_absorbance'
STABILITY_SIGNAL_MODEL_STATUS_INSUFFICIENT = 'insufficient_observations'
STABILITY_SIGNAL_MODEL_STATUS_FITTED = 'fitted'
STABILITY_SIGNAL_MODEL_STATUS_FIT_FAILED = 'fit_failed'
STABILITY_SIGNAL_MODEL_MINIMUM_OBSERVATIONS = 2


class AutoStabilitySignalModelValidationError(ValueError):
    '''Raised when reference-peak model data are ambiguous or invalid.'''


def _finite_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _executed_in_current_run(value):
    if value is False:
        return False
    return str(value).strip().lower() not in ('false', '0', 'no')


def _condition_identity(row, fallback_ordinal):
    return '|'.join([
        str(row.get('origin_run_directory', '')).strip(),
        str(row.get('batch_number', '')).strip(),
        str(row.get('reaction_number', fallback_ordinal)).strip(),
    ])


def _empty_audit_row(row, condition_id):
    return {
        'condition_id': condition_id,
        'origin_run_directory': row.get('origin_run_directory', ''),
        'batch_number': row.get('batch_number', ''),
        'reaction_number': row.get('reaction_number', ''),
        'condition_type': row.get('condition_type', ''),
        'executed_in_current_run': _executed_in_current_run(
            row.get('executed_in_current_run', True)
        ),
        'condition_signal_status': '',
        'reference_peak_eligible_well_count': None,
        'total_well_count': None,
        'reference_peak_absorbance': None,
        'replicate_reference_peak_absorbance_sample_sd': None,
        'model_target_transform': STABILITY_SIGNAL_MODEL_TARGET_TRANSFORM,
        'model_target_reference_peak_absorbance': None,
        'normalized_recipe': None,
        'stability_signal_model_training_status': '',
        'stability_signal_model_training_reason': '',
    }


def _validate_recipe_bounds(variable_reagents, min_concentrations,
                            max_concentrations):
    reagent_names = [str(name) for name in (variable_reagents or [])]
    if not reagent_names:
        raise AutoStabilitySignalModelValidationError(
            'A stability signal model requires at least one variable reagent.'
        )
    if len(set(reagent_names)) != len(reagent_names):
        raise AutoStabilitySignalModelValidationError(
            'Stability signal model variable reagent names must be unique.'
        )
    minimums = np.asarray(min_concentrations, dtype=float).reshape(-1)
    maximums = np.asarray(max_concentrations, dtype=float).reshape(-1)
    if (
            minimums.shape[0] != len(reagent_names)
            or maximums.shape[0] != len(reagent_names)
            or not np.all(np.isfinite(minimums))
            or not np.all(np.isfinite(maximums))
            or np.any(maximums <= minimums)):
        raise AutoStabilitySignalModelValidationError(
            'Stability signal model concentration bounds must be finite, '
            'aligned with variable reagents, and have positive spans.'
        )
    return reagent_names, minimums, maximums


def build_stability_signal_model_training_records(
        condition_summaries,
        condition_rows,
        variable_reagents,
        min_concentrations,
        max_concentrations):
    '''Return explicit current-run reference-peak training audit records.

    A condition may train this signal model when every expected physical
    replicate supplied a finite fixed-reference peak.  This is intentionally
    independent from loss-rate eligibility: a no-decline trajectory, a low
    signal, or a high signal still teaches the future selector about optical
    signal magnitude.  Imported lambda-only history remains excluded because
    it lacks immutable current-run reference-peak provenance.
    '''
    reagent_names, minimums, maximums = _validate_recipe_bounds(
        variable_reagents, min_concentrations, max_concentrations
    )

    summaries_by_id = {}
    for summary in condition_summaries or []:
        if not isinstance(summary, dict):
            continue
        condition_id = str(summary.get('condition_id', '')).strip()
        if not condition_id:
            continue
        if condition_id in summaries_by_id:
            raise AutoStabilitySignalModelValidationError(
                'Duplicate stability signal condition summary identity: {}.'
                .format(condition_id)
            )
        summaries_by_id[condition_id] = dict(summary)

    records = []
    known_ids = set()
    for ordinal, source_row in enumerate(condition_rows or []):
        if not isinstance(source_row, dict):
            continue
        source_row = dict(source_row)
        condition_id = _condition_identity(source_row, ordinal)
        audit = _empty_audit_row(source_row, condition_id)
        if condition_id in known_ids:
            audit['stability_signal_model_training_status'] = (
                'rejected_duplicate_condition'
            )
            audit['stability_signal_model_training_reason'] = (
                'The condition identity occurred more than once in the '
                'performance history and cannot be mapped unambiguously.'
            )
            records.append(audit)
            continue
        known_ids.add(condition_id)

        summary = summaries_by_id.get(condition_id)
        if not audit['executed_in_current_run']:
            audit['stability_signal_model_training_status'] = (
                'rejected_imported_condition'
            )
            audit['stability_signal_model_training_reason'] = (
                'Imported lambda-history conditions have no immutable '
                'current-run reference-peak trajectory and are excluded.'
            )
            records.append(audit)
            continue
        if summary is None:
            audit['stability_signal_model_training_status'] = (
                'rejected_missing_signal_summary'
            )
            audit['stability_signal_model_training_reason'] = (
                'No manifest-linked condition signal summary was available '
                'for this current-run condition.'
            )
            records.append(audit)
            continue

        audit.update({
            'condition_signal_status': summary.get(
                'condition_signal_status', ''
            ),
            'reference_peak_eligible_well_count': summary.get(
                'reference_peak_eligible_well_count'
            ),
            'total_well_count': summary.get('total_well_count'),
            'reference_peak_absorbance': summary.get(
                'condition_reference_peak_absorbance_mean'
            ),
            'replicate_reference_peak_absorbance_sample_sd': summary.get(
                'condition_reference_peak_absorbance_sample_sd'
            ),
        })
        if audit['condition_signal_status'] != 'complete':
            audit['stability_signal_model_training_status'] = (
                'rejected_incomplete_signal_evidence'
            )
            audit['stability_signal_model_training_reason'] = (
                'All expected replicate wells must have a finite '
                'fixed-reference peak before a condition can train the '
                'stability signal model.'
            )
            records.append(audit)
            continue

        peak = _finite_float(audit['reference_peak_absorbance'])
        if peak is None:
            audit['stability_signal_model_training_status'] = (
                'rejected_invalid_reference_peak'
            )
            audit['stability_signal_model_training_reason'] = (
                'A complete signal condition requires a finite '
                'condition-level reference-peak absorbance.'
            )
            records.append(audit)
            continue

        concentrations = []
        for reagent_name in reagent_names:
            concentration = _finite_float(
                source_row.get(reagent_name + '_concentration')
            )
            if concentration is None:
                audit['stability_signal_model_training_status'] = (
                    'rejected_missing_recipe'
                )
                audit['stability_signal_model_training_reason'] = (
                    'A finite executed concentration was not recorded for '
                    'variable reagent {}.'.format(reagent_name)
                )
                break
            concentrations.append(concentration)
            audit['{}_concentration_mM'.format(reagent_name)] = concentration
        else:
            concentration_array = np.asarray(concentrations, dtype=float)
            normalized_recipe = (
                (concentration_array - minimums) / (maximums - minimums)
            )
            tolerance = 1e-9
            if (
                    not np.all(np.isfinite(normalized_recipe))
                    or np.any(normalized_recipe < -tolerance)
                    or np.any(normalized_recipe > 1.0 + tolerance)):
                audit['stability_signal_model_training_status'] = (
                    'rejected_recipe_outside_bounds'
                )
                audit['stability_signal_model_training_reason'] = (
                    'Executed concentrations could not be represented in the '
                    'current normalized variable-reagent bounds.'
                )
            else:
                audit['normalized_recipe'] = np.clip(
                    normalized_recipe, 0.0, 1.0
                ).tolist()
                audit['model_target_reference_peak_absorbance'] = peak
                audit['stability_signal_model_training_status'] = 'accepted'
                audit['stability_signal_model_training_reason'] = (
                    'Complete fixed-reference peak evidence; loss-rate QC is '
                    'deliberately independent.'
                )
        records.append(audit)

    for condition_id, summary in sorted(summaries_by_id.items()):
        if condition_id in known_ids:
            continue
        audit = _empty_audit_row({}, condition_id)
        audit.update({
            'condition_signal_status': summary.get('condition_signal_status', ''),
            'reference_peak_eligible_well_count': summary.get(
                'reference_peak_eligible_well_count'
            ),
            'total_well_count': summary.get('total_well_count'),
            'reference_peak_absorbance': summary.get(
                'condition_reference_peak_absorbance_mean'
            ),
            'replicate_reference_peak_absorbance_sample_sd': summary.get(
                'condition_reference_peak_absorbance_sample_sd'
            ),
            'stability_signal_model_training_status': (
                'rejected_missing_condition_provenance'
            ),
            'stability_signal_model_training_reason': (
                'The manifest-linked signal summary could not be matched to '
                'one immutable Auto condition record.'
            ),
        })
        records.append(audit)
    return records


class AutoStabilitySignalModel(object):
    '''Cumulative GP over accepted raw fixed-reference peak absorbance.'''

    def __init__(self, variable_reagents):
        self.variable_reagents = tuple(str(name) for name in variable_reagents)
        if not self.variable_reagents:
            raise AutoStabilitySignalModelValidationError(
                'A stability signal model requires at least one variable '
                'reagent.'
            )
        if len(set(self.variable_reagents)) != len(self.variable_reagents):
            raise AutoStabilitySignalModelValidationError(
                'Stability signal model variable reagent names must be unique.'
            )
        self.gp_model = None
        self.X = np.empty((0, len(self.variable_reagents)), dtype=float)
        self.Y_reference_peak_absorbance = np.empty((0, 1), dtype=float)
        self.status = STABILITY_SIGNAL_MODEL_STATUS_INSUFFICIENT
        self.last_error = None

    def refresh_from_training_records(self, training_records):
        '''Atomically rebuild from complete accepted cumulative signal history.'''
        accepted = [
            dict(row) for row in (training_records or [])
            if isinstance(row, dict)
            and row.get('stability_signal_model_training_status') == 'accepted'
        ]
        candidate_x = []
        candidate_y = []
        for row in accepted:
            recipe = np.asarray(row.get('normalized_recipe'), dtype=float)
            target = _finite_float(
                row.get('model_target_reference_peak_absorbance')
            )
            if (
                    recipe.ndim != 1
                    or recipe.shape[0] != len(self.variable_reagents)
                    or target is None
                    or not np.all(np.isfinite(recipe))
                    or np.any(recipe < 0.0) or np.any(recipe > 1.0)):
                raise AutoStabilitySignalModelValidationError(
                    'Accepted stability signal training records contain '
                    'invalid normalized recipe or reference-peak values.'
                )
            candidate_x.append(recipe.copy())
            candidate_y.append(target)

        candidate_X = np.asarray(candidate_x, dtype=float)
        if candidate_X.size == 0:
            candidate_X = np.empty((0, len(self.variable_reagents)), dtype=float)
        else:
            candidate_X = candidate_X.reshape(-1, len(self.variable_reagents))
        candidate_Y = np.asarray(candidate_y, dtype=float).reshape(-1, 1)
        result = {
            'status': STABILITY_SIGNAL_MODEL_STATUS_INSUFFICIENT,
            'fitted': False,
            'accepted_condition_count': len(accepted),
            'minimum_condition_count': STABILITY_SIGNAL_MODEL_MINIMUM_OBSERVATIONS,
            'model_target_transform': STABILITY_SIGNAL_MODEL_TARGET_TRANSFORM,
            'variable_reagents': list(self.variable_reagents),
            'error': None,
        }
        if candidate_X.shape[0] < self.X.shape[0]:
            message = (
                'Stability signal model refresh would regress cumulative '
                'accepted history from {} to {} condition(s).'.format(
                    self.X.shape[0], candidate_X.shape[0]
                )
            )
            self.status = STABILITY_SIGNAL_MODEL_STATUS_FIT_FAILED
            self.last_error = message
            result.update({
                'status': STABILITY_SIGNAL_MODEL_STATUS_FIT_FAILED,
                'error': message,
            })
            return result
        if len(accepted) < STABILITY_SIGNAL_MODEL_MINIMUM_OBSERVATIONS:
            self.X = candidate_X.copy()
            self.Y_reference_peak_absorbance = candidate_Y.copy()
            self.gp_model = None
            self.status = STABILITY_SIGNAL_MODEL_STATUS_INSUFFICIENT
            self.last_error = None
            return result

        kernel = GPy.kern.Matern32(
            input_dim=len(self.variable_reagents), variance=1.0,
            lengthscale=1.0, ARD=True,
        )
        try:
            fresh_model = GPy.models.GPRegression(
                candidate_X.copy(), candidate_Y.copy(), kernel, noise_var=1e-6
            )
        except Exception as exc:
            self.status = STABILITY_SIGNAL_MODEL_STATUS_FIT_FAILED
            self.last_error = str(exc)
            result.update({
                'status': STABILITY_SIGNAL_MODEL_STATUS_FIT_FAILED,
                'error': str(exc),
            })
            return result

        self.gp_model = fresh_model
        self.X = candidate_X.copy()
        self.Y_reference_peak_absorbance = candidate_Y.copy()
        self.status = STABILITY_SIGNAL_MODEL_STATUS_FITTED
        self.last_error = None
        result.update({
            'status': STABILITY_SIGNAL_MODEL_STATUS_FITTED,
            'fitted': True,
        })
        return result

    def predict_reference_peak_absorbance_distribution(self, normalized_recipes):
        '''Return raw-absorbance prediction mean and SD without state changes.'''
        if (
                self.status != STABILITY_SIGNAL_MODEL_STATUS_FITTED
                or self.gp_model is None):
            raise AutoStabilitySignalModelValidationError(
                'Cannot predict reference peak before the companion signal GP '
                'is fitted.'
            )
        values = np.asarray(normalized_recipes, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if (
                values.ndim != 2
                or values.shape[1] != len(self.variable_reagents)
                or values.shape[0] == 0
                or not np.all(np.isfinite(values))
                or np.any(values < 0.0) or np.any(values > 1.0)):
            raise AutoStabilitySignalModelValidationError(
                'Stability signal predictions require finite normalized '
                'recipes in [0, 1] with one value per variable reagent.'
            )
        try:
            predicted_mean, predicted_variance = self.gp_model.predict(values)
        except Exception as exc:
            raise AutoStabilitySignalModelValidationError(
                'Companion stability signal GP prediction failed: {}.'
                .format(exc)
            )
        predicted_mean = np.asarray(predicted_mean, dtype=float).reshape(-1)
        predicted_variance = np.asarray(
            predicted_variance, dtype=float
        ).reshape(-1)
        if (
                predicted_mean.shape[0] != values.shape[0]
                or predicted_variance.shape[0] != values.shape[0]
                or not np.all(np.isfinite(predicted_mean))
                or not np.all(np.isfinite(predicted_variance))):
            raise AutoStabilitySignalModelValidationError(
                'Companion stability signal GP returned non-finite or '
                'misaligned prediction values.'
            )
        tolerance = 1e-12
        if np.any(predicted_variance < -tolerance):
            raise AutoStabilitySignalModelValidationError(
                'Companion stability signal GP returned a materially negative '
                'predictive variance.'
            )
        return predicted_mean.copy(), np.sqrt(
            np.maximum(predicted_variance, 0.0)
        ).copy()
