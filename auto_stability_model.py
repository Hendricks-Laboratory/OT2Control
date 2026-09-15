'''Pure cumulative Gaussian-process support for Auto optical stability.

Stage 12E keeps optical-stability learning separate from the existing
target-wavelength ``OptimizationModel``.  This module has no controller,
reader, robot, filesystem, or recipe-selection dependency.  It accepts only
already-derived condition summaries and immutable recipe provenance, then
builds an auditable, cumulative companion GP in normalized recipe space.

The physical response remains ``absorbance loss / second``.  Eligible loss
rates are strictly positive by the Stage-12A metric contract, so their base-10
logarithm is a finite, monotonic internal regression target.  Raw units are
preserved in every audit record; predictions are intentionally not used for
selection until Stage 12F.
'''

from __future__ import division

import math

import GPy
import numpy as np


STABILITY_MODEL_TARGET_TRANSFORM = 'log10_loss_rate_absorbance_per_s'
STABILITY_MODEL_STATUS_DISABLED = 'disabled'
STABILITY_MODEL_STATUS_INSUFFICIENT = 'insufficient_observations'
STABILITY_MODEL_STATUS_FITTED = 'fitted'
STABILITY_MODEL_STATUS_FIT_FAILED = 'fit_failed'
STABILITY_MODEL_MINIMUM_OBSERVATIONS = 2


class AutoStabilityModelValidationError(ValueError):
    '''Raised when condition-level stability model data are ambiguous.'''


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
        'condition_stability_status': '',
        'eligible_well_count': None,
        'total_well_count': None,
        'loss_rate_absorbance_per_s': None,
        'model_target_transform': STABILITY_MODEL_TARGET_TRANSFORM,
        'model_target_log10_loss_rate': None,
        'normalized_recipe': None,
        'stability_model_training_status': '',
        'stability_model_training_reason': '',
    }


def build_stability_model_training_records(
        condition_summaries,
        condition_rows,
        variable_reagents,
        min_concentrations,
        max_concentrations):
    '''Return one explicit model-training audit row per known condition.

    Only a condition with a complete Stage-12D stability summary, every
    expected well eligible, a finite positive loss rate, and a finite
    normalized recipe can enter the companion GP.  Existing wavelength QC is
    deliberately not reused as a stability gate: the two measurements answer
    different scientific questions and are audited independently.

    Imported conditions are retained in the audit with an explicit exclusion.
    A normal checkpoint has no immutable current-run trigger/trajectory record
    for an imported well, so importing its lambda history must never fabricate
    stability training evidence.
    '''
    reagent_names = [str(name) for name in (variable_reagents or [])]
    if not reagent_names:
        raise AutoStabilityModelValidationError(
            'A stability model requires at least one variable reagent.'
        )
    if len(set(reagent_names)) != len(reagent_names):
        raise AutoStabilityModelValidationError(
            'Stability model variable reagent names must be unique.'
        )

    minimums = np.asarray(min_concentrations, dtype=float).reshape(-1)
    maximums = np.asarray(max_concentrations, dtype=float).reshape(-1)
    if (
            minimums.shape[0] != len(reagent_names)
            or maximums.shape[0] != len(reagent_names)
            or not np.all(np.isfinite(minimums))
            or not np.all(np.isfinite(maximums))
            or np.any(maximums <= minimums)):
        raise AutoStabilityModelValidationError(
            'Stability model concentration bounds must be finite, aligned '
            'with variable reagents, and have positive spans.'
        )

    summaries_by_id = {}
    for ordinal, summary in enumerate(condition_summaries or []):
        if not isinstance(summary, dict):
            continue
        condition_id = str(summary.get('condition_id', '')).strip()
        if not condition_id:
            continue
        if condition_id in summaries_by_id:
            raise AutoStabilityModelValidationError(
                'Duplicate stability condition summary identity: {}.'
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
            audit['stability_model_training_status'] = 'rejected_duplicate_condition'
            audit['stability_model_training_reason'] = (
                'The condition identity occurred more than once in the '
                'performance history and cannot be mapped unambiguously.'
            )
            records.append(audit)
            continue
        known_ids.add(condition_id)

        summary = summaries_by_id.get(condition_id)
        if not audit['executed_in_current_run']:
            audit['stability_model_training_status'] = 'rejected_imported_condition'
            audit['stability_model_training_reason'] = (
                'Imported lambda-history conditions have no immutable '
                'current-run stability trajectory and are excluded.'
            )
            records.append(audit)
            continue
        if summary is None:
            audit['stability_model_training_status'] = 'rejected_missing_stability_summary'
            audit['stability_model_training_reason'] = (
                'No manifest-linked condition stability summary was '
                'available for this current-run condition.'
            )
            records.append(audit)
            continue

        audit.update({
            'condition_stability_status': summary.get(
                'condition_stability_status', ''
            ),
            'eligible_well_count': summary.get('eligible_well_count'),
            'total_well_count': summary.get('total_well_count'),
            'loss_rate_absorbance_per_s': summary.get(
                'condition_loss_rate_mean_absorbance_per_s'
            ),
        })
        if audit['condition_stability_status'] != 'complete':
            audit['stability_model_training_status'] = 'rejected_stability_qc'
            audit['stability_model_training_reason'] = (
                'All expected replicate wells must have eligible stability '
                'metrics before a condition can train the stability model.'
            )
            records.append(audit)
            continue

        rate = _finite_float(audit['loss_rate_absorbance_per_s'])
        if rate is None or rate <= 0.0:
            audit['stability_model_training_status'] = 'rejected_invalid_loss_rate'
            audit['stability_model_training_reason'] = (
                'A complete stability condition requires a finite positive '
                'post-peak loss rate.'
            )
            records.append(audit)
            continue

        concentrations = []
        for reagent_name in reagent_names:
            concentration = _finite_float(
                source_row.get(reagent_name + '_concentration')
            )
            if concentration is None:
                audit['stability_model_training_status'] = 'rejected_missing_recipe'
                audit['stability_model_training_reason'] = (
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
                audit['stability_model_training_status'] = 'rejected_recipe_outside_bounds'
                audit['stability_model_training_reason'] = (
                    'Executed concentrations could not be represented in the '
                    'current normalized variable-reagent bounds.'
                )
            else:
                normalized_recipe = np.clip(normalized_recipe, 0.0, 1.0)
                audit['normalized_recipe'] = normalized_recipe.tolist()
                audit['model_target_log10_loss_rate'] = math.log10(rate)
                audit['stability_model_training_status'] = 'accepted'
                audit['stability_model_training_reason'] = (
                    'Complete condition-level stability QC with finite '
                    'positive post-peak loss rate.'
                )
        records.append(audit)

    # A manifest/summary that cannot be matched to a condition row is unsafe
    # to train on. Preserve it as an auditable rejection instead of guessing a
    # recipe from timing or current plate state.
    for condition_id, summary in sorted(summaries_by_id.items()):
        if condition_id in known_ids:
            continue
        audit = _empty_audit_row({}, condition_id)
        audit.update({
            'condition_stability_status': summary.get(
                'condition_stability_status', ''
            ),
            'eligible_well_count': summary.get('eligible_well_count'),
            'total_well_count': summary.get('total_well_count'),
            'loss_rate_absorbance_per_s': summary.get(
                'condition_loss_rate_mean_absorbance_per_s'
            ),
            'stability_model_training_status': 'rejected_missing_condition_provenance',
            'stability_model_training_reason': (
                'The manifest-linked stability summary could not be matched '
                'to one immutable Auto condition record.'
            ),
        })
        records.append(audit)
    return records


class AutoStabilityModel(object):
    '''A cumulative GP over accepted condition-level optical-stability rates.

    This is deliberately not an ``OptimizationModel`` and exposes no proposal
    method.  Stage 12E can therefore fit and audit a companion GP without
    changing the primary lambda GP, its optimizer history, or selection.
    '''

    def __init__(self, variable_reagents):
        self.variable_reagents = tuple(str(name) for name in variable_reagents)
        if not self.variable_reagents:
            raise AutoStabilityModelValidationError(
                'A stability model requires at least one variable reagent.'
            )
        if len(set(self.variable_reagents)) != len(self.variable_reagents):
            raise AutoStabilityModelValidationError(
                'Stability model variable reagent names must be unique.'
            )
        self.gp_model = None
        self.X = np.empty((0, len(self.variable_reagents)), dtype=float)
        self.Y_log10_loss_rate = np.empty((0, 1), dtype=float)
        self.status = STABILITY_MODEL_STATUS_INSUFFICIENT
        self.last_error = None

    def refresh_from_training_records(self, training_records):
        '''Rebuild the GP from the complete accepted cumulative history.

        A fresh ``GPRegression`` avoids partial history updates.  If fitting
        fails, the prior fitted model and arrays are left untouched.  Fewer
        than two accepted condition-level observations are retained as known
        history but intentionally do not claim a fitted GP.
        '''
        accepted = [
            dict(row) for row in (training_records or [])
            if isinstance(row, dict)
            and row.get('stability_model_training_status') == 'accepted'
        ]
        candidate_x = []
        candidate_y = []
        for row in accepted:
            recipe = np.asarray(row.get('normalized_recipe'), dtype=float)
            if recipe.ndim != 1 or recipe.shape[0] != len(self.variable_reagents):
                raise AutoStabilityModelValidationError(
                    'Accepted stability training records require one '
                    'normalized value per variable reagent.'
                )
            target = _finite_float(row.get('model_target_log10_loss_rate'))
            if (
                    target is None or not np.all(np.isfinite(recipe))
                    or np.any(recipe < 0.0) or np.any(recipe > 1.0)):
                raise AutoStabilityModelValidationError(
                    'Accepted stability training records contain invalid '
                    'normalized recipe or log-loss-rate values.'
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
            'status': STABILITY_MODEL_STATUS_INSUFFICIENT,
            'fitted': False,
            'accepted_condition_count': len(accepted),
            'minimum_condition_count': STABILITY_MODEL_MINIMUM_OBSERVATIONS,
            'model_target_transform': STABILITY_MODEL_TARGET_TRANSFORM,
            'variable_reagents': list(self.variable_reagents),
            'error': None,
        }
        # A monitor-mode run only gains completed conditions. If a later raw
        # reload or provenance parse reports fewer accepted conditions than a
        # prior refresh, treat that as a derived-data failure rather than
        # silently discarding cumulative stability history.
        if candidate_X.shape[0] < self.X.shape[0]:
            message = (
                'Stability model refresh would regress cumulative accepted '
                'history from {} to {} condition(s).'.format(
                    self.X.shape[0], candidate_X.shape[0]
                )
            )
            self.status = STABILITY_MODEL_STATUS_FIT_FAILED
            self.last_error = message
            result.update({
                'status': STABILITY_MODEL_STATUS_FIT_FAILED,
                'error': message,
            })
            return result
        if len(accepted) < STABILITY_MODEL_MINIMUM_OBSERVATIONS:
            self.X = candidate_X.copy()
            self.Y_log10_loss_rate = candidate_Y.copy()
            self.gp_model = None
            self.status = STABILITY_MODEL_STATUS_INSUFFICIENT
            self.last_error = None
            return result

        kernel = GPy.kern.Matern32(
            input_dim=len(self.variable_reagents),
            variance=1.0,
            lengthscale=1.0,
            ARD=True,
        )
        try:
            fresh_model = GPy.models.GPRegression(
                candidate_X.copy(), candidate_Y.copy(), kernel,
                noise_var=1e-6
            )
        except Exception as exc:
            self.status = STABILITY_MODEL_STATUS_FIT_FAILED
            self.last_error = str(exc)
            result.update({
                'status': STABILITY_MODEL_STATUS_FIT_FAILED,
                'error': str(exc),
            })
            return result

        self.gp_model = fresh_model
        self.X = candidate_X.copy()
        self.Y_log10_loss_rate = candidate_Y.copy()
        self.status = STABILITY_MODEL_STATUS_FITTED
        self.last_error = None
        result.update({
            'status': STABILITY_MODEL_STATUS_FITTED,
            'fitted': True,
        })
        return result
