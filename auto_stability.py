'''Pure configuration and metric primitives for Auto optical-stability mode.

This module intentionally has no controller, robot, plate-reader, filesystem,
or Gaussian-process imports.  Stage 12A establishes the science-facing
configuration contract and deterministic calculations before later stages add
scan scheduling, data persistence, quality control, or recipe selection.

The primary per-well metric is the post-peak loss rate at one fixed reference
wavelength::

    (A_peak - A_tail_min) / (t_tail_min - t_peak)

``A_tail_min`` is the lowest valid absorbance strictly after the first maximum
within the observation window.  A trajectory without an observed post-peak
decrease is retained as an auditable incomplete observation; it is not
silently treated as a zero-loss stable sample.
'''

from __future__ import division

import math
import statistics


class AutoStabilityValidationError(ValueError):
    '''Raised when stability settings or trajectory data are ambiguous.'''


STABILITY_MODE_OFF = 'off'
STABILITY_MODE_MONITOR = 'monitor'
STABILITY_MODE_TARGET_THEN_STABILITY = 'target_then_stability'
STABILITY_MODE_STABILITY_ONLY = 'stability_only'

STABILITY_SCAN_SCHEDULE_EACH_COMPLETION = 'each_completion'
STABILITY_SCAN_SCHEDULE_CADENCED_ACTIVE_SET = 'cadenced_active_set'

STABILITY_MIXING_MODE_PLATE_SHAKE = 'plate_shake'

METRIC_STATUS_ELIGIBLE = 'eligible'
METRIC_STATUS_INSUFFICIENT_OBSERVATIONS = 'insufficient_observations'
METRIC_STATUS_LOW_SIGNAL = 'low_signal'
METRIC_STATUS_NO_POST_PEAK_DECLINE = 'no_post_peak_decline'


def _is_blank(value):
    '''Return whether a Header/trajectory value is absent without losing zero.'''
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ''


def _canonical_token(value):
    '''Normalize a Header choice without accepting arbitrary free text.'''
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')


def canonical_reagent_name(value):
    '''Match the repository's conventional underscore-separated reagent names.'''
    if _is_blank(value):
        return ''
    return '_'.join(str(value).strip().split())


def _parse_positive_finite(value, header_name):
    '''Parse one finite, strictly positive scientific setting.'''
    if _is_blank(value):
        raise AutoStabilityValidationError(
            'Header value {} is required when auto_stability_mode is '
            'monitor.'.format(header_name)
        )
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityValidationError(
            'Header value {} must be a finite positive number; received '
            '{!r}.'.format(header_name, value)
        )
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise AutoStabilityValidationError(
            'Header value {} must be a finite positive number; received '
            '{!r}.'.format(header_name, value)
        )
    return float(numeric_value)


def parse_auto_stability_header_settings(header_values):
    '''Return canonical Stage-12 stability settings from a Header mapping.

    Missing stability rows return the exact inert ``off`` configuration, so
    older workbooks retain their existing Auto behavior.  ``monitor`` is the
    only enabled Stage-12A mode.  It records an explicit future intent but
    starts no scans and changes no physical or optimizer behavior in this
    stage.  Active ranking modes are rejected clearly until their separate GP
    and selection stages exist.
    '''
    if not isinstance(header_values, dict):
        raise AutoStabilityValidationError(
            'Header settings must be supplied as a dictionary.'
        )

    raw_mode = _canonical_token(
        header_values.get('auto_stability_mode', STABILITY_MODE_OFF)
    )
    mode_aliases = {
        '': STABILITY_MODE_OFF,
        'off': STABILITY_MODE_OFF,
        'none': STABILITY_MODE_OFF,
        'disabled': STABILITY_MODE_OFF,
        'false': STABILITY_MODE_OFF,
        '0': STABILITY_MODE_OFF,
        'monitor': STABILITY_MODE_MONITOR,
        'monitoring': STABILITY_MODE_MONITOR,
        'observe': STABILITY_MODE_MONITOR,
        'target_then_stability': STABILITY_MODE_TARGET_THEN_STABILITY,
        'target_stability': STABILITY_MODE_TARGET_THEN_STABILITY,
        'stability_only': STABILITY_MODE_STABILITY_ONLY,
    }
    if raw_mode not in mode_aliases:
        raise AutoStabilityValidationError(
            'Header value auto_stability_mode must be off or monitor at this '
            'stage. Received: {!r}.'.format(raw_mode)
        )

    mode = mode_aliases[raw_mode]
    if mode in (
            STABILITY_MODE_TARGET_THEN_STABILITY,
            STABILITY_MODE_STABILITY_ONLY):
        raise AutoStabilityValidationError(
            'Header value auto_stability_mode={!r} is reserved for a later '
            'Stage 12 selection implementation. Use monitor or off for now.'
            .format(mode)
        )

    if mode == STABILITY_MODE_OFF:
        return {
            'auto_stability_mode': STABILITY_MODE_OFF,
            'auto_stability_trigger_reagent': None,
            'auto_stability_scan_schedule': None,
            'auto_stability_observation_window_s': None,
            'auto_stability_scan_interval_s': None,
            'auto_stability_min_peak_absorbance': None,
            'auto_stability_mixing_mode': None,
        }

    trigger_reagent = canonical_reagent_name(
        header_values.get('auto_stability_trigger_reagent')
    )
    if not trigger_reagent:
        raise AutoStabilityValidationError(
            'Header value auto_stability_trigger_reagent is required when '
            'auto_stability_mode is monitor.'
        )

    raw_schedule = _canonical_token(
        header_values.get(
            'auto_stability_scan_schedule',
            STABILITY_SCAN_SCHEDULE_CADENCED_ACTIVE_SET
        )
    )
    schedule_aliases = {
        'each_completion': STABILITY_SCAN_SCHEDULE_EACH_COMPLETION,
        'each_complete': STABILITY_SCAN_SCHEDULE_EACH_COMPLETION,
        'cadenced_active_set': STABILITY_SCAN_SCHEDULE_CADENCED_ACTIVE_SET,
        'cadenced': STABILITY_SCAN_SCHEDULE_CADENCED_ACTIVE_SET,
        'active_set': STABILITY_SCAN_SCHEDULE_CADENCED_ACTIVE_SET,
        'default': STABILITY_SCAN_SCHEDULE_CADENCED_ACTIVE_SET,
        '': STABILITY_SCAN_SCHEDULE_CADENCED_ACTIVE_SET,
    }
    if raw_schedule not in schedule_aliases:
        raise AutoStabilityValidationError(
            'Header value auto_stability_scan_schedule must be '
            'each_completion or cadenced_active_set. Received: {!r}.'
            .format(raw_schedule)
        )
    scan_schedule = schedule_aliases[raw_schedule]

    raw_mixing_mode = _canonical_token(
        header_values.get(
            'auto_stability_mixing_mode',
            STABILITY_MIXING_MODE_PLATE_SHAKE
        )
    )
    mixing_aliases = {
        '': STABILITY_MIXING_MODE_PLATE_SHAKE,
        'plate_shake': STABILITY_MIXING_MODE_PLATE_SHAKE,
        'shake': STABILITY_MIXING_MODE_PLATE_SHAKE,
        'plate': STABILITY_MIXING_MODE_PLATE_SHAKE,
    }
    if raw_mixing_mode not in mixing_aliases:
        raise AutoStabilityValidationError(
            'Header value auto_stability_mixing_mode currently supports '
            'only plate_shake. Pipette mixing and no-added-mixing remain '
            'separate future, hardware-validated implementations. Received: '
            '{!r}.'.format(raw_mixing_mode)
        )

    observation_window_s = _parse_positive_finite(
        header_values.get('auto_stability_observation_window_s'),
        'auto_stability_observation_window_s'
    )
    min_peak_absorbance = _parse_positive_finite(
        header_values.get('auto_stability_min_peak_absorbance'),
        'auto_stability_min_peak_absorbance'
    )

    if scan_schedule == STABILITY_SCAN_SCHEDULE_CADENCED_ACTIVE_SET:
        scan_interval_s = _parse_positive_finite(
            header_values.get('auto_stability_scan_interval_s'),
            'auto_stability_scan_interval_s'
        )
    else:
        raw_interval = header_values.get('auto_stability_scan_interval_s')
        scan_interval_s = (
            None if _is_blank(raw_interval)
            else _parse_positive_finite(
                raw_interval,
                'auto_stability_scan_interval_s'
            )
        )

    return {
        'auto_stability_mode': STABILITY_MODE_MONITOR,
        'auto_stability_trigger_reagent': trigger_reagent,
        'auto_stability_scan_schedule': scan_schedule,
        'auto_stability_observation_window_s': observation_window_s,
        'auto_stability_scan_interval_s': scan_interval_s,
        'auto_stability_min_peak_absorbance': min_peak_absorbance,
        'auto_stability_mixing_mode': mixing_aliases[raw_mixing_mode],
    }


def validate_stability_trigger_reagent(trigger_reagent, ordered_transfer_reagents):
    '''Require the configured trigger to be the final non-water transfer.

    The controller supplies transfer reagents in the exact workbook execution
    order.  Water top-off is excluded because it is an accounting transfer,
    not a chemistry-defining completion event.  The returned string is the
    original workbook spelling, so later controller code can match its live
    dataframes without a second normalization step.
    '''
    if not ordered_transfer_reagents:
        raise AutoStabilityValidationError(
            'Stability monitoring requires at least one non-water transfer '
            'in the input template.'
        )

    canonical_trigger = canonical_reagent_name(trigger_reagent).lower()
    canonical_transfers = [
        canonical_reagent_name(reagent).lower()
        for reagent in ordered_transfer_reagents
    ]
    if canonical_trigger not in canonical_transfers:
        raise AutoStabilityValidationError(
            'Header auto_stability_trigger_reagent {!r} is not a '
            'non-water transfer reagent in the input template. Available '
            'transfer reagents in execution order: {}.'.format(
                trigger_reagent,
                ', '.join(str(reagent) for reagent in ordered_transfer_reagents)
            )
        )

    final_trigger = canonical_transfers[-1]
    if canonical_trigger != final_trigger:
        raise AutoStabilityValidationError(
            'Header auto_stability_trigger_reagent {!r} must be the final '
            'non-water transfer in the input template so a completed well is '
            'never monitored before its reaction is complete. The final '
            'non-water transfer is {!r}.'.format(
                trigger_reagent,
                ordered_transfer_reagents[-1]
            )
        )

    return ordered_transfer_reagents[-1]


def select_reference_wavelength_nm(observations):
    '''Return the first valid post-trigger lambda-max as a fixed reference.'''
    for observation in observations:
        if not isinstance(observation, dict):
            raise AutoStabilityValidationError(
                'Each stability observation must be a dictionary.'
            )
        value = observation.get('lambda_max_nm')
        if _is_blank(value):
            continue
        try:
            wavelength_nm = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(wavelength_nm):
            return wavelength_nm
    return None


def _normalized_observations(observations):
    '''Validate ordered reference-wavelength observations without reordering them.'''
    normalized = []
    previous_timestamp_s = None
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise AutoStabilityValidationError(
                'Stability observation {} must be a dictionary.'.format(index)
            )
        try:
            timestamp_s = float(observation['timestamp_s'])
            absorbance = float(observation['reference_absorbance'])
        except (KeyError, TypeError, ValueError):
            raise AutoStabilityValidationError(
                'Stability observation {} requires finite timestamp_s and '
                'reference_absorbance values.'.format(index)
            )
        if not math.isfinite(timestamp_s) or not math.isfinite(absorbance):
            raise AutoStabilityValidationError(
                'Stability observation {} requires finite timestamp_s and '
                'reference_absorbance values.'.format(index)
            )
        if (
                previous_timestamp_s is not None
                and timestamp_s <= previous_timestamp_s):
            raise AutoStabilityValidationError(
                'Stability observation timestamps must be strictly increasing; '
                'observation {} has {} after {}.'.format(
                    index, timestamp_s, previous_timestamp_s
                )
            )
        lambda_max_nm = observation.get('lambda_max_nm')
        if not _is_blank(lambda_max_nm):
            try:
                lambda_max_nm = float(lambda_max_nm)
            except (TypeError, ValueError):
                lambda_max_nm = None
            if lambda_max_nm is not None and not math.isfinite(lambda_max_nm):
                lambda_max_nm = None
        else:
            lambda_max_nm = None
        normalized.append({
            'timestamp_s': timestamp_s,
            'reference_absorbance': absorbance,
            'lambda_max_nm': lambda_max_nm,
        })
        previous_timestamp_s = timestamp_s
    return normalized


def compute_stability_metrics(observations, min_peak_absorbance,
                              trigger_timestamp_s=None,
                              observation_window_s=None):
    '''Compute auditable post-peak stability metrics for one physical well.

    ``observations`` must already represent one fixed per-well reference
    wavelength.  When an observation window is supplied, only timestamps from
    the actual trigger through ``trigger + window`` participate.  The optional
    trigger timestamp defaults to the first supplied observation solely for
    unit-level use; Stage 12B must supply the durable trigger time from its
    manifest. This pure function deliberately does not infer absorbance from a
    spectrum.
    '''
    minimum_peak = _parse_positive_finite(
        min_peak_absorbance,
        'auto_stability_min_peak_absorbance'
    )
    all_trajectory = _normalized_observations(observations)
    if trigger_timestamp_s is None:
        trigger_time_s = (
            None if not all_trajectory
            else all_trajectory[0]['timestamp_s']
        )
    else:
        try:
            trigger_time_s = float(trigger_timestamp_s)
        except (TypeError, ValueError):
            raise AutoStabilityValidationError(
                'trigger_timestamp_s must be a finite timestamp.'
            )
        if not math.isfinite(trigger_time_s):
            raise AutoStabilityValidationError(
                'trigger_timestamp_s must be a finite timestamp.'
            )

    if observation_window_s is None:
        window_s = None
    else:
        window_s = _parse_positive_finite(
            observation_window_s,
            'auto_stability_observation_window_s'
        )

    trajectory = [
        observation for observation in all_trajectory
        if (
            trigger_time_s is None
            or (
                observation['timestamp_s'] >= trigger_time_s
                and (
                    window_s is None
                    or observation['timestamp_s'] <= trigger_time_s + window_s
                )
            )
        )
    ]
    reference_wavelength_nm = select_reference_wavelength_nm(trajectory)
    result = {
        'status': METRIC_STATUS_INSUFFICIENT_OBSERVATIONS,
        'observation_count': len(all_trajectory),
        'observation_count_within_window': len(trajectory),
        'trigger_timestamp_s': trigger_time_s,
        'observation_window_s': window_s,
        'reference_wavelength_nm': reference_wavelength_nm,
        'peak_index': None,
        'peak_time_s': None,
        'peak_absorbance': None,
        'tail_min_index': None,
        'tail_min_time_s': None,
        'tail_min_absorbance': None,
        'absorbance_loss': None,
        'loss_rate_absorbance_per_s': None,
        'post_peak_decline_detected': False,
        'tail_lambda_max_nm': None,
        'lambda_drift_nm': None,
    }
    if len(trajectory) < 2:
        return result

    peak_index = max(
        range(len(trajectory)),
        key=lambda index: trajectory[index]['reference_absorbance']
    )
    peak = trajectory[peak_index]
    result.update({
        'peak_index': peak_index,
        'peak_time_s': peak['timestamp_s'],
        'peak_absorbance': peak['reference_absorbance'],
    })
    if peak['reference_absorbance'] < minimum_peak:
        result['status'] = METRIC_STATUS_LOW_SIGNAL
        return result

    tail = trajectory[peak_index + 1:]
    if not tail:
        return result

    tail_offset, tail_min = min(
        enumerate(tail),
        key=lambda item: item[1]['reference_absorbance']
    )
    tail_min_index = peak_index + 1 + tail_offset
    result.update({
        'tail_min_index': tail_min_index,
        'tail_min_time_s': tail_min['timestamp_s'],
        'tail_min_absorbance': tail_min['reference_absorbance'],
        'tail_lambda_max_nm': tail_min['lambda_max_nm'],
    })
    if reference_wavelength_nm is not None and tail_min['lambda_max_nm'] is not None:
        result['lambda_drift_nm'] = (
            tail_min['lambda_max_nm'] - reference_wavelength_nm
        )

    absorbance_loss = peak['reference_absorbance'] - tail_min['reference_absorbance']
    result['absorbance_loss'] = absorbance_loss
    if absorbance_loss <= 0.0:
        result['status'] = METRIC_STATUS_NO_POST_PEAK_DECLINE
        return result

    elapsed_s = tail_min['timestamp_s'] - peak['timestamp_s']
    if elapsed_s <= 0.0:
        raise AutoStabilityValidationError(
            'Post-peak stability timestamps must span a positive interval.'
        )
    result.update({
        'status': METRIC_STATUS_ELIGIBLE,
        'loss_rate_absorbance_per_s': absorbance_loss / elapsed_s,
        'post_peak_decline_detected': True,
    })
    return result


def aggregate_condition_stability_metrics(well_metrics):
    '''Aggregate only eligible physical-well loss rates without doing QC.

    Stage 12D will add explicit condition-level stability QC.  This helper
    preserves the total/eligible counts and sample SD needed for that later
    audit while avoiding any premature accept/reject decision.
    '''
    metric_rows = list(well_metrics or [])
    eligible_rates = []
    for metric in metric_rows:
        if not isinstance(metric, dict):
            raise AutoStabilityValidationError(
                'Each condition stability metric must be a dictionary.'
            )
        if metric.get('status') != METRIC_STATUS_ELIGIBLE:
            continue
        rate = metric.get('loss_rate_absorbance_per_s')
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            raise AutoStabilityValidationError(
                'An eligible stability metric requires a finite loss rate.'
            )
        if not math.isfinite(rate) or rate < 0.0:
            raise AutoStabilityValidationError(
                'An eligible stability metric requires a finite nonnegative '
                'loss rate.'
            )
        eligible_rates.append(rate)

    result = {
        'total_well_count': len(metric_rows),
        'eligible_well_count': len(eligible_rates),
        'condition_loss_rate_mean_absorbance_per_s': None,
        'condition_loss_rate_sample_sd_absorbance_per_s': None,
    }
    if not eligible_rates:
        return result

    result['condition_loss_rate_mean_absorbance_per_s'] = statistics.mean(
        eligible_rates
    )
    if len(eligible_rates) >= 2:
        result['condition_loss_rate_sample_sd_absorbance_per_s'] = (
            statistics.stdev(eligible_rates)
        )
    return result
