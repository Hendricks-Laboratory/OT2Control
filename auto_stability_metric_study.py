'''Pure analysis comparators for the Stage-14G stability-metric study.

This module is deliberately disconnected from Auto execution.  It does not
read files, issue robot or reader commands, parse workbook settings, write an
artifact, fit a model, or select a recipe.  The established production metric
continues to be calculated only by :func:`auto_stability.compute_stability_metrics`.

Stage 14G uses these comparators to test whether asynchronous scan density can
change a stability ranking.  They are decision-study tools, not alternative
production metrics.  In particular, the fixed-horizon comparator refuses to
interpolate a spectrum that was never acquired.
'''

from __future__ import division

import math
import statistics

from auto_stability import (
    METRIC_STATUS_ELIGIBLE,
    METRIC_STATUS_INSUFFICIENT_OBSERVATIONS,
    METRIC_STATUS_LOW_SIGNAL,
    METRIC_STATUS_NO_POST_PEAK_DECLINE,
    compute_stability_metrics,
    normalize_stability_observations,
)


STUDY_STATUS_ASSESSMENT_WINDOW_UNOBSERVED = 'assessment_window_unobserved'
STUDY_STATUS_INSUFFICIENT_POST_PEAK_OBSERVATIONS = (
    'insufficient_post_peak_observations'
)
STUDY_STATUS_NONPOSITIVE_ENDPOINT_ABSORBANCE = (
    'nonpositive_endpoint_absorbance'
)


class AutoStabilityMetricStudyError(ValueError):
    '''Raised when an analysis-only comparator input is ambiguous.'''


def _positive_finite(value, name):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityMetricStudyError(
            '{} must be a finite positive number.'.format(name)
        )
    if not math.isfinite(numeric_value) or numeric_value <= 0.0:
        raise AutoStabilityMetricStudyError(
            '{} must be a finite positive number.'.format(name)
        )
    return numeric_value


def _nonnegative_finite(value, name):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityMetricStudyError(
            '{} must be a finite nonnegative number.'.format(name)
        )
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise AutoStabilityMetricStudyError(
            '{} must be a finite nonnegative number.'.format(name)
        )
    return numeric_value


def _finite(value, name):
    '''Return a finite signed value while preserving timestamp compatibility.'''
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityMetricStudyError(
            '{} must be a finite number.'.format(name)
        )
    if not math.isfinite(numeric_value):
        raise AutoStabilityMetricStudyError(
            '{} must be a finite number.'.format(name)
        )
    return numeric_value


def _study_trajectory(observations, min_peak_absorbance,
                      trigger_timestamp_s=None, observation_window_s=None):
    '''Normalize a trajectory using the production metric's timing semantics.'''
    minimum_peak = _positive_finite(
        min_peak_absorbance, 'min_peak_absorbance'
    )
    all_observations = normalize_stability_observations(observations)
    if trigger_timestamp_s is None:
        trigger_timestamp_s = (
            None if not all_observations
            else all_observations[0]['timestamp_s']
        )
    else:
        trigger_timestamp_s = _finite(
            trigger_timestamp_s, 'trigger_timestamp_s'
        )
    if observation_window_s is None:
        observation_window_s = None
    else:
        observation_window_s = _positive_finite(
            observation_window_s, 'observation_window_s'
        )
    in_window = [
        observation for observation in all_observations
        if (
            trigger_timestamp_s is None
            or (
                observation['timestamp_s'] >= trigger_timestamp_s
                and (
                    observation_window_s is None
                    or observation['timestamp_s'] <= (
                        trigger_timestamp_s + observation_window_s
                    )
                )
            )
        )
    ]
    result = {
        'observation_count': len(all_observations),
        'observation_count_within_window': len(in_window),
        'trigger_timestamp_s': trigger_timestamp_s,
        'observation_window_s': observation_window_s,
        'peak_index': None,
        'peak_time_s': None,
        'peak_absorbance': None,
        'trajectory': in_window,
        'status': METRIC_STATUS_INSUFFICIENT_OBSERVATIONS,
    }
    if len(in_window) < 2:
        return result
    peak_index = max(
        range(len(in_window)),
        key=lambda index: in_window[index]['reference_absorbance']
    )
    peak = in_window[peak_index]
    result.update({
        'peak_index': peak_index,
        'peak_time_s': peak['timestamp_s'],
        'peak_absorbance': peak['reference_absorbance'],
    })
    if peak['reference_absorbance'] < minimum_peak:
        result['status'] = METRIC_STATUS_LOW_SIGNAL
        return result
    if peak_index == len(in_window) - 1:
        return result
    result['status'] = 'post_peak_observations_available'
    return result


def _comparator_result(metric_name, study_trajectory):
    '''Create a common, JSON-safe analysis-only comparator record.'''
    return {
        'metric_name': metric_name,
        'status': study_trajectory['status'],
        'observation_count': study_trajectory['observation_count'],
        'observation_count_within_window': (
            study_trajectory['observation_count_within_window']
        ),
        'trigger_timestamp_s': study_trajectory['trigger_timestamp_s'],
        'observation_window_s': study_trajectory['observation_window_s'],
        'peak_index': study_trajectory['peak_index'],
        'peak_time_s': study_trajectory['peak_time_s'],
        'peak_absorbance': study_trajectory['peak_absorbance'],
        'loss_rate_absorbance_per_s': None,
        'absorbance_loss': None,
    }


def compute_fixed_horizon_observed_loss_rate(
        observations, min_peak_absorbance, post_peak_horizon_s,
        maximum_endpoint_lateness_s, trigger_timestamp_s=None,
        observation_window_s=None):
    '''Measure one observed, non-interpolated post-peak endpoint loss rate.

    The endpoint is the first eligible scan in the fixed interval
    ``[peak + horizon, peak + horizon + maximum_endpoint_lateness]``.  A
    missing endpoint remains explicitly unavailable, preventing a well with a
    convenient late scan from silently receiving a fabricated measurement.
    '''
    horizon_s = _positive_finite(
        post_peak_horizon_s, 'post_peak_horizon_s'
    )
    maximum_lateness_s = _nonnegative_finite(
        maximum_endpoint_lateness_s, 'maximum_endpoint_lateness_s'
    )
    prepared = _study_trajectory(
        observations, min_peak_absorbance, trigger_timestamp_s,
        observation_window_s
    )
    result = _comparator_result('fixed_horizon_observed_loss_rate', prepared)
    result.update({
        'post_peak_horizon_s': horizon_s,
        'maximum_endpoint_lateness_s': maximum_lateness_s,
        'assessment_target_time_s': None,
        'endpoint_time_s': None,
        'endpoint_absorbance': None,
        'endpoint_index': None,
        'endpoint_selection': (
            'first_observed_in_predeclared_post_peak_assessment_interval'
        ),
    })
    if prepared['status'] != 'post_peak_observations_available':
        return result

    target_time_s = prepared['peak_time_s'] + horizon_s
    latest_time_s = target_time_s + maximum_lateness_s
    endpoint_index = None
    endpoint = None
    for index, observation in enumerate(prepared['trajectory']):
        if (
                observation['timestamp_s'] >= target_time_s
                and observation['timestamp_s'] <= latest_time_s):
            endpoint_index = index
            endpoint = observation
            break
    result['assessment_target_time_s'] = target_time_s
    if endpoint is None:
        result['status'] = STUDY_STATUS_ASSESSMENT_WINDOW_UNOBSERVED
        return result

    absorbance_loss = prepared['peak_absorbance'] - endpoint[
        'reference_absorbance'
    ]
    result.update({
        'endpoint_index': endpoint_index,
        'endpoint_time_s': endpoint['timestamp_s'],
        'endpoint_absorbance': endpoint['reference_absorbance'],
        'absorbance_loss': absorbance_loss,
    })
    if absorbance_loss <= 0.0:
        result['status'] = METRIC_STATUS_NO_POST_PEAK_DECLINE
        return result
    elapsed_s = endpoint['timestamp_s'] - prepared['peak_time_s']
    if elapsed_s <= 0.0:
        raise AutoStabilityMetricStudyError(
            'Fixed-horizon endpoint must follow the identified peak.'
        )
    result.update({
        'status': METRIC_STATUS_ELIGIBLE,
        'loss_rate_absorbance_per_s': absorbance_loss / elapsed_s,
    })
    return result


def compute_robust_post_peak_slope(
        observations, min_peak_absorbance, minimum_observation_count=3,
        trigger_timestamp_s=None, observation_window_s=None):
    '''Estimate decline from the median of post-peak pairwise slopes.

    The peak plus at least two later observations are required.  The returned
    loss rate is the nonnegative inverse of the median signed slope; a flat or
    rising trajectory remains ``no_post_peak_decline`` rather than becoming a
    favorable zero-loss value.
    '''
    if (isinstance(minimum_observation_count, bool)
            or not isinstance(minimum_observation_count, int)
            or minimum_observation_count < 3):
        raise AutoStabilityMetricStudyError(
            'minimum_observation_count must be an integer of at least 3.'
        )
    prepared = _study_trajectory(
        observations, min_peak_absorbance, trigger_timestamp_s,
        observation_window_s
    )
    result = _comparator_result('robust_post_peak_slope', prepared)
    result.update({
        'minimum_observation_count': minimum_observation_count,
        'post_peak_observation_count': 0,
        'pairwise_slope_count': 0,
        'median_signed_slope_absorbance_per_s': None,
    })
    if prepared['status'] != 'post_peak_observations_available':
        return result

    post_peak = prepared['trajectory'][prepared['peak_index']:]
    result['post_peak_observation_count'] = len(post_peak)
    if len(post_peak) < minimum_observation_count:
        result['status'] = STUDY_STATUS_INSUFFICIENT_POST_PEAK_OBSERVATIONS
        return result

    slopes = []
    for left_index, left in enumerate(post_peak[:-1]):
        for right in post_peak[left_index + 1:]:
            elapsed_s = right['timestamp_s'] - left['timestamp_s']
            if elapsed_s <= 0.0:
                raise AutoStabilityMetricStudyError(
                    'Post-peak observation times must be distinct and ordered.'
                )
            slopes.append(
                (right['reference_absorbance'] - left['reference_absorbance'])
                / elapsed_s
            )
    median_signed_slope = statistics.median(slopes)
    result.update({
        'pairwise_slope_count': len(slopes),
        'median_signed_slope_absorbance_per_s': median_signed_slope,
    })
    if median_signed_slope >= 0.0:
        result['status'] = METRIC_STATUS_NO_POST_PEAK_DECLINE
        return result
    result.update({
        'status': METRIC_STATUS_ELIGIBLE,
        'loss_rate_absorbance_per_s': -median_signed_slope,
    })
    return result


def compute_fixed_horizon_normalized_metrics(
        observations, min_peak_absorbance, post_peak_horizon_s,
        endpoint_tolerance_s, trigger_timestamp_s=None,
        observation_window_s=None):
    '''Return co-timed raw, normalized, and log-loss study metrics.

    This is deliberately a study-only comparator.  It selects the observed
    post-peak point nearest the requested horizon inside a symmetric, declared
    tolerance.  It never constructs an interpolated value.  The three returned
    records therefore use the exact same peak, endpoint, and elapsed time.
    A missing, nonpositive, or nondeclining endpoint remains an explicit
    noneligible status rather than becoming an artificial favorable response.
    '''
    horizon_s = _positive_finite(
        post_peak_horizon_s, 'post_peak_horizon_s'
    )
    tolerance_s = _nonnegative_finite(
        endpoint_tolerance_s, 'endpoint_tolerance_s'
    )
    prepared = _study_trajectory(
        observations, min_peak_absorbance, trigger_timestamp_s,
        observation_window_s
    )

    common = {
        'observation_count': prepared['observation_count'],
        'observation_count_within_window': (
            prepared['observation_count_within_window']
        ),
        'trigger_timestamp_s': prepared['trigger_timestamp_s'],
        'observation_window_s': prepared['observation_window_s'],
        'peak_index': prepared['peak_index'],
        'peak_time_s': prepared['peak_time_s'],
        'peak_absorbance': prepared['peak_absorbance'],
        'post_peak_horizon_s': horizon_s,
        'endpoint_tolerance_s': tolerance_s,
        'assessment_target_time_s': None,
        'endpoint_index': None,
        'endpoint_time_s': None,
        'endpoint_absorbance': None,
        'endpoint_timing_offset_s': None,
        'endpoint_selection': (
            'nearest_observed_in_predeclared_symmetric_assessment_interval'
        ),
    }
    names = (
        'fixed_horizon_raw_loss_rate',
        'normalized_retention',
        'log_loss_rate',
    )
    results = {}
    for name in names:
        result = dict(common)
        result.update({
            'metric_name': name,
            'status': prepared['status'],
            'absorbance_loss': None,
            'loss_rate_absorbance_per_s': None,
            'retention_fraction': None,
            'fractional_loss': None,
            'log_loss_rate_per_s': None,
        })
        results[name] = result
    if prepared['status'] != 'post_peak_observations_available':
        return results

    target_time_s = prepared['peak_time_s'] + horizon_s
    endpoint_candidates = []
    for index, observation in enumerate(prepared['trajectory']):
        if index <= prepared['peak_index']:
            continue
        offset_s = observation['timestamp_s'] - target_time_s
        if abs(offset_s) <= tolerance_s:
            endpoint_candidates.append((abs(offset_s), index, observation))
    for result in results.values():
        result['assessment_target_time_s'] = target_time_s
    if not endpoint_candidates:
        for result in results.values():
            result['status'] = STUDY_STATUS_ASSESSMENT_WINDOW_UNOBSERVED
        return results

    _, endpoint_index, endpoint = min(endpoint_candidates)
    elapsed_s = endpoint['timestamp_s'] - prepared['peak_time_s']
    if elapsed_s <= 0.0:
        raise AutoStabilityMetricStudyError(
            'Fixed-horizon endpoint must follow the identified peak.'
        )
    endpoint_absorbance = endpoint['reference_absorbance']
    absorbance_loss = prepared['peak_absorbance'] - endpoint_absorbance
    endpoint_fields = {
        'endpoint_index': endpoint_index,
        'endpoint_time_s': endpoint['timestamp_s'],
        'endpoint_absorbance': endpoint_absorbance,
        'endpoint_timing_offset_s': endpoint['timestamp_s'] - target_time_s,
        'absorbance_loss': absorbance_loss,
    }
    for result in results.values():
        result.update(endpoint_fields)
    if endpoint_absorbance <= 0.0:
        for result in results.values():
            result['status'] = STUDY_STATUS_NONPOSITIVE_ENDPOINT_ABSORBANCE
        return results

    retention_fraction = endpoint_absorbance / prepared['peak_absorbance']
    fractional_loss = 1.0 - retention_fraction
    for result in results.values():
        result.update({
            'retention_fraction': retention_fraction,
            'fractional_loss': fractional_loss,
        })
    if absorbance_loss <= 0.0:
        for result in results.values():
            result['status'] = METRIC_STATUS_NO_POST_PEAK_DECLINE
        return results

    results['fixed_horizon_raw_loss_rate'].update({
        'status': METRIC_STATUS_ELIGIBLE,
        'loss_rate_absorbance_per_s': absorbance_loss / elapsed_s,
    })
    results['normalized_retention'].update({
        'status': METRIC_STATUS_ELIGIBLE,
    })
    results['log_loss_rate'].update({
        'status': METRIC_STATUS_ELIGIBLE,
        'log_loss_rate_per_s': -math.log(retention_fraction) / elapsed_s,
    })
    return results


def compare_stage14g2_metrics(
        observations, min_peak_absorbance, post_peak_horizon_s,
        endpoint_tolerance_s, trigger_timestamp_s=None,
        observation_window_s=None):
    '''Compare the five Stage 14G2 study metrics without choosing a winner.'''
    legacy = compute_stability_metrics(
        observations,
        min_peak_absorbance,
        trigger_timestamp_s=trigger_timestamp_s,
        observation_window_s=observation_window_s,
    )
    legacy = dict(legacy)
    legacy['metric_name'] = 'legacy_tail_minimum_v1'
    results = {'legacy_tail_minimum_v1': legacy}
    results.update(compute_fixed_horizon_normalized_metrics(
        observations,
        min_peak_absorbance,
        post_peak_horizon_s,
        endpoint_tolerance_s,
        trigger_timestamp_s=trigger_timestamp_s,
        observation_window_s=observation_window_s,
    ))
    results['robust_post_peak_slope'] = compute_robust_post_peak_slope(
        observations,
        min_peak_absorbance,
        trigger_timestamp_s=trigger_timestamp_s,
        observation_window_s=observation_window_s,
    )
    return results


def compare_stability_metric_estimators(
        observations, min_peak_absorbance, post_peak_horizon_s,
        maximum_endpoint_lateness_s, trigger_timestamp_s=None,
        observation_window_s=None):
    '''Return production and study-only metric results without selecting one.'''
    return {
        'production_tail_minimum': compute_stability_metrics(
            observations,
            min_peak_absorbance,
            trigger_timestamp_s=trigger_timestamp_s,
            observation_window_s=observation_window_s,
        ),
        'fixed_horizon_observed': compute_fixed_horizon_observed_loss_rate(
            observations,
            min_peak_absorbance,
            post_peak_horizon_s,
            maximum_endpoint_lateness_s,
            trigger_timestamp_s=trigger_timestamp_s,
            observation_window_s=observation_window_s,
        ),
        'robust_post_peak_slope': compute_robust_post_peak_slope(
            observations,
            min_peak_absorbance,
            trigger_timestamp_s=trigger_timestamp_s,
            observation_window_s=observation_window_s,
        ),
    }


def simulate_linear_decay_observations(
        observation_times_s, peak_absorbance, true_loss_rate_absorbance_per_s,
        additive_noise_by_index=None):
    '''Build a deterministic linear-decay trajectory for study-only tests.

    It has no random-number source.  Any supplied noise sequence is explicit
    and index-aligned, preserving exact reproducibility for ranking studies.
    '''
    peak_absorbance = _positive_finite(peak_absorbance, 'peak_absorbance')
    true_loss_rate = _nonnegative_finite(
        true_loss_rate_absorbance_per_s,
        'true_loss_rate_absorbance_per_s'
    )
    times = list(observation_times_s or [])
    if not times:
        raise AutoStabilityMetricStudyError(
            'observation_times_s must contain at least one timestamp.'
        )
    if additive_noise_by_index is None:
        noise = [0.0] * len(times)
    else:
        noise = list(additive_noise_by_index)
        if len(noise) != len(times):
            raise AutoStabilityMetricStudyError(
                'additive_noise_by_index must match observation_times_s.'
            )
    normalized_times = []
    previous_time_s = None
    for index, timestamp_s in enumerate(times):
        timestamp_s = _nonnegative_finite(
            timestamp_s, 'observation_times_s[{}]'.format(index)
        )
        if previous_time_s is not None and timestamp_s <= previous_time_s:
            raise AutoStabilityMetricStudyError(
                'observation_times_s must be strictly increasing.'
            )
        normalized_times.append(timestamp_s)
        previous_time_s = timestamp_s
    normalized_noise = [
        _nonnegative_or_signed_finite(
            value, 'additive_noise_by_index[{}]'.format(index)
        )
        for index, value in enumerate(noise)
    ]
    initial_time_s = normalized_times[0]
    return [
        {
            'timestamp_s': timestamp_s,
            'reference_absorbance': (
                peak_absorbance
                - true_loss_rate * (timestamp_s - initial_time_s)
                + normalized_noise[index]
            ),
        }
        for index, timestamp_s in enumerate(normalized_times)
    ]


def _nonnegative_or_signed_finite(value, name):
    '''Compatibility alias retained to make simulation call sites explicit.'''
    return _finite(value, name)
