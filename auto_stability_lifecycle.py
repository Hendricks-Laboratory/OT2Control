'''Pure Stage-13A lifecycle rules for optical-stability monitoring.

The current Stage-12C controller still operates one fixed observation window.
This module intentionally does not change that scheduler, reader, robot, GP,
or workbook behavior.  It defines and tests the scientific state contract that
later stages will use when targeted per-well mixing and cross-batch active-set
scanning are available.

In particular, a well becomes *decision ready* only after its fixed decision
horizon.  Its condition-level decision metric can then be frozen for GP use,
while the physical well may remain active for extended monitoring.  Replicate
wells remain independent trajectories, but condition readiness is true only
when every unique member is individually decision ready; this prevents an
extra replicate from becoming an extra GP recipe point.
'''

from __future__ import division

import math

from auto_stability import (
    METRIC_STATUS_LOW_SIGNAL,
    compute_stability_metrics,
    normalize_stability_observations,
)


WELL_MONITORING_STATE_PENDING_TRIGGER = 'pending_trigger'
WELL_MONITORING_STATE_ACTIVE = 'active'
WELL_MONITORING_STATE_DECISION_READY = 'decision_ready'
WELL_MONITORING_STATE_PLATEAU_CONFIRMED = 'plateau_confirmed'
WELL_MONITORING_STATE_MAX_WINDOW_EXPIRED = 'max_observation_window_expired'
WELL_MONITORING_STATE_TERMINAL_QC_EXCLUDED = 'terminal_qc_excluded'

RETIREMENT_REASON_PLATEAU_CONFIRMED = 'plateau_confirmed'
RETIREMENT_REASON_MAX_WINDOW_EXPIRED = 'maximum_observation_window_expired'
RETIREMENT_REASON_TERMINAL_LOW_SIGNAL = 'terminal_low_signal'


class AutoStabilityLifecycleError(ValueError):
    '''Raised when a lifecycle policy or state request is scientifically invalid.'''


def _positive_finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityLifecycleError(
            '{} must be a finite positive number.'.format(name)
        )
    if not math.isfinite(number) or number <= 0.0:
        raise AutoStabilityLifecycleError(
            '{} must be a finite positive number.'.format(name)
        )
    return number


def _nonnegative_finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityLifecycleError(
            '{} must be a finite nonnegative number.'.format(name)
        )
    if not math.isfinite(number) or number < 0.0:
        raise AutoStabilityLifecycleError(
            '{} must be a finite nonnegative number.'.format(name)
        )
    return number


def _positive_integer(value, name):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise AutoStabilityLifecycleError(
            '{} must be a positive integer.'.format(name)
        )
    if number <= 0 or float(value) != float(number):
        raise AutoStabilityLifecycleError(
            '{} must be a positive integer.'.format(name)
        )
    return number


def _strict_boolean(value, name):
    '''Accept only a real boolean for a safety-relevant policy switch.'''
    if isinstance(value, bool):
        return value
    raise AutoStabilityLifecycleError(
        '{} must be a boolean.'.format(name)
    )


def _validated_policy(policy):
    '''Return one fully validated policy from a public lifecycle dictionary.'''
    if not isinstance(policy, dict):
        raise AutoStabilityLifecycleError('policy must be a dictionary.')
    required_keys = (
        'decision_horizon_s',
        'max_observation_window_s',
        'adaptive_retirement_enabled',
        'plateau_min_tail_observation_count',
        'plateau_consecutive_interval_count',
        'plateau_max_absorbance_slope_per_s',
    )
    missing = [key for key in required_keys if key not in policy]
    if missing:
        raise AutoStabilityLifecycleError(
            'policy is missing required key(s): {}.'.format(', '.join(missing))
        )
    return build_stability_monitoring_policy(
        decision_horizon_s=policy['decision_horizon_s'],
        max_observation_window_s=policy['max_observation_window_s'],
        adaptive_retirement_enabled=policy['adaptive_retirement_enabled'],
        plateau_min_tail_observation_count=(
            policy['plateau_min_tail_observation_count']
        ),
        plateau_consecutive_interval_count=(
            policy['plateau_consecutive_interval_count']
        ),
        plateau_max_absorbance_slope_per_s=(
            policy['plateau_max_absorbance_slope_per_s']
        ),
    )


def build_stability_monitoring_policy(
        decision_horizon_s,
        max_observation_window_s,
        adaptive_retirement_enabled,
        plateau_min_tail_observation_count,
        plateau_consecutive_interval_count,
        plateau_max_absorbance_slope_per_s):
    '''Build a validated, scheduler-independent Stage-13 monitoring policy.

    ``decision_horizon_s`` defines the fixed comparable evidence window used
    for a future condition-level GP metric.  ``max_observation_window_s`` is
    the longest permitted monitoring interval.  A well cannot retire for a
    plateau before the decision horizon, even if its raw trace looks flat,
    because that would make decision evidence depend on unequal durations.

    The plateau threshold and observation-count requirements are deliberately
    explicit: no unvalidated universal noise threshold is silently assumed.
    Adaptive retirement is represented here for pure testing only. The
    existing controller does not invoke it until a later scheduler stage.
    '''
    decision_horizon_s = _positive_finite(
        decision_horizon_s, 'decision_horizon_s'
    )
    max_observation_window_s = _positive_finite(
        max_observation_window_s, 'max_observation_window_s'
    )
    if max_observation_window_s < decision_horizon_s:
        raise AutoStabilityLifecycleError(
            'max_observation_window_s must be greater than or equal to '
            'decision_horizon_s.'
        )
    plateau_min_tail_observation_count = _positive_integer(
        plateau_min_tail_observation_count,
        'plateau_min_tail_observation_count'
    )
    plateau_consecutive_interval_count = _positive_integer(
        plateau_consecutive_interval_count,
        'plateau_consecutive_interval_count'
    )
    if plateau_min_tail_observation_count < (
            plateau_consecutive_interval_count + 1):
        raise AutoStabilityLifecycleError(
            'plateau_min_tail_observation_count must provide at least one '
            'more observation than plateau_consecutive_interval_count.'
        )
    return {
        'decision_horizon_s': decision_horizon_s,
        'max_observation_window_s': max_observation_window_s,
        'adaptive_retirement_enabled': _strict_boolean(
            adaptive_retirement_enabled, 'adaptive_retirement_enabled'
        ),
        'plateau_min_tail_observation_count': (
            plateau_min_tail_observation_count
        ),
        'plateau_consecutive_interval_count': (
            plateau_consecutive_interval_count
        ),
        'plateau_max_absorbance_slope_per_s': _nonnegative_finite(
            plateau_max_absorbance_slope_per_s,
            'plateau_max_absorbance_slope_per_s'
        ),
    }


def evaluate_stability_plateau(
        observations,
        trigger_timestamp_s,
        minimum_peak_absorbance,
        policy):
    '''Evaluate a strictly post-peak, sustained low-slope plateau criterion.

    The function never treats an absent decline as a plateau.  It requires a
    valid peak above the signal threshold, a later lower absorbance, enough
    post-peak observations, and a configured number of consecutive adjacent
    tail intervals whose absolute slopes are below the configured limit.
    The caller separately enforces the decision horizon before retirement.
    '''
    policy = _validated_policy(policy)
    normalized = normalize_stability_observations(observations)
    metric = compute_stability_metrics(
        normalized,
        minimum_peak_absorbance,
        trigger_timestamp_s=trigger_timestamp_s,
        observation_window_s=policy['max_observation_window_s']
    )
    result = {
        'plateau_confirmed': False,
        'reason': 'insufficient_post_peak_evidence',
        'metric_status': metric['status'],
        'post_peak_observation_count': 0,
        'recent_interval_slopes_absorbance_per_s': [],
    }
    if metric['status'] == METRIC_STATUS_LOW_SIGNAL:
        result['reason'] = RETIREMENT_REASON_TERMINAL_LOW_SIGNAL
        return result
    peak_index = metric.get('peak_index')
    if peak_index is None:
        return result
    trajectory = [
        item for item in normalized
        if (
            item['timestamp_s'] >= metric['trigger_timestamp_s']
            and item['timestamp_s'] <= (
                metric['trigger_timestamp_s'] +
                policy['max_observation_window_s']
            )
        )
    ]
    tail = trajectory[peak_index + 1:]
    result['post_peak_observation_count'] = len(tail)
    if not metric['post_peak_decline_detected']:
        result['reason'] = 'no_post_peak_decline'
        return result
    if len(tail) < policy['plateau_min_tail_observation_count']:
        result['reason'] = 'insufficient_post_peak_observations'
        return result
    consecutive_count = policy['plateau_consecutive_interval_count']
    recent = tail[-(consecutive_count + 1):]
    slopes = []
    for earlier, later in zip(recent, recent[1:]):
        elapsed_s = later['timestamp_s'] - earlier['timestamp_s']
        if elapsed_s <= 0.0:
            raise AutoStabilityLifecycleError(
                'Validated stability observations must have increasing times.'
            )
        slopes.append(
            (later['reference_absorbance'] - earlier['reference_absorbance']) /
            elapsed_s
        )
    result['recent_interval_slopes_absorbance_per_s'] = slopes
    threshold = policy['plateau_max_absorbance_slope_per_s']
    if any(abs(slope) > threshold for slope in slopes):
        result['reason'] = 'recent_slope_above_plateau_threshold'
        return result
    result.update({
        'plateau_confirmed': True,
        'reason': RETIREMENT_REASON_PLATEAU_CONFIRMED,
    })
    return result


def evaluate_well_monitoring_lifecycle(
        trigger_timestamp_s,
        now_timestamp_s,
        observations,
        minimum_peak_absorbance,
        policy,
        trigger_completed=True):
    '''Return one well's non-mutating lifecycle and auditable retirement state.

    This does not schedule a scan, mutate an observer, or decide model
    training.  ``decision_ready`` becomes true at the fixed horizon even when
    a well continues monitoring.  A terminal low-signal result is permitted
    only at or after that horizon, preserving comparable opportunity to form
    signal for every well.
    '''
    policy = _validated_policy(policy)
    if not trigger_completed:
        return {
            'state': WELL_MONITORING_STATE_PENDING_TRIGGER,
            'decision_ready': False,
            'retired': False,
            'retirement_reason': None,
            'elapsed_s': None,
            'plateau': None,
        }
    try:
        trigger_timestamp_s = float(trigger_timestamp_s)
        now_timestamp_s = float(now_timestamp_s)
    except (TypeError, ValueError):
        raise AutoStabilityLifecycleError(
            'trigger_timestamp_s and now_timestamp_s must be numeric.'
        )
    if (
            not math.isfinite(trigger_timestamp_s)
            or not math.isfinite(now_timestamp_s)
            or now_timestamp_s < trigger_timestamp_s):
        raise AutoStabilityLifecycleError(
            'now_timestamp_s must be finite and no earlier than trigger time.'
        )
    elapsed_s = now_timestamp_s - trigger_timestamp_s
    decision_ready = elapsed_s >= policy['decision_horizon_s']
    # A lifecycle result is always evaluated as of ``now_timestamp_s``.  It
    # must never use a later raw observation merely because a caller supplied
    # a complete saved trajectory rather than an incremental one.
    observations_as_of_now = [
        observation for observation in normalize_stability_observations(
            observations
        )
        if observation['timestamp_s'] <= now_timestamp_s
    ]
    plateau = evaluate_stability_plateau(
        observations_as_of_now,
        trigger_timestamp_s,
        minimum_peak_absorbance,
        policy
    )
    if decision_ready and plateau['reason'] == RETIREMENT_REASON_TERMINAL_LOW_SIGNAL:
        return {
            'state': WELL_MONITORING_STATE_TERMINAL_QC_EXCLUDED,
            'decision_ready': True,
            'retired': True,
            'retirement_reason': RETIREMENT_REASON_TERMINAL_LOW_SIGNAL,
            'elapsed_s': elapsed_s,
            'plateau': plateau,
        }
    if (
            decision_ready
            and policy['adaptive_retirement_enabled']
            and plateau['plateau_confirmed']):
        return {
            'state': WELL_MONITORING_STATE_PLATEAU_CONFIRMED,
            'decision_ready': True,
            'retired': True,
            'retirement_reason': RETIREMENT_REASON_PLATEAU_CONFIRMED,
            'elapsed_s': elapsed_s,
            'plateau': plateau,
        }
    if elapsed_s >= policy['max_observation_window_s']:
        return {
            'state': WELL_MONITORING_STATE_MAX_WINDOW_EXPIRED,
            'decision_ready': True,
            'retired': True,
            'retirement_reason': RETIREMENT_REASON_MAX_WINDOW_EXPIRED,
            'elapsed_s': elapsed_s,
            'plateau': plateau,
        }
    return {
        'state': (
            WELL_MONITORING_STATE_DECISION_READY
            if decision_ready else WELL_MONITORING_STATE_ACTIVE
        ),
        'decision_ready': decision_ready,
        'retired': False,
        'retirement_reason': None,
        'elapsed_s': elapsed_s,
        'plateau': plateau,
    }


def evaluate_condition_decision_readiness(well_lifecycle_records):
    '''Return whether every uniquely named replicate has reached a decision.

    Later controller code must aggregate those completed replicate metrics to
    one condition-level GP observation.  This helper intentionally rejects
    repeated well names rather than silently letting a duplicate trajectory
    affect readiness or future model weighting.
    '''
    records = list(well_lifecycle_records or [])
    if not records:
        return {
            'decision_ready': False,
            'well_count': 0,
            'pending_wellnames': [],
        }
    names = []
    pending = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise AutoStabilityLifecycleError(
                'Well lifecycle record {} must be a dictionary.'.format(index)
            )
        name = str(record.get('wellname', '')).strip()
        if not name:
            raise AutoStabilityLifecycleError(
                'Well lifecycle record {} requires wellname.'.format(index)
            )
        names.append(name)
        if record.get('decision_ready') is not True:
            pending.append(name)
    if len(set(names)) != len(names):
        raise AutoStabilityLifecycleError(
            'Condition lifecycle records cannot contain a duplicate wellname.'
        )
    return {
        'decision_ready': not pending,
        'well_count': len(names),
        'pending_wellnames': pending,
    }
