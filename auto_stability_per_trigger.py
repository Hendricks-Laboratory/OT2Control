'''Pure Stage-13D active-set planning for per-trigger stability monitoring.

This module deliberately contains no controller, robot, reader, filesystem,
or workbook access.  It converts the already-observed completion records for
one physical plate into one explicit reader-scan plan after a newly completed
trigger well.  A later controller scheduler may execute that plan only after
the targeted-mix handshake has been acknowledged.

The planner is intentionally strict about physical plate identity.  It never
guesses that two logical wells remain on the same reader plate after a plate
replacement, and it never combines same-named reader coordinates from two
physical generations.  It also keeps the fixed decision horizon distinct from
the longer monitoring window: decision-ready wells remain eligible to be
observed until an explicit lifecycle retirement reason applies.
'''

from __future__ import division

import math

from auto_stability_lifecycle import (
    AutoStabilityLifecycleError,
    evaluate_well_monitoring_lifecycle,
)


PER_TRIGGER_ACTIVE_SET_SCHEMA_VERSION = 1


class AutoStabilityPerTriggerError(ValueError):
    '''Raised when a future per-trigger scan would have ambiguous evidence.'''


def _require_nonnegative_integer(value, field_name):
    if isinstance(value, bool):
        raise AutoStabilityPerTriggerError(
            '{} must be a nonnegative integer.'.format(field_name)
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise AutoStabilityPerTriggerError(
            '{} must be a nonnegative integer.'.format(field_name)
        )
    if parsed < 0 or float(value) != float(parsed):
        raise AutoStabilityPerTriggerError(
            '{} must be a nonnegative integer.'.format(field_name)
        )
    return parsed


def _require_positive_integer(value, field_name):
    parsed = _require_nonnegative_integer(value, field_name)
    if parsed <= 0:
        raise AutoStabilityPerTriggerError(
            '{} must be a positive integer.'.format(field_name)
        )
    return parsed


def _require_finite_nonnegative(value, field_name):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityPerTriggerError(
            '{} must be finite and nonnegative.'.format(field_name)
        )
    if not math.isfinite(parsed) or parsed < 0.0:
        raise AutoStabilityPerTriggerError(
            '{} must be finite and nonnegative.'.format(field_name)
        )
    return parsed


def _require_finite_positive(value, field_name):
    parsed = _require_finite_nonnegative(value, field_name)
    if parsed <= 0.0:
        raise AutoStabilityPerTriggerError(
            '{} must be finite and greater than zero.'.format(field_name)
        )
    return parsed


def _require_nonblank_text(value, field_name):
    text = str(value).strip()
    if not text:
        raise AutoStabilityPerTriggerError(
            '{} cannot be blank.'.format(field_name)
        )
    return text


def normalize_physical_identity(record):
    '''Return one strict physical identity from an active-well record.

    The observer also uses this parser when it first registers the immutable
    mapping.  Keeping it here ensures registration and later scan planning
    cannot drift into accepting different plate-identity representations.
    '''
    if not isinstance(record, dict):
        raise AutoStabilityPerTriggerError(
            'Each active-well record must be a dictionary.'
        )
    return {
        'plate_generation': _require_nonnegative_integer(
            record.get('plate_generation'), 'plate_generation'
        ),
        'plate_mapping_revision': _require_nonnegative_integer(
            record.get('plate_mapping_revision'), 'plate_mapping_revision'
        ),
        'deck_pos': _require_nonnegative_integer(
            record.get('deck_pos'), 'deck_pos'
        ),
        'reader_location': _require_nonblank_text(
            record.get('reader_location'), 'reader_location'
        ).upper(),
    }


def _normalized_observations_by_well(observations_by_well, allowed_wellnames):
    if observations_by_well is None:
        return {}
    if not isinstance(observations_by_well, dict):
        raise AutoStabilityPerTriggerError(
            'observations_by_well must be a dictionary keyed by wellname.'
        )
    normalized = {}
    for raw_wellname, observations in observations_by_well.items():
        wellname = _require_nonblank_text(raw_wellname, 'observation wellname')
        if wellname not in allowed_wellnames:
            raise AutoStabilityPerTriggerError(
                'Observations were supplied for non-active well {}.'.format(
                    wellname
                )
            )
        if observations is None:
            observations = []
        if isinstance(observations, (str, bytes)):
            raise AutoStabilityPerTriggerError(
                'Observations for {} must be an iterable of records.'.format(
                    wellname
                )
            )
        try:
            normalized[wellname] = list(observations)
        except TypeError:
            raise AutoStabilityPerTriggerError(
                'Observations for {} must be an iterable of records.'.format(
                    wellname
                )
            )
    return normalized


def build_per_trigger_active_set(
        active_well_records,
        triggering_wellname,
        now_monotonic_s,
        monitoring_policy,
        minimum_peak_absorbance,
        observations_by_well=None):
    '''Build one cross-batch current-plate active-set scan plan.

    ``active_well_records`` must contain a registered physical identity and a
    unique trigger-completion sequence for every still-active well. The plan
    sorts by that observed sequence rather than trusting packet-dispatch or
    dictionary order. It includes all non-retired wells on that one physical
    plate, including the newly completed triggering well; earlier wells may
    come from earlier batches. Retired wells are reported separately rather
    than silently discarded, so the future scheduler can append an explicit
    retirement event before it stops observing them.

    This function does not mutate a record, reserve a reader path, send a Pi
    packet, or infer an observation.  It is the deterministic contract between
    Stage 13D's later scheduler and the Stage 13A lifecycle evaluator.
    '''
    triggering_wellname = _require_nonblank_text(
        triggering_wellname, 'triggering_wellname'
    )
    now_monotonic_s = _require_finite_nonnegative(
        now_monotonic_s, 'now_monotonic_s'
    )
    minimum_peak_absorbance = _require_finite_positive(
        minimum_peak_absorbance, 'minimum_peak_absorbance'
    )
    try:
        records = list(active_well_records)
    except TypeError:
        raise AutoStabilityPerTriggerError(
            'active_well_records must be an iterable of dictionaries.'
        )
    if not records:
        raise AutoStabilityPerTriggerError(
            'A per-trigger scan requires at least the triggering active well.'
        )

    normalized_records = []
    seen_wellnames = set()
    for record in records:
        if not isinstance(record, dict):
            raise AutoStabilityPerTriggerError(
                'Each active-well record must be a dictionary.'
            )
        wellname = _require_nonblank_text(record.get('wellname'), 'wellname')
        if wellname in seen_wellnames:
            raise AutoStabilityPerTriggerError(
                'Active-well records cannot repeat {}.'.format(wellname)
            )
        seen_wellnames.add(wellname)
        if record.get('activation_status') != 'active':
            raise AutoStabilityPerTriggerError(
                'Per-trigger scans require active wells only: {}.'.format(
                    wellname
                )
            )
        activation_time = _require_finite_nonnegative(
            record.get('activation_monotonic_s'),
            'activation_monotonic_s for {}'.format(wellname)
        )
        if activation_time > now_monotonic_s:
            raise AutoStabilityPerTriggerError(
                'Active well {} has an activation time after the requested '
                'scan time.'.format(wellname)
            )
        normalized = dict(record)
        normalized['wellname'] = wellname
        normalized['activation_monotonic_s'] = activation_time
        normalized['batch_number'] = _require_nonnegative_integer(
            record.get('batch_number'), 'batch_number for {}'.format(wellname)
        )
        normalized['trigger_completion_sequence'] = _require_positive_integer(
            record.get('trigger_completion_sequence'),
            'trigger_completion_sequence for {}'.format(wellname)
        )
        normalized['physical_identity'] = normalize_physical_identity(record)
        normalized_records.append(normalized)

    if triggering_wellname not in seen_wellnames:
        raise AutoStabilityPerTriggerError(
            'Triggering well {} is not an active completed well.'.format(
                triggering_wellname
            )
        )

    completion_sequences = [
        record['trigger_completion_sequence'] for record in normalized_records
    ]
    if len(set(completion_sequences)) != len(completion_sequences):
        raise AutoStabilityPerTriggerError(
            'Active-well records cannot share a trigger completion sequence.'
        )
    triggering_record = next(
        record for record in normalized_records
        if record['wellname'] == triggering_wellname
    )
    if triggering_record['trigger_completion_sequence'] != max(
            completion_sequences):
        raise AutoStabilityPerTriggerError(
            'A per-trigger scan must be anchored to the newest confirmed '
            'trigger completion.'
        )
    # Dispatch order can differ from completion order when the controller has
    # several pending trigger commands. Reader layout and audit order must
    # follow actual completion, never dictionary insertion order.
    normalized_records.sort(
        key=lambda record: record['trigger_completion_sequence']
    )

    identities = {
        (
            record['physical_identity']['plate_generation'],
            record['physical_identity']['plate_mapping_revision'],
            record['physical_identity']['deck_pos'],
        )
        for record in normalized_records
    }
    if len(identities) != 1:
        raise AutoStabilityPerTriggerError(
            'Per-trigger active-set scans refuse active wells from multiple '
            'physical plate identities.'
        )
    reader_locations = [
        record['physical_identity']['reader_location']
        for record in normalized_records
    ]
    if len(set(reader_locations)) != len(reader_locations):
        raise AutoStabilityPerTriggerError(
            'Per-trigger active-set scans cannot map two active wells to one '
            'reader location.'
        )

    observations_by_well = _normalized_observations_by_well(
        observations_by_well, seen_wellnames
    )
    lifecycles = []
    scan_records = []
    retired_wellnames = []
    for record in normalized_records:
        try:
            lifecycle = evaluate_well_monitoring_lifecycle(
                trigger_timestamp_s=record['activation_monotonic_s'],
                now_timestamp_s=now_monotonic_s,
                observations=observations_by_well.get(record['wellname'], []),
                minimum_peak_absorbance=minimum_peak_absorbance,
                policy=monitoring_policy,
                trigger_completed=True,
            )
        except AutoStabilityLifecycleError as exc:
            raise AutoStabilityPerTriggerError(
                'Cannot evaluate active well {}: {}.'.format(
                    record['wellname'], exc
                )
            )
        lifecycle_record = dict(lifecycle)
        lifecycle_record['wellname'] = record['wellname']
        lifecycle_record['batch_number'] = record['batch_number']
        lifecycles.append(lifecycle_record)
        if lifecycle['retired']:
            retired_wellnames.append(record['wellname'])
        else:
            scan_records.append(record)

    scan_wellnames = [record['wellname'] for record in scan_records]
    if triggering_wellname not in scan_wellnames:
        raise AutoStabilityPerTriggerError(
            'Triggering well {} is already retired and cannot define a '
            'completion-triggered scan.'.format(triggering_wellname)
        )
    identity = scan_records[0]['physical_identity']
    return {
        'schema_version': PER_TRIGGER_ACTIVE_SET_SCHEMA_VERSION,
        'scan_reason': 'each_completion',
        'triggering_wellname': triggering_wellname,
        'plate_generation': identity['plate_generation'],
        'plate_mapping_revision': identity['plate_mapping_revision'],
        'deck_pos': identity['deck_pos'],
        'scan_wellnames': scan_wellnames,
        'reader_locations': [
            record['physical_identity']['reader_location']
            for record in scan_records
        ],
        'active_batch_numbers': sorted({
            record['batch_number'] for record in scan_records
        }),
        'decision_ready_wellnames': [
            record['wellname'] for record, lifecycle in zip(
                normalized_records, lifecycles
            )
            if lifecycle['decision_ready'] and not lifecycle['retired']
        ],
        'retired_wellnames': retired_wellnames,
        'well_lifecycles': lifecycles,
    }


def _normalize_acknowledged_targeted_mix_actions(
        active_well_records, acknowledged_mix_actions_by_well):
    '''Bind every active future well to exactly one completed mix action.

    The Stage-13D reader plan is valid only after the Pi has positively
    acknowledged the targeted mix for every well that could be included. This
    is intentionally stricter than the Stage-13D-A planner: it is a separate
    future boundary so current plate-shake callers retain their validated
    contract unchanged.
    '''
    if not isinstance(acknowledged_mix_actions_by_well, dict):
        raise AutoStabilityPerTriggerError(
            'acknowledged_mix_actions_by_well must be a dictionary.'
        )
    active_by_well = {}
    for record in active_well_records:
        wellname = _require_nonblank_text(record.get('wellname'), 'wellname')
        active_by_well[wellname] = record

    if set(acknowledged_mix_actions_by_well) != set(active_by_well):
        missing = sorted(set(active_by_well) - set(
            acknowledged_mix_actions_by_well
        ))
        unexpected = sorted(set(acknowledged_mix_actions_by_well) - set(
            active_by_well
        ))
        raise AutoStabilityPerTriggerError(
            'Targeted-mix acknowledgements must match active wells exactly; '
            'missing={}, unexpected={}.'.format(missing, unexpected)
        )

    normalized = {}
    seen_action_ids = set()
    for wellname, action in acknowledged_mix_actions_by_well.items():
        if not isinstance(action, dict):
            raise AutoStabilityPerTriggerError(
                'Targeted-mix acknowledgement for {} must be a dictionary.'
                .format(wellname)
            )
        action_id = _require_nonblank_text(
            action.get('action_id'),
            'targeted mix action_id for {}'.format(wellname)
        )
        if action_id in seen_action_ids:
            raise AutoStabilityPerTriggerError(
                'Targeted-mix acknowledgement action_id {} is reused.'
                .format(action_id)
            )
        if _require_nonblank_text(
                action.get('wellname'),
                'targeted mix acknowledgement wellname'
        ) != wellname:
            raise AutoStabilityPerTriggerError(
                'Targeted-mix acknowledgement action {} changed its well.'
                .format(action_id)
            )
        if action.get('acknowledged') is not True:
            raise AutoStabilityPerTriggerError(
                'Targeted-mix action {} for {} is not acknowledged.'
                .format(action_id, wellname)
            )
        action_batch = _require_nonnegative_integer(
            action.get('batch_number'),
            'targeted mix batch_number for {}'.format(wellname)
        )
        record_batch = _require_nonnegative_integer(
            active_by_well[wellname].get('batch_number'),
            'active well batch_number for {}'.format(wellname)
        )
        if action_batch != record_batch:
            raise AutoStabilityPerTriggerError(
                'Targeted-mix action {} has a batch mismatch for {}.'
                .format(action_id, wellname)
            )
        seen_action_ids.add(action_id)
        normalized[wellname] = {
            'action_id': action_id,
            'wellname': wellname,
            'batch_number': action_batch,
            'acknowledged': True
        }
    return normalized


def build_per_trigger_post_mix_active_set(
        active_well_records,
        triggering_wellname,
        now_monotonic_s,
        monitoring_policy,
        minimum_peak_absorbance,
        acknowledged_mix_actions_by_well,
        observations_by_well=None):
    '''Build a future per-trigger reader plan only after every mix is confirmed.

    This is a pure contract. It neither sends a mix packet nor reserves or
    starts a reader acquisition. The later runtime scheduler must use this
    stricter function, not the Stage-13D-A foundation, after it has durably
    recorded a successful Stage-13C Pi acknowledgement.
    '''
    records = list(active_well_records)
    acknowledged_actions = _normalize_acknowledged_targeted_mix_actions(
        records, acknowledged_mix_actions_by_well
    )
    plan = build_per_trigger_active_set(
        active_well_records=records,
        triggering_wellname=triggering_wellname,
        now_monotonic_s=now_monotonic_s,
        monitoring_policy=monitoring_policy,
        minimum_peak_absorbance=minimum_peak_absorbance,
        observations_by_well=observations_by_well,
    )
    plan['targeted_mix_acknowledgement_required'] = True
    plan['targeted_mix_action_ids'] = [
        acknowledged_actions[wellname]['action_id']
        for wellname in plan['scan_wellnames']
    ]
    plan['triggering_mix_action_id'] = acknowledged_actions[
        plan['triggering_wellname']
    ]['action_id']
    return plan
