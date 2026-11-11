'''Post-run integrity audit for frozen optical-stability decision snapshots.

Stage 14E1 checks the controller-exported, decision-horizon snapshot files
against the append-only observer manifest.  It is intentionally a pure,
reporting-only component: it cannot schedule work, modify a model, communicate
with the Pi, or reinterpret later audit-only observations as decision evidence.

The audit records relationships that can be established from saved controller
evidence.  A passing status means those relationships are internally
consistent; it is not an independent claim about physical robot or reader
behavior.
'''

from __future__ import division

import math


SNAPSHOT_INTEGRITY_AUDIT_SCHEMA_VERSION = 1


def _text(value):
    return '' if value is None else str(value).strip()


def _as_int(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or int(numeric) != numeric:
        return None
    return int(numeric)


def _as_positive_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _same_number(left, right):
    left_number = _as_positive_float(left)
    right_number = _as_positive_float(right)
    return (
        left_number is not None and right_number is not None and
        abs(left_number - right_number) <= 1e-9
    )


def _append_violation(violations, code, message, **details):
    violation = {'code': code, 'message': message}
    violation.update(details)
    violations.append(violation)


def _snapshot_rows_by_identity(rows, identity_name, row_kind, violations):
    '''Index snapshot rows while retaining every malformed row as evidence.'''
    indexed = {}
    for ordinal, original_row in enumerate(rows or []):
        if not isinstance(original_row, dict):
            _append_violation(
                violations,
                'non_dictionary_{}_snapshot_row'.format(row_kind),
                'Ignored {} snapshot row {} because it is not a dictionary.'.format(
                    row_kind, ordinal
                ),
                row_ordinal=ordinal,
            )
            continue
        row = dict(original_row)
        identity = _text(row.get(identity_name))
        if not identity:
            _append_violation(
                violations,
                'missing_{}_snapshot_identity'.format(row_kind),
                '{} snapshot row {} has no {}.'.format(
                    row_kind.capitalize(), ordinal, identity_name
                ),
                row_ordinal=ordinal,
            )
            continue
        if identity in indexed:
            _append_violation(
                violations,
                'duplicate_{}_snapshot_identity'.format(row_kind),
                '{} snapshot identity {!r} occurs more than once.'.format(
                    row_kind.capitalize(), identity
                ),
                **{identity_name: identity}
            )
            continue
        indexed[identity] = row
    return indexed


def _manifest_evidence(manifest_rows, violations):
    '''Return immutable trigger and decision-horizon evidence keyed by well.'''
    triggers = {}
    decision_horizons = {}
    for ordinal, original_row in enumerate(manifest_rows or []):
        if not isinstance(original_row, dict):
            continue
        row = dict(original_row)
        event_type = _text(row.get('event_type'))
        wellname = _text(row.get('wellname'))
        if not wellname or wellname == '__active_set__':
            continue
        sequence = _as_int(row.get('event_sequence'))
        if event_type == 'trigger_transfer_completed':
            if wellname in triggers:
                _append_violation(
                    violations,
                    'duplicate_manifest_trigger_completion',
                    'Manifest has more than one trigger completion for well {}.'
                    .format(wellname),
                    wellname=wellname,
                    event_sequence=sequence,
                )
                continue
            triggers[wellname] = {
                'batch_number': _as_int(row.get('batch_number')),
                'event_sequence': sequence,
            }
        elif event_type == 'decision_horizon_reached':
            if wellname in decision_horizons:
                _append_violation(
                    violations,
                    'duplicate_manifest_decision_horizon',
                    'Manifest has more than one decision-horizon event for well {}.'
                    .format(wellname),
                    wellname=wellname,
                    event_sequence=sequence,
                )
                continue
            decision_horizons[wellname] = {
                'batch_number': _as_int(row.get('batch_number')),
                'event_sequence': sequence,
                'decision_horizon_s': _as_positive_float(
                    row.get('decision_horizon_s')
                ),
            }
    return triggers, decision_horizons


def build_stability_decision_snapshot_integrity_audit(
        manifest_rows,
        condition_snapshot_rows,
        well_snapshot_rows,
        configuration=None):
    '''Build a pure Stage-14E1 snapshot-integrity audit summary.

    Snapshot records are immutable model-facing evidence.  The final reporting
    CSVs may include later observations from the extended audit window; this
    function deliberately does not compare their values or permit them to
    supplement a frozen snapshot.
    '''
    configuration = dict(configuration or {})
    expected_horizon = _as_positive_float(
        configuration.get('auto_stability_decision_horizon_s')
    )
    maximum_window = _as_positive_float(
        configuration.get('auto_stability_max_observation_window_s')
    )
    violations = []
    warnings = []
    condition_rows = list(condition_snapshot_rows or [])
    well_rows = list(well_snapshot_rows or [])

    applicable = (
        _text(configuration.get('auto_stability_mode')) == 'stability_only' and
        _text(configuration.get('auto_stability_mixing_mode')) == 'pipette_mix' and
        _text(configuration.get('auto_stability_scan_schedule')) ==
        'each_completion' and expected_horizon is not None and
        maximum_window is not None and expected_horizon < maximum_window
    )
    if not applicable:
        return {
            'schema_version': SNAPSHOT_INTEGRITY_AUDIT_SCHEMA_VERSION,
            'status': 'not_applicable',
            'applicable': False,
            'configuration': configuration,
            'message': (
                'Frozen decision-snapshot integrity is applicable only to the '
                'cross-batch stability-only pipette-mix/each-completion path '
                'with a decision horizon shorter than its maximum window.'
            ),
            'snapshot_conditions': {'row_count': len(condition_rows)},
            'snapshot_wells': {'row_count': len(well_rows)},
            'violations': [],
            'warnings': [],
        }

    if expected_horizon is None:
        _append_violation(
            violations,
            'missing_configured_decision_horizon',
            'Cross-batch snapshot integrity requires a positive configured '
            'decision horizon.',
        )
    if not condition_rows:
        _append_violation(
            violations,
            'missing_condition_snapshot_rows',
            'Cross-batch execution produced no condition decision snapshots.',
        )
    if not well_rows:
        _append_violation(
            violations,
            'missing_well_snapshot_rows',
            'Cross-batch execution produced no per-well decision snapshots.',
        )

    conditions = _snapshot_rows_by_identity(
        condition_rows, 'condition_id', 'condition', violations
    )
    wells = _snapshot_rows_by_identity(
        well_rows, 'wellname', 'well', violations
    )
    triggers, decision_horizons = _manifest_evidence(manifest_rows, violations)
    wells_by_condition = {}

    for condition_id, row in conditions.items():
        if _text(row.get('snapshot_status')).lower() != 'frozen':
            _append_violation(
                violations,
                'condition_snapshot_not_frozen',
                'Condition snapshot {} is not marked frozen.'.format(condition_id),
                condition_id=condition_id,
            )
        if expected_horizon is not None and not _same_number(
                row.get('decision_horizon_s'), expected_horizon):
            _append_violation(
                violations,
                'condition_snapshot_horizon_mismatch',
                'Condition snapshot {} does not use the configured decision '
                'horizon.'.format(condition_id),
                condition_id=condition_id,
            )

    for wellname, row in wells.items():
        condition_id = _text(row.get('condition_id'))
        if not condition_id:
            _append_violation(
                violations,
                'well_snapshot_missing_condition_identity',
                'Well snapshot {} has no condition_id.'.format(wellname),
                wellname=wellname,
            )
        else:
            wells_by_condition.setdefault(condition_id, []).append(row)
            if condition_id not in conditions:
                _append_violation(
                    violations,
                    'well_snapshot_condition_not_exported',
                    'Well snapshot {} refers to missing condition snapshot {}.'
                    .format(wellname, condition_id),
                    wellname=wellname,
                    condition_id=condition_id,
                )
        if _text(row.get('snapshot_status')).lower() != 'frozen':
            _append_violation(
                violations,
                'well_snapshot_not_frozen',
                'Well snapshot {} is not marked frozen.'.format(wellname),
                wellname=wellname,
            )
        if expected_horizon is not None and not _same_number(
                row.get('decision_horizon_s'), expected_horizon):
            _append_violation(
                violations,
                'well_snapshot_horizon_mismatch',
                'Well snapshot {} does not use the configured decision horizon.'
                .format(wellname),
                wellname=wellname,
            )
        trigger = triggers.get(wellname)
        if trigger is None:
            _append_violation(
                violations,
                'snapshot_well_missing_trigger_completion',
                'Well snapshot {} has no manifest trigger completion.'.format(
                    wellname
                ),
                wellname=wellname,
            )
        else:
            row_batch = _as_int(row.get('batch_number'))
            if row_batch is None or trigger['batch_number'] is None or (
                    row_batch != trigger['batch_number']):
                _append_violation(
                    violations,
                    'snapshot_well_batch_mismatch',
                    'Well snapshot {} batch identity does not match its manifest '
                    'trigger completion.'.format(wellname),
                    wellname=wellname,
                    snapshot_batch_number=row.get('batch_number'),
                    manifest_batch_number=trigger['batch_number'],
                )
        horizon_event = decision_horizons.get(wellname)
        if horizon_event is None:
            _append_violation(
                violations,
                'snapshot_well_missing_decision_horizon_event',
                'Well snapshot {} has no manifest decision-horizon event.'.format(
                    wellname
                ),
                wellname=wellname,
            )
        elif expected_horizon is not None and (
                horizon_event['decision_horizon_s'] is not None and
                not _same_number(horizon_event['decision_horizon_s'], expected_horizon)):
            _append_violation(
                violations,
                'manifest_decision_horizon_mismatch',
                'Manifest decision-horizon event for {} disagrees with the '
                'configured horizon.'.format(wellname),
                wellname=wellname,
            )

    for condition_id, condition_row in conditions.items():
        linked_wells = wells_by_condition.get(condition_id, [])
        if not linked_wells:
            _append_violation(
                violations,
                'condition_snapshot_missing_well_evidence',
                'Condition snapshot {} has no linked per-well snapshot evidence.'
                .format(condition_id),
                condition_id=condition_id,
            )
            continue
        condition_batch = _as_int(condition_row.get('batch_number'))
        for well_row in linked_wells:
            well_batch = _as_int(well_row.get('batch_number'))
            if condition_batch is None or well_batch is None or (
                    condition_batch != well_batch):
                _append_violation(
                    violations,
                    'condition_well_snapshot_batch_mismatch',
                    'Condition snapshot {} and well snapshot {} do not share a '
                    'batch identity.'.format(condition_id, well_row['wellname']),
                    condition_id=condition_id,
                    wellname=well_row['wellname'],
                )
        expected_count = _as_int(condition_row.get('total_well_count'))
        if expected_count is not None and expected_count != len(linked_wells):
            _append_violation(
                violations,
                'condition_snapshot_well_count_mismatch',
                'Condition snapshot {} expects {} wells but exports {} linked '
                'well snapshots.'.format(
                    condition_id, expected_count, len(linked_wells)
                ),
                condition_id=condition_id,
            )

    if not triggers:
        warnings.append({
            'code': 'manifest_has_no_trigger_completions',
            'message': 'The manifest has no trigger-completion evidence.',
        })
    status = (
        'snapshot_violations_detected' if violations
        else 'no_detected_snapshot_violation'
    )
    return {
        'schema_version': SNAPSHOT_INTEGRITY_AUDIT_SCHEMA_VERSION,
        'status': status,
        'applicable': True,
        'configuration': configuration,
        'snapshot_conditions': {
            'row_count': len(condition_rows),
            'unique_condition_count': len(conditions),
        },
        'snapshot_wells': {
            'row_count': len(well_rows),
            'unique_well_count': len(wells),
        },
        'manifest': {
            'trigger_completion_count': len(triggers),
            'decision_horizon_event_count': len(decision_horizons),
        },
        'violations': violations,
        'warnings': warnings,
        'scope_note': (
            'The frozen decision snapshots remain separate from later '
            'extended-window observations, which are audit-only for the '
            'corresponding decision.'
        ),
    }
