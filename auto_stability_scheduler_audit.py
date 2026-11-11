'''Pure Stage-14D scheduler-audit summary for optical-stability runs.

The append-only observer manifest remains the primary evidence. This module
only summarizes that evidence after a run; it cannot schedule a scan, alter a
model, or communicate with laboratory hardware. A summary status of
``no_detected_manifest_violation`` is therefore not a claim that the reader or
robot physically behaved correctly. It means the controller manifest contains
none of the explicitly checked contradictions.
'''

from __future__ import division

import math


SCHEDULER_AUDIT_SCHEMA_VERSION = 1

_RETIREMENT_EVENT_REASONS = {
    'observation_window_completed': 'maximum_observation_window_expired',
    'observation_plateau_confirmed': 'plateau_confirmed',
    'observation_terminal_qc_excluded': 'terminal_low_signal',
}

_ACTIVE_SET_SENTINEL = '__active_set__'


def _text(value):
    return '' if value is None else str(value).strip()


def _semicolon_values(value):
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = _text(value).split(';')
    return [_text(item) for item in values if _text(item)]


def _as_int(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or int(numeric) != numeric:
        return None
    return int(numeric)


def _event_ordered_rows(manifest_rows):
    '''Return manifest rows in durable event order with explicit warnings.'''
    warnings = []
    ordered = []
    seen_sequences = set()
    for ordinal, original_row in enumerate(manifest_rows or []):
        if not isinstance(original_row, dict):
            warnings.append({
                'code': 'non_dictionary_manifest_row',
                'message': 'Ignored manifest row {} because it is not a dictionary.'.format(
                    ordinal
                ),
            })
            continue
        row = dict(original_row)
        sequence = _as_int(row.get('event_sequence'))
        if sequence is None:
            warnings.append({
                'code': 'missing_event_sequence',
                'message': 'Manifest row {} has no valid event_sequence; file order was used.'.format(
                    ordinal
                ),
            })
            sequence = ordinal + 1
        elif sequence in seen_sequences:
            warnings.append({
                'code': 'duplicate_event_sequence',
                'message': 'Manifest event_sequence {} occurs more than once.'.format(
                    sequence
                ),
            })
        seen_sequences.add(sequence)
        ordered.append((sequence, ordinal, row))
    ordered.sort(key=lambda item: (item[0], item[1]))
    return ordered, warnings


def _active_locations(value):
    '''Parse the observer's immutable ``logical=physical`` mapping.'''
    mapping = {}
    malformed = []
    for item in _semicolon_values(value):
        if '=' not in item:
            malformed.append(item)
            continue
        wellname, location = item.split('=', 1)
        wellname = _text(wellname)
        location = _text(location)
        if not wellname or not location or wellname in mapping:
            malformed.append(item)
            continue
        mapping[wellname] = location
    return mapping, malformed


def _configuration_summary(configuration):
    '''Keep only declared scheduler fields; do not expose unrelated settings.'''
    configuration = configuration or {}
    names = (
        'auto_stability_mode',
        'auto_stability_mixing_mode',
        'auto_stability_scan_schedule',
        'auto_stability_retirement_policy',
        'auto_stability_decision_horizon_s',
        'auto_stability_max_observation_window_s',
        'auto_stability_observation_window_s',
        'auto_stability_min_peak_absorbance',
        'auto_stability_plateau_max_absorbance_slope_per_s',
    )
    return {
        name: configuration.get(name)
        for name in names if name in configuration
    }


def build_stability_scheduler_audit_summary(manifest_rows, configuration=None):
    '''Summarize scheduler evidence from one append-only observer manifest.

    This function deliberately does not open reader files: reader-file loading
    and blank correction remain the established Stage-12D reporting path. It
    checks only durable relationships that can be proven from the manifest.
    '''
    ordered_rows, warnings = _event_ordered_rows(manifest_rows)
    violations = []
    wells = {}
    scans = []
    scan_ids = set()
    retired_wells = {}

    def well_record(wellname):
        return wells.setdefault(wellname, {
            'wellname': wellname,
            'batch_number': None,
            'trigger_completion_event_sequence': None,
            'targeted_mix_acknowledgement_count': 0,
            'targeted_mix_acknowledgement_event_sequences': [],
            'raw_scan_ids': [],
            'decision_horizon_event_sequence': None,
            'retirement': None,
        })

    for sequence, _ordinal, row in ordered_rows:
        event_type = _text(row.get('event_type'))
        wellname = _text(row.get('wellname'))
        # Raw active-set events use this manifest sentinel as their anchor,
        # not as a physical reaction well. The durable per-well membership is
        # represented only by ``active_wellnames`` below.
        if wellname == _ACTIVE_SET_SENTINEL:
            wellname = ''
        batch_number = _as_int(row.get('batch_number'))
        if wellname:
            record = well_record(wellname)
            if batch_number is not None:
                if record['batch_number'] is None:
                    record['batch_number'] = batch_number
                elif record['batch_number'] != batch_number:
                    violations.append({
                        'code': 'well_batch_identity_changed',
                        'wellname': wellname,
                        'event_sequence': sequence,
                        'message': 'Well {} changed from batch {} to batch {} in the manifest.'.format(
                            wellname, record['batch_number'], batch_number
                        ),
                    })

        if event_type == 'trigger_transfer_completed' and wellname:
            record = well_record(wellname)
            if record['trigger_completion_event_sequence'] is not None:
                violations.append({
                    'code': 'duplicate_trigger_completion',
                    'wellname': wellname,
                    'event_sequence': sequence,
                    'message': 'Well {} has more than one trigger completion event.'.format(
                        wellname
                    ),
                })
            else:
                record['trigger_completion_event_sequence'] = sequence

        elif event_type == 'targeted_pipette_mix_acknowledged' and wellname:
            record = well_record(wellname)
            record['targeted_mix_acknowledgement_count'] += 1
            record['targeted_mix_acknowledgement_event_sequences'].append(sequence)
            if record['targeted_mix_acknowledgement_count'] > 1:
                violations.append({
                    'code': 'multiple_targeted_mix_acknowledgements',
                    'wellname': wellname,
                    'event_sequence': sequence,
                    'message': 'Well {} has more than one acknowledged targeted mix.'.format(
                        wellname
                    ),
                })

        elif event_type == 'decision_horizon_reached' and wellname:
            record = well_record(wellname)
            if record['decision_horizon_event_sequence'] is None:
                record['decision_horizon_event_sequence'] = sequence

        elif event_type in _RETIREMENT_EVENT_REASONS and wellname:
            record = well_record(wellname)
            retirement_reason = _text(row.get('retirement_reason')) or (
                _RETIREMENT_EVENT_REASONS[event_type]
            )
            if record['retirement'] is not None:
                violations.append({
                    'code': 'multiple_retirement_events',
                    'wellname': wellname,
                    'event_sequence': sequence,
                    'message': 'Well {} has more than one lifecycle retirement event.'.format(
                        wellname
                    ),
                })
            else:
                record['retirement'] = {
                    'event_sequence': sequence,
                    'event_type': event_type,
                    'reason': retirement_reason,
                    'lifecycle_state': _text(row.get('lifecycle_state')),
                }
                retired_wells[wellname] = sequence

        if event_type != 'raw_scan_completed':
            continue

        scan_id = _text(row.get('raw_scan_id'))
        active_wellnames = _semicolon_values(row.get('active_wellnames'))
        active_batch_numbers = [
            value for value in (
                _as_int(item) for item in _semicolon_values(
                    row.get('active_batch_numbers')
                )
            ) if value is not None
        ]
        locations, malformed_locations = _active_locations(
            row.get('active_well_locations')
        )
        if not scan_id:
            violations.append({
                'code': 'raw_scan_missing_id',
                'event_sequence': sequence,
                'message': 'A raw_scan_completed event has no raw_scan_id.',
            })
        elif scan_id in scan_ids:
            violations.append({
                'code': 'duplicate_raw_scan_id',
                'raw_scan_id': scan_id,
                'event_sequence': sequence,
                'message': 'Raw scan ID {} occurs more than once.'.format(scan_id),
            })
        else:
            scan_ids.add(scan_id)
        if not active_wellnames:
            violations.append({
                'code': 'raw_scan_missing_active_wells',
                'raw_scan_id': scan_id,
                'event_sequence': sequence,
                'message': 'Raw scan {} has no declared active wells.'.format(scan_id),
            })
        if set(active_wellnames) != set(locations) or malformed_locations:
            violations.append({
                'code': 'raw_scan_mapping_mismatch',
                'raw_scan_id': scan_id,
                'event_sequence': sequence,
                'message': 'Raw scan {} does not have one valid immutable reader location per active well.'.format(scan_id),
            })
        if len(set(active_batch_numbers)) != len(active_batch_numbers):
            violations.append({
                'code': 'raw_scan_duplicate_active_batch',
                'raw_scan_id': scan_id,
                'event_sequence': sequence,
                'message': 'Raw scan {} repeats an active batch number.'.format(scan_id),
            })
        for active_wellname in active_wellnames:
            record = well_record(active_wellname)
            record['raw_scan_ids'].append(scan_id)
            retirement_sequence = retired_wells.get(active_wellname)
            if retirement_sequence is not None and sequence > retirement_sequence:
                violations.append({
                    'code': 'retired_well_scanned_again',
                    'wellname': active_wellname,
                    'raw_scan_id': scan_id,
                    'event_sequence': sequence,
                    'message': 'Retired well {} appears in later raw scan {}.'.format(
                        active_wellname, scan_id
                    ),
                })
        scans.append({
            'raw_scan_id': scan_id,
            'event_sequence': sequence,
            'active_wellnames': active_wellnames,
            'active_batch_numbers': active_batch_numbers,
            'raw_scan_relative_path': _text(row.get('raw_scan_relative_path')),
            'observation_reason': _text(row.get('observation_reason')),
            'scan_started_at_utc': _text(row.get('scan_started_at_utc')),
            'scan_completed_at_utc': _text(row.get('scan_completed_at_utc')),
        })

    for wellname, record in sorted(wells.items()):
        if record['targeted_mix_acknowledgement_count'] and (
                record['trigger_completion_event_sequence'] is None):
            violations.append({
                'code': 'mix_acknowledgement_without_trigger_completion',
                'wellname': wellname,
                'message': 'Well {} has an acknowledged mix but no trigger completion event.'.format(
                    wellname
                ),
            })
        if record['retirement'] is not None and (
                record['trigger_completion_event_sequence'] is None):
            violations.append({
                'code': 'retirement_without_trigger_completion',
                'wellname': wellname,
                'message': 'Well {} retired without a trigger completion event.'.format(
                    wellname
                ),
            })

    for scan in scans:
        expected_batches = set()
        for wellname in scan['active_wellnames']:
            record = wells.get(wellname)
            if record is None or record['trigger_completion_event_sequence'] is None:
                violations.append({
                    'code': 'raw_scan_well_without_trigger_completion',
                    'wellname': wellname,
                    'raw_scan_id': scan['raw_scan_id'],
                    'event_sequence': scan['event_sequence'],
                    'message': 'Raw scan {} includes {} without a trigger completion event.'.format(
                        scan['raw_scan_id'], wellname
                    ),
                })
                continue
            if record['batch_number'] is not None:
                expected_batches.add(record['batch_number'])
        if expected_batches and expected_batches != set(
                scan['active_batch_numbers']):
            violations.append({
                'code': 'raw_scan_active_batch_identity_mismatch',
                'raw_scan_id': scan['raw_scan_id'],
                'event_sequence': scan['event_sequence'],
                'message': 'Raw scan {} batch membership does not match its active wells.'.format(
                    scan['raw_scan_id']
                ),
            })

    configuration_summary = _configuration_summary(configuration)
    requires_targeted_mix = (
        configuration_summary.get('auto_stability_mixing_mode') == 'pipette_mix'
        and configuration_summary.get('auto_stability_scan_schedule') == (
            'each_completion'
        )
    )
    if requires_targeted_mix:
        for wellname, record in sorted(wells.items()):
            if record['trigger_completion_event_sequence'] is None:
                continue
            acknowledgements = record['targeted_mix_acknowledgement_count']
            if acknowledgements != 1:
                violations.append({
                    'code': 'targeted_mix_count_not_exactly_one',
                    'wellname': wellname,
                    'message': 'Pipette-mix each-completion well {} has {} acknowledged mixes; exactly one is required.'.format(
                        wellname, acknowledgements
                    ),
                })
            elif record['targeted_mix_acknowledgement_event_sequences'][0] < (
                    record['trigger_completion_event_sequence']):
                violations.append({
                    'code': 'targeted_mix_acknowledgement_before_trigger_completion',
                    'wellname': wellname,
                    'message': 'Well {} has an acknowledged targeted mix before trigger completion.'.format(
                        wellname
                    ),
                })

    mixed_batch_scan_ids = [
        scan['raw_scan_id'] for scan in scans
        if len(set(scan['active_batch_numbers'])) > 1
    ]
    completed_trigger_count = sum(
        record['trigger_completion_event_sequence'] is not None
        for record in wells.values()
    )
    total_acknowledged_mixes = sum(
        record['targeted_mix_acknowledgement_count'] for record in wells.values()
    )
    if violations:
        status = 'manifest_violations_detected'
    elif warnings:
        status = 'manifest_review_required'
    else:
        status = 'no_detected_manifest_violation'
    return {
        'schema_version': SCHEDULER_AUDIT_SCHEMA_VERSION,
        'status': status,
        'scope': (
            'Post-run audit of append-only controller manifest relationships; '
            'does not independently verify physical robot or reader behavior.'
        ),
        'configuration': configuration_summary,
        'manifest': {
            'event_count': len(ordered_rows),
            'schema_versions': sorted(set(
                _text(row.get('schema_version'))
                for _sequence, _ordinal, row in ordered_rows
                if _text(row.get('schema_version'))
            )),
            'raw_scan_count': len(scans),
            'completed_trigger_count': completed_trigger_count,
            'acknowledged_targeted_mix_count': total_acknowledged_mixes,
        },
        'cross_batch_union': {
            'observed': bool(mixed_batch_scan_ids),
            'raw_scan_ids': mixed_batch_scan_ids,
        },
        'wells': [wells[wellname] for wellname in sorted(wells)],
        'raw_scans': scans,
        'violations': violations,
        'warnings': warnings,
    }
