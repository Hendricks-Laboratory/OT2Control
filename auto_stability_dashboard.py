'''Pure derived records for the Stage-14F3 final stability dashboard.

The dashboard is intentionally downstream of completed Stage-12D reporting.
It neither opens raw reader files nor decides QC, scheduling, mixing, model
training, or recipe selection.  Its only job is to make the reporting records'
replicate coverage and observed reader timing easy to inspect without silently
counting one active-set scan more than once.
'''

from __future__ import division

import datetime
import math


def _text(value):
    return '' if value is None else str(value).strip()


def _finite_number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _integer_or_none(value):
    numeric = _finite_number(value)
    if numeric is None or int(numeric) != numeric:
        return None
    return int(numeric)


def _parse_utc(value):
    text = _text(value)
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        timestamp = datetime.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        return None
    return timestamp.astimezone(datetime.timezone.utc)


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / float(len(values)) if values else None


def _condition_key(row):
    condition_id = _text(row.get('condition_id'))
    if condition_id:
        return ('condition_id', condition_id)
    return (
        'batch_reaction',
        _integer_or_none(row.get('batch_number')),
        _integer_or_none(row.get('reaction_number')),
    )


def _condition_sort_key(row):
    batch = _integer_or_none(row.get('batch_number'))
    reaction = _integer_or_none(row.get('reaction_number'))
    return (
        batch is None, batch if batch is not None else 0,
        reaction is None, reaction if reaction is not None else 0,
        _text(row.get('condition_id')),
    )


def _well_sort_key(row):
    return _condition_sort_key(row) + (_text(row.get('wellname')),)


def _best_qc_reason_by_well(qc_rows):
    '''Pick the terminal, well-level reason without discarding scan-level audit.'''
    reasons = {}
    for original_row in qc_rows or []:
        if not isinstance(original_row, dict):
            continue
        wellname = _text(original_row.get('wellname'))
        status = _text(original_row.get('qc_status'))
        reason = _text(original_row.get('qc_reason'))
        if not wellname or not reason:
            continue
        # A blank scan ID denotes the final per-well metric result. It is the
        # clearest concise reason for the completeness table. Earlier scan
        # issues remain available unchanged in stability_qc_audit.csv.
        priority = 0 if not _text(original_row.get('raw_scan_id')) else 1
        candidate = (priority, status, reason)
        existing = reasons.get(wellname)
        if existing is None or candidate[0] < existing[0]:
            reasons[wellname] = candidate
    return reasons


def _condition_summary_index(condition_summaries):
    indexed = {}
    for original_row in condition_summaries or []:
        if not isinstance(original_row, dict):
            continue
        row = dict(original_row)
        key = _condition_key(row)
        # The established reporting builder emits one summary per condition.
        # Retain the first record if malformed input duplicates one rather than
        # silently counting its denominator twice in the final dashboard.
        if key not in indexed:
            indexed[key] = row
    return indexed


def build_stability_dashboard_records(records):
    '''Build immutable, audit-oriented inputs for Stage-14F3 figures/CSVs.

    ``records`` is the completed reporting dictionary produced by
    ``build_stability_reporting_records`` plus controller provenance.  The
    returned data deliberately preserves every planned well metric, including
    wells without an eligible stability loss rate.  Raw-scan rows are reduced
    by immutable raw-scan ID so one multiwell scan is represented exactly once.
    '''
    if not isinstance(records, dict):
        raise ValueError('stability dashboard requires a reporting-record dictionary.')

    qc_by_well = _best_qc_reason_by_well(records.get('qc_rows'))
    summaries_by_condition = _condition_summary_index(
        records.get('condition_summaries')
    )

    replicate_rows = []
    seen_wells = set()
    for original_metric in records.get('well_metrics', []) or []:
        if not isinstance(original_metric, dict):
            continue
        metric = dict(original_metric)
        wellname = _text(metric.get('wellname'))
        if not wellname or wellname in seen_wells:
            # Physical wells must be unique in the established reporting
            # records. Retaining a duplicate would fabricate a replicate.
            continue
        seen_wells.add(wellname)
        summary = summaries_by_condition.get(_condition_key(metric), {})
        metric_status = _text(metric.get('status')) or 'status_unavailable'
        _priority, qc_status, qc_reason = qc_by_well.get(
            wellname, (None, '', '')
        )
        peak_absorbance = _finite_number(metric.get('peak_absorbance'))
        reference_wavelength_nm = _finite_number(
            metric.get('reference_wavelength_nm')
        )
        row = {
            'wellname': wellname,
            'condition_id': _text(metric.get('condition_id')),
            'condition_type': _text(metric.get('condition_type')),
            'batch_number': metric.get('batch_number', ''),
            'reaction_number': metric.get('reaction_number', ''),
            'metric_status': metric_status,
            'metric_qc_status': qc_status or metric_status,
            'metric_qc_reason': qc_reason or _text(metric.get('metric_error')),
            'loss_rate_absorbance_per_s': _finite_number(
                metric.get('loss_rate_absorbance_per_s')
            ),
            'loss_rate_qc_included': metric_status == 'eligible',
            'peak_absorbance': peak_absorbance,
            'reference_wavelength_nm': reference_wavelength_nm,
            'reference_peak_qc_included': (
                peak_absorbance is not None and reference_wavelength_nm is not None
            ),
            'metric_input_scan_count': _integer_or_none(
                metric.get('metric_input_scan_count')
            ) or 0,
            'raw_trajectory_scan_count': _integer_or_none(
                metric.get('raw_trajectory_scan_count')
            ) or 0,
            'condition_eligible_well_count': _integer_or_none(
                summary.get('eligible_well_count')
            ) or 0,
            'condition_total_well_count': _integer_or_none(
                summary.get('total_well_count')
            ) or 0,
            'condition_stability_status': _text(
                summary.get('condition_stability_status')
            ),
            'condition_reference_peak_eligible_well_count': (
                _integer_or_none(summary.get(
                    'reference_peak_eligible_well_count'
                )) or 0
            ),
            'condition_signal_status': _text(summary.get('condition_signal_status')),
            'condition_loss_rate_mean_absorbance_per_s': _finite_number(
                summary.get('condition_loss_rate_mean_absorbance_per_s')
            ),
            'condition_loss_rate_sample_sd_absorbance_per_s': _finite_number(
                summary.get('condition_loss_rate_sample_sd_absorbance_per_s')
            ),
            'condition_reference_peak_absorbance_mean': _finite_number(
                summary.get('condition_reference_peak_absorbance_mean')
            ),
            'condition_reference_peak_absorbance_sample_sd': _finite_number(
                summary.get('condition_reference_peak_absorbance_sample_sd')
            ),
        }
        replicate_rows.append(row)
    replicate_rows.sort(key=_well_sort_key)

    trajectory_by_pair = {}
    for original_row in records.get('trajectory_rows', []) or []:
        if not isinstance(original_row, dict):
            continue
        scan_id = _text(original_row.get('raw_scan_id'))
        wellname = _text(original_row.get('wellname'))
        if scan_id and wellname and (scan_id, wellname) not in trajectory_by_pair:
            trajectory_by_pair[(scan_id, wellname)] = dict(original_row)

    scans = {}
    for original_timing in records.get('timing_rows', []) or []:
        if not isinstance(original_timing, dict):
            continue
        timing = dict(original_timing)
        scan_id = _text(timing.get('raw_scan_id'))
        wellname = _text(timing.get('wellname'))
        if not scan_id or not wellname:
            continue
        scan = scans.setdefault(scan_id, {
            'raw_scan_id': scan_id,
            'rows': [],
            'wellnames': set(),
        })
        if wellname in scan['wellnames']:
            continue
        scan['wellnames'].add(wellname)
        scan['rows'].append(timing)

    first_scan_by_well = {}
    for scan in scans.values():
        for timing in scan['rows']:
            started_at = _parse_utc(timing.get('scan_started_at_utc'))
            trigger_at = _parse_utc(timing.get('trigger_completed_at_utc'))
            if started_at is None or trigger_at is None:
                continue
            wellname = _text(timing.get('wellname'))
            current = first_scan_by_well.get(wellname)
            if current is None or started_at < current['started_at']:
                first_scan_by_well[wellname] = {
                    'raw_scan_id': _text(timing.get('raw_scan_id')),
                    'started_at': started_at,
                    'delay_s': (started_at - trigger_at).total_seconds(),
                }

    throughput_rows = []
    for scan in scans.values():
        timing_rows = scan['rows']
        starts = [_parse_utc(row.get('scan_started_at_utc')) for row in timing_rows]
        starts = [value for value in starts if value is not None]
        completions = [
            _parse_utc(row.get('scan_completed_at_utc')) for row in timing_rows
        ]
        completions = [value for value in completions if value is not None]
        started_at = min(starts) if starts else None
        completed_at = max(completions) if completions else None
        durations = []
        if started_at is not None and completed_at is not None:
            duration = (completed_at - started_at).total_seconds()
            if duration >= 0.0:
                durations.append(duration)
        reasons = sorted({_text(row.get('observation_reason')) for row in timing_rows})
        cadence_delays = [
            _finite_number(row.get('observed_cadence_delay_s'))
            for row in timing_rows
        ]
        cadence_delays = [value for value in cadence_delays if value is not None]
        cadence_statuses = [
            _text(row.get('cadence_status')) for row in timing_rows
            if _text(row.get('cadence_status'))
        ]
        eligible_count = sum(
            bool(trajectory_by_pair.get((scan['raw_scan_id'], wellname), {}).get(
                'metric_eligible_interval'
            )) for wellname in scan['wellnames']
        )
        first_delays = [
            record['delay_s'] for wellname, record in first_scan_by_well.items()
            if record['raw_scan_id'] == scan['raw_scan_id']
        ]
        throughput_rows.append({
            'raw_scan_id': scan['raw_scan_id'],
            'scan_started_at_utc': started_at.isoformat() if started_at else '',
            'scan_completed_at_utc': completed_at.isoformat() if completed_at else '',
            'scan_duration_s': durations[0] if durations else None,
            'observation_reason': reasons[0] if len(reasons) == 1 else (
                'mixed_or_unavailable'
            ),
            'active_well_count': len(scan['wellnames']),
            'decision_eligible_well_count': eligible_count,
            'audit_only_well_count': len(scan['wellnames']) - eligible_count,
            'cadenced_well_count': len(cadence_delays),
            'cadence_delay_mean_s': _mean(cadence_delays),
            'cadence_delay_min_s': min(cadence_delays) if cadence_delays else None,
            'cadence_delay_max_s': max(cadence_delays) if cadence_delays else None,
            'cadence_met_or_early_well_count': cadence_statuses.count(
                'met_or_early'
            ),
            'cadence_deferred_well_count': cadence_statuses.count(
                'deferred_by_serial_reader'
            ),
            'cadence_initial_or_missing_predecessor_well_count': (
                cadence_statuses.count('initial_or_missing_predecessor')
            ),
            'first_observation_well_count': len(first_delays),
            'trigger_to_first_scan_delay_mean_s': _mean(first_delays),
            'trigger_to_first_scan_delay_min_s': (
                min(first_delays) if first_delays else None
            ),
            'trigger_to_first_scan_delay_max_s': (
                max(first_delays) if first_delays else None
            ),
        })
    throughput_rows.sort(key=lambda row: (
        _parse_utc(row.get('scan_started_at_utc')) is None,
        _parse_utc(row.get('scan_started_at_utc')) or datetime.datetime.max.replace(
            tzinfo=datetime.timezone.utc
        ),
        row['raw_scan_id'],
    ))
    for index, row in enumerate(throughput_rows, 1):
        row['scan_ordinal'] = index

    return {
        'replicate_rows': replicate_rows,
        'throughput_rows': throughput_rows,
    }
