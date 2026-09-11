'''Pure, audit-first reporting primitives for Auto optical-stability mode.

Stage 12D turns the append-only Stage-12B/C manifest plus immutable,
blank-corrected raw spectra into per-well trajectories and condition-level
summaries.  This module deliberately has no controller, reader, robot,
filesystem, pandas, or optimizer dependency: it cannot change an experiment,
the fitted GP, or recipe selection.

The reader is serial.  A scan whose *start* was inside a well's configured
window can finish after it.  The manifest preserves that fact, while stability
metrics conservatively use only intervals wholly inside the window.  The
resulting timing audit therefore describes observed cadence; it never claims a
fixed cadence that the reader did not achieve.
'''

from __future__ import division

import datetime
import json
import math

from auto_stability import (
    AutoStabilityValidationError,
    METRIC_STATUS_ELIGIBLE,
    aggregate_condition_stability_metrics,
    compute_stability_metrics,
)


TRAJECTORY_WINDOW_FULLY_WITHIN = 'fully_within_window'
TRAJECTORY_WINDOW_SPANS_END = 'spans_window_end'
TRAJECTORY_WINDOW_STARTS_AFTER = 'starts_after_window'
TRAJECTORY_WINDOW_INVALID_INTERVAL = 'invalid_reader_interval'
TRAJECTORY_WINDOW_MISSING_TIMING = 'missing_reader_timing'
TRAJECTORY_SPECTRUM_AVAILABLE = 'available'
TRAJECTORY_SPECTRUM_MISSING = 'raw_spectrum_unavailable'


def _parse_utc_timestamp(value):
    '''Return a timezone-aware UTC datetime or ``None`` for an empty value.'''
    text = '' if value is None else str(value).strip()
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


def _serialize_utc_timestamp(timestamp):
    return '' if timestamp is None else timestamp.isoformat()


def _as_positive_float(value, name):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityValidationError(
            '{} must be a finite positive number.'.format(name)
        )
    if not math.isfinite(numeric) or numeric <= 0:
        raise AutoStabilityValidationError(
            '{} must be a finite positive number.'.format(name)
        )
    return numeric


def _split_wellnames(value):
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value or '').split(';')
    return [str(item).strip() for item in values if str(item).strip()]


def _json_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or str(value).strip() == '':
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _finite_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _condition_index(condition_rows):
    '''Map each physical replicate sample name to immutable condition metadata.'''
    index = {}
    for ordinal, row in enumerate(condition_rows or []):
        if not isinstance(row, dict):
            continue
        # Imported conditions belong to a different physical run/plate and
        # cannot have a current-run trigger trajectory.  They remain part of
        # the ordinary λmax history but must not be mislabeled as missing
        # stability replicates in this run's condition summary.
        executed_value = row.get('executed_in_current_run', True)
        if executed_value is False or str(executed_value).strip().lower() in (
                'false', '0', 'no'):
            continue
        samples = _json_list(row.get('replicate_sample_names'))
        if not samples:
            continue
        identity = '|'.join([
            str(row.get('origin_run_directory', '')).strip(),
            str(row.get('batch_number', '')).strip(),
            str(row.get('reaction_number', ordinal)).strip(),
        ])
        metadata = {
            'condition_id': identity,
            'condition_type': row.get('condition_type', ''),
            'batch_number': row.get('batch_number', ''),
            'reaction_number': row.get('reaction_number', ordinal),
            'origin_run_directory': row.get('origin_run_directory', ''),
        }
        for sample in samples:
            sample_name = str(sample).strip()
            if sample_name and sample_name not in index:
                index[sample_name] = metadata
    return index


def _scan_interval_status(trigger_at, started_at, completed_at, window_s):
    if trigger_at is None or started_at is None or completed_at is None:
        return TRAJECTORY_WINDOW_MISSING_TIMING
    if completed_at < started_at:
        return TRAJECTORY_WINDOW_INVALID_INTERVAL
    window_end = trigger_at + datetime.timedelta(seconds=window_s)
    if started_at >= window_end:
        return TRAJECTORY_WINDOW_STARTS_AFTER
    if completed_at > window_end:
        return TRAJECTORY_WINDOW_SPANS_END
    return TRAJECTORY_WINDOW_FULLY_WITHIN


def _normalized_spectrum(record):
    '''Return {wavelength_nm: absorbance} from a controller-supplied spectrum.'''
    if not isinstance(record, dict):
        return {}
    source = record.get('spectrum_by_wavelength_nm', {})
    if not isinstance(source, dict):
        return {}
    normalized = {}
    for wavelength, absorbance in source.items():
        wavelength_nm = _finite_float(wavelength)
        absorbance_value = _finite_float(absorbance)
        if wavelength_nm is not None and absorbance_value is not None:
            normalized[wavelength_nm] = absorbance_value
    return normalized


def _lambda_max_from_spectrum(spectrum):
    if not spectrum:
        return None
    return float(max(spectrum, key=spectrum.get))


def _absorbance_at_reference(spectrum, reference_wavelength_nm):
    if not spectrum or reference_wavelength_nm is None:
        return None
    if reference_wavelength_nm in spectrum:
        return spectrum[reference_wavelength_nm]
    nearest_wavelength = min(
        spectrum,
        key=lambda wavelength: abs(wavelength - reference_wavelength_nm)
    )
    # The normal Auto UV-Vis reader emits 1 nm samples. Do not interpolate or
    # silently substitute a distant wavelength if a malformed file omits the
    # fixed reference measurement.
    if abs(nearest_wavelength - reference_wavelength_nm) > 0.01:
        return None
    return spectrum[nearest_wavelength]


def build_stability_reporting_records(
        manifest_rows,
        spectra_by_raw_scan,
        condition_rows,
        observation_window_s,
        scan_interval_s,
        min_peak_absorbance):
    '''Build audit rows, trajectories, metrics, and condition summaries.

    ``spectra_by_raw_scan`` is keyed by raw scan ID, then logical well name.
    Each well record must contain ``spectrum_by_wavelength_nm``.  Spectra are
    supplied after the controller has applied the exact legacy Auto blank
    correction; this pure layer never rereads or changes a raw reader file.
    '''
    window_s = _as_positive_float(
        observation_window_s, 'auto_stability_observation_window_s'
    )
    interval_s = _as_positive_float(
        scan_interval_s, 'auto_stability_scan_interval_s'
    )
    minimum_peak = _as_positive_float(
        min_peak_absorbance, 'auto_stability_min_peak_absorbance'
    )
    manifest_rows = [row for row in (manifest_rows or []) if isinstance(row, dict)]
    spectra_by_raw_scan = spectra_by_raw_scan or {}
    conditions_by_well = _condition_index(condition_rows)

    trigger_by_well = {}
    for row in manifest_rows:
        if row.get('event_type') != 'trigger_transfer_completed':
            continue
        wellname = str(row.get('wellname', '')).strip()
        completed_at = _parse_utc_timestamp(
            row.get('trigger_transfer_completion_observed_at_utc')
        )
        if wellname and completed_at is not None:
            trigger_by_well[wellname] = completed_at

    timing_rows = []
    trajectory_rows = []
    qc_rows = []
    observations_by_well = {}
    last_completed_by_well = {}

    completed_scan_rows = [
        row for row in manifest_rows
        if row.get('event_type') == 'raw_scan_completed'
    ]
    completed_scan_rows.sort(
        key=lambda row: (
            _parse_utc_timestamp(row.get('scan_started_at_utc'))
            or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc),
            str(row.get('raw_scan_id', '')),
        )
    )

    for scan_row in completed_scan_rows:
        scan_id = str(scan_row.get('raw_scan_id', '')).strip()
        started_at = _parse_utc_timestamp(scan_row.get('scan_started_at_utc'))
        completed_at = _parse_utc_timestamp(scan_row.get('scan_completed_at_utc'))
        reason = str(scan_row.get('observation_reason', '')).strip()
        for wellname in _split_wellnames(scan_row.get('active_wellnames')):
            trigger_at = trigger_by_well.get(wellname)
            window_status = _scan_interval_status(
                trigger_at, started_at, completed_at, window_s
            )
            requested_deadline = None
            if reason == 'cadenced_active_set':
                previous = last_completed_by_well.get(wellname)
                if previous is not None:
                    requested_deadline = previous + datetime.timedelta(
                        seconds=interval_s
                    )
            cadence_delay_s = None
            if requested_deadline is not None and started_at is not None:
                cadence_delay_s = (
                    started_at - requested_deadline
                ).total_seconds()
            timing_rows.append({
                'raw_scan_id': scan_id,
                'wellname': wellname,
                'observation_reason': reason,
                'trigger_completed_at_utc': _serialize_utc_timestamp(trigger_at),
                'scan_started_at_utc': _serialize_utc_timestamp(started_at),
                'scan_completed_at_utc': _serialize_utc_timestamp(completed_at),
                'window_interval_status': window_status,
                'requested_cadence_deadline_utc': _serialize_utc_timestamp(
                    requested_deadline
                ),
                'observed_cadence_delay_s': cadence_delay_s,
                'cadence_status': (
                    'not_cadenced' if reason != 'cadenced_active_set'
                    else ('initial_or_missing_predecessor'
                          if requested_deadline is None
                          else ('met_or_early' if cadence_delay_s <= 0
                                else 'deferred_by_serial_reader'))
                ),
                'mixing_mode': scan_row.get('mixing_mode', ''),
                'shake_duration_s': scan_row.get('shake_duration_s', ''),
            })
            if completed_at is not None:
                last_completed_by_well[wellname] = completed_at

            spectrum_record = (
                spectra_by_raw_scan.get(scan_id, {}).get(wellname, {})
            )
            spectrum = _normalized_spectrum(spectrum_record)
            lambda_max_nm = _lambda_max_from_spectrum(spectrum)
            elapsed_s = (
                None if started_at is None or trigger_at is None
                else (started_at - trigger_at).total_seconds()
            )
            trajectory_row = {
                'raw_scan_id': scan_id,
                'raw_scan_relative_path': scan_row.get(
                    'raw_scan_relative_path', ''
                ),
                'wellname': wellname,
                'condition_id': conditions_by_well.get(
                    wellname, {}
                ).get('condition_id', ''),
                'trigger_completed_at_utc': _serialize_utc_timestamp(trigger_at),
                'scan_started_at_utc': _serialize_utc_timestamp(started_at),
                'scan_completed_at_utc': _serialize_utc_timestamp(completed_at),
                'elapsed_s_from_trigger_to_scan_start': elapsed_s,
                'window_interval_status': window_status,
                'metric_eligible_interval': (
                    window_status == TRAJECTORY_WINDOW_FULLY_WITHIN
                ),
                'spectrum_status': (
                    TRAJECTORY_SPECTRUM_AVAILABLE if spectrum
                    else TRAJECTORY_SPECTRUM_MISSING
                ),
                'lambda_max_nm': lambda_max_nm,
                # Filled after the fixed per-well reference is known.
                'reference_wavelength_nm': None,
                'reference_absorbance': None,
            }
            trajectory_rows.append(trajectory_row)
            if spectrum:
                trajectory_row['_spectrum'] = spectrum
            if not spectrum:
                qc_rows.append({
                    'wellname': wellname,
                    'raw_scan_id': scan_id,
                    'qc_status': 'raw_spectrum_unavailable',
                    'qc_reason': (
                        'No controller-loaded blank-corrected spectrum was '
                        'available for this manifest scan/well pair.'
                    ),
                })
            elif window_status != TRAJECTORY_WINDOW_FULLY_WITHIN:
                qc_rows.append({
                    'wellname': wellname,
                    'raw_scan_id': scan_id,
                    'qc_status': 'scan_interval_excluded_from_metric',
                    'qc_reason': (
                        'The reader interval was retained in the raw '
                        'trajectory audit but was not wholly inside this '
                        'well\'s configured observation window.'
                    ),
                })

    by_well = {}
    for row in trajectory_rows:
        by_well.setdefault(row['wellname'], []).append(row)

    well_metrics = []
    for wellname, rows in sorted(by_well.items()):
        rows.sort(key=lambda row: (
            math.inf if row['elapsed_s_from_trigger_to_scan_start'] is None
            else row['elapsed_s_from_trigger_to_scan_start'],
            row['raw_scan_id'],
        ))
        metric_rows = [
            row for row in rows
            if row['metric_eligible_interval'] and '_spectrum' in row
            and row['elapsed_s_from_trigger_to_scan_start'] is not None
        ]
        reference_wavelength_nm = next(
            (row['lambda_max_nm'] for row in metric_rows
             if row['lambda_max_nm'] is not None),
            None
        )
        for row in rows:
            row['reference_wavelength_nm'] = reference_wavelength_nm
            row['reference_absorbance'] = _absorbance_at_reference(
                row.get('_spectrum', {}), reference_wavelength_nm
            )
            row.pop('_spectrum', None)
        metric_observations = [
            {
                'timestamp_s': row['elapsed_s_from_trigger_to_scan_start'],
                'reference_absorbance': row['reference_absorbance'],
                'lambda_max_nm': row['lambda_max_nm'],
            }
            for row in metric_rows
            if row['reference_absorbance'] is not None
        ]
        try:
            metrics = compute_stability_metrics(
                metric_observations,
                minimum_peak,
                trigger_timestamp_s=0.0,
                observation_window_s=window_s,
            )
        except AutoStabilityValidationError as exc:
            metrics = {
                'status': 'invalid_trajectory',
                'loss_rate_absorbance_per_s': None,
                'observation_count': len(metric_observations),
                'observation_count_within_window': len(metric_observations),
                'reference_wavelength_nm': reference_wavelength_nm,
                'metric_error': str(exc),
            }
        condition = conditions_by_well.get(wellname, {})
        metric_row = dict(metrics)
        metric_row.update({
            'wellname': wellname,
            'condition_id': condition.get('condition_id', ''),
            'condition_type': condition.get('condition_type', ''),
            'batch_number': condition.get('batch_number', ''),
            'reaction_number': condition.get('reaction_number', ''),
            'metric_input_scan_count': len(metric_observations),
            'raw_trajectory_scan_count': len(rows),
        })
        well_metrics.append(metric_row)
        qc_rows.append({
            'wellname': wellname,
            'raw_scan_id': '',
            'qc_status': metrics.get('status', 'invalid_trajectory'),
            'qc_reason': (
                'Stability metric is eligible only when the complete reader '
                'interval is inside the configured observation window.'
            ),
        })

    # A condition-level result must distinguish a partial replicate trajectory
    # from a complete condition with no eligible loss rate. Add an explicit
    # audit metric for every expected replicate that never produced a raw
    # stability trajectory; do not silently reduce the denominator.
    metric_by_well = {metric['wellname']: metric for metric in well_metrics}
    expected_wells_by_condition = {}
    condition_metadata_by_id = {}
    for wellname, condition in conditions_by_well.items():
        condition_id = condition.get('condition_id', '')
        if condition_id:
            expected_wells_by_condition.setdefault(condition_id, []).append(
                wellname
            )
            condition_metadata_by_id[condition_id] = condition
    for condition_id, expected_wells in expected_wells_by_condition.items():
        for wellname in expected_wells:
            if wellname in metric_by_well:
                continue
            condition = condition_metadata_by_id[condition_id]
            missing_metric = {
                'status': 'missing_stability_trajectory',
                'loss_rate_absorbance_per_s': None,
                'observation_count': 0,
                'observation_count_within_window': 0,
                'reference_wavelength_nm': None,
                'wellname': wellname,
                'condition_id': condition_id,
                'condition_type': condition.get('condition_type', ''),
                'batch_number': condition.get('batch_number', ''),
                'reaction_number': condition.get('reaction_number', ''),
                'metric_input_scan_count': 0,
                'raw_trajectory_scan_count': 0,
            }
            well_metrics.append(missing_metric)
            qc_rows.append({
                'wellname': wellname,
                'raw_scan_id': '',
                'qc_status': 'missing_stability_trajectory',
                'qc_reason': (
                    'This condition-level replicate had no manifest-linked '
                    'raw stability trajectory.'
                ),
            })

    metrics_by_condition = {}
    metadata_by_condition = {}
    for metric in well_metrics:
        condition_id = metric.get('condition_id', '')
        if not condition_id:
            continue
        metrics_by_condition.setdefault(condition_id, []).append(metric)
        metadata_by_condition[condition_id] = metric
    condition_summaries = []
    for condition_id, metrics in sorted(metrics_by_condition.items()):
        summary = aggregate_condition_stability_metrics(metrics)
        metadata = metadata_by_condition[condition_id]
        summary.update({
            'condition_id': condition_id,
            'condition_type': metadata.get('condition_type', ''),
            'batch_number': metadata.get('batch_number', ''),
            'reaction_number': metadata.get('reaction_number', ''),
            'condition_stability_status': (
                'complete' if summary['eligible_well_count'] ==
                summary['total_well_count'] else (
                    'partial' if summary['eligible_well_count'] > 0
                    else 'no_eligible_wells'
                )
            ),
        })
        condition_summaries.append(summary)

    return {
        'timing_rows': timing_rows,
        'trajectory_rows': trajectory_rows,
        'well_metrics': well_metrics,
        'condition_summaries': condition_summaries,
        'qc_rows': qc_rows,
    }
