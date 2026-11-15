'''Offline, reproducible Stage 14G2 stability-metric sensitivity study.

This module is intentionally outside the controller and optimizer pathways. It
reads only an explicitly supplied, completed ``stability_well_trajectories.csv``
export, generates deterministic synthetic comparison trajectories, and writes a
clearly labeled local study package.  It never accesses a workbook, robot, Pi,
reader, packet client, or live run state.
'''

from __future__ import division

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from auto_stability import METRIC_STATUS_ELIGIBLE
from auto_stability_metric_study import compare_stage14g2_metrics


STUDY_SCHEMA_VERSION = 'stage14g2-v1'
EVIDENCE_SOURCE_SYNTHETIC = 'synthetic'
EVIDENCE_SOURCE_WATER_TIMING = 'water_timing'
EVIDENCE_SOURCE_CHEMISTRY = 'chemistry'
ALLOWED_EVIDENCE_SOURCES = frozenset((
    EVIDENCE_SOURCE_SYNTHETIC,
    EVIDENCE_SOURCE_WATER_TIMING,
    EVIDENCE_SOURCE_CHEMISTRY,
))

METRIC_ORDER = (
    'legacy_tail_minimum_v1',
    'fixed_horizon_raw_loss_rate',
    'normalized_retention',
    'log_loss_rate',
    'robust_post_peak_slope',
)
LOWER_IS_MORE_STABLE = frozenset((
    'legacy_tail_minimum_v1',
    'fixed_horizon_raw_loss_rate',
    'log_loss_rate',
    'robust_post_peak_slope',
))


class AutoStabilityMetricStudyPackageError(ValueError):
    '''Raised when an offline study input or output location is ambiguous.'''


def _finite(value, name):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityMetricStudyPackageError(
            '{} must be a finite number.'.format(name)
        )
    if not math.isfinite(numeric_value):
        raise AutoStabilityMetricStudyPackageError(
            '{} must be a finite number.'.format(name)
        )
    return numeric_value


def _positive(value, name):
    numeric_value = _finite(value, name)
    if numeric_value <= 0.0:
        raise AutoStabilityMetricStudyPackageError(
            '{} must be positive.'.format(name)
        )
    return numeric_value


def _parse_boolean(value, name):
    normalized = str(value).strip().lower()
    if normalized in ('true', '1', 'yes'):
        return True
    if normalized in ('false', '0', 'no', ''):
        return False
    raise AutoStabilityMetricStudyPackageError(
        '{} must be a Boolean-compatible value.'.format(name)
    )


def _validate_evidence_source(value):
    normalized = str(value).strip().lower()
    if normalized not in ALLOWED_EVIDENCE_SOURCES:
        raise AutoStabilityMetricStudyPackageError(
            'evidence_source must be one of {}.'.format(
                ', '.join(sorted(ALLOWED_EVIDENCE_SOURCES))
            )
        )
    return normalized


def _trajectory_id(row, source_path):
    condition_id = str(row.get('condition_id') or '').strip()
    wellname = str(row.get('wellname') or '').strip()
    if not wellname:
        raise AutoStabilityMetricStudyPackageError(
            '{} has a blank wellname.'.format(source_path)
        )
    return '{}|{}'.format(condition_id or 'unlabeled-condition', wellname)


def read_saved_trajectory_export(path, evidence_source=None):
    '''Read a completed trajectory export without touching a live run.

    DEBUG-only rows may be labeled only ``water_timing``.  Non-DEBUG rows
    require an explicit evidence source, preventing accidental presentation of
    an unlabeled chemistry dataset as a scientific conclusion.
    '''
    path = Path(path)
    required_columns = frozenset((
        'wellname',
        'condition_id',
        'elapsed_s_from_trigger_to_scan_start',
        'reference_absorbance',
        'debug_only',
    ))
    try:
        handle = path.open('r', newline='', encoding='utf-8')
    except OSError as exc:
        raise AutoStabilityMetricStudyPackageError(
            'Unable to read completed trajectory export {}: {}.'.format(
                path, exc
            )
        )
    with handle:
        reader = csv.DictReader(handle)
        fieldnames = frozenset(reader.fieldnames or ())
        missing = sorted(required_columns.difference(fieldnames))
        if missing:
            raise AutoStabilityMetricStudyPackageError(
                '{} is missing required trajectory columns: {}.'.format(
                    path, ', '.join(missing)
                )
            )
        grouped = defaultdict(list)
        metadata = {}
        for line_number, row in enumerate(reader, start=2):
            trajectory_id = _trajectory_id(row, path)
            is_debug = _parse_boolean(row.get('debug_only'),
                                      'debug_only at line {}'.format(line_number))
            if is_debug:
                row_source = EVIDENCE_SOURCE_WATER_TIMING
            else:
                if evidence_source is None:
                    raise AutoStabilityMetricStudyPackageError(
                        '{} contains non-DEBUG evidence; provide an explicit '
                        'evidence_source of chemistry.'.format(path)
                    )
                row_source = _validate_evidence_source(evidence_source)
                if row_source != EVIDENCE_SOURCE_CHEMISTRY:
                    raise AutoStabilityMetricStudyPackageError(
                        'Non-DEBUG trajectory evidence must be labeled chemistry.'
                    )
            if evidence_source is not None:
                requested_source = _validate_evidence_source(evidence_source)
                if requested_source != row_source:
                    raise AutoStabilityMetricStudyPackageError(
                        '{} evidence label conflicts with debug_only at line {}.'
                        .format(path, line_number)
                    )
            grouped[trajectory_id].append({
                'timestamp_s': _finite(
                    row.get('elapsed_s_from_trigger_to_scan_start'),
                    'elapsed_s_from_trigger_to_scan_start at line {}'.format(
                        line_number
                    )
                ),
                'reference_absorbance': _finite(
                    row.get('reference_absorbance'),
                    'reference_absorbance at line {}'.format(line_number)
                ),
            })
            prior_metadata = metadata.get(trajectory_id)
            if (prior_metadata is not None
                    and prior_metadata['evidence_source'] != row_source):
                raise AutoStabilityMetricStudyPackageError(
                    '{} mixes DEBUG-only and non-DEBUG evidence for {}.'
                    .format(path, trajectory_id)
                )
            metadata[trajectory_id] = {
                'trajectory_id': trajectory_id,
                'condition_id': str(row.get('condition_id') or '').strip(),
                'wellname': str(row.get('wellname') or '').strip(),
                'evidence_source': row_source,
                'input_path': str(path),
            }
    records = []
    for trajectory_id in sorted(grouped):
        observations = sorted(
            grouped[trajectory_id], key=lambda observation: observation['timestamp_s']
        )
        timestamps = [observation['timestamp_s'] for observation in observations]
        if len(timestamps) != len(set(timestamps)):
            raise AutoStabilityMetricStudyPackageError(
                '{} has duplicate timestamp(s) for {}.'.format(path, trajectory_id)
            )
        record = dict(metadata[trajectory_id])
        record['observations'] = observations
        record['scenario_id'] = None
        record['comparison_group'] = None
        record['true_stability_rank'] = None
        record['trajectory_kind'] = 'saved_export'
        records.append(record)
    return records


def _schedule(name):
    schedules = {
        'dense': (0.0, 60.0, 120.0, 180.0, 240.0, 300.0),
        'sparse': (0.0, 120.0, 240.0),
        'reader_delay': (0.0, 111.0, 221.0),
        'missing_endpoint': (0.0, 60.0, 360.0),
    }
    return schedules[name]


def _simulate_trajectory(schedule_name, peak_absorbance, decline_kind,
                         decline_rate, peak_time_s, noise_sd, rng):
    '''Build one explicit synthetic trajectory with a known post-peak model.'''
    observations = []
    for timestamp_s in _schedule(schedule_name):
        elapsed_s = timestamp_s - peak_time_s
        if elapsed_s <= 0.0:
            # Deliberately retain a growth phase so peak timing is exercised.
            fraction = max(0.15, 1.0 + elapsed_s / max(peak_time_s, 1.0))
            absorbance = peak_absorbance * fraction
        elif decline_kind == 'linear':
            absorbance = peak_absorbance - decline_rate * elapsed_s
        elif decline_kind == 'exponential':
            absorbance = peak_absorbance * math.exp(-decline_rate * elapsed_s)
        elif decline_kind == 'accelerating':
            absorbance = peak_absorbance * math.exp(
                -decline_rate * elapsed_s
                - 0.0000015 * elapsed_s * elapsed_s
            )
        elif decline_kind == 'flat':
            absorbance = peak_absorbance
        else:
            raise AutoStabilityMetricStudyPackageError(
                'Unsupported synthetic decline_kind {}.'.format(decline_kind)
            )
        observations.append({
            'timestamp_s': timestamp_s,
            'reference_absorbance': max(
                0.000001, absorbance + rng.gauss(0.0, noise_sd)
            ),
        })
    return observations


def build_synthetic_scenarios(seed=20261114, noise_replicates=12):
    '''Return a deterministic, ground-truth-labeled Stage 14G2 scenario set.'''
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AutoStabilityMetricStudyPackageError('seed must be an integer.')
    if (isinstance(noise_replicates, bool)
            or not isinstance(noise_replicates, int)
            or noise_replicates < 1):
        raise AutoStabilityMetricStudyPackageError(
            'noise_replicates must be an integer of at least one.'
        )
    rng = random.Random(seed)
    definitions = (
        # paired scenarios: lower true rank is more stable
        ('linear_dense_sparse_a', 'linear_density_a', 'dense', 'linear', .00016, .65, 60.0, .006, 1),
        ('linear_dense_sparse_b', 'linear_density_a', 'sparse', 'linear', .00046, .65, 60.0, .006, 2),
        ('linear_sparse_dense_a', 'linear_density_b', 'sparse', 'linear', .00016, .65, 60.0, .006, 1),
        ('linear_sparse_dense_b', 'linear_density_b', 'dense', 'linear', .00046, .65, 60.0, .006, 2),
        ('exponential_delayed_a', 'exponential_delay', 'reader_delay', 'exponential', .00018, .65, 60.0, .006, 1),
        ('exponential_delayed_b', 'exponential_delay', 'dense', 'exponential', .00052, .65, 60.0, .006, 2),
        ('accelerating_a', 'nonlinear_decline', 'dense', 'accelerating', .00008, .65, 60.0, .006, 1),
        ('accelerating_b', 'nonlinear_decline', 'sparse', 'accelerating', .00024, .65, 60.0, .006, 2),
        ('robust_dense_a', 'robust_dense', 'dense', 'linear', .00016, .65, 60.0, .006, 1),
        ('robust_dense_b', 'robust_dense', 'dense', 'linear', .00046, .65, 60.0, .006, 2),
        # amplitude pair has equal fractional behavior and no truth ranking.
        ('equal_decay_low_amplitude', 'amplitude_confounding', 'dense', 'exponential', .00035, .25, 60.0, .006, None),
        ('equal_decay_high_amplitude', 'amplitude_confounding', 'dense', 'exponential', .00035, .80, 60.0, .006, None),
        ('low_signal', None, 'dense', 'linear', .00005, .05, 60.0, .003, None),
        ('no_decline', None, 'dense', 'flat', 0.0, .65, 60.0, 0.0, None),
        ('missing_endpoint', None, 'missing_endpoint', 'linear', .00035, .65, 60.0, .003, None),
        ('late_peak', None, 'dense', 'exponential', .00035, .65, 120.0, .006, None),
    )
    records = []
    for replicate_index in range(noise_replicates):
        for definition in definitions:
            (base_scenario_id, comparison_group, schedule_name, decline_kind,
             decline_rate, peak_absorbance, peak_time_s, noise_sd,
             true_stability_rank) = definition
            scenario_seed = rng.randrange(0, 2 ** 31)
            scenario_id = '{}__r{:02d}'.format(
                base_scenario_id, replicate_index + 1
            )
            records.append({
                'trajectory_id': 'synthetic|{}'.format(scenario_id),
                'scenario_id': scenario_id,
                'base_scenario_id': base_scenario_id,
                'scenario_seed': scenario_seed,
                'comparison_group': (
                    '{}__r{:02d}'.format(comparison_group, replicate_index + 1)
                    if comparison_group else None
                ),
                'true_stability_rank': true_stability_rank,
                'trajectory_kind': decline_kind,
                'schedule_name': schedule_name,
                'true_decline_rate': decline_rate,
                'peak_absorbance_ground_truth': peak_absorbance,
                'peak_time_ground_truth_s': peak_time_s,
                'noise_sd': noise_sd,
                'condition_id': 'synthetic',
                'wellname': scenario_id,
                'evidence_source': EVIDENCE_SOURCE_SYNTHETIC,
                'input_path': None,
                'observations': _simulate_trajectory(
                    schedule_name, peak_absorbance, decline_kind, decline_rate,
                    peak_time_s, noise_sd, random.Random(scenario_seed)
                ),
            })
    return records


def _metric_value(metric_name, result):
    if result.get('status') != METRIC_STATUS_ELIGIBLE:
        return None
    if metric_name == 'normalized_retention':
        return result.get('retention_fraction')
    if metric_name == 'log_loss_rate':
        return result.get('log_loss_rate_per_s')
    return result.get('loss_rate_absorbance_per_s')


def _direction(metric_name):
    return 'lower_is_more_stable' if metric_name in LOWER_IS_MORE_STABLE else 'higher_is_more_stable'


def evaluate_trajectory_records(records, min_peak_absorbance,
                                post_peak_horizon_s, endpoint_tolerance_s):
    '''Flatten study metric results while retaining complete provenance.'''
    minimum_peak = _positive(min_peak_absorbance, 'min_peak_absorbance')
    horizon_s = _positive(post_peak_horizon_s, 'post_peak_horizon_s')
    tolerance_s = _finite(endpoint_tolerance_s, 'endpoint_tolerance_s')
    if tolerance_s < 0.0:
        raise AutoStabilityMetricStudyPackageError(
            'endpoint_tolerance_s must be nonnegative.'
        )
    rows = []
    for record in records:
        source = _validate_evidence_source(record.get('evidence_source'))
        results = compare_stage14g2_metrics(
            record.get('observations'), minimum_peak, horizon_s, tolerance_s,
            trigger_timestamp_s=0.0,
        )
        for metric_name in METRIC_ORDER:
            result = results[metric_name]
            row = {
                'schema_version': STUDY_SCHEMA_VERSION,
                'evidence_source': source,
                'trajectory_id': record.get('trajectory_id'),
                'scenario_id': record.get('scenario_id') or '',
                'base_scenario_id': record.get('base_scenario_id') or '',
                'scenario_seed': record.get('scenario_seed'),
                'comparison_group': record.get('comparison_group') or '',
                'true_stability_rank': record.get('true_stability_rank'),
                'trajectory_kind': record.get('trajectory_kind') or '',
                'schedule_name': record.get('schedule_name') or '',
                'true_decline_rate': record.get('true_decline_rate'),
                'peak_absorbance_ground_truth': record.get(
                    'peak_absorbance_ground_truth'
                ),
                'noise_sd': record.get('noise_sd'),
                'condition_id': record.get('condition_id') or '',
                'wellname': record.get('wellname') or '',
                'input_path': record.get('input_path') or '',
                'metric_name': metric_name,
                'stability_direction': _direction(metric_name),
                'status': result.get('status'),
                'comparison_value': _metric_value(metric_name, result),
                'observation_count': result.get('observation_count'),
                'peak_time_s': result.get('peak_time_s'),
                'peak_absorbance': result.get('peak_absorbance'),
                'endpoint_time_s': result.get('endpoint_time_s'),
                'endpoint_timing_offset_s': result.get('endpoint_timing_offset_s'),
                'endpoint_absorbance': result.get('endpoint_absorbance'),
                'retention_fraction': result.get('retention_fraction'),
                'fractional_loss': result.get('fractional_loss'),
                'legacy_tail_min_time_s': result.get('tail_min_time_s'),
                'post_peak_horizon_s': horizon_s,
                'endpoint_tolerance_s': tolerance_s,
            }
            rows.append(row)
    return rows


def summarize_metric_rows(rows):
    '''Summarize availability by metric and evidence source without inference.'''
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['evidence_source'], row['metric_name'])].append(row)
    summary = []
    for evidence_source, metric_name in sorted(grouped):
        metric_rows = grouped[(evidence_source, metric_name)]
        status_counts = defaultdict(int)
        values = []
        for row in metric_rows:
            status_counts[row['status']] += 1
            if row['comparison_value'] is not None:
                values.append(float(row['comparison_value']))
        summary.append({
            'schema_version': STUDY_SCHEMA_VERSION,
            'evidence_source': evidence_source,
            'metric_name': metric_name,
            'stability_direction': _direction(metric_name),
            'trajectory_count': len(metric_rows),
            'eligible_count': len(values),
            'eligible_fraction': (
                float(len(values)) / len(metric_rows) if metric_rows else 0.0
            ),
            'mean_comparison_value': (
                sum(values) / len(values) if values else None
            ),
            'status_counts_json': json.dumps(
                dict(sorted(status_counts.items())), sort_keys=True
            ),
        })
    return summary


def summarize_known_rankings(rows):
    '''Compare synthetic paired rankings only where a truth label exists.'''
    groups = defaultdict(list)
    for row in rows:
        if (row['evidence_source'] == EVIDENCE_SOURCE_SYNTHETIC
                and row['comparison_group']
                and row['true_stability_rank'] is not None):
            groups[(row['comparison_group'], row['metric_name'])].append(row)
    summary = []
    grouped_results = defaultdict(lambda: [0, 0])
    for (group_name, metric_name), group_rows in sorted(groups.items()):
        group_rows = sorted(group_rows, key=lambda row: row['true_stability_rank'])
        result = grouped_results[metric_name]
        if len(group_rows) != 2:
            continue
        left, right = group_rows
        left_value = left['comparison_value']
        right_value = right['comparison_value']
        if left_value is None or right_value is None:
            continue
        result[1] += 1
        if metric_name in LOWER_IS_MORE_STABLE:
            correct = left_value < right_value
        else:
            correct = left_value > right_value
        result[0] += int(correct)
    for metric_name in METRIC_ORDER:
        correct, available = grouped_results[metric_name]
        summary.append({
            'schema_version': STUDY_SCHEMA_VERSION,
            'evidence_source': EVIDENCE_SOURCE_SYNTHETIC,
            'metric_name': metric_name,
            'known_truth_pair_count': available,
            'correct_ranking_count': correct,
            'correct_ranking_fraction': (
                float(correct) / available if available else None
            ),
        })
    return summary


def _fieldnames(rows):
    fields = set()
    for row in rows:
        fields.update(row)
    return sorted(fields)


def _write_csv(path, rows):
    fieldnames = _fieldnames(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_markdown_report(path, metadata, summary_rows, ranking_rows):
    lines = [
        '# Stage 14G2 stability-metric sensitivity study',
        '',
        '## Evidence boundary',
        '',
        'This is an offline comparison package. It does not change a controller,',
        'optimizer, spreadsheet, packet, robot, reader, or production response.',
        'Synthetic rows demonstrate estimator behavior only. `water_timing` rows',
        'demonstrate saved scheduler timing only. Neither establishes chemical',
        'nanocrystal stability or selects a production response version.',
        '',
        '## Reproducibility metadata',
        '',
        '```json',
        json.dumps(metadata, sort_keys=True, indent=2),
        '```',
        '',
        '## Metric availability',
        '',
        '| Evidence source | Metric | Eligible / total | Eligible fraction |',
        '| --- | --- | ---: | ---: |',
    ]
    for row in summary_rows:
        lines.append('| {evidence_source} | {metric_name} | {eligible_count} / '
                     '{trajectory_count} | {eligible_fraction:.3f} |'.format(**row))
    lines.extend((
        '',
        '## Known-truth synthetic ranking checks',
        '',
        '| Metric | Correct / available pairs | Correct fraction |',
        '| --- | ---: | ---: |',
    ))
    for row in ranking_rows:
        fraction = row['correct_ranking_fraction']
        fraction_text = 'unavailable' if fraction is None else '{:.3f}'.format(fraction)
        lines.append('| {metric_name} | {correct_ranking_count} / '
                     '{known_truth_pair_count} | {fraction} |'.format(
                         fraction=fraction_text, **row
                     ))
    lines.extend((
        '',
        '## Interpretation boundary',
        '',
        'These outputs support Stage 14G4 planning only. They do not determine',
        'the chemical assessment horizon, timing tolerance, spectral QC limits,',
        'or active optimizer response. Those require the monitor-only chemistry',
        'evidence defined in Stage 14G3.',
        '',
    ))
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def _write_diagnostic_figures(output_dir, rows, summary_rows):
    '''Write source-labeled offline diagnostic figures using a noninteractive backend.'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    metric_display_names = {
        'fixed_horizon_raw_loss_rate': 'Fixed-horizon\nraw loss rate',
        'legacy_tail_minimum_v1': 'Legacy tail\nminimum v1',
        'normalized_retention': 'Normalized\nretention',
        'log_loss_rate': 'Log-loss\nrate',
        'robust_post_peak_slope': 'Robust post-peak\nslope',
    }
    synthetic = [row for row in rows if row['evidence_source'] == EVIDENCE_SOURCE_SYNTHETIC]
    amplitude_rows = [
        row for row in synthetic
        if row['comparison_group'].startswith('amplitude_confounding__')
        and row['comparison_value'] is not None
    ]
    metric_names = (
        'fixed_horizon_raw_loss_rate', 'normalized_retention', 'log_loss_rate'
    )
    fig, axes = plt.subplots(1, len(metric_names), figsize=(12, 3.9))
    for axis, metric_name in zip(axes, metric_names):
        metric_rows = [row for row in amplitude_rows if row['metric_name'] == metric_name]
        axis.plot(
            [row['peak_absorbance_ground_truth'] for row in metric_rows],
            [row['comparison_value'] for row in metric_rows],
            marker='o', linestyle='none', alpha=0.7,
        )
        axis.set_xlabel('Known peak absorbance')
        axis.set_title(metric_display_names[metric_name], fontsize=10)
    axes[0].set_ylabel('Metric value (metric-specific units)')
    fig.suptitle('Synthetic amplitude sensitivity; not chemistry evidence')
    fig.tight_layout()
    fig.savefig(output_dir / 'amplitude_confounding.png', dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    schedule_rows = [
        row for row in synthetic
        if row['metric_name'] == 'legacy_tail_minimum_v1'
        and row['comparison_value'] is not None
    ]
    schedule_names = sorted({row['schedule_name'] for row in schedule_rows})
    values = [
        [row['comparison_value'] for row in schedule_rows if row['schedule_name'] == name]
        for name in schedule_names
    ]
    axis.boxplot(values)
    axis.set_xticklabels(schedule_names)
    axis.set_xlabel('Synthetic scan schedule')
    axis.set_ylabel('Legacy tail-minimum loss rate (Abs/s)')
    axis.set_title('Synthetic scan-density sensitivity; not chemistry evidence')
    fig.tight_layout()
    fig.savefig(output_dir / 'scan_density_sensitivity.png', dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    status_counts = defaultdict(int)
    for row in synthetic:
        if row['metric_name'] in (
                'fixed_horizon_raw_loss_rate', 'normalized_retention',
                'log_loss_rate'):
            status_counts[row['status']] += 1
    labels = sorted(status_counts)
    axis.bar(range(len(labels)), [status_counts[label] for label in labels])
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=25, ha='right')
    axis.set_ylabel('Metric rows')
    axis.set_title('Missing-endpoint and status behavior; synthetic evidence')
    fig.tight_layout()
    fig.savefig(output_dir / 'missing_endpoint_behavior.png', dpi=160)
    plt.close(fig)

    evidence_sources = sorted({row['evidence_source'] for row in summary_rows})
    fig, axes = plt.subplots(1, len(evidence_sources), figsize=(11, 4.5),
                             sharey=True)
    if len(evidence_sources) == 1:
        axes = (axes,)
    for axis, evidence_source in zip(axes, evidence_sources):
        source_rows = [
            row for row in summary_rows
            if row['evidence_source'] == evidence_source
        ]
        labels = [metric_display_names[row['metric_name']]
                  for row in source_rows]
        axis.bar(range(len(source_rows)), [
            row['eligible_fraction'] for row in source_rows
        ])
        axis.set_xticks(range(len(source_rows)))
        axis.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
        axis.set_title(evidence_source)
        axis.set_ylim(0.0, 1.05)
    axes[0].set_ylabel('Eligible fraction')
    fig.suptitle('Estimator availability by evidence source')
    fig.tight_layout()
    fig.savefig(output_dir / 'estimator_availability.png', dpi=160)
    plt.close(fig)


def _content_hash(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def write_study_package(output_dir, saved_trajectory_path=None,
                        saved_evidence_source=None, seed=20261114,
                        min_peak_absorbance=0.1,
                        post_peak_horizon_s=120.0,
                        endpoint_tolerance_s=60.0):
    '''Create one non-overwriting Stage 14G2 package in an empty directory.'''
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AutoStabilityMetricStudyPackageError(
            'Refusing to overwrite nonempty study output directory {}.'.format(
                output_dir
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_records = build_synthetic_scenarios(seed=seed)
    saved_records = []
    if saved_trajectory_path is not None:
        saved_records = read_saved_trajectory_export(
            saved_trajectory_path, evidence_source=saved_evidence_source
        )
    records = synthetic_records + saved_records
    metric_rows = evaluate_trajectory_records(
        records, min_peak_absorbance, post_peak_horizon_s, endpoint_tolerance_s
    )
    summary_rows = summarize_metric_rows(metric_rows)
    ranking_rows = summarize_known_rankings(metric_rows)
    metadata = {
        'study_schema_version': STUDY_SCHEMA_VERSION,
        'seed': seed,
        'min_peak_absorbance': min_peak_absorbance,
        'post_peak_horizon_s': post_peak_horizon_s,
        'endpoint_tolerance_s': endpoint_tolerance_s,
        'synthetic_trajectory_count': len(synthetic_records),
        'saved_trajectory_count': len(saved_records),
        'saved_trajectory_path': (
            str(saved_trajectory_path) if saved_trajectory_path is not None
            else None
        ),
        'metric_rows_sha256': _content_hash(metric_rows),
    }
    _write_csv(output_dir / 'scenario_metric_results.csv', metric_rows)
    _write_csv(output_dir / 'metric_availability_summary.csv', summary_rows)
    _write_csv(output_dir / 'known_truth_ranking_summary.csv', ranking_rows)
    (output_dir / 'study_metadata.json').write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + '\n', encoding='utf-8'
    )
    _write_markdown_report(
        output_dir / 'stage14g2_metric_study_report.md', metadata,
        summary_rows, ranking_rows
    )
    _write_diagnostic_figures(output_dir, metric_rows, summary_rows)
    return metadata


def _parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description='Create an offline Stage 14G2 stability-metric study package.'
    )
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--saved-trajectories')
    parser.add_argument(
        '--saved-evidence-source', choices=(EVIDENCE_SOURCE_WATER_TIMING,
                                            EVIDENCE_SOURCE_CHEMISTRY)
    )
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--min-peak-absorbance', type=float, default=0.1)
    parser.add_argument('--post-peak-horizon-s', type=float, default=120.0)
    parser.add_argument('--endpoint-tolerance-s', type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv=None):
    '''CLI entry point for offline analysis only; it never launches Auto.'''
    args = _parse_arguments(argv)
    metadata = write_study_package(
        args.output_dir,
        saved_trajectory_path=args.saved_trajectories,
        saved_evidence_source=args.saved_evidence_source,
        seed=args.seed,
        min_peak_absorbance=args.min_peak_absorbance,
        post_peak_horizon_s=args.post_peak_horizon_s,
        endpoint_tolerance_s=args.endpoint_tolerance_s,
    )
    print(json.dumps(metadata, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
