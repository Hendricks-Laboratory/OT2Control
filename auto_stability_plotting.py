'''Read-only final plotting for Auto optical-stability reporting.

This module consumes only the derived records produced after a completed Auto
run.  It never opens a reader file, changes a manifest, fits a model, sends a
packet, or schedules laboratory work.  Its figures distinguish model-facing
decision-window evidence from later audit-only observations rather than
combining the two into a visually plausible but scientifically misleading
trajectory.
'''

from __future__ import division

import datetime
import math
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize


STABILITY_PLOT_SUITE_VERSION = 1
_ATLAS_WELLS_PER_PAGE = 6


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


def _well_sort_key(wellname):
    return str(wellname or '').strip()


def _safe_title(value, fallback):
    text = str(value or '').strip()
    return text if text else fallback


def _chunks(values, size):
    return [values[index:index + size] for index in range(0, len(values), size)]


def build_stability_plot_data(records):
    '''Normalize completed reporting records for read-only plotting.

    The returned structure is deliberately small and serializable.  It keeps
    raw trajectory membership separate from metric eligibility and preserves
    the final controller-reported QC/condition metadata.  Missing values are
    retained as absent observations rather than interpolated.
    '''
    if not isinstance(records, dict):
        raise ValueError('stability plotting requires a reporting-record dictionary.')

    trajectories_by_well = {}
    for source_row in records.get('trajectory_rows', []) or []:
        if not isinstance(source_row, dict):
            continue
        row = dict(source_row)
        wellname = str(row.get('wellname', '')).strip()
        elapsed_s = _finite_number(
            row.get('elapsed_s_from_trigger_to_scan_start')
        )
        absorbance = _finite_number(row.get('reference_absorbance'))
        if not wellname or elapsed_s is None or absorbance is None:
            continue
        row['wellname'] = wellname
        row['_elapsed_s'] = elapsed_s
        row['_absorbance'] = absorbance
        trajectories_by_well.setdefault(wellname, []).append(row)
    for rows in trajectories_by_well.values():
        rows.sort(key=lambda row: (row['_elapsed_s'], str(row.get('raw_scan_id', ''))))

    metrics_by_well = {}
    for source_row in records.get('well_metrics', []) or []:
        if not isinstance(source_row, dict):
            continue
        wellname = str(source_row.get('wellname', '')).strip()
        if wellname:
            metrics_by_well[wellname] = dict(source_row)

    lifecycle_by_well = {}
    for source_row in records.get('lifecycle_rows', []) or []:
        if not isinstance(source_row, dict):
            continue
        wellname = str(source_row.get('wellname', '')).strip()
        if not wellname:
            continue
        row = dict(source_row)
        row['_timestamp'] = _parse_utc(row.get('recorded_at_utc'))
        lifecycle_by_well.setdefault(wellname, []).append(row)
    for rows in lifecycle_by_well.values():
        rows.sort(key=lambda row: (
            row['_timestamp'] is None,
            row['_timestamp'] or datetime.datetime.max.replace(
                tzinfo=datetime.timezone.utc
            ),
            str(row.get('event_type', '')),
        ))

    all_wellnames = sorted(
        set(trajectories_by_well) | set(metrics_by_well) | set(lifecycle_by_well),
        key=_well_sort_key,
    )
    well_metadata = {}
    for wellname in all_wellnames:
        metric = metrics_by_well.get(wellname, {})
        lifecycle = lifecycle_by_well.get(wellname, [])
        trigger = next((
            row for row in lifecycle
            if row.get('event_type') == 'trigger_transfer_completed'
        ), None)
        well_metadata[wellname] = {
            'wellname': wellname,
            'batch_number': _integer_or_none(metric.get('batch_number')),
            'condition_id': str(metric.get('condition_id', '')).strip(),
            'condition_type': str(metric.get('condition_type', '')).strip(),
            'metric_status': str(metric.get('status', '')).strip(),
            'trigger_timestamp': (
                trigger.get('_timestamp') if isinstance(trigger, dict) else None
            ),
        }

    scans = {}
    for source_row in records.get('timing_rows', []) or []:
        if not isinstance(source_row, dict):
            continue
        scan_id = str(source_row.get('raw_scan_id', '')).strip()
        wellname = str(source_row.get('wellname', '')).strip()
        if not scan_id or not wellname:
            continue
        scan = scans.setdefault(scan_id, {
            'raw_scan_id': scan_id,
            'wellnames': [],
            'observation_reason': str(
                source_row.get('observation_reason', '')
            ).strip(),
            'started_at': _parse_utc(source_row.get('scan_started_at_utc')),
        })
        if wellname not in scan['wellnames']:
            scan['wellnames'].append(wellname)
    ordered_scans = sorted(scans.values(), key=lambda scan: (
        scan['started_at'] is None,
        scan['started_at'] or datetime.datetime.max.replace(
            tzinfo=datetime.timezone.utc
        ),
        scan['raw_scan_id'],
    ))
    for scan in ordered_scans:
        scan['wellnames'].sort(key=_well_sort_key)

    return {
        'wellnames': all_wellnames,
        'well_metadata': well_metadata,
        'trajectories_by_well': trajectories_by_well,
        'lifecycle_by_well': lifecycle_by_well,
        'condition_summaries': [
            dict(row) for row in records.get('condition_summaries', []) or []
            if isinstance(row, dict)
        ],
        'scans': ordered_scans,
        'debug_only': bool(records.get('debug_evidence_rows')),
        'evidence_source': str(
            records.get('stability_evidence_source', 'raw_reader')
        ).strip() or 'raw_reader',
    }


def _add_debug_notice(figure, plot_data, minimum_bottom=0.16):
    if plot_data['debug_only']:
        # Reserve a true footer rather than painting the provenance warning over
        # axis labels.  This matters especially for the membership matrix,
        # whose long rotated well labels already use more bottom margin.
        figure.subplots_adjust(
            bottom=max(float(figure.subplotpars.bottom), float(minimum_bottom))
        )
        figure.text(
            0.5, 0.025,
            'DEBUG ONLY — synthetic companion evidence; not measured chemistry.',
            ha='center', va='bottom', color='#9b1c1c', fontsize=8,
        )


def _save(figure, output_path):
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return output_path


def _well_label(wellname, metadata):
    parts = [str(wellname)]
    if metadata.get('batch_number') is not None:
        parts.append('batch {}'.format(metadata['batch_number']))
    if metadata.get('condition_type'):
        parts.append(str(metadata['condition_type']))
    if metadata.get('metric_status'):
        parts.append(str(metadata['metric_status']))
    return ' — '.join(parts)


def _plot_trajectory_atlas(plot_data, output_dir, decision_horizon_s,
                           maximum_window_s):
    paths = []
    pages = _chunks(plot_data['wellnames'], _ATLAS_WELLS_PER_PAGE)
    for page_number, wells in enumerate(pages, 1):
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for axis, wellname in zip(axes.flat, wells):
            metadata = plot_data['well_metadata'][wellname]
            rows = plot_data['trajectories_by_well'].get(wellname, [])
            eligible = [row for row in rows if row.get('metric_eligible_interval')]
            audit_only = [row for row in rows if not row.get('metric_eligible_interval')]
            if eligible:
                axis.plot(
                    [row['_elapsed_s'] for row in eligible],
                    [row['_absorbance'] for row in eligible],
                    color='#1f77b4', marker='o', linewidth=1.4,
                    markersize=4, label='Decision-eligible observation',
                )
            if audit_only:
                axis.scatter(
                    [row['_elapsed_s'] for row in audit_only],
                    [row['_absorbance'] for row in audit_only],
                    marker='x', color='0.35', s=28, zorder=3,
                    label='Audit-only / excluded interval',
                )
            if decision_horizon_s is not None:
                axis.axvline(
                    decision_horizon_s, color='#0f766e', linestyle='--',
                    linewidth=1.0, label='Decision horizon',
                )
            if maximum_window_s is not None:
                axis.axvline(
                    maximum_window_s, color='#7c3aed', linestyle=':',
                    linewidth=1.0, label='Maximum window',
                )
            axis.set_title(_well_label(wellname, metadata), fontsize=9)
            axis.set_xlabel('Seconds from trigger to scan start')
            axis.set_ylabel('Fixed-reference absorbance')
            axis.grid(alpha=0.22)
            if not rows:
                axis.text(
                    0.5, 0.5, 'No plottable reference-absorbance observations',
                    ha='center', va='center', transform=axis.transAxes,
                    fontsize=8,
                )
        for axis in list(axes.flat)[len(wells):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles, labels, loc='upper center', ncol=2, fontsize=8,
                bbox_to_anchor=(0.5, 0.94),
            )
        figure.suptitle(
            'Optical-stability well trajectories — page {}/{}'.format(
                page_number, len(pages)
            ), fontsize=14, y=0.99,
        )
        figure.subplots_adjust(top=0.82)
        _add_debug_notice(figure, plot_data, minimum_bottom=0.16)
        path = os.path.join(
            output_dir, 'well_trajectories',
            'stability_well_trajectories_page_{:02d}_final.png'.format(
                page_number
            ),
        )
        paths.append(_save(figure, path))
    return paths


def _plot_legacy_trajectory_overview(plot_data, output_dir):
    '''Retain the established one-file trajectory overview for compatibility.'''
    plotted_wells = [
        wellname for wellname in plot_data['wellnames']
        if plot_data['trajectories_by_well'].get(wellname)
    ]
    if not plotted_wells:
        return []
    figure, axis = plt.subplots(figsize=(11, 6))
    for wellname in plotted_wells:
        rows = plot_data['trajectories_by_well'][wellname]
        eligible = [row for row in rows if row.get('metric_eligible_interval')]
        audit_only = [row for row in rows if not row.get('metric_eligible_interval')]
        if eligible:
            axis.plot(
                [row['_elapsed_s'] for row in eligible],
                [row['_absorbance'] for row in eligible],
                marker='o', linewidth=1.2, markersize=3.5,
                label=str(wellname),
            )
        if audit_only:
            axis.scatter(
                [row['_elapsed_s'] for row in audit_only],
                [row['_absorbance'] for row in audit_only], marker='x',
                color='0.35', s=24, zorder=3,
            )
    axis.set_title('Optical-stability trajectories (fixed per-well reference)')
    axis.set_xlabel('Elapsed time from trigger to reader scan start (s)')
    axis.set_ylabel('Blank-corrected absorbance')
    axis.grid(alpha=0.25)
    if len(plotted_wells) <= 16:
        axis.legend(loc='best', fontsize=8)
    _add_debug_notice(figure, plot_data)
    return [_save(
        figure,
        os.path.join(output_dir, 'stability_per_well_trajectories_final.png'),
    )]


def _plot_condition_loss_rates(plot_data, output_dir):
    summaries = [
        row for row in plot_data['condition_summaries']
        if _finite_number(
            row.get('condition_loss_rate_mean_absorbance_per_s')
        ) is not None
    ]
    if not summaries:
        return []
    summaries.sort(key=lambda row: (
        _integer_or_none(row.get('batch_number')) is None,
        _integer_or_none(row.get('batch_number')) or 0,
        _integer_or_none(row.get('reaction_number')) is None,
        _integer_or_none(row.get('reaction_number')) or 0,
    ))
    labels = [
        'B{} C{}'.format(
            _safe_title(row.get('batch_number'), '?'),
            _safe_title(row.get('reaction_number'), '?'),
        ) for row in summaries
    ]
    values = [
        _finite_number(row.get('condition_loss_rate_mean_absorbance_per_s'))
        for row in summaries
    ]
    errors = [
        _finite_number(row.get('condition_loss_rate_sample_sd_absorbance_per_s'))
        or 0.0 for row in summaries
    ]
    figure, axis = plt.subplots(figsize=(max(8, len(values) * 1.4), 6))
    positions = list(range(len(values)))
    axis.bar(positions, values, yerr=errors, capsize=4, color='#60a5fa')
    for position, row in zip(positions, summaries):
        axis.text(
            position, values[position],
            'n={}/{}'.format(
                row.get('eligible_well_count', 0), row.get('total_well_count', '?')
            ),
            ha='center', va='bottom', fontsize=8,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha='right')
    axis.set_ylabel('Post-peak absorbance loss rate (absorbance/s)')
    axis.set_title(
        'Condition-level optical-stability summary\n'
        '(mean ± sample SD across eligible replicate wells)'
    )
    axis.grid(axis='y', alpha=0.25)
    _add_debug_notice(figure, plot_data)
    paths = []
    detailed_path = os.path.join(
        output_dir, 'condition_summaries',
        'stability_condition_loss_rate_final.png',
    )
    paths.append(_save(figure, detailed_path))

    # Preserve the established Stage-12D filename for report compatibility.
    figure, axis = plt.subplots(figsize=(max(8, len(values) * 1.4), 6))
    axis.bar(positions, values, yerr=errors, capsize=4, color='#60a5fa')
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha='right')
    axis.set_ylabel('Post-peak absorbance loss rate (absorbance/s)')
    axis.set_title('Condition-level optical-stability summary')
    axis.grid(axis='y', alpha=0.25)
    _add_debug_notice(figure, plot_data)
    paths.append(_save(
        figure,
        os.path.join(output_dir, 'stability_condition_loss_rate_final.png'),
    ))
    return paths


def _plot_lifecycle_timeline(plot_data, output_dir):
    trigger_times = [
        metadata['trigger_timestamp']
        for metadata in plot_data['well_metadata'].values()
        if metadata.get('trigger_timestamp') is not None
    ]
    if not trigger_times:
        return []
    baseline = min(trigger_times)
    figure, axis = plt.subplots(
        figsize=(13, max(4, len(plot_data['wellnames']) * 0.55 + 2))
    )
    marker_styles = {
        'trigger_transfer_completed': ('o', '#1d4ed8', 'trigger'),
        'targeted_pipette_mix_acknowledged': ('s', '#059669', 'targeted mix'),
        'decision_horizon_reached': ('D', '#0f766e', 'decision horizon'),
        'observation_window_completed': ('X', '#7c3aed', 'maximum-window retirement'),
        'observation_plateau_confirmed': ('X', '#7c3aed', 'plateau retirement'),
        'observation_terminal_qc_excluded': ('X', '#7c3aed', 'terminal QC retirement'),
    }
    shown = set()
    for index, wellname in enumerate(plot_data['wellnames']):
        metadata = plot_data['well_metadata'][wellname]
        trigger = metadata.get('trigger_timestamp')
        if trigger is None:
            continue
        trigger_x = (trigger - baseline).total_seconds()
        lifecycle_rows = plot_data['lifecycle_by_well'].get(wellname, [])
        scan_rows = plot_data['trajectories_by_well'].get(wellname, [])
        event_timestamps = [
            row.get('_timestamp') for row in lifecycle_rows
            if row.get('_timestamp') is not None
        ] + [
            _parse_utc(row.get('scan_started_at_utc')) for row in scan_rows
            if _parse_utc(row.get('scan_started_at_utc')) is not None
        ]
        final_x = max(
            [(timestamp - baseline).total_seconds() for timestamp in event_timestamps]
            or [trigger_x]
        )
        axis.hlines(index, trigger_x, final_x, color='0.75', linewidth=2.5)
        for row in lifecycle_rows:
            event_type = row.get('event_type')
            timestamp = row.get('_timestamp')
            if event_type not in marker_styles or timestamp is None:
                continue
            marker, color, label = marker_styles[event_type]
            axis.scatter(
                [(timestamp - baseline).total_seconds()], [index],
                marker=marker, color=color, s=45, zorder=3,
                label=label if label not in shown else None,
            )
            shown.add(label)
        for row in scan_rows:
            timestamp = _parse_utc(row.get('scan_started_at_utc'))
            if timestamp is None:
                continue
            axis.scatter(
                [(timestamp - baseline).total_seconds()], [index], marker='|',
                color=('#1f77b4' if row.get('metric_eligible_interval') else '0.4'),
                s=120, zorder=2,
                label=(
                    'decision-eligible raw scan'
                    if 'decision-eligible raw scan' not in shown
                    and row.get('metric_eligible_interval') else (
                        'audit-only raw scan'
                        if 'audit-only raw scan' not in shown else None
                    )
                ),
            )
            shown.add(
                'decision-eligible raw scan'
                if row.get('metric_eligible_interval') else 'audit-only raw scan'
            )
    axis.set_yticks(range(len(plot_data['wellnames'])))
    axis.set_yticklabels(plot_data['wellnames'])
    axis.set_xlabel('Seconds from first trigger completion in this run')
    axis.set_title('Optical-stability well lifecycle timeline')
    axis.grid(axis='x', alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc='best', fontsize=8)
    _add_debug_notice(figure, plot_data, minimum_bottom=0.16)
    return [_save(
        figure,
        os.path.join(
            output_dir, 'lifecycle_timelines',
            'stability_well_lifecycle_timeline_final.png',
        ),
    )]


def _plot_scan_membership(plot_data, output_dir):
    scans = plot_data['scans']
    wells = plot_data['wellnames']
    if not scans or not wells:
        return []
    batches = sorted({
        metadata['batch_number'] for metadata in plot_data['well_metadata'].values()
        if metadata.get('batch_number') is not None
    })
    batch_to_code = {batch: index + 1 for index, batch in enumerate(batches)}
    matrix = []
    row_labels = []
    for scan in scans:
        membership = set(scan['wellnames'])
        matrix.append([
            batch_to_code.get(
                plot_data['well_metadata'][wellname].get('batch_number'), 0
            ) if wellname in membership else 0
            for wellname in wells
        ])
        row_labels.append('{} ({})'.format(
            scan['raw_scan_id'],
            _safe_title(scan.get('observation_reason'), 'unspecified'),
        ))
    colors = ['#ffffff'] + [
        plt.get_cmap('tab10')((index - 1) % 10)
        for index in range(1, len(batches) + 1)
    ]
    figure, axis = plt.subplots(
        figsize=(max(8, len(wells) * 0.7 + 3), max(4, len(scans) * 0.65 + 2))
    )
    image = axis.imshow(matrix, aspect='auto', cmap=ListedColormap(colors),
                         vmin=0, vmax=max(1, len(batches)))
    del image
    axis.set_xticks(range(len(wells)))
    axis.set_xticklabels(wells, rotation=45, ha='right')
    axis.set_yticks(range(len(scans)))
    axis.set_yticklabels(row_labels, fontsize=8)
    axis.set_xlabel('Physical reaction well')
    axis.set_ylabel('Manifest-linked raw scan')
    axis.set_title('Raw scan membership by well and batch')
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            if value:
                axis.text(
                    column_index, row_index, 'B{}'.format(batches[value - 1]),
                    ha='center', va='center', fontsize=7,
                )
    _add_debug_notice(figure, plot_data, minimum_bottom=0.34)
    return [_save(
        figure,
        os.path.join(
            output_dir, 'scan_membership',
            'stability_raw_scan_membership_final.png',
        ),
    )]


def _plot_evidence_boundaries(plot_data, output_dir, decision_horizon_s,
                              maximum_window_s):
    wells = [
        wellname for wellname in plot_data['wellnames']
        if plot_data['trajectories_by_well'].get(wellname)
    ]
    if not wells:
        return []
    figure, axis = plt.subplots(
        figsize=(13, max(4, len(wells) * 0.55 + 2))
    )
    if decision_horizon_s is not None:
        axis.axvspan(0, decision_horizon_s, color='#d1fae5', alpha=0.6,
                    label='Frozen decision-evidence window')
        axis.axvline(decision_horizon_s, color='#0f766e', linestyle='--',
                    linewidth=1.0)
    if maximum_window_s is not None:
        axis.axvline(maximum_window_s, color='#7c3aed', linestyle=':',
                    linewidth=1.2, label='Maximum observation window')
    for index, wellname in enumerate(wells):
        rows = plot_data['trajectories_by_well'][wellname]
        maximum_observed = max(row['_elapsed_s'] for row in rows)
        axis.hlines(index, 0, maximum_observed, color='0.7', linewidth=2)
        eligible = [row for row in rows if row.get('metric_eligible_interval')]
        audit_only = [row for row in rows if not row.get('metric_eligible_interval')]
        if eligible:
            axis.scatter(
                [row['_elapsed_s'] for row in eligible], [index] * len(eligible),
                color='#1f77b4', marker='o', s=28, zorder=3,
                label='Decision-eligible observation' if index == 0 else None,
            )
        if audit_only:
            axis.scatter(
                [row['_elapsed_s'] for row in audit_only],
                [index] * len(audit_only), color='0.35', marker='x', s=32,
                zorder=3, label='Later audit-only observation' if index == 0 else None,
            )
    axis.set_yticks(range(len(wells)))
    axis.set_yticklabels(wells)
    axis.set_xlabel('Seconds from each well\'s own trigger completion')
    axis.set_title('Frozen decision evidence versus extended audit observations')
    axis.grid(axis='x', alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc='best', fontsize=8)
    _add_debug_notice(figure, plot_data, minimum_bottom=0.16)
    return [_save(
        figure,
        os.path.join(
            output_dir, 'evidence_boundaries',
            'stability_decision_vs_audit_final.png',
        ),
    )]


def write_stability_plot_suite(records, output_dir, decision_horizon_s=None,
                               maximum_window_s=None):
    '''Write the final stability plot suite from completed reporting records.

    ``output_dir`` is normally ``<run>/Plots/stability``.  The function makes
    only new or replacement derived PNG outputs under that directory; it does
    not alter raw scans, manifests, CSV exports, model state, or execution
    state.  It returns every successfully requested path in deterministic
    order.  A run with no plottable evidence returns an empty list.
    '''
    plot_data = build_stability_plot_data(records)
    decision_horizon_s = _finite_number(decision_horizon_s)
    maximum_window_s = _finite_number(maximum_window_s)
    if decision_horizon_s is not None and decision_horizon_s <= 0:
        decision_horizon_s = None
    if maximum_window_s is not None and maximum_window_s <= 0:
        maximum_window_s = None
    if (decision_horizon_s is not None and maximum_window_s is not None and
            decision_horizon_s > maximum_window_s):
        raise ValueError(
            'stability decision horizon cannot exceed the maximum window.'
        )
    if not plot_data['wellnames']:
        return []
    for directory in (
            output_dir,
            os.path.join(output_dir, 'well_trajectories'),
            os.path.join(output_dir, 'condition_summaries'),
            os.path.join(output_dir, 'lifecycle_timelines'),
            os.path.join(output_dir, 'scan_membership'),
            os.path.join(output_dir, 'evidence_boundaries')):
        if not os.path.isdir(directory):
            os.makedirs(directory)
    paths = []
    paths.extend(_plot_legacy_trajectory_overview(plot_data, output_dir))
    paths.extend(_plot_trajectory_atlas(
        plot_data, output_dir, decision_horizon_s, maximum_window_s
    ))
    paths.extend(_plot_condition_loss_rates(plot_data, output_dir))
    paths.extend(_plot_lifecycle_timeline(plot_data, output_dir))
    paths.extend(_plot_scan_membership(plot_data, output_dir))
    paths.extend(_plot_evidence_boundaries(
        plot_data, output_dir, decision_horizon_s, maximum_window_s
    ))
    return paths


def _as_boolean(value):
    if isinstance(value, bool):
        return value
    return str(value or '').strip().lower() in ('true', '1', 'yes')


def _build_spectral_evidence_plot_data(records):
    '''Normalize observed raw-reader spectral evidence without resampling it.'''
    evidence = records.get('raw_spectral_evidence', {}) if isinstance(
        records, dict
    ) else {}
    if not isinstance(evidence, dict):
        return {
            'by_well': {},
            'debug_companion_active': False,
            'decision_horizon_s': None,
            'maximum_window_s': None,
        }
    rows = evidence.get('evidence_rows', []) or []
    spectra_by_pair = evidence.get('spectra_by_pair', {}) or {}
    by_well = {}
    for original_row in rows:
        if not isinstance(original_row, dict):
            continue
        row = dict(original_row)
        wellname = str(row.get('wellname', '')).strip()
        scan_id = str(row.get('raw_scan_id', '')).strip()
        elapsed_s = _finite_number(
            row.get('elapsed_s_from_trigger_to_scan_start')
        )
        points = spectra_by_pair.get((scan_id, wellname), [])
        cleaned_points = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            wavelength = _finite_number(point[0])
            absorbance = _finite_number(point[1])
            if wavelength is not None and absorbance is not None:
                cleaned_points.append((wavelength, absorbance))
        cleaned_points.sort(key=lambda point: point[0])
        if not wellname or not scan_id or elapsed_s is None or not cleaned_points:
            continue
        row['_elapsed_s'] = elapsed_s
        row['_points'] = cleaned_points
        row['_eligible'] = _as_boolean(row.get('metric_eligible_interval'))
        by_well.setdefault(wellname, []).append(row)
    for rows_for_well in by_well.values():
        rows_for_well.sort(key=lambda row: (
            row['_elapsed_s'], str(row.get('raw_scan_id', ''))
        ))
    return {
        'by_well': by_well,
        'debug_companion_active': bool(
            isinstance(records, dict) and records.get('debug_evidence_rows')
        ),
        'decision_horizon_s': _finite_number(
            records.get('spectral_evidence_decision_horizon_s')
        ) if isinstance(records, dict) else None,
        'maximum_window_s': _finite_number(
            records.get('spectral_evidence_maximum_window_s')
        ) if isinstance(records, dict) else None,
    }


def _spectral_well_label(wellname, rows):
    first = rows[0] if rows else {}
    parts = [wellname]
    batch = _integer_or_none(first.get('batch_number'))
    if batch is not None:
        parts.append('batch {}'.format(batch))
    reader_location = str(first.get('reader_location', '')).strip()
    if reader_location:
        parts.append('reader {}'.format(reader_location))
    return ' — '.join(parts)


def _add_observed_reader_notice(figure, plot_data):
    figure.subplots_adjust(bottom=max(float(figure.subplotpars.bottom), 0.13))
    notice = 'Observed blank-corrected raw-reader spectra; solid = metric-eligible, dashed = audit-only.'
    if plot_data['debug_companion_active']:
        notice += ' DEBUG companion evidence was not used in these figures.'
    figure.text(0.5, 0.02, notice, ha='center', va='bottom', fontsize=8)


def _add_spectral_window_boundaries(axis, plot_data):
    '''Mark controller-configured timing boundaries without altering evidence.'''
    decision_horizon_s = plot_data.get('decision_horizon_s')
    maximum_window_s = plot_data.get('maximum_window_s')
    if decision_horizon_s is not None:
        axis.axvline(
            decision_horizon_s, color='#d97706', linestyle=':', linewidth=1.0,
            zorder=0,
        )
    if maximum_window_s is not None:
        axis.axvline(
            maximum_window_s, color='#b91c1c', linestyle=':', linewidth=1.0,
            zorder=0,
        )


def _spectral_color_scale(rows):
    elapsed = [row['_elapsed_s'] for row in rows]
    lower = min(elapsed)
    upper = max(elapsed)
    if lower == upper:
        upper = lower + 1.0
    return Normalize(vmin=lower, vmax=upper)


def _plot_spectral_waterfall_atlas(plot_data, output_dir, normalized):
    '''Render paginated native-grid spectra, preserving every measured point.'''
    paths = []
    wells = sorted(plot_data['by_well'], key=_well_sort_key)
    folder_name = 'normalized_shape' if normalized else 'raw_amplitude'
    filename_prefix = (
        'stability_normalized_spectra' if normalized
        else 'stability_raw_spectra'
    )
    title = (
        'Normalized observed spectral shape' if normalized
        else 'Observed blank-corrected spectral amplitude'
    )
    for page_number, page_wells in enumerate(_chunks(wells, _ATLAS_WELLS_PER_PAGE), 1):
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        all_rows = [
            row for wellname in page_wells
            for row in plot_data['by_well'][wellname]
        ]
        normalizer = _spectral_color_scale(all_rows)
        color_map = plt.get_cmap('viridis')
        for axis, wellname in zip(axes.flat, page_wells):
            rows = plot_data['by_well'][wellname]
            for row in rows:
                points = row['_points']
                wavelengths = [point[0] for point in points]
                absorbance = [point[1] for point in points]
                if normalized:
                    peak = max(absorbance)
                    if peak <= 0:
                        continue
                    absorbance = [value / peak for value in absorbance]
                axis.plot(
                    wavelengths, absorbance,
                    color=color_map(normalizer(row['_elapsed_s'])),
                    linestyle='-' if row['_eligible'] else '--',
                    linewidth=1.2,
                    alpha=0.95 if row['_eligible'] else 0.55,
                )
            axis.set_title(_spectral_well_label(wellname, rows), fontsize=9)
            axis.set_xlabel('Wavelength (nm)')
            axis.set_ylabel(
                'Normalized absorbance' if normalized
                else 'Blank-corrected absorbance'
            )
            axis.grid(False)
        for axis in axes.flat[len(page_wells):]:
            axis.set_visible(False)
        scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        scalar_map.set_array([])
        colorbar = figure.colorbar(
            scalar_map, ax=list(axes.flat), shrink=0.78, pad=0.02
        )
        colorbar.set_label('Seconds from trigger to scan start')
        figure.suptitle(
            '{} (page {})'.format(title, page_number), fontsize=14
        )
        _add_observed_reader_notice(figure, plot_data)
        paths.append(_save(
            figure,
            os.path.join(
                output_dir, folder_name,
                '{}_page_{:02d}_final.png'.format(filename_prefix, page_number)
            )
        ))
    return paths


def _plot_spectral_feature_atlases(plot_data, output_dir):
    '''Render one honest feature panel per feature type, leaving undefined blank.'''
    paths = []
    wells = sorted(plot_data['by_well'], key=_well_sort_key)
    feature_specs = (
        ('lambda_max_nm', 'λmax (nm)', 'stability_lambda_max'),
        ('peak_absorbance', 'Peak blank-corrected absorbance', 'stability_peak_height'),
        ('fwhm_nm', 'FWHM (nm)', 'stability_fwhm'),
    )
    for field_name, y_label, filename_prefix in feature_specs:
        for page_number, page_wells in enumerate(
                _chunks(wells, _ATLAS_WELLS_PER_PAGE), 1):
            figure, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
            for axis, wellname in zip(axes.flat, page_wells):
                rows = plot_data['by_well'][wellname]
                eligible_x, eligible_y = [], []
                audit_x, audit_y = [], []
                for row in rows:
                    value = _finite_number(row.get(field_name))
                    if value is None:
                        continue
                    target = (eligible_x, eligible_y) if row['_eligible'] else (
                        audit_x, audit_y
                    )
                    target[0].append(row['_elapsed_s'])
                    target[1].append(value)
                if eligible_x:
                    axis.plot(
                        eligible_x, eligible_y, color='#1f77b4', marker='o',
                        linewidth=1.3, markersize=3.5,
                    )
                if audit_x:
                    axis.plot(
                        audit_x, audit_y, color='#7f7f7f', marker='o',
                        linestyle='--', linewidth=1.1, markersize=3.5,
                    )
                if not eligible_x and not audit_x:
                    status = 'No defined FWHM' if field_name == 'fwhm_nm' else (
                        'No available feature'
                    )
                    axis.text(
                        0.5, 0.5, status, ha='center', va='center',
                        transform=axis.transAxes, fontsize=9,
                    )
                _add_spectral_window_boundaries(axis, plot_data)
                axis.set_title(_spectral_well_label(wellname, rows), fontsize=9)
                axis.set_xlabel('Seconds from trigger to scan start')
                axis.set_ylabel(y_label)
                axis.grid(False)
            for axis in axes.flat[len(page_wells):]:
                axis.set_visible(False)
            figure.suptitle(
                'Observed spectral {} (page {})'.format(y_label, page_number),
                fontsize=14,
            )
            if (
                    plot_data.get('decision_horizon_s') is not None
                    or plot_data.get('maximum_window_s') is not None):
                figure.text(
                    0.5, 0.055,
                    'Dotted orange = decision horizon; dotted red = maximum observation window.',
                    ha='center', va='bottom', fontsize=8,
                )
            _add_observed_reader_notice(figure, plot_data)
            paths.append(_save(
                figure,
                os.path.join(
                    output_dir, 'features',
                    '{}_page_{:02d}_final.png'.format(
                        filename_prefix, page_number
                    )
                )
            ))
    return paths


def write_stability_spectral_evidence_plot_suite(records, output_dir):
    '''Write Stage-14F2 observed-spectral figures beneath ``Plots/stability``.

    This is strictly reporting-only.  It consumes the raw-reader spectral
    evidence retained before any DEBUG companion substitution; it cannot open
    a reader file, alter model-facing records, or affect execution.
    '''
    plot_data = _build_spectral_evidence_plot_data(records)
    if not plot_data['by_well']:
        return []
    spectral_dir = os.path.join(output_dir, 'spectral_evidence')
    for directory in (
            spectral_dir,
            os.path.join(spectral_dir, 'raw_amplitude'),
            os.path.join(spectral_dir, 'normalized_shape'),
            os.path.join(spectral_dir, 'features')):
        if not os.path.isdir(directory):
            os.makedirs(directory)
    paths = []
    paths.extend(_plot_spectral_waterfall_atlas(
        plot_data, spectral_dir, normalized=False
    ))
    paths.extend(_plot_spectral_waterfall_atlas(
        plot_data, spectral_dir, normalized=True
    ))
    paths.extend(_plot_spectral_feature_atlases(plot_data, spectral_dir))
    return paths
