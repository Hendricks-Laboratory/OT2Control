'''Read-only Stage-14F3 replicate and reader-throughput dashboard plots.'''

from __future__ import division

import math
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from auto_stability_dashboard import build_stability_dashboard_records


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


def _condition_key(row):
    condition_id = str(row.get('condition_id', '') or '').strip()
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
        str(row.get('condition_id', '') or '').strip(),
    )


def _condition_label(row):
    batch = row.get('batch_number', '')
    reaction = row.get('reaction_number', '')
    if str(batch).strip() or str(reaction).strip():
        return 'B{} C{}'.format(
            str(batch).strip() or '?', str(reaction).strip() or '?'
        )
    return str(row.get('condition_id', '') or 'Condition')


def _save(figure, path):
    figure.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return path


def _add_debug_notice(figure, records, minimum_bottom=0.14):
    if not isinstance(records, dict) or not records.get('debug_evidence_rows'):
        return
    figure.subplots_adjust(
        bottom=max(float(figure.subplotpars.bottom), float(minimum_bottom))
    )
    figure.text(
        0.5, 0.02,
        'DEBUG ONLY — synthetic companion evidence; not measured chemistry.',
        ha='center', va='bottom', color='#9b1c1c', fontsize=8,
    )


def _condition_groups(replicate_rows):
    groups = {}
    for row in replicate_rows:
        groups.setdefault(_condition_key(row), []).append(row)
    ordered = sorted(
        groups.values(), key=lambda rows: _condition_sort_key(rows[0])
    )
    return ordered


def _jitter(index, count):
    if count <= 1:
        return 0.0
    return (index - ((count - 1) / 2.0)) * 0.10


def _plot_replicate_loss_rates(replicate_rows, records, output_dir):
    groups = _condition_groups(replicate_rows)
    if not groups:
        return []
    figure, axis = plt.subplots(figsize=(max(8, len(groups) * 1.5), 6.4))
    shown = set()
    for position, rows in enumerate(groups):
        plottable_count = 0
        for ordinal, row in enumerate(rows):
            value = _finite_number(row.get('loss_rate_absorbance_per_s'))
            if value is None:
                continue
            plottable_count += 1
            included = bool(row.get('loss_rate_qc_included'))
            label = (
                'Eligible physical well' if included
                else 'Excluded physical well'
            )
            axis.scatter(
                position + _jitter(ordinal, len(rows)), value,
                color='#2563eb' if included else '#9ca3af',
                marker='o' if included else 'x', s=38,
                zorder=3, label=label if label not in shown else None,
            )
            shown.add(label)
        summary = rows[0]
        mean = _finite_number(
            summary.get('condition_loss_rate_mean_absorbance_per_s')
        )
        sample_sd = _finite_number(
            summary.get('condition_loss_rate_sample_sd_absorbance_per_s')
        )
        if mean is not None:
            axis.errorbar(
                position, mean, yerr=sample_sd, color='#111827', marker='_',
                markersize=16, capsize=4, linewidth=1.3, zorder=4,
                label='Mean ± sample SD' if 'Mean ± sample SD' not in shown else None,
            )
            shown.add('Mean ± sample SD')
        planned = summary.get('condition_total_well_count', len(rows))
        eligible = summary.get('condition_eligible_well_count', 0)
        axis.text(
            position, 0.01, 'n={}/{}'.format(eligible, planned),
            transform=axis.get_xaxis_transform(), ha='center', va='bottom',
            fontsize=8,
        )
        if len(rows) > plottable_count:
            axis.text(
                position, 0.045,
                '{} excluded / no rate'.format(len(rows) - plottable_count),
                transform=axis.get_xaxis_transform(), ha='center', va='bottom',
                color='#6b7280', fontsize=7,
            )
    axis.axhline(0.0, color='0.45', linewidth=0.8)
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels([_condition_label(rows[0]) for rows in groups],
                         rotation=25, ha='right')
    axis.set_xlabel('Condition')
    axis.set_ylabel('Post-peak absorbance loss rate (absorbance/s)')
    axis.set_title(
        'Condition stability evidence: physical replicate wells\n'
        'Mean ± sample SD uses eligible wells only; n = eligible / planned'
    )
    axis.grid(axis='y', alpha=0.22)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc='best', fontsize=8)
    _add_debug_notice(figure, records)
    return [_save(figure, os.path.join(
        output_dir, 'replicate_completeness',
        'stability_loss_rate_replicates_final.png',
    ))]


def _plot_replicate_signal(replicate_rows, records, output_dir):
    groups = _condition_groups(replicate_rows)
    if not groups:
        return []
    figure, axis = plt.subplots(figsize=(max(8, len(groups) * 1.5), 6.4))
    shown = set()
    for position, rows in enumerate(groups):
        plottable_count = 0
        for ordinal, row in enumerate(rows):
            value = _finite_number(row.get('peak_absorbance'))
            if value is None:
                continue
            plottable_count += 1
            included = bool(row.get('reference_peak_qc_included'))
            label = (
                'Reference-peak observation' if included
                else 'Excluded / unavailable observation'
            )
            axis.scatter(
                position + _jitter(ordinal, len(rows)), value,
                color='#0891b2' if included else '#9ca3af',
                marker='o' if included else 'x', s=38,
                zorder=3, label=label if label not in shown else None,
            )
            shown.add(label)
        summary = rows[0]
        mean = _finite_number(
            summary.get('condition_reference_peak_absorbance_mean')
        )
        sample_sd = _finite_number(
            summary.get('condition_reference_peak_absorbance_sample_sd')
        )
        if mean is not None:
            axis.errorbar(
                position, mean, yerr=sample_sd, color='#111827', marker='_',
                markersize=16, capsize=4, linewidth=1.3, zorder=4,
                label='Mean ± sample SD' if 'Mean ± sample SD' not in shown else None,
            )
            shown.add('Mean ± sample SD')
        planned = summary.get('condition_total_well_count', len(rows))
        eligible = summary.get('condition_reference_peak_eligible_well_count', 0)
        axis.text(
            position, 0.01, 'n={}/{}'.format(eligible, planned),
            transform=axis.get_xaxis_transform(), ha='center', va='bottom',
            fontsize=8,
        )
        if len(rows) > plottable_count:
            axis.text(
                position, 0.045,
                '{} unavailable'.format(len(rows) - plottable_count),
                transform=axis.get_xaxis_transform(), ha='center', va='bottom',
                color='#6b7280', fontsize=7,
            )
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels([_condition_label(rows[0]) for rows in groups],
                         rotation=25, ha='right')
    axis.set_xlabel('Condition')
    axis.set_ylabel('Fixed-reference peak absorbance')
    axis.set_title(
        'Condition signal evidence: physical replicate wells\n'
        'Reported separately from loss-rate variability; n = observed / planned'
    )
    axis.grid(axis='y', alpha=0.22)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, loc='best', fontsize=8)
    _add_debug_notice(figure, records)
    return [_save(figure, os.path.join(
        output_dir, 'replicate_completeness',
        'stability_reference_peak_replicates_final.png',
    ))]


def _plot_well_completeness_matrix(replicate_rows, records, output_dir):
    if not replicate_rows:
        return []
    statuses = sorted({
        str(row.get('metric_status', '') or 'status_unavailable')
        for row in replicate_rows
    })
    status_codes = {status: index for index, status in enumerate(statuses)}
    colors = [
        '#2563eb', '#d97706', '#dc2626', '#7c3aed', '#6b7280', '#0f766e',
        '#be123c', '#4b5563',
    ]
    color_map = ListedColormap([
        colors[index % len(colors)] for index in range(len(statuses))
    ])
    height = max(4.5, len(replicate_rows) * 0.45 + 2.2)
    figure, axis = plt.subplots(figsize=(12, height))
    matrix = [[status_codes[str(row.get('metric_status', '') or 'status_unavailable')]]
              for row in replicate_rows]
    image = axis.imshow(matrix, aspect='auto', cmap=color_map,
                        vmin=-0.5, vmax=max(0.5, len(statuses) - 0.5))
    axis.set_xticks([0])
    axis.set_xticklabels(['Final stability metric status'])
    axis.set_yticks(range(len(replicate_rows)))
    axis.set_yticklabels([
        '{} — {}{}'.format(
            _condition_label(row), row.get('wellname', ''),
            (' ({})'.format(row.get('metric_qc_reason'))
             if row.get('metric_qc_reason') else '')
        ) for row in replicate_rows
    ], fontsize=8)
    for index, row in enumerate(replicate_rows):
        status = str(row.get('metric_status', '') or 'status_unavailable')
        axis.text(0, index, status.replace('_', '\n'), ha='center', va='center',
                  color='white', fontsize=8, fontweight='bold')
    colorbar = figure.colorbar(image, ax=axis, ticks=range(len(statuses)), pad=0.02)
    colorbar.ax.set_yticklabels([status.replace('_', ' ') for status in statuses])
    axis.set_title(
        'Per-well stability completeness and final QC status\n'
        'Full reasons are retained in stability_replicate_completeness.csv'
    )
    figure.subplots_adjust(left=0.42)
    _add_debug_notice(figure, records, minimum_bottom=0.12)
    return [_save(figure, os.path.join(
        output_dir, 'replicate_completeness',
        'stability_well_completeness_qc_final.png',
    ))]


def _plot_reader_throughput(throughput_rows, records, output_dir):
    if not throughput_rows:
        return []
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    ordinal = [row['scan_ordinal'] for row in throughput_rows]
    active = [row['active_well_count'] for row in throughput_rows]
    eligible = [row['decision_eligible_well_count'] for row in throughput_rows]
    audit_only = [row['audit_only_well_count'] for row in throughput_rows]
    axes[0][0].bar(ordinal, eligible, color='#2563eb', label='Decision-eligible wells')
    axes[0][0].bar(
        ordinal, audit_only, bottom=eligible, color='#9ca3af',
        label='Audit-only wells',
    )
    axes[0][0].set_title('Active-set membership per raw reader scan')
    axes[0][0].set_xlabel('Raw-scan order')
    axes[0][0].set_ylabel('Physical wells')
    axes[0][0].legend(fontsize=8)

    durations = [_finite_number(row.get('scan_duration_s')) for row in throughput_rows]
    valid_duration = [index for index, value in enumerate(durations) if value is not None]
    if valid_duration:
        axes[0][1].plot(
            [ordinal[index] for index in valid_duration],
            [durations[index] for index in valid_duration],
            marker='o', color='#0f766e', linewidth=1.3,
        )
    else:
        axes[0][1].text(0.5, 0.5, 'No complete scan timestamps',
                        ha='center', va='center', transform=axes[0][1].transAxes)
    axes[0][1].set_title('Observed reader scan duration')
    axes[0][1].set_xlabel('Raw-scan order')
    axes[0][1].set_ylabel('Seconds from scan start to completion')

    cadence = [_finite_number(row.get('cadence_delay_mean_s')) for row in throughput_rows]
    valid_cadence = [index for index, value in enumerate(cadence) if value is not None]
    if valid_cadence:
        axes[1][0].scatter(
            [ordinal[index] for index in valid_cadence],
            [cadence[index] for index in valid_cadence],
            color='#d97706', s=36, label='Mean per-well cadence delay',
        )
        axes[1][0].axhline(0.0, color='0.45', linewidth=0.8)
        axes[1][0].legend(fontsize=8)
    else:
        axes[1][0].text(0.5, 0.5, 'No cadence deadline applied',
                        ha='center', va='center', transform=axes[1][0].transAxes)
    axes[1][0].set_title('Requested versus achieved cadence')
    axes[1][0].set_xlabel('Raw-scan order')
    axes[1][0].set_ylabel('Seconds late (+) or early (−)')

    first_delays = [
        _finite_number(row.get('trigger_to_first_scan_delay_mean_s'))
        for row in throughput_rows
    ]
    valid_first = [index for index, value in enumerate(first_delays) if value is not None]
    if valid_first:
        axes[1][1].scatter(
            [ordinal[index] for index in valid_first],
            [first_delays[index] for index in valid_first],
            color='#7c3aed', s=36,
        )
    else:
        axes[1][1].text(0.5, 0.5, 'No trigger-to-first-scan pairs',
                        ha='center', va='center', transform=axes[1][1].transAxes)
    axes[1][1].set_title('Trigger-to-first-scan delay')
    axes[1][1].set_xlabel('Raw-scan order')
    axes[1][1].set_ylabel('Seconds')

    for axis in axes.flat:
        axis.grid(axis='y', alpha=0.20)
    figure.suptitle(
        'Observed reader throughput and timing\n'
        'One multiwell raw scan is counted exactly once', fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    _add_debug_notice(figure, records, minimum_bottom=0.10)
    return [_save(figure, os.path.join(
        output_dir, 'reader_throughput',
        'stability_reader_throughput_final.png',
    ))]


def write_stability_dashboard_plot_suite(records, output_dir):
    '''Write Stage-14F3 dashboard figures after a completed stability run.'''
    dashboard = records.get('dashboard_records', {}) if isinstance(records, dict) else {}
    if not isinstance(dashboard, dict):
        dashboard = build_stability_dashboard_records(records)
    replicate_rows = dashboard.get('replicate_rows', []) or []
    throughput_rows = dashboard.get('throughput_rows', []) or []
    if not replicate_rows and not throughput_rows:
        return []
    for directory in (
            os.path.join(output_dir, 'replicate_completeness'),
            os.path.join(output_dir, 'reader_throughput')):
        if not os.path.isdir(directory):
            os.makedirs(directory)
    paths = []
    paths.extend(_plot_replicate_loss_rates(replicate_rows, records, output_dir))
    paths.extend(_plot_replicate_signal(replicate_rows, records, output_dir))
    paths.extend(_plot_well_completeness_matrix(
        replicate_rows, records, output_dir
    ))
    paths.extend(_plot_reader_throughput(throughput_rows, records, output_dir))
    return paths
