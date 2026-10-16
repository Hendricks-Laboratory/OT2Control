'''Review exports for one supervised P300 well-mix calibration.

The calibration acquires eleven independent, unmerged raw scans. This module
turns those immutable reader observations into an audit manifest, a compact
per-cycle CSV, and human-review plots. It deliberately does not select a
final mix count: the observed convergence evidence is exported for a
scientist to review before any production stability default changes.
'''

import csv
import datetime
import json
import math
import os
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import pandas as pd

from auto_mix_calibration import MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM


class AutoMixCalibrationReportingError(RuntimeError):
    '''Raised when a raw calibration observation cannot be audited safely.'''


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _atomic_json_dump(value, path):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise AutoMixCalibrationReportingError(
            'Calibration report directory does not exist: {}.'.format(
                directory
            )
        )
    temporary_path = None
    try:
        with NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=directory,
                prefix='.mix_calibration_', suffix='.tmp', delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write('\n')
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except (IOError, OSError, TypeError, ValueError) as exc:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise AutoMixCalibrationReportingError(
            'Could not write calibration JSON artifact: {}.'.format(exc)
        )
    return path


def write_auto_mix_calibration_execution_record(value, path):
    '''Atomically persist the controller-owned execution summary.

    This small public boundary keeps ``controller.py`` from depending on a
    private reporting helper while retaining the same no-partial-file write
    semantics used by the manifest.
    '''
    return _atomic_json_dump(value, path)


def _atomic_csv_rows(rows, path):
    rows = [dict(row) for row in rows]
    columns = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    if not columns:
        columns = ['scan_index']
    directory = os.path.dirname(os.path.abspath(path))
    temporary_path = None
    try:
        with NamedTemporaryFile(
                mode='w', encoding='utf-8', newline='', dir=directory,
                prefix='.mix_calibration_', suffix='.tmp', delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            writer = csv.DictWriter(temporary_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except (IOError, OSError, ValueError) as exc:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise AutoMixCalibrationReportingError(
            'Could not write calibration CSV artifact: {}.'.format(exc)
        )
    return path


def _normalized_spectrum(spectrum):
    '''Convert one reader series into an ordered finite wavelength series.'''
    if isinstance(spectrum, pd.DataFrame):
        if spectrum.shape[1] != 1:
            raise AutoMixCalibrationReportingError(
                'Calibration reader scan must contain exactly one selected well.'
            )
        spectrum = spectrum.iloc[:, 0]
    if not isinstance(spectrum, pd.Series):
        raise AutoMixCalibrationReportingError(
            'Calibration reader scan must be provided as a pandas Series.'
        )
    try:
        wavelengths = pd.to_numeric(spectrum.index, errors='raise').astype(float)
        absorbance = pd.to_numeric(spectrum, errors='raise').astype(float)
    except (TypeError, ValueError):
        raise AutoMixCalibrationReportingError(
            'Calibration reader scan has nonnumeric wavelength or absorbance data.'
        )
    normalized = pd.Series(absorbance.values, index=wavelengths)
    normalized = normalized[normalized.index.notnull()]
    normalized = normalized[normalized.apply(math.isfinite)]
    if normalized.empty or normalized.index.duplicated().any():
        raise AutoMixCalibrationReportingError(
            'Calibration reader scan has no usable unique spectrum.'
        )
    return normalized.sort_index()


def _exact_wavelength_value(spectrum, wavelength_nm):
    matches = spectrum.index[abs(spectrum.index - float(wavelength_nm)) < 1e-6]
    if len(matches) != 1:
        raise AutoMixCalibrationReportingError(
            'Calibration reader scan is missing required {} nm data.'
            .format(wavelength_nm)
        )
    return float(spectrum.loc[matches[0]])


def _series_difference_metrics(previous, current):
    shared = previous.index.intersection(current.index)
    if shared.empty:
        raise AutoMixCalibrationReportingError(
            'Successive calibration scans have no shared wavelength grid.'
        )
    difference = current.loc[shared] - previous.loc[shared]
    return {
        'max_absorbance_change_from_previous': float(difference.abs().max()),
        'rms_absorbance_change_from_previous': float(
            math.sqrt(float((difference ** 2).mean()))
        ),
        'shared_wavelength_count': int(len(shared)),
    }


class AutoMixCalibrationReporter(object):
    '''Persist transparent evidence for a human mix-count decision.'''

    def __init__(self, output_root, plan, preflight_record):
        if not isinstance(output_root, str) or not output_root.strip():
            raise AutoMixCalibrationReportingError(
                'Calibration output root must be a nonblank path.'
            )
        self.output_root = os.path.abspath(output_root)
        self.data_root = os.path.join(
            self.output_root, 'pr_data', 'mix_calibration'
        )
        self.raw_scan_root = os.path.join(self.data_root, 'raw_scans')
        self.plot_root = os.path.join(
            self.output_root, 'Plots', 'mix_calibration'
        )
        for path in (self.data_root, self.raw_scan_root, self.plot_root):
            os.makedirs(path, exist_ok=True)
        self.plan = dict(plan)
        self.preflight_record = dict(preflight_record)
        self.observations = []

    def raw_scan_relative_path(self, scan_index):
        return os.path.join(
            'mix_calibration', 'raw_scans', 'scan_{:02d}.csv'.format(scan_index)
        )

    def record_scan(self, scan_step, spectrum, scan_started_at_utc,
                    scan_completed_at_utc):
        '''Add one parsed reader result and update durable review artifacts.'''
        spectrum = _normalized_spectrum(spectrum)
        scan_index = scan_step['scan_index']
        if scan_index != len(self.observations):
            raise AutoMixCalibrationReportingError(
                'Calibration scans must be recorded sequentially from 0.'
            )
        metric = {
            'scan_index': int(scan_index),
            'cycle_index': int(scan_step['cycle_index']),
            'observation_reason': str(scan_step['observation_reason']),
            'reader_well': str(scan_step['reader_well']),
            'logical_wellname': str(scan_step['wellname']),
            'raw_scan_relative_path': self.raw_scan_relative_path(scan_index),
            'scan_started_at_utc': str(scan_started_at_utc),
            'scan_completed_at_utc': str(scan_completed_at_utc),
            'wavelength_count': int(len(spectrum)),
        }
        try:
            scan_start = datetime.datetime.fromisoformat(
                str(scan_started_at_utc).replace('Z', '+00:00')
            )
            baseline_start = (
                scan_start if not self.observations else
                datetime.datetime.fromisoformat(
                    str(self.observations[0]['metric']['scan_started_at_utc'])
                    .replace('Z', '+00:00')
                )
            )
            metric['elapsed_s_from_baseline_scan_start'] = float(
                (scan_start - baseline_start).total_seconds()
            )
        except (TypeError, ValueError):
            raise AutoMixCalibrationReportingError(
                'Calibration scan timestamps must be ISO-8601 values.'
            )
        for wavelength_nm in MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM:
            column = 'absorbance_{}nm'.format(wavelength_nm)
            try:
                metric[column] = _exact_wavelength_value(spectrum, wavelength_nm)
            except AutoMixCalibrationReportingError:
                metric[column] = None
        if self.observations:
            previous = self.observations[-1]
            metric.update(_series_difference_metrics(previous['spectrum'], spectrum))
            for wavelength_nm in MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM:
                column = 'absorbance_{}nm'.format(wavelength_nm)
                previous_value = previous['metric'].get(column)
                current_value = metric.get(column)
                metric['delta_{}nm_from_previous'.format(wavelength_nm)] = (
                    None if previous_value is None or current_value is None else
                    float(current_value - previous_value)
                )
        else:
            metric['max_absorbance_change_from_previous'] = None
            metric['rms_absorbance_change_from_previous'] = None
            metric['shared_wavelength_count'] = None
            for wavelength_nm in MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM:
                metric['delta_{}nm_from_previous'.format(wavelength_nm)] = None
        self.observations.append({'metric': metric, 'spectrum': spectrum})
        self.write_progress()
        return dict(metric)

    def scan_record(self, scan_step, metric):
        return {
            'scan_index': int(scan_step['scan_index']),
            'cycle_index': int(scan_step['cycle_index']),
            'reader_well': str(scan_step['reader_well']),
            'logical_wellname': str(scan_step['wellname']),
            'raw_scan_relative_path': str(metric['raw_scan_relative_path']),
            'scan_started_at_utc': str(metric['scan_started_at_utc']),
            'scan_completed_at_utc': str(metric['scan_completed_at_utc']),
            'wavelengths_nm': [float(value) for value in self.observations[-1]['spectrum'].index],
        }

    def _manifest(self, status):
        return {
            'record_type': 'auto_mix_calibration_manifest',
            'manifest_version': 1,
            'generated_at_utc': _utc_now(),
            'status': status,
            'calibration_id': self.plan['calibration_id'],
            'assay_profile': self.plan['assay_profile'],
            'target': dict(self.plan['target']),
            'expected_scan_count': self.plan['expected_scan_count'],
            'completed_scan_count': len(self.observations),
            'automatic_retry_permitted': False,
            'human_mix_count_selection_required': True,
            'preflight_record_path': self.preflight_record.get('record_path'),
            'scan_records': [dict(item['metric']) for item in self.observations],
        }

    def write_progress(self):
        _atomic_csv_rows(
            [item['metric'] for item in self.observations],
            os.path.join(self.data_root, 'mix_calibration_cycle_metrics.csv')
        )
        _atomic_json_dump(
            self._manifest('in_progress'),
            os.path.join(self.data_root, 'mix_calibration_manifest.json')
        )

    def finalize(self):
        '''Write final plots and a completion manifest; never choose a cycle.'''
        if not self.observations:
            raise AutoMixCalibrationReportingError(
                'Cannot finalize a calibration with no reader observations.'
            )
        spectral_path = os.path.join(
            self.plot_root, 'spectral_overlay_by_cycle.png'
        )
        fig, axis = plt.subplots(figsize=(10, 6), dpi=180)
        for observation in self.observations:
            metric = observation['metric']
            axis.plot(
                observation['spectrum'].index,
                observation['spectrum'].values,
                linewidth=1.2,
                label='baseline' if metric['cycle_index'] == 0 else
                'mix {}'.format(metric['cycle_index'])
            )
        axis.set_title('P300 Well-Mix Calibration: Raw Spectra by Cycle')
        axis.set_xlabel('Wavelength (nm)')
        axis.set_ylabel('Absorbance (AU)')
        axis.legend(ncol=2, fontsize=8)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(spectral_path)
        plt.close(fig)

        convergence_path = os.path.join(
            self.plot_root, 'absorbance_convergence_520_730nm.png'
        )
        fig, axis = plt.subplots(figsize=(9, 5), dpi=180)
        cycles = [item['metric']['cycle_index'] for item in self.observations]
        plotted = False
        for wavelength_nm in MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM:
            values = [
                item['metric'].get('absorbance_{}nm'.format(wavelength_nm))
                for item in self.observations
            ]
            if any(value is not None for value in values):
                axis.plot(cycles, values, marker='o', label='{} nm'.format(wavelength_nm))
                plotted = True
        axis.set_title('P300 Well-Mix Calibration: Absorbance by Mix Cycle')
        axis.set_xlabel('Completed P300 mix cycles')
        axis.set_ylabel('Absorbance (AU)')
        if plotted:
            axis.legend()
        else:
            axis.text(
                0.5, 0.5,
                '520/730 nm review is not applicable to the water-safety profile.',
                ha='center', va='center', transform=axis.transAxes
            )
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(convergence_path)
        plt.close(fig)
        manifest_path = _atomic_json_dump(
            self._manifest('completed'),
            os.path.join(self.data_root, 'mix_calibration_manifest.json')
        )
        return {
            'manifest_path': manifest_path,
            'metrics_path': os.path.join(
                self.data_root, 'mix_calibration_cycle_metrics.csv'
            ),
            'spectral_overlay_path': spectral_path,
            'convergence_path': convergence_path,
        }
