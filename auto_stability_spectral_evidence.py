'''Pure, raw-reader spectral evidence for final stability reporting.

Stage 14F2 retains the observed, blank-corrected spectra that the controller
already loaded through the established reader path.  It deliberately does not
open a reader file, alter a manifest, interpolate a wavelength grid, or use
DEBUG synthetic companion spectra.  The resulting records are reporting-only
and cannot affect stability metrics, companion models, or recipe selection.
'''

from __future__ import division

import math


RAW_READER_SPECTRAL_EVIDENCE_SOURCE = 'raw_reader'


def _text(value):
    return '' if value is None else str(value).strip()


def _finite_number(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _split_semicolon(value):
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = _text(value).split(';')
    return [_text(item) for item in values if _text(item)]


def _active_locations(value):
    '''Return the saved logical-well to physical-reader mapping without guesswork.'''
    locations = {}
    for item in _split_semicolon(value):
        if '=' not in item:
            continue
        wellname, location = item.split('=', 1)
        wellname = _text(wellname)
        location = _text(location)
        if wellname and location and wellname not in locations:
            locations[wellname] = location
    return locations


def _normalized_spectrum(spectrum_record):
    '''Return finite observed points in native wavelength order, never resampled.'''
    if not isinstance(spectrum_record, dict):
        return []
    source = spectrum_record.get('spectrum_by_wavelength_nm', {})
    if not isinstance(source, dict):
        return []
    points = []
    for wavelength, absorbance in source.items():
        parsed_wavelength = _finite_number(wavelength)
        parsed_absorbance = _finite_number(absorbance)
        if parsed_wavelength is None or parsed_absorbance is None:
            continue
        points.append((parsed_wavelength, parsed_absorbance))
    points.sort(key=lambda item: item[0])
    return points


def _interpolated_crossing(left, right, threshold):
    '''Return one linear threshold crossing only within an observed bracket.'''
    left_wavelength, left_absorbance = left
    right_wavelength, right_absorbance = right
    if left_absorbance == right_absorbance:
        return None
    fraction = (threshold - left_absorbance) / (
        right_absorbance - left_absorbance
    )
    if fraction < 0.0 or fraction > 1.0:
        return None
    return left_wavelength + fraction * (right_wavelength - left_wavelength)


def _local_peak_indices(points):
    '''Return strict interior local maxima; boundary maxima are never accepted.'''
    indices = []
    for index in range(1, len(points) - 1):
        previous = points[index - 1][1]
        current = points[index][1]
        following = points[index + 1][1]
        if current > previous and current > following:
            indices.append(index)
    return indices


def extract_observed_spectral_features(points):
    '''Extract conservative plotting features from one native-grid spectrum.

    FWHM is reported only for one strict, interior local maximum with observed
    half-height crossings on both sides.  Crossings are linearly located
    inside their immediately adjacent measured bracket; missing grids and
    ambiguous peak shapes remain explicitly undefined rather than inferred.
    '''
    feature = {
        'spectrum_status': 'available' if points else 'raw_spectrum_unavailable',
        'native_point_count': len(points),
        'wavelength_min_nm': points[0][0] if points else None,
        'wavelength_max_nm': points[-1][0] if points else None,
        'lambda_max_nm': None,
        'peak_absorbance': None,
        'fwhm_nm': None,
        'fwhm_status': 'raw_spectrum_unavailable',
        'half_height_absorbance': None,
    }
    if not points:
        return feature
    if len(points) < 3:
        feature['fwhm_status'] = 'insufficient_native_wavelength_points'
        return feature

    peak_absorbance = max(point[1] for point in points)
    maximum_indices = [
        index for index, point in enumerate(points)
        if point[1] == peak_absorbance
    ]
    # A tied maximum cannot supply one physically defensible FWHM centre.
    if len(maximum_indices) != 1:
        feature['fwhm_status'] = 'nonunique_global_maximum'
        return feature
    peak_index = maximum_indices[0]
    feature['lambda_max_nm'] = points[peak_index][0]
    feature['peak_absorbance'] = peak_absorbance
    if peak_index == 0 or peak_index == len(points) - 1:
        feature['fwhm_status'] = 'peak_at_wavelength_boundary'
        return feature
    if peak_absorbance <= 0.0:
        feature['fwhm_status'] = 'nonpositive_peak_absorbance'
        return feature

    local_peaks = _local_peak_indices(points)
    if local_peaks != [peak_index]:
        feature['fwhm_status'] = 'ambiguous_or_multiple_local_peaks'
        return feature

    half_height = peak_absorbance / 2.0
    feature['half_height_absorbance'] = half_height
    left_crossing = None
    for index in range(peak_index - 1, -1, -1):
        if points[index][1] <= half_height:
            left_crossing = _interpolated_crossing(
                points[index], points[index + 1], half_height
            )
            break
    if left_crossing is None:
        feature['fwhm_status'] = 'left_half_height_crossing_missing'
        return feature

    right_crossing = None
    for index in range(peak_index + 1, len(points)):
        if points[index][1] <= half_height:
            right_crossing = _interpolated_crossing(
                points[index - 1], points[index], half_height
            )
            break
    if right_crossing is None:
        feature['fwhm_status'] = 'right_half_height_crossing_missing'
        return feature

    feature['fwhm_nm'] = right_crossing - left_crossing
    feature['fwhm_status'] = 'defined'
    return feature


def build_raw_spectral_evidence_records(
        manifest_rows, observed_spectra_by_raw_scan, trajectory_rows,
        well_metrics=None):
    '''Build immutable-context observed-spectral records for Stage 14F2.

    ``trajectory_rows`` contribute timing and decision-window status only.
    They may be derived from DEBUG synthetic companion evidence, but the
    spectra here always originate from ``observed_spectra_by_raw_scan`` before
    any DEBUG substitution.  The input collections are never mutated.
    '''
    trajectory_by_pair = {}
    for original_row in trajectory_rows or []:
        if not isinstance(original_row, dict):
            continue
        row = dict(original_row)
        scan_id = _text(row.get('raw_scan_id'))
        wellname = _text(row.get('wellname'))
        if scan_id and wellname:
            trajectory_by_pair[(scan_id, wellname)] = row

    metrics_by_well = {}
    for original_row in well_metrics or []:
        if not isinstance(original_row, dict):
            continue
        wellname = _text(original_row.get('wellname'))
        if wellname:
            metrics_by_well[wellname] = dict(original_row)

    completed_scans = [
        dict(row) for row in manifest_rows or []
        if isinstance(row, dict) and row.get('event_type') == 'raw_scan_completed'
    ]
    completed_scans.sort(key=lambda row: (
        _text(row.get('scan_started_at_utc')),
        _text(row.get('raw_scan_id')),
    ))

    evidence_rows = []
    spectra_by_pair = {}
    for scan_row in completed_scans:
        scan_id = _text(scan_row.get('raw_scan_id'))
        if not scan_id:
            continue
        locations = _active_locations(scan_row.get('active_well_locations'))
        for wellname in _split_semicolon(scan_row.get('active_wellnames')):
            trajectory = trajectory_by_pair.get((scan_id, wellname), {})
            metric = metrics_by_well.get(wellname, {})
            points = _normalized_spectrum(
                (observed_spectra_by_raw_scan or {}).get(scan_id, {}).get(
                    wellname, {}
                )
            )
            features = extract_observed_spectral_features(points)
            row = {
                'raw_scan_id': scan_id,
                'raw_scan_relative_path': _text(
                    scan_row.get('raw_scan_relative_path')
                ),
                'wellname': wellname,
                'reader_location': locations.get(wellname, ''),
                'condition_id': _text(trajectory.get('condition_id')),
                'condition_type': _text(metric.get('condition_type')),
                'batch_number': metric.get('batch_number', ''),
                'reaction_number': metric.get('reaction_number', ''),
                'trigger_completed_at_utc': _text(
                    trajectory.get('trigger_completed_at_utc')
                ),
                'scan_started_at_utc': _text(
                    trajectory.get('scan_started_at_utc')
                ),
                'scan_completed_at_utc': _text(
                    trajectory.get('scan_completed_at_utc')
                ),
                'elapsed_s_from_trigger_to_scan_start': trajectory.get(
                    'elapsed_s_from_trigger_to_scan_start'
                ),
                'window_interval_status': _text(
                    trajectory.get('window_interval_status')
                ),
                'metric_eligible_interval': trajectory.get(
                    'metric_eligible_interval'
                ),
                'stability_evidence_source': RAW_READER_SPECTRAL_EVIDENCE_SOURCE,
                'debug_only': False,
            }
            row.update(features)
            evidence_rows.append(row)
            spectra_by_pair[(scan_id, wellname)] = list(points)

    return {
        'evidence_rows': evidence_rows,
        'spectra_by_pair': spectra_by_pair,
        'evidence_source': RAW_READER_SPECTRAL_EVIDENCE_SOURCE,
    }
