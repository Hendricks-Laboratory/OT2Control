'''Pure DEBUG-ONLY synthetic evidence for the Stage-12F dry-debug harness.

This module never replaces, edits, or suppresses a reader file. The
controller first loads and blank-corrects every manifest-linked raw spectrum
through the normal reader path. Only after that succeeds may this module build
deterministic *synthetic companion evidence* keyed to the same raw scan IDs,
logical wells, and recorded scan ordering.

The resulting spectra exist solely to exercise stability reporting, companion
model fitting, and stability-only selection without claiming that an empty
plate produced nanocrystal optical evidence. Callers must label every derived
artifact as DEBUG ONLY and must never use these records for checkpoints,
imports, or scientific reporting.
'''

from __future__ import division

import json
import math


SYNTHETIC_COMPANION_EVIDENCE_SOURCE = 'synthetic_companion_evidence'
SYNTHETIC_REFERENCE_WAVELENGTH_NM = 625.0


class AutoStabilityDebugEvidenceError(ValueError):
    '''Raised when real dry-debug scan provenance cannot support a fixture.'''


def _split_wellnames(value):
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value or '').split(';')
    return [str(item).strip() for item in values if str(item).strip()]


def _condition_identity(row, fallback_ordinal):
    return '|'.join([
        str(row.get('origin_run_directory', '')).strip(),
        str(row.get('batch_number', '')).strip(),
        str(row.get('reaction_number', fallback_ordinal)).strip(),
    ])


def _condition_ids_by_well(condition_rows):
    '''Return one immutable condition identity for every expected replicate.'''
    identities = {}
    for ordinal, row in enumerate(condition_rows or []):
        if not isinstance(row, dict):
            continue
        executed = row.get('executed_in_current_run', True)
        if executed is False or str(executed).strip().lower() in (
                'false', '0', 'no'):
            continue
        try:
            samples = json.loads(row.get('replicate_sample_names', '[]'))
        except (TypeError, ValueError):
            samples = []
        if not isinstance(samples, list):
            continue
        condition_id = _condition_identity(row, ordinal)
        for sample in samples:
            wellname = str(sample).strip()
            if not wellname:
                continue
            if wellname in identities and identities[wellname] != condition_id:
                raise AutoStabilityDebugEvidenceError(
                    'DEBUG synthetic evidence found one logical well in '
                    'multiple condition identities: {}.'.format(wellname)
                )
            identities[wellname] = condition_id
    return identities


def _completed_scan_pairs(manifest_rows):
    '''Return raw-scan/well pairs in the reader's durable recorded order.'''
    pairs_by_well = {}
    for ordinal, row in enumerate(manifest_rows or []):
        if not isinstance(row, dict) or row.get('event_type') != 'raw_scan_completed':
            continue
        raw_scan_id = str(row.get('raw_scan_id', '')).strip()
        if not raw_scan_id:
            raise AutoStabilityDebugEvidenceError(
                'DEBUG synthetic evidence requires every completed raw scan '
                'to have a durable raw_scan_id.'
            )
        scan_order_key = (
            str(row.get('scan_started_at_utc', '')).strip(),
            raw_scan_id,
            ordinal,
        )
        for wellname in _split_wellnames(row.get('active_wellnames')):
            pairs_by_well.setdefault(wellname, []).append((
                scan_order_key, raw_scan_id
            ))
    for pairs in pairs_by_well.values():
        pairs.sort(key=lambda item: item[0])
    return pairs_by_well


def _has_finite_observed_spectrum(observed_spectra, raw_scan_id, wellname):
    record = observed_spectra.get(raw_scan_id, {}).get(wellname, {})
    spectrum = record.get('spectrum_by_wavelength_nm', {})
    if not isinstance(spectrum, dict):
        return False
    for wavelength, absorbance in spectrum.items():
        try:
            if math.isfinite(float(wavelength)) and math.isfinite(float(absorbance)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def build_synthetic_companion_evidence(
        manifest_rows, observed_spectra_by_raw_scan, condition_rows):
    '''Create deterministic, condition-consistent test spectra after raw load.

    Each real manifest pair must have a finite blank-corrected reader spectrum
    before a synthetic spectrum is permitted. Replicates of one condition
    receive identical trajectories, while different condition identities
    receive bounded, deterministic peak/loss values. This gives the two
    companion GPs valid, distinguishable inputs without manufacturing an
    interpretation of the actual blank spectra.
    '''
    observed_spectra = observed_spectra_by_raw_scan or {}
    condition_by_well = _condition_ids_by_well(condition_rows)
    pairs_by_well = _completed_scan_pairs(manifest_rows)
    if not pairs_by_well:
        raise AutoStabilityDebugEvidenceError(
            'DEBUG synthetic evidence requires at least one completed '
            'manifest-linked raw stability scan.'
        )

    missing_conditions = sorted(
        wellname for wellname in pairs_by_well if wellname not in condition_by_well
    )
    if missing_conditions:
        raise AutoStabilityDebugEvidenceError(
            'DEBUG synthetic evidence could not map manifest well(s) to '
            'current-run condition provenance: {}.'.format(
                ', '.join(missing_conditions)
            )
        )

    condition_ids = sorted(set(condition_by_well.values()))
    condition_rank = {
        condition_id: index for index, condition_id in enumerate(condition_ids)
    }
    synthetic_spectra = {}
    evidence_rows = []
    for wellname, pairs in sorted(pairs_by_well.items()):
        condition_id = condition_by_well[wellname]
        rank = condition_rank[condition_id]
        # These bounded values are deliberately simple and documented. They
        # are not a chemistry emulator: they provide a high, non-saturated
        # initial peak followed by a positive post-peak decline. Distinct
        # conditions supply distinguishable model targets; replicate wells
        # remain exactly concordant so replicate-QC plumbing is testable.
        initial_peak = 0.42 + (0.10 * (rank % 4))
        per_scan_decline = 0.025 + (0.005 * (rank % 3))
        for scan_ordinal, (_scan_key, raw_scan_id) in enumerate(pairs):
            if not _has_finite_observed_spectrum(
                    observed_spectra, raw_scan_id, wellname):
                raise AutoStabilityDebugEvidenceError(
                    'DEBUG synthetic evidence refuses to bypass missing or '
                    'non-finite reader data for raw scan {} well {}.'
                    .format(raw_scan_id, wellname)
                )
            reference_absorbance = initial_peak - (
                per_scan_decline * scan_ordinal
            )
            # Keep the maximum at the fixed reference wavelength, with a
            # minimal three-point spectrum sufficient for the established
            # reader-independent reporting parser.
            synthetic_spectra.setdefault(raw_scan_id, {})[wellname] = {
                'spectrum_by_wavelength_nm': {
                    SYNTHETIC_REFERENCE_WAVELENGTH_NM - 1.0: (
                        reference_absorbance - 0.05
                    ),
                    SYNTHETIC_REFERENCE_WAVELENGTH_NM: reference_absorbance,
                    SYNTHETIC_REFERENCE_WAVELENGTH_NM + 1.0: (
                        reference_absorbance - 0.05
                    ),
                },
            }
            evidence_rows.append({
                'stability_evidence_source': SYNTHETIC_COMPANION_EVIDENCE_SOURCE,
                'debug_only': True,
                'raw_scan_id': raw_scan_id,
                'wellname': wellname,
                'condition_id': condition_id,
                'synthetic_scan_ordinal': scan_ordinal,
                'synthetic_reference_wavelength_nm': (
                    SYNTHETIC_REFERENCE_WAVELENGTH_NM
                ),
                'synthetic_reference_absorbance': reference_absorbance,
                'synthetic_initial_peak_absorbance': initial_peak,
                'synthetic_per_scan_decline_absorbance': per_scan_decline,
                'raw_reader_spectrum_verified': True,
                'disclaimer': (
                    'DEBUG ONLY: deterministic synthetic companion evidence; '
                    'not a measurement and never scientific model history.'
                ),
            })
    return synthetic_spectra, evidence_rows
