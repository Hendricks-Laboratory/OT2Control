'''Pure contracts for the Stage 14G3 monitor-only chemistry-pilot route.

The monitor-only route may execute an explicitly supplied set of physical
conditions, but it must never generate, score, or select a recipe.  This
module deliberately contains no controller, spreadsheet client, robot,
plate-reader, or optimization imports so workbook data can be rejected before
any hardware-adjacent action is possible.
'''

from __future__ import division

import hashlib
import json
import math


STABILITY_MONITOR_MODE_OFF = 'off'
STABILITY_MONITOR_MODE_REQUIRED = 'required'
STABILITY_MONITOR_WORKSHEET_NAME = 'stability_monitor_conditions'


class AutoStabilityMonitorValidationError(ValueError):
    '''Raised when a monitor-only run is incomplete or ambiguous.'''


def _blank(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ''


def _token(value):
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')


def parse_auto_stability_monitor_header_settings(header_values):
    '''Parse the opt-in monitor-only Header setting without side effects.'''
    if not isinstance(header_values, dict):
        raise AutoStabilityMonitorValidationError(
            'Header settings must be supplied as a dictionary.'
        )
    raw = _token(header_values.get('auto_stability_monitor_only', ''))
    aliases = {
        '': STABILITY_MONITOR_MODE_OFF,
        'off': STABILITY_MONITOR_MODE_OFF,
        'false': STABILITY_MONITOR_MODE_OFF,
        '0': STABILITY_MONITOR_MODE_OFF,
        'required': STABILITY_MONITOR_MODE_REQUIRED,
        'on': STABILITY_MONITOR_MODE_REQUIRED,
        'true': STABILITY_MONITOR_MODE_REQUIRED,
        '1': STABILITY_MONITOR_MODE_REQUIRED,
    }
    if raw not in aliases:
        raise AutoStabilityMonitorValidationError(
            'Header value auto_stability_monitor_only must be off or '
            'required. Received: {!r}.'.format(raw)
        )
    return {
        'auto_stability_monitor_only': aliases[raw],
        'stability_monitor_conditions_worksheet': (
            STABILITY_MONITOR_WORKSHEET_NAME
        ),
    }


def _finite_nonnegative(value, label):
    if _blank(value):
        raise AutoStabilityMonitorValidationError(
            '{} is required.'.format(label)
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise AutoStabilityMonitorValidationError(
            '{} must be a finite nonnegative concentration; received {!r}.'
            .format(label, value)
        )
    if not math.isfinite(numeric) or numeric < 0.0:
        raise AutoStabilityMonitorValidationError(
            '{} must be a finite nonnegative concentration; received {!r}.'
            .format(label, value)
        )
    return numeric


def _positive_integer(value, label):
    if _blank(value):
        raise AutoStabilityMonitorValidationError(
            '{} is required.'.format(label)
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float('nan')
    if not math.isfinite(numeric) or numeric < 1 or int(numeric) != numeric:
        raise AutoStabilityMonitorValidationError(
            '{} must be a positive whole number; received {!r}.'.format(
                label, value
            )
        )
    return int(numeric)


def build_stability_monitor_condition_manifest(rows, variable_reagents):
    '''Validate explicit chemistry-monitor conditions into an immutable plan.

    Required columns are ``condition_id``, ``replicate_count``, and one exact
    concentration column for every variable reagent in the Input Template.
    Fixed-reagent concentrations remain defined by that existing template.
    ``notes`` is optional audit text. No concentrations are inferred or
    optimized here.
    '''
    variable_reagents = [str(value) for value in variable_reagents]
    if not variable_reagents:
        raise AutoStabilityMonitorValidationError(
            'Monitor-only chemistry requires at least one variable reagent.'
        )
    if not isinstance(rows, (list, tuple)):
        raise AutoStabilityMonitorValidationError(
            'Monitor-only condition rows must be a list.'
        )
    required = set(['condition_id', 'replicate_count'] + variable_reagents)
    conditions = []
    seen_ids = set()
    for row_number, raw_row in enumerate(rows, start=2):
        if not isinstance(raw_row, dict):
            raise AutoStabilityMonitorValidationError(
                'Monitor-only row {} is not a table row.'.format(row_number)
            )
        if all(_blank(value) for value in raw_row.values()):
            continue
        missing = sorted(
            column for column in required if column not in raw_row
        )
        if missing:
            raise AutoStabilityMonitorValidationError(
                'Monitor-only row {} is missing required column(s): {}.'
                .format(row_number, ', '.join(missing))
            )
        condition_id = str(raw_row.get('condition_id') or '').strip()
        if not condition_id:
            raise AutoStabilityMonitorValidationError(
                'Monitor-only row {} has a blank condition_id.'.format(
                    row_number
                )
            )
        if condition_id in seen_ids:
            raise AutoStabilityMonitorValidationError(
                'Monitor-only condition_id {!r} is duplicated.'.format(
                    condition_id
                )
            )
        seen_ids.add(condition_id)
        concentrations = {
            reagent: _finite_nonnegative(
                raw_row.get(reagent),
                '{} for condition {!r}'.format(reagent, condition_id)
            )
            for reagent in variable_reagents
        }
        conditions.append({
            'condition_id': condition_id,
            'replicate_count': _positive_integer(
                raw_row.get('replicate_count'),
                'replicate_count for condition {!r}'.format(condition_id)
            ),
            'variable_concentrations_mM': concentrations,
            'notes': str(raw_row.get('notes') or '').strip(),
            'source_row_number': row_number,
        })
    if not conditions:
        raise AutoStabilityMonitorValidationError(
            'No nonblank monitor-only condition rows were supplied.'
        )
    payload = {
        'schema_version': 'stage14g3-monitor-v1',
        'worksheet_name': STABILITY_MONITOR_WORKSHEET_NAME,
        'variable_reagents': variable_reagents,
        'conditions': conditions,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    payload['manifest_sha256'] = hashlib.sha256(
        canonical.encode('utf-8')
    ).hexdigest()
    return payload
