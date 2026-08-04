'''Validated, hardware-free planning primitives for Auto reagent preparation.

This module deliberately contains no controller, spreadsheet, portal, or robot
imports.  It is the Stage 9A contract for an opt-in preparation phase that will
run *once before* Auto seed and optimizer batches.  Keeping the requested
preparations as a pure manifest makes the chemistry calculation independently
testable and prevents an incomplete preparation feature from silently changing
an Auto run.

The calculation mirrors the established manual-controller dilution semantics:

    C_stock * V_stock = C_working * V_final

The future execution stage will use the existing controller convention of
adding water first, then stock reagent, then mixing the prepared destination.
'''

from __future__ import division

import hashlib
import json
import math


class AutoPreparationValidationError(ValueError):
    '''Raised when a requested Auto preparation is chemically unsafe or vague.'''


PREPARATION_WORKSHEET_NAME = 'auto_preparation'
PREPARATION_WORKSHEET_COLUMNS = (
    'enabled',
    'stock_reagent',
    'stock_concentration_mM',
    'working_concentration_mM',
    'final_volume_uL',
)
PREPARATION_SCHEMA_VERSION = 1
MIX_CYCLES = 2
MIN_EXECUTABLE_TRANSFER_UL = 5.0


def _canonical_reagent_name(value):
    '''Return the repository's conventional underscore-separated reagent name.'''
    if value is None:
        return ''
    return '_'.join(str(value).strip().split())


def _finite_positive_number(value, field_name):
    '''Parse one finite, strictly positive numeric spreadsheet value.'''
    try:
        parsed_value = float(value)
    except (TypeError, ValueError):
        raise AutoPreparationValidationError(
            '{} must be a finite positive number; received {!r}.'.format(
                field_name,
                value
            )
        )
    if not math.isfinite(parsed_value) or parsed_value <= 0:
        raise AutoPreparationValidationError(
            '{} must be a finite positive number; received {!r}.'.format(
                field_name,
                value
            )
        )
    return parsed_value


def _is_enabled(value):
    '''Interpret the optional worksheet enabled field conservatively.'''
    if value is None:
        return True
    # Spreadsheet downloads represent blank cells as floating-point NaN.  A
    # blank ``enabled`` cell is intentionally a skipped request, never an
    # unrecognised affirmative value.  This keeps template rows inert until an
    # operator explicitly enables them.
    if isinstance(value, float) and math.isnan(value):
        return False
    normalized_value = str(value).strip().lower()
    if normalized_value in ('', '0', 'false', 'no', 'n', 'off', 'disabled'):
        return False
    if normalized_value in ('1', 'true', 'yes', 'y', 'on', 'enabled'):
        return True
    raise AutoPreparationValidationError(
        'enabled must be yes/on/true/1 or no/off/false/0; received {!r}.'.format(
            value
        )
    )


def _concentration_label(value):
    '''Match the existing controller's ``str(float(...))`` chemical-name style.'''
    return str(float(value))


def _chemical_name(reagent, concentration_mM):
    '''Build the source/product identifier used by controller and robot mappings.'''
    return '{}C{}'.format(
        _canonical_reagent_name(reagent),
        _concentration_label(concentration_mM)
    )


def _validate_executable_transfer(volume_uL, field_name):
    '''Reject non-zero transfers that cannot be executed by the validated OT-2 rule.'''
    if 0 < volume_uL < MIN_EXECUTABLE_TRANSFER_UL:
        raise AutoPreparationValidationError(
            '{} is {:.6g} uL, which lies in the non-executable 0–5 uL '
            'interval.'.format(field_name, volume_uL)
        )


def build_preparation_manifest(rows, destination_container, destination_capacity_uL):
    '''Build and validate a deterministic Auto-preparation manifest.

    Parameters
    ----------
    rows : iterable of mapping
        Rows from the future ``auto_preparation`` worksheet.  The required
        columns are listed in :data:`PREPARATION_WORKSHEET_COLUMNS`.  Disabled
        rows are retained nowhere and have no effect.
    destination_container : str
        Empty-container type already configured by the existing ``dilution_cont``
        Header setting.
    destination_capacity_uL : number
        The maximum permitted prepared volume, from ``dilution_vol``.

    Returns
    -------
    dict
        JSON-serializable manifest.  It describes the requested chemistry only;
        it does not allocate a physical destination, contact the robot, or issue
        a transfer command.

    Notes
    -----
    Stage 9A intentionally permits only one destination per resulting working
    chemical name.  The later execution stage must explicitly support grouped
    duplicate working sources before it can safely prepare multiple same-name
    backup tubes.
    '''
    normalized_destination = str(destination_container).strip()
    if not normalized_destination:
        raise AutoPreparationValidationError(
            'destination_container must be provided by dilution_cont.'
        )
    capacity_uL = _finite_positive_number(
        destination_capacity_uL,
        'destination_capacity_uL'
    )

    preparations = []
    working_names = set()
    prepared_reagents = set()
    for row_number, row in enumerate(rows, start=2):
        if not isinstance(row, dict):
            raise AutoPreparationValidationError(
                'auto_preparation row {} must be a mapping.'.format(row_number)
            )
        if not _is_enabled(row.get('enabled')):
            continue

        missing_columns = [
            column for column in PREPARATION_WORKSHEET_COLUMNS[1:]
            if column not in row or str(row[column]).strip() == ''
        ]
        if missing_columns:
            raise AutoPreparationValidationError(
                'auto_preparation row {} is missing required field(s): {}.'.format(
                    row_number,
                    ', '.join(missing_columns)
                )
            )

        stock_reagent = _canonical_reagent_name(row['stock_reagent'])
        if not stock_reagent:
            raise AutoPreparationValidationError(
                'auto_preparation row {} has an empty stock_reagent.'.format(
                    row_number
                )
            )
        if stock_reagent in prepared_reagents:
            raise AutoPreparationValidationError(
                'auto_preparation requests more than one working source for '
                'reagent {!r}. One Auto reagent may use only one prepared '
                'working concentration per run.'.format(stock_reagent)
            )
        stock_concentration_mM = _finite_positive_number(
            row['stock_concentration_mM'],
            'stock_concentration_mM (row {})'.format(row_number)
        )
        working_concentration_mM = _finite_positive_number(
            row['working_concentration_mM'],
            'working_concentration_mM (row {})'.format(row_number)
        )
        final_volume_uL = _finite_positive_number(
            row['final_volume_uL'],
            'final_volume_uL (row {})'.format(row_number)
        )
        if working_concentration_mM >= stock_concentration_mM:
            raise AutoPreparationValidationError(
                'auto_preparation row {} must dilute to a concentration below '
                'the stock concentration ({} mM >= {} mM).'.format(
                    row_number,
                    working_concentration_mM,
                    stock_concentration_mM
                )
            )
        if final_volume_uL > capacity_uL:
            raise AutoPreparationValidationError(
                'auto_preparation row {} requests {:.6g} uL, exceeding the '
                'configured {} capacity of {:.6g} uL.'.format(
                    row_number,
                    final_volume_uL,
                    normalized_destination,
                    capacity_uL
                )
            )

        stock_transfer_uL = (
            final_volume_uL * working_concentration_mM /
            stock_concentration_mM
        )
        water_transfer_uL = final_volume_uL - stock_transfer_uL
        _validate_executable_transfer(stock_transfer_uL, 'stock transfer')
        _validate_executable_transfer(water_transfer_uL, 'water transfer')

        stock_chemical_name = _chemical_name(
            stock_reagent,
            stock_concentration_mM
        )
        working_chemical_name = _chemical_name(
            stock_reagent,
            working_concentration_mM
        )
        if working_chemical_name in working_names:
            raise AutoPreparationValidationError(
                'auto_preparation creates duplicate working source {!r}. '
                'Multiple same-concentration backup destinations require the '
                'future grouped-source execution stage.'.format(
                    working_chemical_name
                )
            )
        working_names.add(working_chemical_name)
        prepared_reagents.add(stock_reagent)
        preparations.append({
            'row_number': row_number,
            'stock_reagent': stock_reagent,
            'stock_chemical_name': stock_chemical_name,
            'stock_concentration_mM': stock_concentration_mM,
            'working_chemical_name': working_chemical_name,
            'working_concentration_mM': working_concentration_mM,
            'final_volume_uL': final_volume_uL,
            'stock_transfer_uL': stock_transfer_uL,
            'water_transfer_uL': water_transfer_uL,
            'mix_cycles': MIX_CYCLES,
            'water_source_policy': 'match_stock_temperature_module',
            'destination_container': normalized_destination,
            'destination_capacity_uL': capacity_uL
        })

    manifest = {
        'schema_version': PREPARATION_SCHEMA_VERSION,
        'worksheet_name': PREPARATION_WORKSHEET_NAME,
        'preparations': preparations
    }
    canonical_manifest = json.dumps(
        manifest,
        sort_keys=True,
        separators=(',', ':')
    ).encode('utf-8')
    manifest['manifest_sha256'] = hashlib.sha256(canonical_manifest).hexdigest()
    return manifest


def validate_manifest_source_names(manifest, available_source_names):
    '''Fail closed unless every planned stock exists and no output collides.

    The manifest intentionally uses the controller/Pi chemical-name format
    (for example ``sodium_borohydrideC130.0``).  This pure check is performed
    before a robot connection is created, so a misspelled source or a working
    product that would overwrite an existing source cannot reach execution.
    '''
    if not isinstance(manifest, dict):
        raise AutoPreparationValidationError('preparation manifest must be a mapping.')
    preparations = manifest.get('preparations')
    if not isinstance(preparations, list):
        raise AutoPreparationValidationError(
            'preparation manifest has invalid preparations data.'
        )

    source_names = {str(name) for name in available_source_names}
    for preparation in preparations:
        stock_name = preparation['stock_chemical_name']
        working_name = preparation['working_chemical_name']
        if stock_name not in source_names:
            raise AutoPreparationValidationError(
                'auto_preparation stock source {!r} is not present in '
                'reagent_info.'.format(stock_name)
            )
        if working_name in source_names:
            raise AutoPreparationValidationError(
                'auto_preparation working source {!r} already exists in '
                'reagent_info; refusing to overwrite or ambiguously reuse it.'
                .format(working_name)
            )
    return manifest
