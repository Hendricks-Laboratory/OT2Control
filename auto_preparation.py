'''Hardware-free planning primitives for grouped Auto working-solution preparation.

This module deliberately contains no controller, spreadsheet, portal, or robot
imports.  It turns the ``auto_preparation`` worksheet into a deterministic
manifest before an Auto run connects to the Pi.  The manifest records the
chemistry requested by the operator, but it does not allocate tubes, reserve
source liquid, or issue a transfer command.  Those physical steps belong to
the later Auto-main protocol stage and must use the Pi's runtime labware
geometry rather than a misleading legacy container-class name.

The planned dilution follows the established controller convention::

    C_stock * V_stock = C_working * V_final

For every working tube, execution will add the selected water first, add stock
second, then mix.  Cold water selection is determined by the *stock source's*
temperature-module placement, matching the existing manual dilution workflow;
the destination tube placement does not decide which water source is used.
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
    'stock_source_group',
    'stock_concentration_mM',
    'working_concentration_mM',
    'tube_count',
    'final_volume_per_tube_uL',
    'destination_labware',
    'destination_container',
)
PREPARATION_SCHEMA_VERSION = 2
MIX_CYCLES = 2
MIN_EXECUTABLE_TRANSFER_UL = 5.0


def _canonical_reagent_name(value):
    '''Return the repository's conventional underscore-separated reagent name.'''
    if value is None:
        return ''
    return '_'.join(str(value).strip().split())


def _is_blank(value):
    '''Return whether a spreadsheet value is absent without treating zero as blank.'''
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ''


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


def _positive_integer(value, field_name):
    '''Parse a finite, strictly positive integer spreadsheet value.'''
    numeric_value = _finite_positive_number(value, field_name)
    rounded_value = int(numeric_value)
    if numeric_value != rounded_value:
        raise AutoPreparationValidationError(
            '{} must be a whole number; received {!r}.'.format(
                field_name,
                value
            )
        )
    return rounded_value


def _is_enabled(value):
    '''Interpret the optional worksheet enabled field conservatively.'''
    if _is_blank(value):
        return False
    normalized_value = str(value).strip().lower()
    if normalized_value in ('0', 'false', 'no', 'n', 'off', 'disabled'):
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
    '''Reject non-zero transfers that violate the validated OT-2 5-uL rule.'''
    if 0 < volume_uL < MIN_EXECUTABLE_TRANSFER_UL:
        raise AutoPreparationValidationError(
            '{} is {:.6g} uL, which lies in the non-executable 0–5 uL '
            'interval.'.format(field_name, volume_uL)
        )


def _resolve_destination_capacity(destination_container, capacity_resolver):
    '''Resolve an optional test-time capacity without inventing labware geometry.

    Auto-main will perform the authoritative physical capacity check against the
    Pi's runtime labware geometry.  The pure planner accepts this optional
    resolver so unit tests and future callers can reject an impossible request
    without hard-coding a capacity from a legacy class name such as
    ``Tube20000uL``.
    '''
    if capacity_resolver is None:
        return None
    try:
        capacity_uL = capacity_resolver(destination_container)
    except TypeError:
        try:
            capacity_uL = capacity_resolver[destination_container]
        except (KeyError, TypeError):
            raise AutoPreparationValidationError(
                'No destination capacity is available for {!r}.'.format(
                    destination_container
                )
            )
    return _finite_positive_number(
        capacity_uL,
        'destination capacity for {!r}'.format(destination_container)
    )


def build_preparation_manifest(rows, destination_capacity_resolver=None):
    '''Build a deterministic grouped working-solution preparation manifest.

    Parameters
    ----------
    rows : iterable of mapping
        Rows from the ``auto_preparation`` worksheet.  Each enabled row defines
        one stock source group and a set of identical destination working tubes.
    destination_capacity_resolver : mapping or callable, optional
        Optional capacity source for pure validation.  Production execution
        intentionally leaves this ``None`` so Auto-main can validate the true
        runtime tube geometry and its safe fill margin.

    Returns
    -------
    dict
        JSON-serializable requested chemistry and per-tube transfer plan.  It
        contains no physical source allocation and no robot commands.
    '''
    preparations = []
    source_groups = set()
    working_names = set()

    for row_number, row in enumerate(rows, start=2):
        if not isinstance(row, dict):
            raise AutoPreparationValidationError(
                'auto_preparation row {} must be a mapping.'.format(row_number)
            )
        if not _is_enabled(row.get('enabled')):
            continue

        missing_columns = [
            column for column in PREPARATION_WORKSHEET_COLUMNS[1:]
            if column not in row or _is_blank(row[column])
        ]
        if missing_columns:
            raise AutoPreparationValidationError(
                'auto_preparation row {} is missing required field(s): {}.'.format(
                    row_number,
                    ', '.join(missing_columns)
                )
            )

        stock_source_group = _canonical_reagent_name(
            row['stock_source_group']
        )
        if not stock_source_group:
            raise AutoPreparationValidationError(
                'auto_preparation row {} has an empty stock_source_group.'.format(
                    row_number
                )
            )
        if stock_source_group in source_groups:
            raise AutoPreparationValidationError(
                'auto_preparation has more than one enabled row for stock '
                'source group {!r}. Define one working concentration and one '
                'destination-tube plan per source group.'.format(
                    stock_source_group
                )
            )

        stock_concentration_mM = _finite_positive_number(
            row['stock_concentration_mM'],
            'stock_concentration_mM (row {})'.format(row_number)
        )
        working_concentration_mM = _finite_positive_number(
            row['working_concentration_mM'],
            'working_concentration_mM (row {})'.format(row_number)
        )
        tube_count = _positive_integer(
            row['tube_count'],
            'tube_count (row {})'.format(row_number)
        )
        final_volume_per_tube_uL = _finite_positive_number(
            row['final_volume_per_tube_uL'],
            'final_volume_per_tube_uL (row {})'.format(row_number)
        )
        destination_labware = str(row['destination_labware']).strip()
        destination_container = str(row['destination_container']).strip()
        if not destination_labware or not destination_container:
            raise AutoPreparationValidationError(
                'auto_preparation row {} requires destination_labware and '
                'destination_container.'.format(row_number)
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

        destination_capacity_uL = _resolve_destination_capacity(
            destination_container,
            destination_capacity_resolver
        )
        if (
            destination_capacity_uL is not None
            and final_volume_per_tube_uL > destination_capacity_uL
        ):
            raise AutoPreparationValidationError(
                'auto_preparation row {} requests {:.6g} uL per tube, '
                'exceeding the {} capacity of {:.6g} uL.'.format(
                    row_number,
                    final_volume_per_tube_uL,
                    destination_container,
                    destination_capacity_uL
                )
            )

        stock_transfer_per_tube_uL = (
            final_volume_per_tube_uL * working_concentration_mM /
            stock_concentration_mM
        )
        water_transfer_per_tube_uL = (
            final_volume_per_tube_uL - stock_transfer_per_tube_uL
        )
        _validate_executable_transfer(
            stock_transfer_per_tube_uL,
            'stock transfer per tube (row {})'.format(row_number)
        )
        _validate_executable_transfer(
            water_transfer_per_tube_uL,
            'water transfer per tube (row {})'.format(row_number)
        )

        stock_chemical_name = _chemical_name(
            stock_source_group,
            stock_concentration_mM
        )
        working_chemical_name = _chemical_name(
            stock_source_group,
            working_concentration_mM
        )
        if working_chemical_name in working_names:
            raise AutoPreparationValidationError(
                'auto_preparation creates duplicate working source {!r}.'.format(
                    working_chemical_name
                )
            )

        tube_plan = []
        for tube_index in range(1, tube_count + 1):
            tube_plan.append({
                'tube_index': tube_index,
                'final_volume_uL': final_volume_per_tube_uL,
                'stock_transfer_uL': stock_transfer_per_tube_uL,
                'water_transfer_uL': water_transfer_per_tube_uL,
                'destination_labware': destination_labware,
                'destination_container': destination_container,
            })

        source_groups.add(stock_source_group)
        working_names.add(working_chemical_name)
        preparations.append({
            'row_number': row_number,
            'stock_source_group': stock_source_group,
            'stock_chemical_name': stock_chemical_name,
            'stock_concentration_mM': stock_concentration_mM,
            'working_source_group': working_chemical_name,
            'working_chemical_name': working_chemical_name,
            'working_concentration_mM': working_concentration_mM,
            'tube_count': tube_count,
            'final_volume_per_tube_uL': final_volume_per_tube_uL,
            'total_final_volume_uL': final_volume_per_tube_uL * tube_count,
            'stock_transfer_per_tube_uL': stock_transfer_per_tube_uL,
            'water_transfer_per_tube_uL': water_transfer_per_tube_uL,
            'total_stock_transfer_uL': stock_transfer_per_tube_uL * tube_count,
            'total_water_transfer_uL': water_transfer_per_tube_uL * tube_count,
            'mix_cycles': MIX_CYCLES,
            'water_source_policy': 'stock_temperature_module_selects_water',
            'destination_labware': destination_labware,
            'destination_container': destination_container,
            'destination_capacity_uL': destination_capacity_uL,
            'requires_runtime_destination_capacity_check': (
                destination_capacity_uL is None
            ),
            'tube_plan': tube_plan,
        })

    manifest = {
        'schema_version': PREPARATION_SCHEMA_VERSION,
        'worksheet_name': PREPARATION_WORKSHEET_NAME,
        'execution_status': 'planning_only_pending_auto_main_group_protocol',
        'preparations': preparations,
    }
    canonical_manifest = json.dumps(
        manifest,
        sort_keys=True,
        separators=(',', ':')
    ).encode('utf-8')
    manifest['manifest_sha256'] = hashlib.sha256(canonical_manifest).hexdigest()
    return manifest


def validate_manifest_source_names(manifest, available_source_names):
    '''Fail closed unless every planned stock exists and output names are unused.

    Physical group membership, source volume, water selection, destination
    allocation, and runtime labware capacity are deliberately deferred to the
    future Auto-main group protocol.  This pure check only establishes that
    preparation cannot begin from a misspelled controller-side stock name or
    overwrite an existing chemical name.
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
