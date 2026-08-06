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
second, then mix.  The optional per-row water policy defaults to automatic
selection from the *stock source's* temperature-module placement, matching the
existing manual dilution workflow. A row may explicitly require
temperature-controlled or ambient water when that chemistry requires an
intentional exception.
'''

from __future__ import division

import hashlib
import json
import math
import re


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
# Optional rather than required so existing validated preparation worksheets
# retain their Auto-main allocation behavior. When supplied, the ordered lists
# form an exact physical-destination contract rather than a preference. They
# intentionally mirror the separate ``loc`` and ``deck_pos`` fields used by
# reagent information rather than encoding both values in one token.
DESTINATION_LOCS_COLUMN = 'destination_locs'
DESTINATION_DECK_POSITIONS_COLUMN = 'destination_deck_positions'
# Kept as a read-only migration fallback for worksheets created during the
# short-lived combined-column implementation. New worksheets must use the two
# explicit columns above.
LEGACY_DESTINATION_TUBE_LOCATIONS_COLUMN = 'destination_tube_locations'
WATER_SOURCE_POLICY_COLUMN = 'water_source_policy'
PREPARATION_SCHEMA_VERSION = 6
MIX_CYCLES = 2
MIN_EXECUTABLE_TRANSFER_UL = 5.0
VARIABLE_SOURCE_CONCENTRATION_COLUMN = 'variable_source_concentration_mM'


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


def _parse_water_source_policy(value, row_number):
    '''Return the canonical water-selection policy for one preparation row.

    ``auto`` preserves the established manual-dilution convention: a stock
    source in the temperature module uses the temperature-controlled-water
    source (whose legacy internal key is ``ColdWaterC1.0``), and any other
    stock source uses ``WaterC1.0``.  The canonical explicit value is
    ``temperature_controlled`` because the module may heat or cool water.
    Explicit values are deliberately narrow so accidental free text cannot
    silently change reagent handling.
    '''
    normalized = 'auto' if _is_blank(value) else str(value).strip().lower()
    normalized = normalized.replace('-', '_').replace(' ', '_')
    aliases = {
        'auto': 'auto',
        'default': 'auto',
        'stock_location': 'auto',
        'temperature_controlled': 'temperature_controlled',
        'temp_controlled': 'temperature_controlled',
        'temperature_module': 'temperature_controlled',
        # Read legacy values without perpetuating an inaccurate scientific
        # label. The canonical manifest never stores ``cold``.
        'cold': 'temperature_controlled',
        'cold_water': 'temperature_controlled',
        'ambient': 'ambient',
        'room_temperature': 'ambient',
        'standard': 'ambient',
    }
    if normalized not in aliases:
        raise AutoPreparationValidationError(
            'auto_preparation row {} has invalid {} {!r}. Use auto, '
            'temperature_controlled, or ambient.'.format(
                row_number, WATER_SOURCE_POLICY_COLUMN, value
            )
        )
    return aliases[normalized]


def _concentration_label(value):
    '''Match the existing controller's ``str(float(...))`` chemical-name style.'''
    return str(float(value))


def _chemical_name(reagent, concentration_mM):
    '''Build the source/product identifier used by controller and robot mappings.'''
    return '{}C{}'.format(
        _canonical_reagent_name(reagent),
        _concentration_label(concentration_mM)
    )


def _source_base_name(chemical_name):
    '''Return a source's canonical reagent root from a chemical-name key.'''
    chemical_name = str(chemical_name)
    concentration_match = re.search(r'C\d*\.\d*$', chemical_name)
    if concentration_match is None:
        return _canonical_reagent_name(chemical_name)
    return _canonical_reagent_name(chemical_name[:concentration_match.start()])


def _source_concentration_from_name(chemical_name):
    '''Return the concentration encoded in one canonical source name or None.'''
    chemical_name = str(chemical_name)
    concentration_match = re.search(r'C(\d*\.\d*)$', chemical_name)
    if concentration_match is None:
        return None
    try:
        return float(concentration_match.group(1))
    except (TypeError, ValueError):
        return None


def build_variable_source_bindings(variable_rows, available_sources,
                                   preparations=None, require_bindings=False):
    '''Resolve explicit physical working sources for Auto variable reagents.

    ``concentration (mM)`` remains blank for an Auto variable because that is
    the established workbook marker for a model-controlled final-reaction
    concentration.  This helper instead consumes the separate optional
    ``variable source concentration (mM)`` field.  Its value identifies the
    *physical* source concentration used to create model coordinates and
    convert a selected final concentration into a transfer volume.

    Parameters are plain mappings/lists so the contract is independently
    testable without importing controller or robot code.  If the optional
    column is present, every variable must provide one finite positive source
    concentration.  In required preparation mode, that concentration must
    equal the declared working concentration for any prepared source group.
    '''
    source_rows = list(available_sources or [])
    preparation_by_group = {
        _canonical_reagent_name(preparation['stock_source_group']): preparation
        for preparation in (preparations or [])
    }

    values_by_reagent = {}
    for row_number, row in enumerate(variable_rows or [], start=1):
        if not isinstance(row, dict):
            raise AutoPreparationValidationError(
                'Variable-source row {} must be a mapping.'.format(row_number)
            )
        reagent = _canonical_reagent_name(row.get('reagent'))
        if not reagent:
            continue
        values_by_reagent.setdefault(reagent, []).append(
            row.get(VARIABLE_SOURCE_CONCENTRATION_COLUMN)
        )

    bindings = {}
    for reagent, values in sorted(values_by_reagent.items()):
        nonblank_values = [value for value in values if not _is_blank(value)]
        if not nonblank_values:
            if require_bindings:
                raise AutoPreparationValidationError(
                    'Auto variable {!r} requires {} in every variable '
                    'transfer row. Leave concentration (mM) blank; enter '
                    'the physical working-source concentration here.'
                    .format(reagent, VARIABLE_SOURCE_CONCENTRATION_COLUMN)
                )
            continue
        if len(nonblank_values) != len(values):
            raise AutoPreparationValidationError(
                'Auto variable {!r} has a partial {} declaration. Every '
                'variable transfer row must state the same physical source '
                'concentration.'.format(
                    reagent, VARIABLE_SOURCE_CONCENTRATION_COLUMN
                )
            )

        source_concentrations = [
            _finite_positive_number(
                value,
                '{} for Auto variable {!r}'.format(
                    VARIABLE_SOURCE_CONCENTRATION_COLUMN, reagent
                )
            )
            for value in nonblank_values
        ]
        reference_concentration = source_concentrations[0]
        if any(
            not math.isclose(
                concentration, reference_concentration,
                rel_tol=1e-9, abs_tol=1e-12
            )
            for concentration in source_concentrations[1:]
        ):
            raise AutoPreparationValidationError(
                'Auto variable {!r} has inconsistent {} values. It must '
                'bind to exactly one physical working-source concentration.'
                .format(reagent, VARIABLE_SOURCE_CONCENTRATION_COLUMN)
            )

        preparation = preparation_by_group.get(reagent)
        if preparation is not None:
            working_concentration = float(
                preparation['working_concentration_mM']
            )
            if not math.isclose(
                reference_concentration, working_concentration,
                rel_tol=1e-9, abs_tol=1e-12
            ):
                raise AutoPreparationValidationError(
                    'Auto variable {!r} binds to {:.12g} mM, but its '
                    'auto_preparation working source is {:.12g} mM. Bind '
                    'the variable to the working concentration, not the '
                    'stock concentration.'.format(
                        reagent,
                        reference_concentration,
                        working_concentration
                    )
                )
            bindings[reagent] = {
                'source_concentration_mM': working_concentration,
                'chemical_name': preparation['working_chemical_name'],
                'source_role': 'prepared_working_source'
            }
            continue

        matching_source_names = []
        for source in source_rows:
            if isinstance(source, dict):
                chemical_name = source.get('chemical_name')
                source_concentration = source.get('conc')
            else:
                chemical_name = source
                source_concentration = _source_concentration_from_name(source)
            if chemical_name is None:
                continue
            if _source_base_name(chemical_name) != reagent:
                continue
            if source_concentration is None:
                source_concentration = _source_concentration_from_name(chemical_name)
            try:
                source_concentration = float(source_concentration)
            except (TypeError, ValueError):
                continue
            if math.isclose(
                source_concentration, reference_concentration,
                rel_tol=1e-9, abs_tol=1e-12
            ):
                matching_source_names.append(str(chemical_name))

        if not matching_source_names:
            raise AutoPreparationValidationError(
                'Auto variable {!r} binds to {:.12g} mM, but reagent_info '
                'contains no matching physical source.'.format(
                    reagent, reference_concentration
                )
            )

        bindings[reagent] = {
            'source_concentration_mM': reference_concentration,
            'chemical_name': matching_source_names[0],
            'source_role': 'existing_deck_source'
        }

    return bindings


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


def _parse_destination_tube_locations(
        loc_value, deck_position_value, legacy_value, tube_count, row_number):
    '''Parse optional explicit preparation destinations from separate columns.

    ``destination_locs`` and ``destination_deck_positions`` are ordered,
    semicolon-separated lists, for example ``A1;A2`` and ``3;3``. Both fields
    must be blank to retain Pi-selected compatible-empty allocation, or both
    must name every requested destination exactly once. The short-lived legacy
    combined column remains a migration fallback only when both new fields are
    blank; it is not emitted or documented for new worksheets.
    '''
    loc_is_blank = _is_blank(loc_value)
    deck_position_is_blank = _is_blank(deck_position_value)
    if loc_is_blank != deck_position_is_blank:
        raise AutoPreparationValidationError(
            'auto_preparation row {} must provide both {} and {} or leave '
            'both blank.'.format(
                row_number, DESTINATION_LOCS_COLUMN,
                DESTINATION_DECK_POSITIONS_COLUMN
            )
        )

    if loc_is_blank:
        if _is_blank(legacy_value):
            return []
        return _parse_legacy_destination_tube_locations(
            legacy_value, tube_count, row_number
        )

    if not _is_blank(legacy_value):
        raise AutoPreparationValidationError(
            'auto_preparation row {} cannot combine deprecated {} with {} '
            'and {}. Remove the combined-column value.'
            .format(
                row_number, LEGACY_DESTINATION_TUBE_LOCATIONS_COLUMN,
                DESTINATION_LOCS_COLUMN, DESTINATION_DECK_POSITIONS_COLUMN
            )
        )

    locations = [token.strip().upper() for token in str(loc_value).split(';')]
    # Excel commonly returns a one-cell deck position such as ``3`` as the
    # numeric value ``3.0``.  Preserve the ordered-list interface while
    # accepting that lossless numeric representation; fractional deck
    # positions remain invalid.
    deck_positions = [
        token.strip() for token in str(deck_position_value).split(';')
    ]
    if len(locations) != tube_count or len(deck_positions) != tube_count:
        raise AutoPreparationValidationError(
            'auto_preparation row {} must name exactly {} {} and {} entries '
            'to match tube_count.'.format(
                row_number, tube_count, DESTINATION_LOCS_COLUMN,
                DESTINATION_DECK_POSITIONS_COLUMN
            )
        )

    requested_tubes = []
    seen_locations = set()
    for location, deck_position_text in zip(locations, deck_positions):
        if not re.match(r'^[A-Z]+\d+$', location):
            raise AutoPreparationValidationError(
                'auto_preparation row {} has invalid {} entry {!r}. Use '
                'well locations such as A1;A2.'.format(
                    row_number, DESTINATION_LOCS_COLUMN, location
                )
            )
        try:
            deck_position_number = float(deck_position_text)
        except (TypeError, ValueError):
            deck_position_number = None
        if (
                deck_position_number is None
                or not math.isfinite(deck_position_number)
                or deck_position_number < 0
                or deck_position_number != int(deck_position_number)
        ):
            raise AutoPreparationValidationError(
                'auto_preparation row {} has invalid {} entry {!r}. Use '
                'integer deck positions such as 3;3.'.format(
                    row_number, DESTINATION_DECK_POSITIONS_COLUMN,
                    deck_position_text
                )
            )
        requested_tube = {
            'deck_pos': int(deck_position_number),
            'loc': location
        }
        location_key = (requested_tube['deck_pos'], requested_tube['loc'])
        if location_key in seen_locations:
            raise AutoPreparationValidationError(
                'auto_preparation row {} repeats destination {}:{}.'
                .format(
                    row_number, requested_tube['deck_pos'],
                    requested_tube['loc']
                )
            )
        seen_locations.add(location_key)
        requested_tubes.append(requested_tube)
    return requested_tubes


def _parse_legacy_destination_tube_locations(value, tube_count, row_number):
    '''Read legacy combined destinations without advertising that input form.'''
    requested_tubes = []
    seen_locations = set()
    for token in str(value).split(';'):
        token = token.strip()
        match = re.match(r'^(\d+)\s*:\s*([A-Za-z]+\d+)$', token)
        if match is None:
            raise AutoPreparationValidationError(
                'auto_preparation row {} has invalid legacy {} entry {!r}. '
                'Use {} and {} as separate columns for new worksheets.'
                .format(
                    row_number, LEGACY_DESTINATION_TUBE_LOCATIONS_COLUMN,
                    token, DESTINATION_LOCS_COLUMN,
                    DESTINATION_DECK_POSITIONS_COLUMN
                )
            )
        requested_tube = {
            'deck_pos': int(match.group(1)),
            'loc': match.group(2).upper()
        }
        location_key = (requested_tube['deck_pos'], requested_tube['loc'])
        if location_key in seen_locations:
            raise AutoPreparationValidationError(
                'auto_preparation row {} repeats destination {}:{}.'
                .format(
                    row_number, requested_tube['deck_pos'],
                    requested_tube['loc']
                )
            )
        seen_locations.add(location_key)
        requested_tubes.append(requested_tube)
    if len(requested_tubes) != tube_count:
        raise AutoPreparationValidationError(
            'auto_preparation row {} names {} destination tube(s), but '
            'tube_count is {}. Name every destination exactly once.'
            .format(row_number, len(requested_tubes), tube_count)
        )
    return requested_tubes


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
    requested_destination_locations = set()

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
        requested_destination_tubes = _parse_destination_tube_locations(
            row.get(DESTINATION_LOCS_COLUMN),
            row.get(DESTINATION_DECK_POSITIONS_COLUMN),
            row.get(LEGACY_DESTINATION_TUBE_LOCATIONS_COLUMN),
            tube_count,
            row_number
        )
        water_source_policy = _parse_water_source_policy(
            row.get(WATER_SOURCE_POLICY_COLUMN),
            row_number
        )
        for requested_tube in requested_destination_tubes:
            location_key = (
                requested_tube['deck_pos'], requested_tube['loc']
            )
            if location_key in requested_destination_locations:
                raise AutoPreparationValidationError(
                    'auto_preparation repeats destination {}:{} across '
                    'enabled preparation groups.'.format(*location_key)
                )
            requested_destination_locations.add(location_key)
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
            requested_tube = (
                requested_destination_tubes[tube_index - 1]
                if requested_destination_tubes else None
            )
            tube_plan.append({
                'tube_index': tube_index,
                'final_volume_uL': final_volume_per_tube_uL,
                'stock_transfer_uL': stock_transfer_per_tube_uL,
                'water_transfer_uL': water_transfer_per_tube_uL,
                'destination_labware': destination_labware,
                'destination_container': destination_container,
                'requested_destination_tube': requested_tube,
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
            'water_source_policy': water_source_policy,
            'destination_labware': destination_labware,
            'destination_container': destination_container,
            'requested_destination_tubes': requested_destination_tubes,
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
