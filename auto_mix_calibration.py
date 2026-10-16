'''Pure configuration contract for the supervised 96-well mix calibration.

This module deliberately contains no controller, reader, spreadsheet-I/O, or
robot imports. It defines the narrow configuration boundary for a future
calibration-only workflow that characterizes targeted P300 mixing before any
production stability default is changed.

The two-dye profile is an explicitly adapted version of the laboratory QC dye
assay: Acid Red 1 is dispensed first, then copper sulfate dye is dispensed
second by the robot. The later P300 mixing/scan study is therefore controlled
and reproducible while retaining the PDF's reagent order.
'''

import math
import re


MIX_CALIBRATION_MODE_OFF = 'off'
MIX_CALIBRATION_MODE_WATER_SAFETY = 'water_safety'
MIX_CALIBRATION_MODE_TWO_DYE = 'two_dye'

MIX_CALIBRATION_ASSAY_PROFILE_HEADER = 'auto_mix_calibration_assay_profile'
MIX_CALIBRATION_FIRST_REAGENT_HEADER = 'auto_mix_calibration_first_reagent'
MIX_CALIBRATION_SECOND_REAGENT_HEADER = 'auto_mix_calibration_second_reagent'
MIX_CALIBRATION_TARGET_READER_WELL_HEADER = (
    'auto_mix_calibration_target_reader_well'
)

MIX_CALIBRATION_PROFILE_WATER_SAFETY = 'water_safety_20ul_first'
MIX_CALIBRATION_PROFILE_QC_TWO_DYE_20UL_RED = 'qc_two_dye_20ul_red'

# These are fixed, reviewed calibration conditions rather than ordinary
# worksheet-tunable production settings. The first/second order is deliberate:
# it matches the QC dye assay's Acid Red 1 then copper sulfate dye order.
MIX_CALIBRATION_FINAL_WELL_VOLUME_UL = 200.0
MIX_CALIBRATION_FIRST_VOLUME_UL = 20.0
MIX_CALIBRATION_SECOND_VOLUME_UL = 180.0
MIX_CALIBRATION_P300_VOLUME_UL = 150.0
MIX_CALIBRATION_CYCLE_COUNT = 10
MIX_CALIBRATION_WELL_CLEARANCE_MM = 2.0

MIX_CALIBRATION_QC_ACID_RED_1_CONCENTRATION_MM = 0.36
MIX_CALIBRATION_QC_COPPER_SULFATE_CONCENTRATION_MM = 8.0
MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM = (520, 730)

MIX_CALIBRATION_READER_WELL_PATTERN = re.compile(
    r'^[A-H](?:[1-9]|1[0-2])$'
)


class AutoMixCalibrationValidationError(ValueError):
    '''Raised when the opt-in calibration Header contract is ambiguous.'''


def _header_value(header, key, default=''):
    if not isinstance(header, dict):
        raise AutoMixCalibrationValidationError(
            'Calibration Header settings must be a dictionary.'
        )
    return header.get(key, default)


def _canonical_text(value):
    if value is None:
        return ''
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')


def _require_name(value, field_name):
    if value is None or not str(value).strip():
        raise AutoMixCalibrationValidationError(
            '{} is required when auto_mix_calibration_mode is enabled.'
            .format(field_name)
        )
    return str(value).strip()


def _require_reader_well(value):
    normalized = _require_name(
        value, MIX_CALIBRATION_TARGET_READER_WELL_HEADER
    ).upper()
    if not MIX_CALIBRATION_READER_WELL_PATTERN.match(normalized):
        raise AutoMixCalibrationValidationError(
            '{} must be a plate-reader well from A1 through H12. Received '
            '{!r}.'.format(
                MIX_CALIBRATION_TARGET_READER_WELL_HEADER, value
            )
        )
    return normalized


def _require_stability_mode_is_off(header):
    '''Prevent an opt-in calibration from being mistaken for Auto stability.'''
    raw_value = _canonical_text(_header_value(
        header, 'auto_stability_mode', MIX_CALIBRATION_MODE_OFF
    ))
    if raw_value not in ('', 'off', 'false', 'no', '0'):
        raise AutoMixCalibrationValidationError(
            'auto_mix_calibration_mode is an independent calibration-only '
            'mode and cannot be combined with auto_stability_mode={!r}.'
            .format(_header_value(header, 'auto_stability_mode', ''))
        )


def _require_exact_fixed_configuration():
    '''Defend reviewed fixed assay constants against accidental invalid edits.'''
    numeric_values = (
        (MIX_CALIBRATION_FINAL_WELL_VOLUME_UL, 'final well volume'),
        (MIX_CALIBRATION_FIRST_VOLUME_UL, 'first transfer volume'),
        (MIX_CALIBRATION_SECOND_VOLUME_UL, 'second transfer volume'),
        (MIX_CALIBRATION_P300_VOLUME_UL, 'P300 mix volume'),
        (MIX_CALIBRATION_WELL_CLEARANCE_MM, 'well clearance'),
        (MIX_CALIBRATION_QC_ACID_RED_1_CONCENTRATION_MM,
         'Acid Red 1 concentration'),
        (MIX_CALIBRATION_QC_COPPER_SULFATE_CONCENTRATION_MM,
         'copper sulfate dye concentration'),
    )
    for value, label in numeric_values:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AutoMixCalibrationValidationError(
                'Reviewed calibration {} must be finite.'.format(label)
            )
    if (MIX_CALIBRATION_FIRST_VOLUME_UL <= 0
            or MIX_CALIBRATION_SECOND_VOLUME_UL <= 0
            or not math.isclose(
                MIX_CALIBRATION_FIRST_VOLUME_UL
                + MIX_CALIBRATION_SECOND_VOLUME_UL,
                MIX_CALIBRATION_FINAL_WELL_VOLUME_UL,
                rel_tol=0,
                abs_tol=1e-9)):
        raise AutoMixCalibrationValidationError(
            'Reviewed calibration transfers must exactly equal the final '
            'well volume.'
        )
    if (not isinstance(MIX_CALIBRATION_CYCLE_COUNT, int)
            or MIX_CALIBRATION_CYCLE_COUNT < 1):
        raise AutoMixCalibrationValidationError(
            'Reviewed calibration cycle count must be a positive integer.'
        )


def _canonical_profile(mode, value):
    raw_profile = _canonical_text(value)
    if mode == MIX_CALIBRATION_MODE_WATER_SAFETY:
        aliases = {
            '': MIX_CALIBRATION_PROFILE_WATER_SAFETY,
            'water': MIX_CALIBRATION_PROFILE_WATER_SAFETY,
            'water_safety': MIX_CALIBRATION_PROFILE_WATER_SAFETY,
            MIX_CALIBRATION_PROFILE_WATER_SAFETY: (
                MIX_CALIBRATION_PROFILE_WATER_SAFETY
            ),
        }
    elif mode == MIX_CALIBRATION_MODE_TWO_DYE:
        aliases = {
            MIX_CALIBRATION_PROFILE_QC_TWO_DYE_20UL_RED: (
                MIX_CALIBRATION_PROFILE_QC_TWO_DYE_20UL_RED
            ),
        }
    else:
        raise AutoMixCalibrationValidationError(
            'Calibration profile cannot be selected while calibration is off.'
        )
    if raw_profile not in aliases:
        expected = (MIX_CALIBRATION_PROFILE_WATER_SAFETY
                    if mode == MIX_CALIBRATION_MODE_WATER_SAFETY
                    else MIX_CALIBRATION_PROFILE_QC_TWO_DYE_20UL_RED)
        raise AutoMixCalibrationValidationError(
            '{} must be {!r} when auto_mix_calibration_mode is {!r}. '
            'Received {!r}.'.format(
                MIX_CALIBRATION_ASSAY_PROFILE_HEADER, expected, mode, value
            )
        )
    return aliases[raw_profile]


def _settings_template(mode):
    return {
        'auto_mix_calibration_mode': mode,
        MIX_CALIBRATION_ASSAY_PROFILE_HEADER: None,
        MIX_CALIBRATION_FIRST_REAGENT_HEADER: None,
        MIX_CALIBRATION_SECOND_REAGENT_HEADER: None,
        MIX_CALIBRATION_TARGET_READER_WELL_HEADER: None,
        'auto_mix_calibration_final_well_volume_uL': (
            MIX_CALIBRATION_FINAL_WELL_VOLUME_UL
        ),
        'auto_mix_calibration_first_volume_uL': MIX_CALIBRATION_FIRST_VOLUME_UL,
        'auto_mix_calibration_second_volume_uL': (
            MIX_CALIBRATION_SECOND_VOLUME_UL
        ),
        'auto_mix_calibration_p300_volume_uL': MIX_CALIBRATION_P300_VOLUME_UL,
        'auto_mix_calibration_cycle_count': MIX_CALIBRATION_CYCLE_COUNT,
        'auto_mix_calibration_well_clearance_mm': (
            MIX_CALIBRATION_WELL_CLEARANCE_MM
        ),
        'auto_mix_calibration_analysis_wavelengths_nm': (
            MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM
            if mode == MIX_CALIBRATION_MODE_TWO_DYE else None
        ),
    }


def parse_auto_mix_calibration_header_settings(header):
    '''Return inert or fully specified PDF-aligned calibration settings.

    Header values identify source reagents and one reader coordinate. The
    selected assay profile, rather than ordinary Input Template transfers,
    owns the fixed transfer quantities. This prevents an arbitrary recipe from
    being confused with the supervised mixing calibration.
    '''
    _require_exact_fixed_configuration()
    raw_mode = _canonical_text(_header_value(
        header, 'auto_mix_calibration_mode', MIX_CALIBRATION_MODE_OFF
    ))
    aliases = {
        '': MIX_CALIBRATION_MODE_OFF,
        'off': MIX_CALIBRATION_MODE_OFF,
        'false': MIX_CALIBRATION_MODE_OFF,
        'no': MIX_CALIBRATION_MODE_OFF,
        '0': MIX_CALIBRATION_MODE_OFF,
        'water': MIX_CALIBRATION_MODE_WATER_SAFETY,
        'water_safety': MIX_CALIBRATION_MODE_WATER_SAFETY,
        'water_only': MIX_CALIBRATION_MODE_WATER_SAFETY,
        'motion_check': MIX_CALIBRATION_MODE_WATER_SAFETY,
        'two_dye': MIX_CALIBRATION_MODE_TWO_DYE,
        'dye': MIX_CALIBRATION_MODE_TWO_DYE,
        'dye_assay': MIX_CALIBRATION_MODE_TWO_DYE,
    }
    if raw_mode not in aliases:
        raise AutoMixCalibrationValidationError(
            'auto_mix_calibration_mode must be off, water_safety, or '
            'two_dye. Received {!r}.'.format(
                _header_value(header, 'auto_mix_calibration_mode', '')
            )
        )
    mode = aliases[raw_mode]
    settings = _settings_template(mode)
    if mode == MIX_CALIBRATION_MODE_OFF:
        return settings

    _require_stability_mode_is_off(header)
    profile = _canonical_profile(
        mode, _header_value(header, MIX_CALIBRATION_ASSAY_PROFILE_HEADER)
    )
    first_reagent = _require_name(
        _header_value(header, MIX_CALIBRATION_FIRST_REAGENT_HEADER),
        MIX_CALIBRATION_FIRST_REAGENT_HEADER
    )
    second_reagent = _require_name(
        _header_value(header, MIX_CALIBRATION_SECOND_REAGENT_HEADER),
        MIX_CALIBRATION_SECOND_REAGENT_HEADER
    )
    if first_reagent.casefold() == second_reagent.casefold():
        raise AutoMixCalibrationValidationError(
            'Calibration first and second reagents must be separately named '
            'sources so both assay additions remain auditable.'
        )
    settings[MIX_CALIBRATION_ASSAY_PROFILE_HEADER] = profile
    settings[MIX_CALIBRATION_FIRST_REAGENT_HEADER] = first_reagent
    settings[MIX_CALIBRATION_SECOND_REAGENT_HEADER] = second_reagent
    settings[MIX_CALIBRATION_TARGET_READER_WELL_HEADER] = _require_reader_well(
        _header_value(header, MIX_CALIBRATION_TARGET_READER_WELL_HEADER)
    )
    return settings


def _canonical_reagent_role(value):
    return _canonical_text(value).replace('__', '_')


def _require_source_metadata(source_metadata, slot):
    if not isinstance(source_metadata, dict):
        raise AutoMixCalibrationValidationError(
            '{} source metadata must be a dictionary.'.format(slot)
        )
    name = _require_name(source_metadata.get('reagent_name'),
                         '{} source reagent_name'.format(slot))
    try:
        concentration = float(source_metadata.get('concentration_mM'))
    except (TypeError, ValueError):
        raise AutoMixCalibrationValidationError(
            '{} source concentration_mM must be finite.'.format(slot)
        )
    if not math.isfinite(concentration) or concentration <= 0:
        raise AutoMixCalibrationValidationError(
            '{} source concentration_mM must be positive and finite.'
            .format(slot)
        )
    return name, concentration


def validate_auto_mix_calibration_profile_source_metadata(
        settings, source_metadata):
    '''Validate resolved source identities/concentrations against a profile.

    This pure helper is intentionally separate from Header parsing because
    concentration and concrete source resolution come from Reagent Info later
    in the future controller path. It neither resolves sources nor accesses a
    workbook.
    '''
    if not isinstance(settings, dict):
        raise AutoMixCalibrationValidationError(
            'Calibration settings must be a dictionary.'
        )
    if settings.get('auto_mix_calibration_mode') != MIX_CALIBRATION_MODE_TWO_DYE:
        return None
    if (settings.get(MIX_CALIBRATION_ASSAY_PROFILE_HEADER)
            != MIX_CALIBRATION_PROFILE_QC_TWO_DYE_20UL_RED):
        raise AutoMixCalibrationValidationError(
            'Two-dye calibration requires the reviewed QC assay profile.'
        )
    if not isinstance(source_metadata, dict) or set(source_metadata) != set(
            ('first', 'second')):
        raise AutoMixCalibrationValidationError(
            'Two-dye source metadata must contain exactly first and second.'
        )
    first_name, first_concentration = _require_source_metadata(
        source_metadata['first'], 'first'
    )
    second_name, second_concentration = _require_source_metadata(
        source_metadata['second'], 'second'
    )
    if (first_name.casefold()
            != settings[MIX_CALIBRATION_FIRST_REAGENT_HEADER].casefold()
            or second_name.casefold()
            != settings[MIX_CALIBRATION_SECOND_REAGENT_HEADER].casefold()):
        raise AutoMixCalibrationValidationError(
            'Resolved two-dye sources do not match the configured first/second '
            'reagent names.'
        )
    allowed_first = set(('acid_red_1', 'acid_red1'))
    allowed_second = set(('copper_sulfate_dye', 'copper_ii_sulfate_dye'))
    if _canonical_reagent_role(first_name) not in allowed_first:
        raise AutoMixCalibrationValidationError(
            'The first QC two-dye source must be Acid Red 1.'
        )
    if _canonical_reagent_role(second_name) not in allowed_second:
        raise AutoMixCalibrationValidationError(
            'The second QC two-dye source must be copper sulfate dye.'
        )
    expected_values = (
        (first_concentration, MIX_CALIBRATION_QC_ACID_RED_1_CONCENTRATION_MM,
         'first Acid Red 1'),
        (second_concentration,
         MIX_CALIBRATION_QC_COPPER_SULFATE_CONCENTRATION_MM,
         'second copper sulfate dye'),
    )
    for actual, expected, label in expected_values:
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
            raise AutoMixCalibrationValidationError(
                '{} concentration_mM must equal {} for profile {}. '
                'Received {}.'.format(
                    label, expected,
                    MIX_CALIBRATION_PROFILE_QC_TWO_DYE_20UL_RED, actual
                )
            )
    return {
        'first_reagent_name': first_name,
        'first_concentration_mM': first_concentration,
        'second_reagent_name': second_name,
        'second_concentration_mM': second_concentration,
        'analysis_wavelengths_nm': MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM,
    }
