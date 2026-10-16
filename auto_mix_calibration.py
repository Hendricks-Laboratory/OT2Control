'''Pure configuration contract for the supervised 96-well mix calibration.

This module deliberately contains no controller, reader, spreadsheet-I/O, or
robot imports.  It defines the narrow configuration boundary for a future
calibration-only workflow that characterizes targeted P300 mixing before any
production stability default is changed.

The calibration is intentionally distinct from ``auto_stability_mode``:
it must not train a GP, select a recipe, create an active-well trajectory, or
silently alter normal Auto execution.
'''

import math


MIX_CALIBRATION_MODE_OFF = 'off'
MIX_CALIBRATION_MODE_WATER_SAFETY = 'water_safety'
MIX_CALIBRATION_MODE_TWO_DYE = 'two_dye'

# These are fixed, reviewed calibration conditions rather than ordinary
# worksheet-tunable production settings.  The eventual Pi command must still
# verify the actual target well volume and loaded P300 before any movement.
MIX_CALIBRATION_FINAL_WELL_VOLUME_UL = 200.0
MIX_CALIBRATION_BASE_VOLUME_UL = 180.0
MIX_CALIBRATION_TRIGGER_VOLUME_UL = 20.0
MIX_CALIBRATION_P300_VOLUME_UL = 150.0
MIX_CALIBRATION_CYCLE_COUNT = 10
MIX_CALIBRATION_WELL_CLEARANCE_MM = 2.0


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


def _require_exact_fixed_configuration():
    '''Defend the reviewed constants against accidental invalid edits.'''
    numeric_values = (
        (MIX_CALIBRATION_FINAL_WELL_VOLUME_UL, 'final well volume'),
        (MIX_CALIBRATION_BASE_VOLUME_UL, 'base transfer volume'),
        (MIX_CALIBRATION_TRIGGER_VOLUME_UL, 'trigger transfer volume'),
        (MIX_CALIBRATION_P300_VOLUME_UL, 'P300 mix volume'),
        (MIX_CALIBRATION_WELL_CLEARANCE_MM, 'well clearance'),
    )
    for value, label in numeric_values:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AutoMixCalibrationValidationError(
                'Reviewed calibration {} must be finite.'.format(label)
            )
    if (MIX_CALIBRATION_BASE_VOLUME_UL <= 0
            or MIX_CALIBRATION_TRIGGER_VOLUME_UL <= 0
            or not math.isclose(
                MIX_CALIBRATION_BASE_VOLUME_UL
                + MIX_CALIBRATION_TRIGGER_VOLUME_UL,
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


def parse_auto_mix_calibration_header_settings(header):
    '''Return an inert or fully specified calibration-only configuration.

    Header keys intentionally remain minimal.  The mode identifies whether a
    supervised water-only motion check or the later two-dye optical assay is
    requested; source identities are named explicitly so the eventual
    controller cannot substitute an ordinary Auto reagent by accident.
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
    settings = {
        'auto_mix_calibration_mode': mode,
        'auto_mix_calibration_base_reagent': None,
        'auto_mix_calibration_trigger_reagent': None,
        'auto_mix_calibration_final_well_volume_uL': (
            MIX_CALIBRATION_FINAL_WELL_VOLUME_UL
        ),
        'auto_mix_calibration_base_volume_uL': MIX_CALIBRATION_BASE_VOLUME_UL,
        'auto_mix_calibration_trigger_volume_uL': (
            MIX_CALIBRATION_TRIGGER_VOLUME_UL
        ),
        'auto_mix_calibration_p300_volume_uL': MIX_CALIBRATION_P300_VOLUME_UL,
        'auto_mix_calibration_cycle_count': MIX_CALIBRATION_CYCLE_COUNT,
        'auto_mix_calibration_well_clearance_mm': (
            MIX_CALIBRATION_WELL_CLEARANCE_MM
        ),
    }
    if mode == MIX_CALIBRATION_MODE_OFF:
        return settings

    base_reagent = _require_name(
        _header_value(header, 'auto_mix_calibration_base_reagent'),
        'auto_mix_calibration_base_reagent'
    )
    trigger_reagent = _require_name(
        _header_value(header, 'auto_mix_calibration_trigger_reagent'),
        'auto_mix_calibration_trigger_reagent'
    )
    if base_reagent.casefold() == trigger_reagent.casefold():
        raise AutoMixCalibrationValidationError(
            'Calibration base and trigger reagents must be separately named '
            'sources so the final trigger transfer remains auditable.'
        )
    settings['auto_mix_calibration_base_reagent'] = base_reagent
    settings['auto_mix_calibration_trigger_reagent'] = trigger_reagent
    return settings
