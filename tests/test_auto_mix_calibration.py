'''Hardware-free tests for the calibration-mode Header contract.'''

import unittest

from auto_mix_calibration import (
    AutoMixCalibrationValidationError,
    MIX_CALIBRATION_BASE_VOLUME_UL,
    MIX_CALIBRATION_CYCLE_COUNT,
    MIX_CALIBRATION_FINAL_WELL_VOLUME_UL,
    MIX_CALIBRATION_MODE_OFF,
    MIX_CALIBRATION_MODE_TWO_DYE,
    MIX_CALIBRATION_MODE_WATER_SAFETY,
    MIX_CALIBRATION_P300_VOLUME_UL,
    MIX_CALIBRATION_TRIGGER_VOLUME_UL,
    MIX_CALIBRATION_WELL_CLEARANCE_MM,
    parse_auto_mix_calibration_header_settings,
)


class AutoMixCalibrationHeaderTests(unittest.TestCase):
    def test_missing_header_setting_is_inert_and_backward_compatible(self):
        settings = parse_auto_mix_calibration_header_settings({})
        self.assertEqual(
            MIX_CALIBRATION_MODE_OFF,
            settings['auto_mix_calibration_mode']
        )
        self.assertIsNone(settings['auto_mix_calibration_base_reagent'])
        self.assertIsNone(settings['auto_mix_calibration_trigger_reagent'])

    def test_water_safety_mode_has_reviewed_fixed_motion_conditions(self):
        settings = parse_auto_mix_calibration_header_settings({
            'auto_mix_calibration_mode': 'water safety',
            'auto_mix_calibration_base_reagent': 'Calibration Base Water',
            'auto_mix_calibration_trigger_reagent': 'Calibration Trigger Water',
        })
        self.assertEqual(
            MIX_CALIBRATION_MODE_WATER_SAFETY,
            settings['auto_mix_calibration_mode']
        )
        self.assertEqual(
            MIX_CALIBRATION_FINAL_WELL_VOLUME_UL,
            settings['auto_mix_calibration_final_well_volume_uL']
        )
        self.assertEqual(
            MIX_CALIBRATION_BASE_VOLUME_UL,
            settings['auto_mix_calibration_base_volume_uL']
        )
        self.assertEqual(
            MIX_CALIBRATION_TRIGGER_VOLUME_UL,
            settings['auto_mix_calibration_trigger_volume_uL']
        )
        self.assertEqual(
            MIX_CALIBRATION_P300_VOLUME_UL,
            settings['auto_mix_calibration_p300_volume_uL']
        )
        self.assertEqual(
            MIX_CALIBRATION_CYCLE_COUNT,
            settings['auto_mix_calibration_cycle_count']
        )
        self.assertEqual(
            MIX_CALIBRATION_WELL_CLEARANCE_MM,
            settings['auto_mix_calibration_well_clearance_mm']
        )

    def test_two_dye_mode_requires_two_explicit_distinct_sources(self):
        settings = parse_auto_mix_calibration_header_settings({
            'auto_mix_calibration_mode': 'dye',
            'auto_mix_calibration_base_reagent': 'Copper Sulfate Dye',
            'auto_mix_calibration_trigger_reagent': 'Acid Red 1',
        })
        self.assertEqual(
            MIX_CALIBRATION_MODE_TWO_DYE,
            settings['auto_mix_calibration_mode']
        )

        for header in (
                {
                    'auto_mix_calibration_mode': 'two_dye',
                    'auto_mix_calibration_base_reagent': 'Water',
                },
                {
                    'auto_mix_calibration_mode': 'two_dye',
                    'auto_mix_calibration_base_reagent': 'Water',
                    'auto_mix_calibration_trigger_reagent': 'water',
                },
                {'auto_mix_calibration_mode': 'unknown'},
        ):
            with self.subTest(header=header):
                with self.assertRaises(AutoMixCalibrationValidationError):
                    parse_auto_mix_calibration_header_settings(header)


if __name__ == '__main__':
    unittest.main()
