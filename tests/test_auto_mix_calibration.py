'''Hardware-free tests for the calibration-mode Header/profile contract.'''

import unittest

from auto_mix_calibration import (
    AutoMixCalibrationValidationError,
    MIX_CALIBRATION_ASSAY_PROFILE_HEADER,
    MIX_CALIBRATION_CYCLE_COUNT,
    MIX_CALIBRATION_FINAL_WELL_VOLUME_UL,
    MIX_CALIBRATION_FIRST_REAGENT_HEADER,
    MIX_CALIBRATION_MODE_OFF,
    MIX_CALIBRATION_MODE_TWO_DYE,
    MIX_CALIBRATION_MODE_WATER_SAFETY,
    MIX_CALIBRATION_P300_VOLUME_UL,
    MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED,
    MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM,
    MIX_CALIBRATION_QC_TWO_DYE_FIRST_VOLUME_UL,
    MIX_CALIBRATION_QC_TWO_DYE_SECOND_VOLUME_UL,
    MIX_CALIBRATION_SECOND_REAGENT_HEADER,
    MIX_CALIBRATION_TARGET_READER_WELL_HEADER,
    MIX_CALIBRATION_WATER_SAFETY_FIRST_VOLUME_UL,
    MIX_CALIBRATION_WATER_SAFETY_SECOND_VOLUME_UL,
    MIX_CALIBRATION_WELL_CLEARANCE_MM,
    parse_auto_mix_calibration_header_settings,
    validate_auto_mix_calibration_profile_source_metadata,
)


def _two_dye_header():
    return {
        'auto_mix_calibration_mode': 'two_dye',
        MIX_CALIBRATION_ASSAY_PROFILE_HEADER: (
            MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED
        ),
        MIX_CALIBRATION_FIRST_REAGENT_HEADER: 'acid_red_1',
        MIX_CALIBRATION_SECOND_REAGENT_HEADER: 'copper_sulfate_dye',
        MIX_CALIBRATION_TARGET_READER_WELL_HEADER: 'C4',
    }


class AutoMixCalibrationHeaderTests(unittest.TestCase):
    def test_missing_header_setting_is_inert_and_backward_compatible(self):
        settings = parse_auto_mix_calibration_header_settings({})
        self.assertEqual(
            MIX_CALIBRATION_MODE_OFF,
            settings['auto_mix_calibration_mode']
        )
        self.assertIsNone(settings[MIX_CALIBRATION_ASSAY_PROFILE_HEADER])
        self.assertIsNone(settings[MIX_CALIBRATION_FIRST_REAGENT_HEADER])
        self.assertIsNone(settings[MIX_CALIBRATION_SECOND_REAGENT_HEADER])
        self.assertIsNone(settings[MIX_CALIBRATION_TARGET_READER_WELL_HEADER])

    def test_water_safety_mode_has_reviewed_fixed_motion_conditions(self):
        settings = parse_auto_mix_calibration_header_settings({
            'auto_mix_calibration_mode': 'water safety',
            MIX_CALIBRATION_FIRST_REAGENT_HEADER: 'Calibration First Water',
            MIX_CALIBRATION_SECOND_REAGENT_HEADER: 'Calibration Second Water',
            MIX_CALIBRATION_TARGET_READER_WELL_HEADER: 'c4',
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
            MIX_CALIBRATION_WATER_SAFETY_FIRST_VOLUME_UL,
            settings['auto_mix_calibration_first_volume_uL']
        )
        self.assertEqual(
            MIX_CALIBRATION_WATER_SAFETY_SECOND_VOLUME_UL,
            settings['auto_mix_calibration_second_volume_uL']
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
            'C4', settings[MIX_CALIBRATION_TARGET_READER_WELL_HEADER]
        )
        self.assertEqual(
            MIX_CALIBRATION_WELL_CLEARANCE_MM,
            settings['auto_mix_calibration_well_clearance_mm']
        )

    def test_two_dye_profile_is_pdf_ordered_and_locks_assay_values(self):
        settings = parse_auto_mix_calibration_header_settings(_two_dye_header())
        self.assertEqual(
            MIX_CALIBRATION_MODE_TWO_DYE,
            settings['auto_mix_calibration_mode']
        )
        self.assertEqual(
            MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED,
            settings[MIX_CALIBRATION_ASSAY_PROFILE_HEADER]
        )
        self.assertEqual(
            'acid_red_1', settings[MIX_CALIBRATION_FIRST_REAGENT_HEADER]
        )
        self.assertEqual(
            'copper_sulfate_dye',
            settings[MIX_CALIBRATION_SECOND_REAGENT_HEADER]
        )
        self.assertEqual(
            MIX_CALIBRATION_QC_ANALYSIS_WAVELENGTHS_NM,
            settings['auto_mix_calibration_analysis_wavelengths_nm']
        )
        self.assertEqual(
            MIX_CALIBRATION_QC_TWO_DYE_FIRST_VOLUME_UL,
            settings['auto_mix_calibration_first_volume_uL']
        )
        self.assertEqual(
            MIX_CALIBRATION_QC_TWO_DYE_SECOND_VOLUME_UL,
            settings['auto_mix_calibration_second_volume_uL']
        )

    def test_two_dye_profile_rejects_ambiguous_or_incompatible_header(self):
        invalid_headers = (
            {
                'auto_mix_calibration_mode': 'two_dye',
                MIX_CALIBRATION_FIRST_REAGENT_HEADER: 'acid_red_1',
                MIX_CALIBRATION_SECOND_REAGENT_HEADER: 'copper_sulfate_dye',
                MIX_CALIBRATION_TARGET_READER_WELL_HEADER: 'C4',
            },
            {
                'auto_mix_calibration_mode': 'two_dye',
                MIX_CALIBRATION_ASSAY_PROFILE_HEADER: 'unknown_profile',
                MIX_CALIBRATION_FIRST_REAGENT_HEADER: 'acid_red_1',
                MIX_CALIBRATION_SECOND_REAGENT_HEADER: 'copper_sulfate_dye',
                MIX_CALIBRATION_TARGET_READER_WELL_HEADER: 'C4',
            },
            {
                'auto_mix_calibration_mode': 'two_dye',
                MIX_CALIBRATION_ASSAY_PROFILE_HEADER: 'qc_two_dye_20ul_red',
                MIX_CALIBRATION_FIRST_REAGENT_HEADER: 'acid_red_1',
                MIX_CALIBRATION_SECOND_REAGENT_HEADER: 'copper_sulfate_dye',
                MIX_CALIBRATION_TARGET_READER_WELL_HEADER: 'C4',
            },
            {
                'auto_mix_calibration_mode': 'two_dye',
                MIX_CALIBRATION_ASSAY_PROFILE_HEADER: (
                    MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED
                ),
                'auto_mix_calibration_base_reagent': 'copper_sulfate_dye',
                'auto_mix_calibration_trigger_reagent': 'acid_red_1',
                MIX_CALIBRATION_TARGET_READER_WELL_HEADER: 'C4',
            },
            {
                'auto_mix_calibration_mode': 'two_dye',
                MIX_CALIBRATION_ASSAY_PROFILE_HEADER: (
                    MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED
                ),
                MIX_CALIBRATION_FIRST_REAGENT_HEADER: 'acid_red_1',
                MIX_CALIBRATION_SECOND_REAGENT_HEADER: 'acid_red_1',
                MIX_CALIBRATION_TARGET_READER_WELL_HEADER: 'C4',
            },
            {
                'auto_mix_calibration_mode': 'two_dye',
                MIX_CALIBRATION_ASSAY_PROFILE_HEADER: (
                    MIX_CALIBRATION_PROFILE_QC_TWO_DYE_5UL_RED
                ),
                MIX_CALIBRATION_FIRST_REAGENT_HEADER: 'acid_red_1',
                MIX_CALIBRATION_SECOND_REAGENT_HEADER: 'copper_sulfate_dye',
                MIX_CALIBRATION_TARGET_READER_WELL_HEADER: 'Q99',
            },
            dict(_two_dye_header(), auto_stability_mode='monitor'),
        )
        for header in invalid_headers:
            with self.subTest(header=header):
                with self.assertRaises(AutoMixCalibrationValidationError):
                    parse_auto_mix_calibration_header_settings(header)

    def test_two_dye_resolved_source_metadata_requires_pdf_identity_and_values(self):
        settings = parse_auto_mix_calibration_header_settings(_two_dye_header())
        accepted = validate_auto_mix_calibration_profile_source_metadata(
            settings,
            {
                'first': {
                    'reagent_name': 'acid_red_1',
                    'concentration_mM': 1.44,
                },
                'second': {
                    'reagent_name': 'copper_sulfate_dye',
                    'concentration_mM': 8.0,
                },
            }
        )
        self.assertEqual(1.44, accepted['first_concentration_mM'])
        self.assertEqual(8.0, accepted['second_concentration_mM'])

        for source_metadata in (
                {
                    'first': {
                        'reagent_name': 'copper_sulfate_dye',
                        'concentration_mM': 8.0,
                    },
                    'second': {
                        'reagent_name': 'acid_red_1',
                        'concentration_mM': 1.44,
                    },
                },
                {
                    'first': {
                        'reagent_name': 'acid_red_1',
                        'concentration_mM': 1.43,
                    },
                    'second': {
                        'reagent_name': 'copper_sulfate_dye',
                        'concentration_mM': 8.0,
                    },
                },
                {
                    'first': {
                        'reagent_name': 'acid_red_1',
                        'concentration_mM': 1.44,
                    },
                    'second': {
                        'reagent_name': 'copper_sulfate_dye',
                        'concentration_mM': 7.9,
                    },
                },
        ):
            with self.subTest(source_metadata=source_metadata):
                with self.assertRaises(AutoMixCalibrationValidationError):
                    validate_auto_mix_calibration_profile_source_metadata(
                        settings, source_metadata
                    )


if __name__ == '__main__':
    unittest.main()
