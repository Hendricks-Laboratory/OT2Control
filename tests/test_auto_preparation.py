'''Pure unit tests for the Stage 9A Auto-preparation manifest contract.'''

import unittest
import math

from auto_preparation import (
    AutoPreparationValidationError,
    build_preparation_manifest,
    validate_manifest_source_names
)


class AutoPreparationManifestTests(unittest.TestCase):
    '''Validate planned C1V1=C2V2 preparations without controller imports.'''

    @staticmethod
    def _valid_row():
        return {
            'enabled': 'yes',
            'stock_reagent': 'sodium borohydride',
            'stock_concentration_mM': 130,
            'working_concentration_mM': 6.25,
            'final_volume_uL': 1000
        }

    def test_manifest_matches_existing_dilution_equation_and_naming(self):
        manifest = build_preparation_manifest(
            [self._valid_row()],
            destination_container='Tube2000uL',
            destination_capacity_uL=1500
        )

        self.assertEqual(manifest['schema_version'], 1)
        self.assertEqual(manifest['worksheet_name'], 'auto_preparation')
        self.assertEqual(len(manifest['preparations']), 1)
        preparation = manifest['preparations'][0]
        self.assertEqual(
            preparation['stock_chemical_name'],
            'sodium_borohydrideC130.0'
        )
        self.assertEqual(
            preparation['working_chemical_name'],
            'sodium_borohydrideC6.25'
        )
        self.assertAlmostEqual(preparation['stock_transfer_uL'], 48.0769230769)
        self.assertAlmostEqual(preparation['water_transfer_uL'], 951.9230769231)
        self.assertEqual(preparation['mix_cycles'], 2)
        self.assertEqual(
            preparation['water_source_policy'],
            'match_stock_temperature_module'
        )
        self.assertTrue(manifest['manifest_sha256'])

    def test_disabled_rows_do_not_create_preparations(self):
        row = self._valid_row()
        row['enabled'] = 'off'
        manifest = build_preparation_manifest(
            [row],
            destination_container='Tube2000uL',
            destination_capacity_uL=1500
        )
        self.assertEqual(manifest['preparations'], [])

    def test_blank_spreadsheet_enabled_cell_is_skipped(self):
        row = self._valid_row()
        row['enabled'] = float('nan')
        manifest = build_preparation_manifest(
            [row],
            destination_container='Tube2000uL',
            destination_capacity_uL=1500
        )
        self.assertTrue(math.isnan(row['enabled']))
        self.assertEqual(manifest['preparations'], [])

    def test_rejects_non_dilution_capacity_and_non_executable_requests(self):
        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'below the stock concentration'
        ):
            row = self._valid_row()
            row['working_concentration_mM'] = 130
            build_preparation_manifest([row], 'Tube2000uL', 1500)

        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'exceeding the configured'
        ):
            row = self._valid_row()
            row['final_volume_uL'] = 1600
            build_preparation_manifest([row], 'Tube2000uL', 1500)

        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'non-executable 0–5 uL'
        ):
            row = self._valid_row()
            row['final_volume_uL'] = 100
            row['working_concentration_mM'] = 1
            build_preparation_manifest([row], 'Tube2000uL', 1500)

    def test_rejects_ambiguous_duplicate_working_sources(self):
        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'only one prepared working concentration'
        ):
            build_preparation_manifest(
                [self._valid_row(), self._valid_row()],
                'Tube2000uL',
                1500
            )

    def test_rejects_multiple_working_concentrations_for_one_reagent(self):
        second_row = self._valid_row()
        second_row['working_concentration_mM'] = 3.125
        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'only one prepared working concentration'
        ):
            build_preparation_manifest(
                [self._valid_row(), second_row],
                'Tube2000uL',
                1500
            )

    def test_source_name_validation_is_exact_and_refuses_collisions(self):
        manifest = build_preparation_manifest(
            [self._valid_row()],
            destination_container='Tube2000uL',
            destination_capacity_uL=1500
        )
        validate_manifest_source_names(
            manifest,
            ['sodium_borohydrideC130.0', 'WaterC1.0']
        )

        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'not present'
        ):
            validate_manifest_source_names(manifest, ['WaterC1.0'])

        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'already exists'
        ):
            validate_manifest_source_names(
                manifest,
                ['sodium_borohydrideC130.0', 'sodium_borohydrideC6.25']
            )


if __name__ == '__main__':
    unittest.main()
