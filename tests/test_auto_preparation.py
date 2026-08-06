'''Pure unit tests for grouped Auto working-solution preparation planning.'''

import math
import unittest

from auto_preparation import (
    AutoPreparationValidationError,
    build_preparation_manifest,
    build_variable_source_bindings,
    validate_manifest_source_names
)


class AutoPreparationManifestTests(unittest.TestCase):
    '''Validate grouped C1V1=C2V2 planning without controller imports.'''

    @staticmethod
    def _valid_row():
        return {
            'enabled': 'yes',
            'stock_source_group': 'sodium borohydride',
            'stock_concentration_mM': 130,
            'working_concentration_mM': 6.25,
            'tube_count': 10,
            'final_volume_per_tube_uL': 1200,
            'destination_labware': 'temp_mod_24_tube',
            'destination_container': 'Tube2000uL'
        }

    def test_manifest_has_per_tube_and_group_totals(self):
        manifest = build_preparation_manifest([self._valid_row()])

        self.assertEqual(manifest['schema_version'], 2)
        self.assertEqual(
            manifest['execution_status'],
            'planning_only_pending_auto_main_group_protocol'
        )
        preparation = manifest['preparations'][0]
        self.assertEqual(
            preparation['stock_chemical_name'],
            'sodium_borohydrideC130.0'
        )
        self.assertEqual(
            preparation['working_chemical_name'],
            'sodium_borohydrideC6.25'
        )
        self.assertEqual(preparation['tube_count'], 10)
        self.assertEqual(len(preparation['tube_plan']), 10)
        self.assertAlmostEqual(
            preparation['stock_transfer_per_tube_uL'],
            57.6923076923
        )
        self.assertAlmostEqual(
            preparation['water_transfer_per_tube_uL'],
            1142.3076923077
        )
        self.assertAlmostEqual(preparation['total_final_volume_uL'], 12000)
        self.assertAlmostEqual(preparation['total_stock_transfer_uL'], 576.923076923)
        self.assertAlmostEqual(preparation['total_water_transfer_uL'], 11423.076923077)
        self.assertEqual(
            preparation['water_source_policy'],
            'stock_temperature_module_selects_water'
        )
        self.assertTrue(
            preparation['requires_runtime_destination_capacity_check']
        )
        self.assertTrue(manifest['manifest_sha256'])

    def test_capacity_resolver_can_reject_an_impossible_per_tube_volume(self):
        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'exceeding the Tube2000uL capacity'
        ):
            build_preparation_manifest(
                [self._valid_row()],
                destination_capacity_resolver={'Tube2000uL': 1000}
            )

    def test_disabled_and_blank_rows_are_inert(self):
        disabled = self._valid_row()
        disabled['enabled'] = 'off'
        blank = self._valid_row()
        blank['enabled'] = float('nan')
        manifest = build_preparation_manifest([disabled, blank])
        self.assertTrue(math.isnan(blank['enabled']))
        self.assertEqual(manifest['preparations'], [])

    def test_rejects_invalid_dilution_count_and_transfer_size(self):
        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'below the stock concentration'
        ):
            row = self._valid_row()
            row['working_concentration_mM'] = 130
            build_preparation_manifest([row])

        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'whole number'
        ):
            row = self._valid_row()
            row['tube_count'] = 1.5
            build_preparation_manifest([row])

        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'non-executable 0–5 uL'
        ):
            row = self._valid_row()
            row['final_volume_per_tube_uL'] = 100
            row['working_concentration_mM'] = 1
            build_preparation_manifest([row])

    def test_rejects_duplicate_source_group(self):
        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'more than one enabled row for stock source group'
        ):
            build_preparation_manifest([self._valid_row(), self._valid_row()])

    def test_source_name_validation_refuses_unknown_or_colliding_names(self):
        manifest = build_preparation_manifest([self._valid_row()])
        validate_manifest_source_names(
            manifest,
            ['sodium_borohydrideC130.0', 'WaterC1.0']
        )

        with self.assertRaisesRegex(AutoPreparationValidationError, 'not present'):
            validate_manifest_source_names(manifest, ['WaterC1.0'])

        with self.assertRaisesRegex(AutoPreparationValidationError, 'already exists'):
            validate_manifest_source_names(
                manifest,
                ['sodium_borohydrideC130.0', 'sodium_borohydrideC6.25']
            )


class AutoVariableSourceBindingTests(unittest.TestCase):
    '''Exercise physical-source provenance without controller/robot imports.'''

    @staticmethod
    def _variable_row(source_concentration):
        return {
            'reagent': 'sodium borohydride',
            'variable_source_concentration_mM': source_concentration
        }

    @staticmethod
    def _source_rows():
        return [
            {'chemical_name': 'sodium_borohydrideC130.0', 'conc': 130.0},
            {'chemical_name': 'sodium_borohydrideC6.25', 'conc': 6.25},
            {'chemical_name': 'silver_nitrateC0.375', 'conc': 0.375}
        ]

    def test_prepared_variable_binds_to_working_not_stock_concentration(self):
        manifest = build_preparation_manifest([
            AutoPreparationManifestTests._valid_row()
        ])
        bindings = build_variable_source_bindings(
            [self._variable_row(6.25)],
            self._source_rows(),
            preparations=manifest['preparations'],
            require_bindings=True
        )

        self.assertEqual(
            bindings['sodium_borohydride']['chemical_name'],
            'sodium_borohydrideC6.25'
        )
        self.assertEqual(
            bindings['sodium_borohydride']['source_role'],
            'prepared_working_source'
        )

    def test_prepared_variable_rejects_its_stock_concentration(self):
        manifest = build_preparation_manifest([
            AutoPreparationManifestTests._valid_row()
        ])
        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'working concentration, not the stock concentration'
        ):
            build_variable_source_bindings(
                [self._variable_row(130.0)],
                self._source_rows(),
                preparations=manifest['preparations'],
                require_bindings=True
            )

    def test_existing_variable_source_binding_selects_exact_deck_source(self):
        bindings = build_variable_source_bindings(
            [{
                'reagent': 'silver nitrate',
                'variable_source_concentration_mM': 0.375
            }],
            self._source_rows(),
            require_bindings=True
        )
        self.assertEqual(
            bindings['silver_nitrate']['chemical_name'],
            'silver_nitrateC0.375'
        )
        self.assertEqual(
            bindings['silver_nitrate']['source_role'],
            'existing_deck_source'
        )

    def test_present_binding_column_cannot_be_blank_for_a_variable(self):
        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'requires variable_source_concentration_mM'
        ):
            build_variable_source_bindings(
                [self._variable_row(None)],
                self._source_rows(),
                require_bindings=True
            )

    def test_variable_rows_must_not_disagree_about_one_source(self):
        with self.assertRaisesRegex(
            AutoPreparationValidationError,
            'inconsistent variable_source_concentration_mM'
        ):
            build_variable_source_bindings(
                [self._variable_row(6.25), self._variable_row(130.0)],
                self._source_rows(),
                require_bindings=True
            )


if __name__ == '__main__':
    unittest.main()
