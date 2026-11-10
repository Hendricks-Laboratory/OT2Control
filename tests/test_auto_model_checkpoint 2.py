'''Hardware-free validation for portable Auto model checkpoint packages.'''

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings
import zipfile

import numpy as np

from auto_model_checkpoint import (
    ModelCheckpointError,
    build_import_run_context_lineage_manifest,
    get_model_checkpoint_import_inbox,
    get_model_checkpoint_file_sha256,
    prepare_model_checkpoint_import,
    prepare_model_checkpoint_import_from_path,
    read_model_checkpoint,
    read_run_context_lineage_manifest,
    validate_run_context_lineage_manifest,
    write_model_checkpoint_import_provenance,
    write_run_context_lineage_manifest,
    write_model_checkpoint
)


class ModelCheckpointPackageTests(unittest.TestCase):
    '''Exercises only JSON/NumPy checkpoint persistence, never controller I/O.'''

    def _manifest(self):
        return {
            'checkpoint_stage': 'after_seed',
            'run_id': 'DEBUGRTG_checkpoint_test',
            'variable_reagents': ['silver_nitrate', 'potassium_bromide'],
            'input_coordinate_system': {
                'name': 'normalized_final_reaction_concentration'
            }
        }

    def _arrays(self):
        return {
            'gp_training_X': np.asarray([
                [0.10, 0.20],
                [0.70, 0.80]
            ]),
            'gp_training_Y': np.asarray([
                [0.50],
                [0.60]
            ]),
            'usable_spectrum_X': np.asarray([
                [0.10, 0.20],
                [0.70, 0.80]
            ]),
            'usable_spectrum_Y': np.asarray([
                [1.0],
                [0.0]
            ])
        }

    def test_round_trip_preserves_arrays_and_json_safe_history(self):
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = write_model_checkpoint(
                temporary_directory,
                'model_after_seed',
                self._manifest(),
                self._arrays(),
                [{'actual_lambda_mean_nm': np.nan, 'condition_number': 0}]
            )

            restored = read_model_checkpoint(checkpoint_path)

            self.assertTrue(Path(checkpoint_path).is_file())
            self.assertEqual(
                restored['manifest']['checkpoint_stage'],
                'after_seed'
            )
            self.assertEqual(
                restored['condition_history'][0]['actual_lambda_mean_nm'],
                None
            )
            for array_name, expected_array in self._arrays().items():
                np.testing.assert_allclose(
                    restored['model_arrays'][array_name],
                    expected_array
                )

    def test_writer_never_overwrites_a_prior_checkpoint(self):
        with TemporaryDirectory() as temporary_directory:
            first_path = write_model_checkpoint(
                temporary_directory,
                'model_after_batch_001',
                self._manifest(),
                self._arrays(),
                []
            )
            second_path = write_model_checkpoint(
                temporary_directory,
                'model_after_batch_001',
                self._manifest(),
                self._arrays(),
                []
            )

            self.assertNotEqual(first_path, second_path)
            self.assertTrue(Path(first_path).is_file())
            self.assertTrue(Path(second_path).is_file())

    def test_reader_rejects_checksum_mismatch(self):
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = write_model_checkpoint(
                temporary_directory,
                'model_after_seed',
                self._manifest(),
                self._arrays(),
                []
            )

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(checkpoint_path, mode='a') as archive:
                    archive.writestr(
                        'manifest.json',
                        json.dumps(self._manifest())
                    )

            with self.assertRaisesRegex(ModelCheckpointError, 'contain exactly'):
                read_model_checkpoint(checkpoint_path)

    def test_writer_rejects_nonbinary_usable_spectrum_history(self):
        arrays = self._arrays()
        arrays['usable_spectrum_Y'] = np.asarray([[0.5], [1.0]])

        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ModelCheckpointError, 'binary'):
                write_model_checkpoint(
                    temporary_directory,
                    'model_after_seed',
                    self._manifest(),
                    arrays,
                    []
                )

    def test_reader_rejects_archive_members_outside_the_schema(self):
        with TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / 'invalid.zip'
            with zipfile.ZipFile(checkpoint_path, mode='w') as archive:
                archive.writestr('../unexpected.txt', 'not a checkpoint')

            with self.assertRaisesRegex(ModelCheckpointError, 'contain exactly'):
                read_model_checkpoint(checkpoint_path)

    def test_import_requires_one_valid_package_and_preserves_a_lineage_copy(self):
        with TemporaryDirectory() as temporary_directory:
            inbox_directory = get_model_checkpoint_import_inbox(
                temporary_directory
            )
            source_path = write_model_checkpoint(
                inbox_directory,
                'model_final',
                self._manifest(),
                self._arrays(),
                []
            )

            prepared = prepare_model_checkpoint_import(temporary_directory)

            self.assertEqual(
                Path(prepared['source_checkpoint_path']).resolve(),
                Path(source_path).resolve()
            )
            self.assertTrue(Path(source_path).is_file())
            self.assertTrue(
                Path(prepared['archived_checkpoint_path']).is_file()
            )
            self.assertEqual(
                prepared['manifest']['run_id'],
                'DEBUGRTG_checkpoint_test'
            )
            self.assertEqual(prepared['import_method'], 'manual_inbox')

    def test_selected_source_import_archives_metadata_and_provenance(self):
        with TemporaryDirectory() as temporary_directory:
            source_directory = Path(temporary_directory) / 'Prior Run'
            destination_directory = Path(temporary_directory) / 'New Run'
            source_directory.mkdir()
            source_path = write_model_checkpoint(
                source_directory,
                'model_final',
                self._manifest(),
                self._arrays(),
                []
            )

            prepared = prepare_model_checkpoint_import_from_path(
                destination_directory,
                source_path,
                'existing_output_run',
                {
                    'source_run_folder': 'RTG_020',
                    'source_checkpoint_filename': 'model_final.zip'
                }
            )
            provenance_path = write_model_checkpoint_import_provenance(
                destination_directory,
                {
                    'import_method': prepared['import_method'],
                    'archive': prepared['archived_checkpoint_path'],
                    'seed_design_skipped': True
                }
            )

            self.assertEqual(
                prepared['import_method'],
                'existing_output_run'
            )
            self.assertEqual(
                prepared['import_source_metadata']['source_run_folder'],
                'RTG_020'
            )
            self.assertTrue(Path(prepared['archived_checkpoint_path']).is_file())
            self.assertEqual(
                prepared['source_checkpoint_sha256'],
                prepared['archived_checkpoint_sha256']
            )
            self.assertEqual(
                prepared['source_checkpoint_sha256'],
                get_model_checkpoint_file_sha256(source_path)
            )
            persisted = json.loads(Path(provenance_path).read_text())
            self.assertTrue(persisted['seed_design_skipped'])
            self.assertEqual(persisted['import_method'], 'existing_output_run')

    def test_flat_lineage_appends_each_sequential_source_once(self):
        source_a_hash = 'a' * 64
        source_b_hash = 'b' * 64
        source_c_hash = 'c' * 64

        lineage_b = build_import_run_context_lineage_manifest(
            current_run_id='RTG_021',
            source_run_id='RTG_020',
            source_checkpoint_sha256=source_a_hash,
            source_checkpoint_stage='final',
            source_checkpoint_filename='model_final.zip',
            import_method='existing_output_run',
            source_run_folder='RTG_020'
        )
        lineage_c = build_import_run_context_lineage_manifest(
            current_run_id='RTG_022',
            source_run_id='RTG_021',
            source_checkpoint_sha256=source_b_hash,
            source_checkpoint_stage='final',
            source_checkpoint_filename='model_final.zip',
            import_method='existing_output_run',
            source_run_folder='RTG_021',
            inherited_manifest=lineage_b
        )
        lineage_d = build_import_run_context_lineage_manifest(
            current_run_id='RTG_023',
            source_run_id='RTG_022',
            source_checkpoint_sha256=source_c_hash,
            source_checkpoint_stage='final',
            source_checkpoint_filename='model_final.zip',
            import_method='existing_output_run',
            source_run_folder='RTG_022',
            inherited_manifest=lineage_c
        )

        self.assertEqual(
            [entry['run_id'] for entry in lineage_d['source_runs']],
            ['RTG_020', 'RTG_021', 'RTG_022']
        )
        self.assertEqual(
            lineage_d['source_runs'][1]['parent_run_id'],
            'RTG_020'
        )
        self.assertEqual(
            lineage_d['source_runs'][2]['parent_run_id'],
            'RTG_021'
        )

    def test_branching_from_an_earlier_run_excludes_later_descendants(self):
        lineage_b = build_import_run_context_lineage_manifest(
            current_run_id='RTG_021',
            source_run_id='RTG_020',
            source_checkpoint_sha256='a' * 64,
            source_checkpoint_stage='final',
            source_checkpoint_filename='model_final.zip',
            import_method='existing_output_run',
            source_run_folder='RTG_020'
        )
        branch_lineage = build_import_run_context_lineage_manifest(
            current_run_id='RTG_branch',
            source_run_id='RTG_021',
            source_checkpoint_sha256='b' * 64,
            source_checkpoint_stage='final',
            source_checkpoint_filename='model_final.zip',
            import_method='existing_output_run',
            source_run_folder='RTG_021',
            inherited_manifest=lineage_b
        )

        self.assertEqual(
            [entry['run_id'] for entry in branch_lineage['source_runs']],
            ['RTG_020', 'RTG_021']
        )
        self.assertNotIn('RTG_022', str(branch_lineage))

    def test_lineage_manifest_is_immutable_and_legacy_absence_is_supported(self):
        with TemporaryDirectory() as temporary_directory:
            self.assertIsNone(
                read_run_context_lineage_manifest(temporary_directory)
            )
            manifest = build_import_run_context_lineage_manifest(
                current_run_id='RTG_021',
                source_run_id='RTG_020',
                source_checkpoint_sha256='a' * 64,
                source_checkpoint_stage='final',
                source_checkpoint_filename='model_final.zip',
                import_method='manual_inbox'
            )
            manifest_path = write_run_context_lineage_manifest(
                temporary_directory,
                manifest
            )

            self.assertTrue(Path(manifest_path).is_file())
            self.assertEqual(
                read_run_context_lineage_manifest(temporary_directory),
                manifest
            )
            with self.assertRaisesRegex(ModelCheckpointError, 'already exists'):
                write_run_context_lineage_manifest(
                    temporary_directory,
                    manifest
                )

    def test_lineage_rejects_conflicting_duplicate_run_identity(self):
        manifest = {
            'schema_version': 1,
            'current_run_id': 'RTG_022',
            'direct_import': {
                'source_run_id': 'RTG_021',
                'source_checkpoint_sha256': 'b' * 64,
                'import_method': 'existing_output_run'
            },
            'source_runs': [
                {
                    'run_id': 'RTG_020',
                    'run_folder': 'RTG_020',
                    'checkpoint_filename': 'model_final.zip',
                    'checkpoint_sha256': 'a' * 64,
                    'checkpoint_stage': 'final',
                    'import_method': 'existing_output_run',
                    'parent_run_id': None
                },
                {
                    'run_id': 'RTG_020',
                    'run_folder': 'RTG_020',
                    'checkpoint_filename': 'model_final.zip',
                    'checkpoint_sha256': 'c' * 64,
                    'checkpoint_stage': 'final',
                    'import_method': 'existing_output_run',
                    'parent_run_id': None
                }
            ]
        }

        with self.assertRaisesRegex(ModelCheckpointError, 'conflicting'):
            validate_run_context_lineage_manifest(manifest)

    def test_import_rejects_missing_or_multiple_packages(self):
        with TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ModelCheckpointError, 'exactly one'):
                prepare_model_checkpoint_import(temporary_directory)

            inbox_directory = get_model_checkpoint_import_inbox(
                temporary_directory
            )
            write_model_checkpoint(
                inbox_directory,
                'model_one',
                self._manifest(),
                self._arrays(),
                []
            )
            write_model_checkpoint(
                inbox_directory,
                'model_two',
                self._manifest(),
                self._arrays(),
                []
            )

            with self.assertRaisesRegex(ModelCheckpointError, 'exactly one'):
                prepare_model_checkpoint_import(temporary_directory)


if __name__ == '__main__':
    unittest.main()
