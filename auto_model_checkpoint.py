'''Portable, versioned Auto-RTG model checkpoint packages.

This module intentionally stores numeric model history rather than serializing
live GPy or GPyOpt objects.  A later process can therefore validate the
package and rebuild a fresh model from the saved cumulative observations.
Checkpoint packages contain only JSON and NumPy ``.npz`` payloads; they never
load pickle or dill data.
'''

import hashlib
import io
import json
import os
import re
import tempfile
import zipfile

import numpy as np


CHECKPOINT_SCHEMA_VERSION = 1

_PACKAGE_MEMBERS = (
    'manifest.json',
    'model_arrays.npz',
    'condition_history.json',
    'integrity.json'
)

_MODEL_ARRAY_NAMES = (
    'gp_training_X',
    'gp_training_Y',
    'usable_spectrum_X',
    'usable_spectrum_Y'
)

# Checkpoint arrays are small relative to raw scan files.  This bound protects
# later import stages from accidentally accepting a malformed or unexpectedly
# large archive while remaining far above realistic Auto model history sizes.
_MAX_UNCOMPRESSED_PACKAGE_BYTES = 256 * 1024 * 1024


class ModelCheckpointError(ValueError):
    '''Raised when a checkpoint cannot be safely written or read.'''


def _sha256(payload):
    '''Returns the SHA-256 digest for one bytes payload.'''
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value):
    '''Converts audit metadata into strict, portable JSON-compatible values.'''
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        # Performance-row audit fields can legitimately be unavailable. JSON
        # has no portable NaN representation, so preserve absence as null.
        return numeric_value if np.isfinite(numeric_value) else None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    raise ModelCheckpointError(
        'Checkpoint metadata contains a non-JSON value of type {}.'.format(
            type(value).__name__
        )
    )


def _json_bytes(value):
    '''Serializes one value as strict, deterministic UTF-8 JSON.'''
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False
    ).encode('utf-8')


def _validate_finite_array(array, array_name):
    '''Returns a finite float copy of one checkpoint array.'''
    try:
        normalized_array = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelCheckpointError(
            'Checkpoint array {!r} must be numeric.'.format(array_name)
        ) from exc

    if not np.all(np.isfinite(normalized_array)):
        raise ModelCheckpointError(
            'Checkpoint array {!r} contains non-finite values.'.format(
                array_name
            )
        )

    return np.array(normalized_array, dtype=float, copy=True)


def validate_model_arrays(model_arrays):
    '''Validates and normalizes the complete portable Auto model state.

    The primary GP stores normalized recipe coordinates and normalized lambda
    responses.  The usable-spectrum companion model stores the same recipe
    coordinates with binary labels.  Both histories are required so a future
    import can rebuild the same model family without opaque object pickles.
    '''
    if not isinstance(model_arrays, dict):
        raise ModelCheckpointError('Checkpoint model arrays must be a dict.')

    supplied_names = set(model_arrays.keys())
    expected_names = set(_MODEL_ARRAY_NAMES)
    if supplied_names != expected_names:
        raise ModelCheckpointError(
            'Checkpoint model arrays must contain exactly: {}. Received: {}.'
            .format(
                ', '.join(sorted(expected_names)),
                ', '.join(sorted(supplied_names))
            )
        )

    arrays = {
        array_name: _validate_finite_array(
            model_arrays[array_name],
            array_name
        )
        for array_name in _MODEL_ARRAY_NAMES
    }

    gp_training_x = arrays['gp_training_X']
    gp_training_y = arrays['gp_training_Y']
    usable_x = arrays['usable_spectrum_X']
    usable_y = arrays['usable_spectrum_Y']

    if gp_training_x.ndim != 2 or gp_training_x.shape[0] == 0:
        raise ModelCheckpointError(
            'gp_training_X must be a non-empty two-dimensional array.'
        )

    if gp_training_y.ndim != 2 or gp_training_y.shape[1:] != (1,):
        raise ModelCheckpointError(
            'gp_training_Y must have shape (n_observations, 1).'
        )

    if gp_training_x.shape[0] != gp_training_y.shape[0]:
        raise ModelCheckpointError(
            'gp_training_X and gp_training_Y must have equal row counts.'
        )

    if usable_x.ndim != 2 or usable_x.shape[1] != gp_training_x.shape[1]:
        raise ModelCheckpointError(
            'usable_spectrum_X must be two-dimensional with the same '
            'feature count as gp_training_X.'
        )

    if usable_y.ndim != 2 or usable_y.shape[1:] != (1,):
        raise ModelCheckpointError(
            'usable_spectrum_Y must have shape (n_observations, 1).'
        )

    if usable_x.shape[0] != usable_y.shape[0]:
        raise ModelCheckpointError(
            'usable_spectrum_X and usable_spectrum_Y must have equal row '
            'counts.'
        )

    if usable_y.size > 0 and not np.all(
        np.logical_or(usable_y == 0.0, usable_y == 1.0)
    ):
        raise ModelCheckpointError(
            'usable_spectrum_Y must contain only binary zero/one labels.'
        )

    return arrays


def _validated_manifest(manifest, expected_dimension):
    '''Validates the minimum schema required for a portable model package.'''
    if not isinstance(manifest, dict):
        raise ModelCheckpointError('Checkpoint manifest must be a dict.')

    normalized_manifest = _json_safe(manifest)
    if normalized_manifest.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise ModelCheckpointError(
            'Checkpoint manifest has an unsupported schema version.'
        )

    variable_reagents = normalized_manifest.get('variable_reagents')

    if not isinstance(variable_reagents, list) or len(variable_reagents) == 0:
        raise ModelCheckpointError(
            'Checkpoint manifest must contain non-empty variable_reagents.'
        )

    if len(variable_reagents) != expected_dimension:
        raise ModelCheckpointError(
            'Checkpoint manifest variable_reagents length does not match '
            'saved model dimension.'
        )

    checkpoint_stage = normalized_manifest.get('checkpoint_stage')
    if not isinstance(checkpoint_stage, str) or not checkpoint_stage.strip():
        raise ModelCheckpointError(
            'Checkpoint manifest must contain checkpoint_stage.'
        )

    run_id = normalized_manifest.get('run_id')
    if not isinstance(run_id, str) or not run_id.strip():
        raise ModelCheckpointError(
            'Checkpoint manifest must contain run_id.'
        )

    return normalized_manifest


def _validate_condition_history(condition_history):
    '''Returns JSON-safe condition history used for later audit/import work.'''
    if not isinstance(condition_history, list):
        raise ModelCheckpointError('condition_history must be a list.')

    normalized_history = _json_safe(condition_history)
    if not all(isinstance(row, dict) for row in normalized_history):
        raise ModelCheckpointError(
            'condition_history must contain only dictionary rows.'
        )

    return normalized_history


def _safe_checkpoint_stem(checkpoint_stem):
    '''Validates a filesystem-safe checkpoint filename stem.'''
    normalized_stem = str(checkpoint_stem).strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', normalized_stem):
        raise ModelCheckpointError(
            'Checkpoint filename stem may contain only letters, numbers, '
            'underscores, and hyphens.'
        )

    return normalized_stem


def _unique_checkpoint_path(checkpoint_directory, checkpoint_stem):
    '''Returns an unused ZIP destination without overwriting prior models.'''
    base_path = os.path.join(checkpoint_directory, checkpoint_stem + '.zip')
    if not os.path.exists(base_path):
        return base_path

    suffix = 1
    while True:
        candidate_path = os.path.join(
            checkpoint_directory,
            '{}_{:03d}.zip'.format(checkpoint_stem, suffix)
        )
        if not os.path.exists(candidate_path):
            return candidate_path
        suffix += 1


def write_model_checkpoint(
    checkpoint_directory,
    checkpoint_stem,
    manifest,
    model_arrays,
    condition_history
):
    '''Writes one immutable, checksummed Auto model checkpoint package.

    The temporary archive is created in the destination directory and replaced
    atomically only after all payloads have been written.  Existing model
    packages are never overwritten.
    '''
    checkpoint_directory = os.fspath(checkpoint_directory)
    checkpoint_stem = _safe_checkpoint_stem(checkpoint_stem)
    arrays = validate_model_arrays(model_arrays)
    manifest_with_schema = dict(manifest)
    manifest_with_schema['schema_version'] = CHECKPOINT_SCHEMA_VERSION
    normalized_manifest = _validated_manifest(
        manifest_with_schema,
        arrays['gp_training_X'].shape[1]
    )
    normalized_history = _validate_condition_history(condition_history)

    os.makedirs(checkpoint_directory, exist_ok=True)
    destination_path = _unique_checkpoint_path(
        checkpoint_directory,
        checkpoint_stem
    )

    arrays_buffer = io.BytesIO()
    np.savez_compressed(arrays_buffer, **arrays)
    payloads = {
        'manifest.json': _json_bytes(normalized_manifest),
        'model_arrays.npz': arrays_buffer.getvalue(),
        'condition_history.json': _json_bytes(normalized_history)
    }
    integrity = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'payload_sha256': {
            member_name: _sha256(payload)
            for member_name, payload in payloads.items()
        }
    }
    payloads['integrity.json'] = _json_bytes(integrity)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb',
            prefix='.auto_model_checkpoint_',
            suffix='.tmp',
            dir=checkpoint_directory,
            delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name

        with zipfile.ZipFile(
            temporary_path,
            mode='w',
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6
        ) as archive:
            for member_name in _PACKAGE_MEMBERS:
                archive.writestr(member_name, payloads[member_name])

        os.replace(temporary_path, destination_path)
        temporary_path = None

    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return destination_path


def _read_json_member(archive, member_name):
    '''Reads one UTF-8 JSON archive member as a parsed value.'''
    try:
        return json.loads(archive.read(member_name).decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCheckpointError(
            'Checkpoint member {!r} is not valid UTF-8 JSON.'.format(
                member_name
            )
        ) from exc


def read_model_checkpoint(checkpoint_path):
    '''Reads and validates one portable Auto model checkpoint package.

    This function performs no model fitting and executes no serialized code.
    It returns only validated metadata and numeric arrays for a caller to use
    in a later, explicitly controlled fresh-model reconstruction step.
    '''
    checkpoint_path = os.fspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise ModelCheckpointError(
            'Checkpoint package does not exist: {!r}.'.format(checkpoint_path)
        )

    try:
        with zipfile.ZipFile(checkpoint_path, mode='r') as archive:
            member_names = [member.filename for member in archive.infolist()]
            if (
                len(member_names) != len(set(member_names))
                or set(member_names) != set(_PACKAGE_MEMBERS)
            ):
                raise ModelCheckpointError(
                    'Checkpoint package must contain exactly: {}. Received: {}.'
                    .format(
                        ', '.join(_PACKAGE_MEMBERS),
                        ', '.join(sorted(member_names))
                    )
                )

            total_uncompressed_size = sum(
                member.file_size for member in archive.infolist()
            )
            if total_uncompressed_size > _MAX_UNCOMPRESSED_PACKAGE_BYTES:
                raise ModelCheckpointError(
                    'Checkpoint package exceeds the maximum supported '
                    'uncompressed size.'
                )

            integrity = _read_json_member(archive, 'integrity.json')
            if not isinstance(integrity, dict) or integrity.get(
                'schema_version'
            ) != CHECKPOINT_SCHEMA_VERSION:
                raise ModelCheckpointError(
                    'Checkpoint integrity metadata has an unsupported schema.'
                )

            expected_hashes = integrity.get('payload_sha256')
            expected_members = _PACKAGE_MEMBERS[:-1]
            if not isinstance(expected_hashes, dict) or set(
                expected_hashes.keys()
            ) != set(expected_members):
                raise ModelCheckpointError(
                    'Checkpoint integrity metadata does not cover every '
                    'payload member.'
                )

            payloads = {
                member_name: archive.read(member_name)
                for member_name in expected_members
            }
            for member_name, payload in payloads.items():
                if _sha256(payload) != expected_hashes[member_name]:
                    raise ModelCheckpointError(
                        'Checkpoint checksum mismatch for {!r}.'.format(
                            member_name
                        )
                    )

    except zipfile.BadZipFile as exc:
        raise ModelCheckpointError('Checkpoint package is not a valid ZIP.') from exc

    try:
        manifest = json.loads(payloads['manifest.json'].decode('utf-8'))
        condition_history = json.loads(
            payloads['condition_history.json'].decode('utf-8')
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCheckpointError(
            'Checkpoint JSON payload is malformed.'
        ) from exc

    try:
        with np.load(
            io.BytesIO(payloads['model_arrays.npz']),
            allow_pickle=False
        ) as archive_arrays:
            model_arrays = {
                array_name: np.array(archive_arrays[array_name], copy=True)
                for array_name in archive_arrays.files
            }
    except (OSError, ValueError, KeyError) as exc:
        raise ModelCheckpointError(
            'Checkpoint NumPy payload cannot be loaded safely.'
        ) from exc

    arrays = validate_model_arrays(model_arrays)
    normalized_manifest = _validated_manifest(
        manifest,
        arrays['gp_training_X'].shape[1]
    )
    normalized_history = _validate_condition_history(condition_history)

    return {
        'manifest': normalized_manifest,
        'model_arrays': arrays,
        'condition_history': normalized_history,
        'checkpoint_path': os.path.abspath(checkpoint_path)
    }
