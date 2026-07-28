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
import shutil
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

# Import is intentionally an explicit, local handoff: an operator places one
# previously exported package in this inbox before completing the new run's
# reagent sheet.  The original is retained in the inbox and a copy is kept in
# the new run's lineage archive for reproducibility.
IMPORT_INBOX_DIRECTORY_NAME = 'Import_Here'
IMPORTED_ARCHIVE_DIRECTORY_NAME = 'Imported_Archives'

# A model checkpoint contains the numeric state required to rebuild a GP.  An
# imported run may additionally record a *flat* lineage of the prior runs
# whose data informed that state.  This lightweight manifest deliberately
# contains identities and checksums only: raw scan/CSV collection and
# cross-run plotting are later stages, not a hidden side effect of model
# restoration.
IMPORTED_RUN_CONTEXT_DIRECTORY_NAME = 'Imported_Run_Context'
RUN_CONTEXT_LINEAGE_FILENAME = 'lineage_manifest.json'
RUN_CONTEXT_LINEAGE_SCHEMA_VERSION = 1


class ModelCheckpointError(ValueError):
    '''Raised when a checkpoint cannot be safely written or read.'''


def _sha256(payload):
    '''Returns the SHA-256 digest for one bytes payload.'''
    return hashlib.sha256(payload).hexdigest()


def get_model_checkpoint_file_sha256(checkpoint_path):
    '''Returns the SHA-256 identity of one validated checkpoint archive.

    The package-internal integrity file protects its individual payloads.
    This archive-level digest additionally gives import lineage a stable,
    human-auditable identity without loading or serializing live model state.
    '''
    checkpoint_path = os.fspath(checkpoint_path)
    digest = hashlib.sha256()
    try:
        with open(checkpoint_path, 'rb') as checkpoint_file:
            while True:
                chunk = checkpoint_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ModelCheckpointError(
            'Checkpoint archive cannot be hashed: {}.'.format(exc)
        ) from exc

    return digest.hexdigest()


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


def get_model_checkpoint_import_inbox(checkpoint_directory):
    '''Returns and creates the dedicated one-package import inbox.'''
    inbox_directory = os.path.join(
        os.fspath(checkpoint_directory),
        IMPORT_INBOX_DIRECTORY_NAME
    )
    os.makedirs(inbox_directory, exist_ok=True)
    return inbox_directory


def prepare_model_checkpoint_import(checkpoint_directory):
    '''Validates and archives exactly one operator-supplied package.

    The input archive is never moved or modified.  A second immutable copy is
    stored under the new run's ``Imported_Archives`` directory, giving later
    reports and audits a durable record of the exact imported source.

    returns:
        dict:
            Validated checkpoint payload plus source and archive paths.
    '''
    checkpoint_directory = os.fspath(checkpoint_directory)
    inbox_directory = get_model_checkpoint_import_inbox(checkpoint_directory)

    candidate_paths = sorted(
        os.path.join(inbox_directory, entry_name)
        for entry_name in os.listdir(inbox_directory)
        if (
            entry_name.lower().endswith('.zip')
            and os.path.isfile(os.path.join(inbox_directory, entry_name))
        )
    )

    if len(candidate_paths) != 1:
        raise ModelCheckpointError(
            'Checkpoint import requires exactly one .zip package in {!r}. '
            'Found {}.'.format(inbox_directory, len(candidate_paths))
        )

    return prepare_model_checkpoint_import_from_path(
        checkpoint_directory=checkpoint_directory,
        source_path=candidate_paths[0],
        import_method='manual_inbox',
        source_metadata={
            'import_inbox_path': os.path.abspath(inbox_directory)
        }
    )


def prepare_model_checkpoint_import_from_path(
    checkpoint_directory,
    source_path,
    import_method,
    source_metadata=None
):
    '''Validates and archives one explicitly selected checkpoint package.

    This is the shared safe boundary for both supported import routes: a
    package placed in this run's manual inbox and a package selected from a
    previous ``Protocol_Outputs`` run.  The source archive is never modified;
    the new run receives an immutable copy before any model reconstruction is
    attempted.
    '''
    checkpoint_directory = os.fspath(checkpoint_directory)
    source_path = os.path.abspath(os.fspath(source_path))
    import_method = str(import_method).strip()

    if not import_method:
        raise ModelCheckpointError('Checkpoint import method must be recorded.')

    if not source_path.lower().endswith('.zip') or not os.path.isfile(source_path):
        raise ModelCheckpointError(
            'Selected checkpoint package is not a readable .zip file: {!r}. '
            .format(source_path)
        )

    checkpoint = read_model_checkpoint(source_path)
    source_checkpoint_sha256 = get_model_checkpoint_file_sha256(source_path)

    archive_directory = os.path.join(
        checkpoint_directory,
        IMPORTED_ARCHIVE_DIRECTORY_NAME
    )
    os.makedirs(archive_directory, exist_ok=True)
    source_stem = os.path.splitext(os.path.basename(source_path))[0]
    archive_stem = re.sub(r'[^A-Za-z0-9_-]+', '_', source_stem).strip('_')
    if not archive_stem:
        archive_stem = 'checkpoint'
    archive_path = _unique_checkpoint_path(
        archive_directory,
        'imported_{}'.format(_safe_checkpoint_stem(archive_stem))
    )

    try:
        shutil.copy2(source_path, archive_path)
    except OSError as exc:
        raise ModelCheckpointError(
            'Validated checkpoint could not be archived in the new run: {}.'
            .format(exc)
        ) from exc

    archived_checkpoint_sha256 = get_model_checkpoint_file_sha256(archive_path)
    if archived_checkpoint_sha256 != source_checkpoint_sha256:
        raise ModelCheckpointError(
            'Archived checkpoint checksum differs from the validated source '
            'package.'
        )

    checkpoint['source_checkpoint_path'] = os.path.abspath(source_path)
    checkpoint['archived_checkpoint_path'] = os.path.abspath(archive_path)
    checkpoint['source_checkpoint_sha256'] = source_checkpoint_sha256
    checkpoint['archived_checkpoint_sha256'] = archived_checkpoint_sha256
    checkpoint['import_method'] = import_method
    checkpoint['import_source_metadata'] = _json_safe(
        source_metadata if source_metadata is not None else {}
    )
    return checkpoint


def _validate_run_context_identifier(value, field_name):
    '''Returns one non-empty lineage identifier without treating it as a path.'''
    normalized_value = str(value).strip()
    if not normalized_value:
        raise ModelCheckpointError(
            'Run-context lineage {} must be a non-empty string.'.format(
                field_name
            )
        )
    return normalized_value


def _validate_run_context_folder_name(folder_name):
    '''Returns an optional safe output-folder basename for one lineage row.'''
    if folder_name is None:
        return None

    normalized_folder = str(folder_name).strip()
    if not normalized_folder:
        return None

    if (
        normalized_folder in ('.', '..')
        or os.path.basename(normalized_folder) != normalized_folder
    ):
        raise ModelCheckpointError(
            'Run-context lineage run_folder must be one output-folder name, '
            'not a path.'
        )
    return normalized_folder


def _validate_run_context_sha256(value):
    '''Returns one normalized archive SHA-256 digest for lineage identity.'''
    normalized_value = str(value).strip().lower()
    if re.fullmatch(r'[0-9a-f]{64}', normalized_value) is None:
        raise ModelCheckpointError(
            'Run-context lineage checkpoint_sha256 must be a SHA-256 digest.'
        )
    return normalized_value


def _normalize_run_context_entry(entry):
    '''Validates one flat, non-path-bearing imported-run lineage entry.'''
    if not isinstance(entry, dict):
        raise ModelCheckpointError(
            'Run-context lineage source_runs entries must be dictionaries.'
        )

    checkpoint_filename = str(entry.get('checkpoint_filename', '')).strip()
    if (
        not checkpoint_filename
        or os.path.basename(checkpoint_filename) != checkpoint_filename
        or not checkpoint_filename.lower().endswith('.zip')
    ):
        raise ModelCheckpointError(
            'Run-context lineage checkpoint_filename must be one ZIP '
            'basename.'
        )

    parent_run_id = entry.get('parent_run_id')
    if parent_run_id is not None:
        parent_run_id = _validate_run_context_identifier(
            parent_run_id,
            'parent_run_id'
        )

    checkpoint_stage = str(entry.get('checkpoint_stage', '')).strip()
    import_method = str(entry.get('import_method', '')).strip()
    if not checkpoint_stage or not import_method:
        raise ModelCheckpointError(
            'Run-context lineage entries must record checkpoint_stage and '
            'import_method.'
        )

    return {
        'run_id': _validate_run_context_identifier(
            entry.get('run_id'),
            'run_id'
        ),
        'run_folder': _validate_run_context_folder_name(
            entry.get('run_folder')
        ),
        'checkpoint_filename': checkpoint_filename,
        'checkpoint_sha256': _validate_run_context_sha256(
            entry.get('checkpoint_sha256')
        ),
        'checkpoint_stage': checkpoint_stage,
        'import_method': import_method,
        'parent_run_id': parent_run_id
    }


def _deduplicate_run_context_entries(entries):
    '''Returns flat lineage entries once, rejecting conflicting run identity.'''
    normalized_entries = []
    entries_by_run_id = {}

    for entry in entries:
        normalized_entry = _normalize_run_context_entry(entry)
        run_id = normalized_entry['run_id']
        previous_entry = entries_by_run_id.get(run_id)
        if previous_entry is None:
            entries_by_run_id[run_id] = normalized_entry
            normalized_entries.append(normalized_entry)
            continue

        if (
            previous_entry['checkpoint_sha256']
            != normalized_entry['checkpoint_sha256']
        ):
            raise ModelCheckpointError(
                'Run-context lineage contains conflicting checkpoint '
                'identities for run {!r}.'.format(run_id)
            )

        # Identical run/checkpoint entries can occur when a legacy source was
        # manually reconstructed.  Keep the earliest entry so the flattened
        # lineage remains stable rather than duplicating a physical run.

    return normalized_entries


def validate_run_context_lineage_manifest(manifest):
    '''Validates and normalizes one import-only flat lineage manifest.

    A manifest lists *prior* source runs only.  It deliberately excludes the
    run that owns the manifest; a future continuation appends that source run
    exactly once.  This is what prevents six sequential imports from becoming
    nested or double-counted.
    '''
    if not isinstance(manifest, dict):
        raise ModelCheckpointError(
            'Run-context lineage manifest must be a dictionary.'
        )

    if manifest.get('schema_version') != RUN_CONTEXT_LINEAGE_SCHEMA_VERSION:
        raise ModelCheckpointError(
            'Run-context lineage manifest has an unsupported schema version.'
        )

    source_runs = manifest.get('source_runs')
    if not isinstance(source_runs, list):
        raise ModelCheckpointError(
            'Run-context lineage manifest source_runs must be a list.'
        )

    direct_import = manifest.get('direct_import')
    if not isinstance(direct_import, dict):
        raise ModelCheckpointError(
            'Run-context lineage manifest must record direct_import.'
        )

    normalized_direct_import = {
        'source_run_id': _validate_run_context_identifier(
            direct_import.get('source_run_id'),
            'direct_import.source_run_id'
        ),
        'source_checkpoint_sha256': _validate_run_context_sha256(
            direct_import.get('source_checkpoint_sha256')
        ),
        'import_method': str(direct_import.get('import_method', '')).strip()
    }
    if not normalized_direct_import['import_method']:
        raise ModelCheckpointError(
            'Run-context lineage direct_import must record import_method.'
        )

    normalized_manifest = {
        'schema_version': RUN_CONTEXT_LINEAGE_SCHEMA_VERSION,
        'current_run_id': _validate_run_context_identifier(
            manifest.get('current_run_id'),
            'current_run_id'
        ),
        'direct_import': normalized_direct_import,
        'source_runs': _deduplicate_run_context_entries(source_runs)
    }

    if (
        normalized_manifest['current_run_id']
        == normalized_direct_import['source_run_id']
    ):
        raise ModelCheckpointError(
            'Run-context lineage cannot import a checkpoint from its own '
            'run identity.'
        )

    entries_by_run_id = {
        entry['run_id']: entry
        for entry in normalized_manifest['source_runs']
    }
    direct_source_entry = entries_by_run_id.get(
        normalized_direct_import['source_run_id']
    )
    if direct_source_entry is None:
        raise ModelCheckpointError(
            'Run-context lineage direct_import source is absent from '
            'source_runs.'
        )
    if (
        direct_source_entry['checkpoint_sha256']
        != normalized_direct_import['source_checkpoint_sha256']
    ):
        raise ModelCheckpointError(
            'Run-context lineage direct_import checksum does not match its '
            'source_runs entry.'
        )

    for entry in normalized_manifest['source_runs']:
        parent_run_id = entry['parent_run_id']
        if parent_run_id is not None and parent_run_id not in entries_by_run_id:
            raise ModelCheckpointError(
                'Run-context lineage parent {!r} is absent from source_runs.'
                .format(parent_run_id)
            )

    return normalized_manifest


def get_imported_run_context_directory(run_output_directory):
    '''Returns and creates one run's import-only context directory.'''
    context_directory = os.path.join(
        os.fspath(run_output_directory),
        IMPORTED_RUN_CONTEXT_DIRECTORY_NAME
    )
    os.makedirs(context_directory, exist_ok=True)
    return os.path.abspath(context_directory)


def read_run_context_lineage_manifest(run_output_directory):
    '''Reads an existing source run's lineage manifest, if it has one.

    A missing file is expected for runs created before this optional feature
    and is represented by ``None``.  A present but malformed file fails
    closed rather than allowing an ambiguous history to drive later plotting.
    '''
    manifest_path = os.path.join(
        os.fspath(run_output_directory),
        IMPORTED_RUN_CONTEXT_DIRECTORY_NAME,
        RUN_CONTEXT_LINEAGE_FILENAME
    )
    if not os.path.exists(manifest_path):
        return None
    if not os.path.isfile(manifest_path):
        raise ModelCheckpointError(
            'Run-context lineage manifest is not a file: {!r}.'.format(
                manifest_path
            )
        )

    try:
        with open(manifest_path, 'r', encoding='utf-8') as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCheckpointError(
            'Run-context lineage manifest cannot be read safely: {}.'
            .format(exc)
        ) from exc

    return validate_run_context_lineage_manifest(manifest)


def build_import_run_context_lineage_manifest(
    current_run_id,
    source_run_id,
    source_checkpoint_sha256,
    source_checkpoint_stage,
    source_checkpoint_filename,
    import_method,
    source_run_folder=None,
    inherited_manifest=None
):
    '''Builds the flat prior-run lineage for one checkpoint import.

    ``inherited_manifest`` is the source run's own manifest when available.
    Its source rows are copied, then the source run itself is appended once.
    Manual checkpoint imports pass ``None`` and correctly become a verified
    model-only lineage with no assumed filesystem context.
    '''
    current_run_id = _validate_run_context_identifier(
        current_run_id,
        'current_run_id'
    )
    source_run_id = _validate_run_context_identifier(
        source_run_id,
        'source_run_id'
    )
    source_checkpoint_sha256 = _validate_run_context_sha256(
        source_checkpoint_sha256
    )
    source_checkpoint_stage = str(source_checkpoint_stage).strip()
    import_method = str(import_method).strip()
    source_checkpoint_filename = str(source_checkpoint_filename).strip()
    if not source_checkpoint_stage or not import_method:
        raise ModelCheckpointError(
            'Run-context lineage requires checkpoint_stage and import_method.'
        )
    if (
        not source_checkpoint_filename
        or os.path.basename(source_checkpoint_filename)
        != source_checkpoint_filename
        or not source_checkpoint_filename.lower().endswith('.zip')
    ):
        raise ModelCheckpointError(
            'Run-context lineage requires one ZIP checkpoint filename.'
        )

    inherited_source_runs = []
    parent_run_id = None
    if inherited_manifest is not None:
        normalized_inherited = validate_run_context_lineage_manifest(
            inherited_manifest
        )
        if normalized_inherited['current_run_id'] != source_run_id:
            raise ModelCheckpointError(
                'Source run-context lineage identifies {!r}, but the selected '
                'checkpoint identifies {!r}.'.format(
                    normalized_inherited['current_run_id'],
                    source_run_id
                )
            )
        inherited_source_runs = normalized_inherited['source_runs']
        parent_run_id = normalized_inherited['direct_import'][
            'source_run_id'
        ]

    source_entry = {
        'run_id': source_run_id,
        'run_folder': _validate_run_context_folder_name(source_run_folder),
        'checkpoint_filename': source_checkpoint_filename,
        'checkpoint_sha256': source_checkpoint_sha256,
        'checkpoint_stage': source_checkpoint_stage,
        'import_method': import_method,
        'parent_run_id': parent_run_id
    }

    return validate_run_context_lineage_manifest({
        'schema_version': RUN_CONTEXT_LINEAGE_SCHEMA_VERSION,
        'current_run_id': current_run_id,
        'direct_import': {
            'source_run_id': source_run_id,
            'source_checkpoint_sha256': source_checkpoint_sha256,
            'import_method': import_method
        },
        'source_runs': inherited_source_runs + [source_entry]
    })


def write_run_context_lineage_manifest(run_output_directory, manifest):
    '''Writes one immutable import-lineage manifest without overwriting it.'''
    normalized_manifest = validate_run_context_lineage_manifest(manifest)
    context_directory = get_imported_run_context_directory(run_output_directory)
    destination_path = os.path.join(
        context_directory,
        RUN_CONTEXT_LINEAGE_FILENAME
    )
    if os.path.exists(destination_path):
        raise ModelCheckpointError(
            'Run-context lineage manifest already exists: {!r}.'.format(
                destination_path
            )
        )

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb',
            prefix='.auto_run_context_lineage_',
            suffix='.tmp',
            dir=context_directory,
            delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(_json_bytes(normalized_manifest))

        os.replace(temporary_path, destination_path)
        temporary_path = None
    except OSError as exc:
        raise ModelCheckpointError(
            'Run-context lineage manifest could not be written: {}.'.format(
                exc
            )
        ) from exc
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    return os.path.abspath(destination_path)


def write_model_checkpoint_import_provenance(checkpoint_directory, provenance):
    '''Writes one immutable JSON record describing a successful model import.

    A separate plain-JSON provenance file keeps the imported archive, its
    selection route, and reconstruction counts inspectable without unpacking
    the checkpoint.  It deliberately uses the same strict JSON rules as the
    package and never overwrites an earlier record.
    '''
    checkpoint_directory = os.fspath(checkpoint_directory)
    os.makedirs(checkpoint_directory, exist_ok=True)
    payload = _json_bytes(provenance)

    base_path = os.path.join(
        checkpoint_directory,
        'import_provenance.json'
    )
    if not os.path.exists(base_path):
        destination_path = base_path
    else:
        suffix = 1
        while True:
            candidate_path = os.path.join(
                checkpoint_directory,
                'import_provenance_{:03d}.json'.format(suffix)
            )
            if not os.path.exists(candidate_path):
                destination_path = candidate_path
                break
            suffix += 1

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb',
            prefix='.auto_model_checkpoint_import_',
            suffix='.tmp',
            dir=checkpoint_directory,
            delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            temporary_file.write(payload)

        os.replace(temporary_path, destination_path)
    except OSError as exc:
        if temporary_path is not None and os.path.exists(temporary_path):
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        raise ModelCheckpointError(
            'Checkpoint import provenance could not be written: {}.'.format(
                exc
            )
        ) from exc

    return os.path.abspath(destination_path)


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
