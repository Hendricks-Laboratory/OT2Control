'''Offline contract for future immutable Auto Live workbook publication.

This module deliberately contains no Google Drive, network, credential, or
controller integration. It converts one already-verified local queue item into
a deterministic publication request and validates the receipt returned by an
injected future publisher. The contract permits exactly one remote operation:
creating a new immutable snapshot. It does not permit replacing, moving,
deleting, or editing remote files.

The local ``AutoLiveRunSyncQueue`` remains the authority for FIFO ordering and
for immutable local workbook copies. A later reviewed integration may adapt
this contract to a remote service without changing those local rules.
'''

import copy
import hashlib
import os


class AutoLiveRunRemoteBoundaryError(RuntimeError):
    '''Raised when a future remote snapshot boundary contract is violated.'''


REMOTE_SNAPSHOT_SCHEMA_VERSION = 1
REMOTE_SNAPSHOT_REQUEST_TYPE = 'auto_live_run_remote_snapshot_request'
REMOTE_SNAPSHOT_RECEIPT_TYPE = 'auto_live_run_remote_snapshot_receipt'
REMOTE_CREATE_OPERATION = 'create_immutable_snapshot'


def _validate_nonempty_string(value, field_name):
    if not isinstance(value, str) or not value:
        raise AutoLiveRunRemoteBoundaryError(
            '{} must be a nonempty string.'.format(field_name)
        )


def _validate_sha256(value, field_name):
    _validate_nonempty_string(value, field_name)
    if (
        len(value) != 64
        or any(character not in '0123456789abcdef' for character in value)
    ):
        raise AutoLiveRunRemoteBoundaryError(
            '{} must be a lowercase SHA-256 digest.'.format(field_name)
        )


def _validate_state_revision(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutoLiveRunRemoteBoundaryError(
            '{} must be a nonnegative integer.'.format(field_name)
        )


def _safe_run_component(run_id):
    '''Returns a readable, collision-resistant component derived from run_id.'''
    _validate_nonempty_string(run_id, 'run_id')
    readable = ''.join(
        character.lower()
        if character.isascii() and character.isalnum()
        else '-'
        for character in run_id
    )
    readable = '-'.join(
        component for component in readable.split('-') if component
    )
    if not readable:
        readable = 'run'
    readable = readable[:48].rstrip('-')
    digest_prefix = hashlib.sha256(run_id.encode('utf-8')).hexdigest()[:12]
    return '{}-{}'.format(readable, digest_prefix)


def _queue_item_value(queue_item, field_name):
    if not isinstance(queue_item, dict):
        raise AutoLiveRunRemoteBoundaryError(
            'Local Live workbook queue item must be a dictionary.'
        )
    try:
        return queue_item[field_name]
    except KeyError:
        raise AutoLiveRunRemoteBoundaryError(
            'Local Live workbook queue item is missing {}.'.format(field_name)
        )


def build_immutable_snapshot_request(queue_item):
    '''Builds one deterministic, create-only request from a local queue item.

    The remote relative path is fully derived from the run identifier, state
    revision, and immutable workbook digest. Callers cannot supply an arbitrary
    destination path or an overwrite flag.
    '''
    run_id = _queue_item_value(queue_item, 'run_id')
    state_revision = _queue_item_value(queue_item, 'state_revision')
    content_sha256 = _queue_item_value(
        queue_item,
        'local_workbook_sha256'
    )
    _validate_nonempty_string(run_id, 'run_id')
    _validate_state_revision(state_revision, 'state_revision')
    _validate_sha256(content_sha256, 'local_workbook_sha256')

    relative_path = 'Auto_Live_Runs/{}/status/revision_{:06d}_{}.xlsx'.format(
        _safe_run_component(run_id),
        state_revision,
        content_sha256[:12]
    )
    return {
        'schema_version': REMOTE_SNAPSHOT_SCHEMA_VERSION,
        'record_type': REMOTE_SNAPSHOT_REQUEST_TYPE,
        'operation': REMOTE_CREATE_OPERATION,
        'run_id': run_id,
        'state_revision': state_revision,
        'content_sha256': content_sha256,
        'remote_relative_path': relative_path,
        'overwrite': False
    }


def _validate_request(request):
    required_keys = {
        'schema_version', 'record_type', 'operation', 'run_id',
        'state_revision', 'content_sha256', 'remote_relative_path',
        'overwrite'
    }
    if not isinstance(request, dict) or set(request) != required_keys:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote immutable-snapshot request has an invalid schema.'
        )
    if request['schema_version'] != REMOTE_SNAPSHOT_SCHEMA_VERSION:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote immutable-snapshot request has an unsupported schema version.'
        )
    if request['record_type'] != REMOTE_SNAPSHOT_REQUEST_TYPE:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote immutable-snapshot request has an invalid record type.'
        )
    if request['operation'] != REMOTE_CREATE_OPERATION:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote boundary permits only immutable snapshot creation.'
        )
    if request['overwrite'] is not False:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote immutable-snapshot requests must refuse overwrite.'
        )
    _validate_nonempty_string(request['run_id'], 'request.run_id')
    _validate_state_revision(
        request['state_revision'],
        'request.state_revision'
    )
    _validate_sha256(request['content_sha256'], 'request.content_sha256')
    expected_path = 'Auto_Live_Runs/{}/status/revision_{:06d}_{}.xlsx'.format(
        _safe_run_component(request['run_id']),
        request['state_revision'],
        request['content_sha256'][:12]
    )
    if request['remote_relative_path'] != expected_path:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote immutable-snapshot path must be derived by this contract.'
        )


def _validate_receipt(request, receipt):
    required_keys = {
        'schema_version', 'record_type', 'operation', 'run_id',
        'state_revision', 'content_sha256', 'remote_relative_path',
        'created', 'remote_object_id'
    }
    if not isinstance(receipt, dict) or set(receipt) != required_keys:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote immutable-snapshot receipt has an invalid schema.'
        )
    if receipt['schema_version'] != REMOTE_SNAPSHOT_SCHEMA_VERSION:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote immutable-snapshot receipt has an unsupported schema version.'
        )
    if receipt['record_type'] != REMOTE_SNAPSHOT_RECEIPT_TYPE:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote immutable-snapshot receipt has an invalid record type.'
        )
    if receipt['created'] is not True:
        raise AutoLiveRunRemoteBoundaryError(
            'Remote publisher did not confirm immutable snapshot creation.'
        )
    _validate_nonempty_string(receipt['remote_object_id'], 'remote_object_id')
    for field_name in (
        'operation', 'run_id', 'state_revision', 'content_sha256',
        'remote_relative_path'
    ):
        if receipt[field_name] != request[field_name]:
            raise AutoLiveRunRemoteBoundaryError(
                'Remote immutable-snapshot receipt changed {}.'.format(
                    field_name
                )
            )


def publish_immutable_snapshot(publisher, queue_item, workbook_path):
    '''Publishes one immutable snapshot through an injected future publisher.

    This function is intentionally unused by the controller in Stage 11A. It
    is exercised only with a local fake publisher. A later adapter must expose
    ``create_immutable_snapshot(request, workbook_path)`` and return a receipt
    conforming to this module's strict schema.
    '''
    if (
        publisher is None
        or not callable(getattr(publisher, 'create_immutable_snapshot', None))
    ):
        raise AutoLiveRunRemoteBoundaryError(
            'Remote publication requires create_immutable_snapshot on its publisher.'
        )

    request = build_immutable_snapshot_request(queue_item)
    _validate_request(request)
    workbook_path = os.path.abspath(workbook_path)
    if not os.path.isfile(workbook_path):
        raise AutoLiveRunRemoteBoundaryError(
            'Immutable local workbook snapshot is missing: {}.'.format(
                workbook_path
            )
        )
    try:
        with open(workbook_path, 'rb') as input_file:
            observed_sha256 = hashlib.sha256(input_file.read()).hexdigest()
    except OSError as exc:
        raise AutoLiveRunRemoteBoundaryError(
            'Could not read immutable local workbook snapshot: {}.'.format(exc)
        )
    if observed_sha256 != request['content_sha256']:
        raise AutoLiveRunRemoteBoundaryError(
            'Immutable local workbook snapshot checksum changed before publication.'
        )

    try:
        receipt = publisher.create_immutable_snapshot(
            copy.deepcopy(request),
            workbook_path
        )
    except Exception as exc:
        raise AutoLiveRunRemoteBoundaryError(
            'Immutable remote snapshot creation failed: {}.'.format(exc)
        )
    _validate_receipt(request, receipt)
    return copy.deepcopy(receipt)
