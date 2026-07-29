'''Durable, ordered, offline-safe queue for future Auto Live workbook mirrors.

The queue is local and dependency-free.  It records rendered workbook
snapshots in order and can replay them through an injected adapter.  Stage 2
does not instantiate a Drive adapter or access credentials; controller code
only writes the local queue.  A later, separately reviewed integration may
provide an adapter without changing the queue's ordering or durability rules.
'''

import copy
import datetime
import hashlib
import json
import os
import tempfile


class AutoLiveRunSyncQueueError(RuntimeError):
    '''Raised when the local cloud-sync queue cannot be updated safely.'''


class AutoLiveRunSyncQueue:
    '''Stores pending Live workbook mirror snapshots in strict FIFO order.'''

    PENDING_FILENAME = 'pending_cloud_sync.jsonl'
    SNAPSHOT_DIRECTORYNAME = 'pending_sync_snapshots'
    QUEUE_ITEM_RECORD_TYPE = 'auto_live_workbook_sync'
    SCHEMA_VERSION = 1

    def __init__(self, run_state_directory):
        self.run_state_directory = os.path.abspath(run_state_directory)
        self.pending_path = os.path.join(
            self.run_state_directory,
            self.PENDING_FILENAME
        )

    @staticmethod
    def _utc_now_string():
        return datetime.datetime.now(
            datetime.timezone.utc
        ).replace(microsecond=0).isoformat()

    @staticmethod
    def _canonical_json_line_bytes(value):
        try:
            serialized = json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
                separators=(',', ':')
            )
        except (TypeError, ValueError) as exc:
            raise AutoLiveRunSyncQueueError(
                'Cloud-sync queue data must be JSON serializable: {}.'.format(
                    exc
                )
            )
        return (serialized + '\n').encode('utf-8')

    @classmethod
    def _atomic_write_bytes(cls, destination_path, payload_bytes):
        file_descriptor = None
        temporary_path = None
        destination_directory = os.path.dirname(destination_path)
        try:
            file_descriptor, temporary_path = tempfile.mkstemp(
                prefix='.{0}.'.format(os.path.basename(destination_path)),
                suffix='.tmp',
                dir=destination_directory
            )
            with os.fdopen(file_descriptor, 'wb') as output_file:
                file_descriptor = None
                output_file.write(payload_bytes)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary_path, destination_path)
            temporary_path = None
        except OSError as exc:
            raise AutoLiveRunSyncQueueError(
                'Could not update cloud-sync queue {}: {}.'.format(
                    destination_path,
                    exc
                )
            )
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            if temporary_path is not None and os.path.exists(temporary_path):
                os.remove(temporary_path)

    @classmethod
    def initialize(cls, run_state_directory):
        '''Creates the Stage 2 queue without overwriting an existing queue.'''
        queue = cls(run_state_directory)
        if not os.path.isdir(queue.run_state_directory):
            raise AutoLiveRunSyncQueueError(
                'Run_State directory does not exist: {}.'.format(
                    queue.run_state_directory
                )
            )
        if not os.path.exists(queue.pending_path):
            cls._atomic_write_bytes(queue.pending_path, b'')
        return queue

    @classmethod
    def _validate_item(cls, item):
        required_keys = {
            'schema_version',
            'record_type',
            'queue_id',
            'run_id',
            'state_revision',
            'created_at_utc',
            'local_workbook_relative_path',
            'local_workbook_sha256'
        }
        if not isinstance(item, dict) or set(item) != required_keys:
            raise AutoLiveRunSyncQueueError(
                'Cloud-sync queue item has an invalid schema.'
            )
        if item['schema_version'] != cls.SCHEMA_VERSION:
            raise AutoLiveRunSyncQueueError(
                'Cloud-sync queue item has an unsupported schema version.'
            )
        if item['record_type'] != cls.QUEUE_ITEM_RECORD_TYPE:
            raise AutoLiveRunSyncQueueError(
                'Cloud-sync queue item has an invalid record type.'
            )
        for field_name in (
            'queue_id',
            'run_id',
            'created_at_utc',
            'local_workbook_relative_path',
            'local_workbook_sha256'
        ):
            if not isinstance(item[field_name], str) or not item[field_name]:
                raise AutoLiveRunSyncQueueError(
                    'Cloud-sync queue item field {} must be nonempty.'.format(
                        field_name
                    )
                )
        if (
            isinstance(item['state_revision'], bool)
            or not isinstance(item['state_revision'], int)
            or item['state_revision'] < 0
        ):
            raise AutoLiveRunSyncQueueError(
                'Cloud-sync queue item state_revision must be nonnegative.'
            )
        digest = item['local_workbook_sha256']
        if (
            len(digest) != 64
            or any(character not in '0123456789abcdef' for character in digest)
        ):
            raise AutoLiveRunSyncQueueError(
                'Cloud-sync queue item must contain a lowercase SHA-256 digest.'
            )

    def read_pending(self):
        '''Returns FIFO queue records without modifying the queue.'''
        try:
            with open(self.pending_path, 'rb') as input_file:
                lines = input_file.readlines()
        except OSError as exc:
            raise AutoLiveRunSyncQueueError(
                'Could not read cloud-sync queue: {}.'.format(exc)
            )

        pending = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line.decode('utf-8'))
            except (UnicodeDecodeError, ValueError) as exc:
                raise AutoLiveRunSyncQueueError(
                    'Cloud-sync queue line {} is invalid: {}.'.format(
                        line_number,
                        exc
                    )
                )
            self._validate_item(item)
            pending.append(item)
        return pending

    def enqueue_rendered_workbook(self, render_result):
        '''Queues one rendered state revision unless that exact revision exists.'''
        required_keys = {
            'workbook_path',
            'workbook_sha256',
            'state_revision',
            'run_id'
        }
        if not isinstance(render_result, dict) or set(render_result) != required_keys:
            raise AutoLiveRunSyncQueueError(
                'Live workbook render result has an invalid schema.'
            )

        workbook_path = os.path.abspath(render_result['workbook_path'])
        if not os.path.isfile(workbook_path):
            raise AutoLiveRunSyncQueueError(
                'Rendered Live workbook is missing: {}.'.format(workbook_path)
            )
        try:
            relative_path = os.path.relpath(
                workbook_path,
                os.path.dirname(self.run_state_directory)
            )
        except ValueError as exc:
            raise AutoLiveRunSyncQueueError(
                'Live workbook path is not compatible with its run directory: {}.'.format(
                    exc
                )
            )
        if relative_path.startswith(os.pardir + os.sep) or relative_path == os.pardir:
            raise AutoLiveRunSyncQueueError(
                'Live workbook must be stored beneath its run directory.'
            )

        state_revision = render_result['state_revision']
        run_id = render_result['run_id']
        queue_id = '{}:{}'.format(run_id, state_revision)
        pending = self.read_pending()
        if any(item['queue_id'] == queue_id for item in pending):
            return None

        # The current Live workbook is intentionally replaced at every state
        # revision. Preserve an immutable per-revision copy for the queue so
        # a later refresh cannot change what an earlier remote replay means.
        snapshot_directory = os.path.join(
            os.path.dirname(self.run_state_directory),
            'Live_Run',
            self.SNAPSHOT_DIRECTORYNAME
        )
        try:
            if not os.path.isdir(snapshot_directory):
                os.makedirs(snapshot_directory)
            with open(workbook_path, 'rb') as input_file:
                workbook_bytes = input_file.read()
        except OSError as exc:
            raise AutoLiveRunSyncQueueError(
                'Could not snapshot rendered Live workbook: {}.'.format(exc)
            )
        observed_sha256 = hashlib.sha256(workbook_bytes).hexdigest()
        if observed_sha256 != render_result['workbook_sha256']:
            raise AutoLiveRunSyncQueueError(
                'Rendered Live workbook changed before it could be queued.'
            )
        snapshot_filename = '{}.revision_{:06d}.xlsx'.format(
            os.path.splitext(os.path.basename(workbook_path))[0],
            state_revision
        )
        snapshot_path = os.path.join(snapshot_directory, snapshot_filename)
        if os.path.exists(snapshot_path):
            try:
                with open(snapshot_path, 'rb') as input_file:
                    existing_sha256 = hashlib.sha256(
                        input_file.read()
                    ).hexdigest()
            except OSError as exc:
                raise AutoLiveRunSyncQueueError(
                    'Could not validate queued Live workbook snapshot: {}.'.format(
                        exc
                    )
                )
            if existing_sha256 != observed_sha256:
                raise AutoLiveRunSyncQueueError(
                    'Refusing to replace a different queued Live workbook snapshot.'
                )
        else:
            self._atomic_write_bytes(snapshot_path, workbook_bytes)

        item = {
            'schema_version': self.SCHEMA_VERSION,
            'record_type': self.QUEUE_ITEM_RECORD_TYPE,
            'queue_id': queue_id,
            'run_id': run_id,
            'state_revision': state_revision,
            'created_at_utc': self._utc_now_string(),
            'local_workbook_relative_path': os.path.relpath(
                snapshot_path,
                os.path.dirname(self.run_state_directory)
            ).replace(os.sep, '/'),
            'local_workbook_sha256': observed_sha256
        }
        self._validate_item(item)
        try:
            with open(self.pending_path, 'ab') as output_file:
                output_file.write(self._canonical_json_line_bytes(item))
                output_file.flush()
                os.fsync(output_file.fileno())
        except OSError as exc:
            raise AutoLiveRunSyncQueueError(
                'Could not append cloud-sync queue item: {}.'.format(exc)
            )
        return copy.deepcopy(item)

    def replay(self, adapter):
        '''Replays pending workbooks in order through one injected adapter.

        ``adapter`` must implement ``upload_live_workbook(item, workbook_path)``.
        A failure intentionally stops replay at the first unsynchronized item;
        later items must not overtake an earlier state revision.
        '''
        if adapter is None or not hasattr(adapter, 'upload_live_workbook'):
            raise AutoLiveRunSyncQueueError(
                'Cloud-sync replay requires an adapter with upload_live_workbook.'
            )

        replayed = []
        for item in self.read_pending():
            run_directory = os.path.dirname(self.run_state_directory)
            workbook_path = os.path.abspath(os.path.join(
                run_directory,
                item['local_workbook_relative_path']
            ))
            try:
                if os.path.commonpath([run_directory, workbook_path]) != run_directory:
                    raise AutoLiveRunSyncQueueError(
                        'Queued Live workbook path escapes its run directory.'
                    )
            except ValueError as exc:
                raise AutoLiveRunSyncQueueError(
                    'Queued Live workbook path is invalid: {}.'.format(exc)
                )
            if not os.path.isfile(workbook_path):
                raise AutoLiveRunSyncQueueError(
                    'Queued Live workbook is missing: {}.'.format(workbook_path)
                )
            with open(workbook_path, 'rb') as input_file:
                observed_sha256 = hashlib.sha256(input_file.read()).hexdigest()
            if observed_sha256 != item['local_workbook_sha256']:
                raise AutoLiveRunSyncQueueError(
                    'Queued Live workbook checksum changed: {}.'.format(
                        workbook_path
                    )
                )
            try:
                adapter.upload_live_workbook(copy.deepcopy(item), workbook_path)
            except Exception as exc:
                raise AutoLiveRunSyncQueueError(
                    'Cloud-sync replay stopped at {}: {}.'.format(
                        item['queue_id'],
                        exc
                    )
                )
            remaining = self.read_pending()
            if not remaining or remaining[0]['queue_id'] != item['queue_id']:
                raise AutoLiveRunSyncQueueError(
                    'Cloud-sync queue changed during ordered replay.'
                )
            remaining_bytes = b''.join(
                self._canonical_json_line_bytes(queued_item)
                for queued_item in remaining[1:]
            )
            self._atomic_write_bytes(self.pending_path, remaining_bytes)
            replayed.append(copy.deepcopy(item))
        return replayed
