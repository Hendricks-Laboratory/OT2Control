'''Durable local journal primitives for future Auto live-run recovery.

The journal is controller-local and filesystem-only. It intentionally knows
nothing about Google Drive, robot commands, workbook edits, or recovery
actions. Those integrations are deferred to later reviewed stages.
'''

import copy
import datetime
import hashlib
import json
import os
import tempfile

from auto_live_run_state import (
    CURRENT_STATE_RECORD_TYPE,
    EVENT_RECORD_TYPE,
    LIFECYCLE_READY_FOR_BATCH,
    MANIFEST_RECORD_TYPE,
    LIVE_RUN_STATE_SCHEMA_VERSION,
    assert_valid_lifecycle_transition,
    make_initial_current_state,
    validate_current_state,
    validate_event,
    validate_run_manifest
)


class AutoLiveRunJournalError(RuntimeError):
    '''Raised when a durable journal operation cannot complete safely.'''


class AutoLiveRunJournal:
    '''Writes one local Auto live-run journal using the versioned contract.'''

    INPUT_SNAPSHOT_FILENAME = 'input_snapshot.json'
    HEADER_SNAPSHOT_FILENAME = 'header_snapshot.json'
    RUNTIME_BASELINE_FILENAME = 'runtime_baseline.json'
    MANIFEST_FILENAME = 'run_manifest.json'
    CURRENT_STATE_FILENAME = 'current_state.json'
    EVENTS_FILENAME = 'events.jsonl'

    def __init__(self, run_state_directory, current_state):
        self.run_state_directory = os.path.abspath(run_state_directory)
        self.current_state_path = os.path.join(
            self.run_state_directory,
            self.CURRENT_STATE_FILENAME
        )
        self.events_path = os.path.join(
            self.run_state_directory,
            self.EVENTS_FILENAME
        )
        validate_current_state(current_state)
        self.current_state = copy.deepcopy(current_state)

    @staticmethod
    def _canonical_json_bytes(value):
        try:
            text = json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise AutoLiveRunJournalError(
                'Live-run journal data must be JSON serializable: {}.'.format(
                    exc
                )
            )

        return (text + '\n').encode('utf-8')

    @staticmethod
    def _canonical_json_line_bytes(value):
        '''Returns one compact JSON line suitable for append-only JSONL.'''
        try:
            text = json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
                separators=(',', ':')
            )
        except (TypeError, ValueError) as exc:
            raise AutoLiveRunJournalError(
                'Live-run event data must be JSON serializable: {}.'.format(
                    exc
                )
            )

        return (text + '\n').encode('utf-8')

    @classmethod
    def _sha256_json(cls, value):
        return hashlib.sha256(cls._canonical_json_bytes(value)).hexdigest()

    @staticmethod
    def _utc_now_string():
        return datetime.datetime.now(
            datetime.timezone.utc
        ).replace(microsecond=0).isoformat()

    @staticmethod
    def _atomic_write_bytes(destination_path, payload_bytes):
        destination_directory = os.path.dirname(destination_path)
        file_descriptor = None
        temporary_path = None

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
            raise AutoLiveRunJournalError(
                'Could not atomically write {}: {}.'.format(
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
    def _atomic_write_json(cls, destination_path, value):
        cls._atomic_write_bytes(
            destination_path,
            cls._canonical_json_bytes(value)
        )

    @classmethod
    def initialize(
        cls,
        run_state_directory,
        run_id,
        input_snapshot,
        header_snapshot,
        runtime_baseline,
        git_branch,
        git_commit,
        created_at_utc=None
    ):
        '''Creates a new journal without overwriting prior live-run state.'''
        run_state_directory = os.path.abspath(run_state_directory)

        if os.path.exists(run_state_directory):
            existing_entries = os.listdir(run_state_directory)
            if existing_entries:
                raise AutoLiveRunJournalError(
                    'Refusing to overwrite existing live-run state in {}.'.format(
                        run_state_directory
                    )
                )
        else:
            try:
                os.makedirs(run_state_directory)
            except OSError as exc:
                raise AutoLiveRunJournalError(
                    'Could not create live-run state directory {}: {}.'.format(
                        run_state_directory,
                        exc
                    )
                )

        created_at_utc = created_at_utc or cls._utc_now_string()
        manifest = {
            'schema_version': LIVE_RUN_STATE_SCHEMA_VERSION,
            'record_type': MANIFEST_RECORD_TYPE,
            'run_id': run_id,
            'created_at_utc': created_at_utc,
            'input_snapshot_sha256': cls._sha256_json(input_snapshot),
            'header_snapshot_sha256': cls._sha256_json(header_snapshot),
            'runtime_baseline_sha256': cls._sha256_json(runtime_baseline),
            'git_branch': git_branch,
            'git_commit': git_commit
        }
        validate_run_manifest(manifest)

        try:
            cls._atomic_write_json(
                os.path.join(run_state_directory, cls.INPUT_SNAPSHOT_FILENAME),
                input_snapshot
            )
            cls._atomic_write_json(
                os.path.join(run_state_directory, cls.HEADER_SNAPSHOT_FILENAME),
                header_snapshot
            )
            cls._atomic_write_json(
                os.path.join(
                    run_state_directory,
                    cls.RUNTIME_BASELINE_FILENAME
                ),
                runtime_baseline
            )
            cls._atomic_write_json(
                os.path.join(run_state_directory, cls.MANIFEST_FILENAME),
                manifest
            )
            cls._atomic_write_json(
                os.path.join(
                    run_state_directory,
                    cls.CURRENT_STATE_FILENAME
                ),
                make_initial_current_state(run_id)
            )
        except (AutoLiveRunJournalError, OSError):
            raise

        journal = cls(run_state_directory, make_initial_current_state(run_id))
        journal.record_event(
            'run_initialized',
            {
                'manifest_filename': cls.MANIFEST_FILENAME,
                'input_snapshot_filename': cls.INPUT_SNAPSHOT_FILENAME,
                'header_snapshot_filename': cls.HEADER_SNAPSHOT_FILENAME,
                'runtime_baseline_filename': cls.RUNTIME_BASELINE_FILENAME
            }
        )
        journal.record_transition(
            LIFECYCLE_READY_FOR_BATCH,
            'input_snapshot_created',
            {'manifest_sha256': cls._sha256_json(manifest)},
            active_batch_number=None
        )
        return journal

    def _record(self, next_state, event_type, payload):
        next_state = copy.deepcopy(next_state)
        next_state['revision'] = self.current_state['revision'] + 1
        next_state['last_event_sequence'] = (
            self.current_state['last_event_sequence'] + 1
        )

        assert_valid_lifecycle_transition(self.current_state, next_state)

        event = {
            'schema_version': LIVE_RUN_STATE_SCHEMA_VERSION,
            'record_type': EVENT_RECORD_TYPE,
            'run_id': next_state['run_id'],
            'sequence': next_state['last_event_sequence'],
            'timestamp_utc': self._utc_now_string(),
            'event_type': event_type,
            'state_revision': next_state['revision'],
            'payload': copy.deepcopy(payload)
        }
        validate_event(event)

        # Snapshot hashes deliberately use readable canonical JSON. Events
        # are JSONL and therefore must occupy exactly one physical line.
        event_bytes = self._canonical_json_line_bytes(event)
        try:
            with open(self.events_path, 'ab') as event_file:
                event_file.write(event_bytes)
                event_file.flush()
                os.fsync(event_file.fileno())
        except OSError as exc:
            raise AutoLiveRunJournalError(
                'Could not append live-run event: {}.'.format(exc)
            )

        self._atomic_write_json(self.current_state_path, next_state)
        self.current_state = next_state
        return copy.deepcopy(event)

    def record_event(self, event_type, payload):
        '''Appends one revisioned event without changing lifecycle phase.'''
        return self._record(
            self.current_state,
            event_type,
            payload
        )

    def record_transition(
        self,
        lifecycle_state,
        event_type,
        payload,
        active_batch_number=None,
        hold_action_id=None,
        fault_id=None
    ):
        '''Appends one event and atomically advances lifecycle state.'''
        next_state = copy.deepcopy(self.current_state)
        next_state['lifecycle_state'] = lifecycle_state
        next_state['active_batch_number'] = active_batch_number
        next_state['hold_action_id'] = hold_action_id
        next_state['fault_id'] = fault_id
        return self._record(next_state, event_type, payload)
