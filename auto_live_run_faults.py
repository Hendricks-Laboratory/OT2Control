'''Pure, local evidence packages for unexpected Auto live-run faults.

This module deliberately has no controller, robot, workbook, cloud, or model
imports.  Stage 10A defines the immutable evidence contract only; later stages
will decide when the controller enters a fault state and presents it to an
operator.  In particular, this module never retries, resumes, trains a model,
or claims to know an unacknowledged physical transfer outcome.
'''

import copy
import csv
import datetime
import json
import os
import tempfile

from auto_live_run_state import (
    LIFECYCLE_EXECUTING_BATCH,
    LIFECYCLE_FAULTED_PARTIAL_BATCH,
    LIFECYCLE_FAULTED_PREPARATION,
    LIFECYCLE_MEASURING_BATCH,
    LIFECYCLE_PROCESSING_BATCH,
    LIFECYCLE_READY_FOR_BATCH,
    LIVE_RUN_STATE_SCHEMA_VERSION,
)


FAULT_EVIDENCE_RECORD_TYPE = 'auto_live_run_fault_evidence'

FAULT_SCOPE_AUTO_PREPARATION = 'auto_preparation'
FAULT_SCOPE_BATCH = 'batch'

FAULT_DISPOSITION_HUMAN_REVIEW_REQUIRED = (
    'human_review_required_no_automatic_resume'
)

FAULT_CERTAINTY_PREPARATION_UNKNOWN_OR_PARTIAL = (
    'preparation_execution_unknown_or_partial'
)
FAULT_CERTAINTY_TRANSFER_UNKNOWN_OR_PARTIAL = (
    'transfer_execution_unknown_or_partial'
)
FAULT_CERTAINTY_TRANSFER_COMPLETE_MEASUREMENT_UNKNOWN = (
    'transfer_complete_measurement_unknown'
)
FAULT_CERTAINTY_MEASUREMENT_COMPLETE_PROCESSING_UNKNOWN = (
    'measurement_complete_processing_unknown'
)

_FAULT_SCOPE_RULES = {
    FAULT_SCOPE_AUTO_PREPARATION: {
        'lifecycle_state': LIFECYCLE_FAULTED_PREPARATION,
        'preceding_state_by_certainty': {
            FAULT_CERTAINTY_PREPARATION_UNKNOWN_OR_PARTIAL:
                LIFECYCLE_READY_FOR_BATCH
        },
        'requires_active_batch': False
    },
    FAULT_SCOPE_BATCH: {
        'lifecycle_state': LIFECYCLE_FAULTED_PARTIAL_BATCH,
        'preceding_state_by_certainty': {
            FAULT_CERTAINTY_TRANSFER_UNKNOWN_OR_PARTIAL:
                LIFECYCLE_EXECUTING_BATCH,
            FAULT_CERTAINTY_TRANSFER_COMPLETE_MEASUREMENT_UNKNOWN:
                LIFECYCLE_MEASURING_BATCH,
            FAULT_CERTAINTY_MEASUREMENT_COMPLETE_PROCESSING_UNKNOWN:
                LIFECYCLE_PROCESSING_BATCH
        },
        'requires_active_batch': True
    }
}


def classify_fault_lifecycle(
    preceding_lifecycle_state,
    preparation_execution_may_have_started=False
):
    '''Classifies one potentially physical Auto interruption conservatively.

    This helper intentionally has no knowledge of controller exceptions,
    portal packets, or hardware.  The controller supplies the last durable
    lifecycle state and whether it had already dispatched grouped
    preparation.  A returned mapping is sufficient to construct a fault
    record; ``None`` means the failure occurred before this stage has
    evidence that liquid handling could have started.

    Preparation takes precedence only while the journal is still at its
    pre-batch ``ready_for_batch`` boundary.  A later batch lifecycle state is
    always classified from that durable state, even if a stale controller flag
    exists, so the record cannot hide an active experimental batch.
    '''
    if (
        preparation_execution_may_have_started
        and preceding_lifecycle_state == LIFECYCLE_READY_FOR_BATCH
    ):
        return {
            'fault_scope': FAULT_SCOPE_AUTO_PREPARATION,
            'certainty': FAULT_CERTAINTY_PREPARATION_UNKNOWN_OR_PARTIAL,
            'lifecycle_state': LIFECYCLE_FAULTED_PREPARATION
        }

    batch_certainty_by_state = {
        LIFECYCLE_EXECUTING_BATCH:
            FAULT_CERTAINTY_TRANSFER_UNKNOWN_OR_PARTIAL,
        LIFECYCLE_MEASURING_BATCH:
            FAULT_CERTAINTY_TRANSFER_COMPLETE_MEASUREMENT_UNKNOWN,
        LIFECYCLE_PROCESSING_BATCH:
            FAULT_CERTAINTY_MEASUREMENT_COMPLETE_PROCESSING_UNKNOWN
    }
    certainty = batch_certainty_by_state.get(preceding_lifecycle_state)
    if certainty is None:
        return None
    return {
        'fault_scope': FAULT_SCOPE_BATCH,
        'certainty': certainty,
        'lifecycle_state': LIFECYCLE_FAULTED_PARTIAL_BATCH
    }


class AutoLiveRunFaultError(RuntimeError):
    '''Raised when fault evidence is malformed or cannot be durably written.'''


def _require_nonempty_string(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise AutoLiveRunFaultError(
            '{} must be a nonempty string.'.format(field_name)
        )


def _require_nonnegative_integer(value, field_name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AutoLiveRunFaultError(
            '{} must be a nonnegative integer.'.format(field_name)
        )


def _require_utc_timestamp(value, field_name):
    _require_nonempty_string(value, field_name)
    try:
        parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise AutoLiveRunFaultError(
            '{} must be an ISO-8601 timestamp.'.format(field_name)
        )

    if parsed.tzinfo is None or parsed.utcoffset() != datetime.timedelta(0):
        raise AutoLiveRunFaultError(
            '{} must be expressed in UTC.'.format(field_name)
        )


def _require_json_mapping(value, field_name):
    if not isinstance(value, dict):
        raise AutoLiveRunFaultError('{} must be a dictionary.'.format(field_name))
    _canonical_json_bytes(value, field_name)


def _canonical_json_bytes(value, field_name):
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise AutoLiveRunFaultError(
            '{} must be JSON serializable: {}.'.format(field_name, exc)
        )
    return (text + '\n').encode('utf-8')


def _utc_now_string():
    return datetime.datetime.now(
        datetime.timezone.utc
    ).replace(microsecond=0).isoformat()


def build_fault_record(
    run_id,
    fault_id,
    fault_scope,
    certainty,
    preceding_lifecycle_state,
    lifecycle_state,
    active_batch_number,
    exception_class,
    exception_message,
    last_event_sequence,
    batch_context,
    recorded_at_utc=None
):
    '''Builds and validates one immutable, conservative fault summary.

    The record describes what is known at the time of a fault, not what the
    robot certainly did.  Physical outcome uncertainty remains explicit and
    later controller stages must treat every record as terminal.
    '''
    record = {
        'schema_version': LIVE_RUN_STATE_SCHEMA_VERSION,
        'record_type': FAULT_EVIDENCE_RECORD_TYPE,
        'run_id': run_id,
        'fault_id': fault_id,
        'recorded_at_utc': recorded_at_utc or _utc_now_string(),
        'fault_scope': fault_scope,
        'certainty': certainty,
        'preceding_lifecycle_state': preceding_lifecycle_state,
        'lifecycle_state': lifecycle_state,
        'disposition': FAULT_DISPOSITION_HUMAN_REVIEW_REQUIRED,
        'active_batch_number': active_batch_number,
        'exception_class': exception_class,
        'exception_message': exception_message,
        'last_event_sequence': last_event_sequence,
        'batch_context': copy.deepcopy(batch_context)
    }
    validate_fault_record(record)
    return copy.deepcopy(record)


def validate_fault_record(record):
    '''Validates one immutable fault-summary record without modifying it.'''
    if not isinstance(record, dict):
        raise AutoLiveRunFaultError('fault record must be a dictionary.')

    expected_keys = {
        'schema_version',
        'record_type',
        'run_id',
        'fault_id',
        'recorded_at_utc',
        'fault_scope',
        'certainty',
        'preceding_lifecycle_state',
        'lifecycle_state',
        'disposition',
        'active_batch_number',
        'exception_class',
        'exception_message',
        'last_event_sequence',
        'batch_context'
    }
    actual_keys = set(record.keys())
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        details = []
        if missing:
            details.append('missing {}'.format(', '.join(missing)))
        if unexpected:
            details.append('unexpected {}'.format(', '.join(unexpected)))
        raise AutoLiveRunFaultError(
            'fault record has {}.'.format('; '.join(details))
        )

    if record['schema_version'] != LIVE_RUN_STATE_SCHEMA_VERSION:
        raise AutoLiveRunFaultError(
            'fault record must use schema_version {}.'.format(
                LIVE_RUN_STATE_SCHEMA_VERSION
            )
        )
    if record['record_type'] != FAULT_EVIDENCE_RECORD_TYPE:
        raise AutoLiveRunFaultError('fault record has an invalid record_type.')
    if record['disposition'] != FAULT_DISPOSITION_HUMAN_REVIEW_REQUIRED:
        raise AutoLiveRunFaultError(
            'fault record.disposition must require human review without '
            'automatic resume.'
        )

    for field_name in (
            'run_id', 'fault_id', 'exception_class', 'exception_message'):
        _require_nonempty_string(record[field_name], 'fault record.{}'.format(
            field_name
        ))
    _require_utc_timestamp(record['recorded_at_utc'], 'fault record.recorded_at_utc')
    _require_nonnegative_integer(
        record['last_event_sequence'],
        'fault record.last_event_sequence'
    )
    _require_json_mapping(record['batch_context'], 'fault record.batch_context')

    fault_scope = record['fault_scope']
    if fault_scope not in _FAULT_SCOPE_RULES:
        raise AutoLiveRunFaultError(
            'fault record.fault_scope is invalid: {!r}.'.format(fault_scope)
        )
    scope_rule = _FAULT_SCOPE_RULES[fault_scope]
    if record['lifecycle_state'] != scope_rule['lifecycle_state']:
        raise AutoLiveRunFaultError(
            'fault record.lifecycle_state does not match {} scope.'.format(
                fault_scope
            )
        )
    expected_preceding_state = scope_rule['preceding_state_by_certainty'].get(
        record['certainty']
    )
    if expected_preceding_state is None:
        raise AutoLiveRunFaultError(
            'fault record.certainty is invalid for {} scope.'.format(
                fault_scope
            )
        )
    if record['preceding_lifecycle_state'] != expected_preceding_state:
        raise AutoLiveRunFaultError(
            'fault record.preceding_lifecycle_state is incompatible with '
            'its stated certainty.'
        )

    active_batch_number = record['active_batch_number']
    if scope_rule['requires_active_batch']:
        _require_nonnegative_integer(
            active_batch_number,
            'fault record.active_batch_number'
        )
    elif active_batch_number is not None:
        raise AutoLiveRunFaultError(
            'auto_preparation faults must not claim active_batch_number.'
        )


class AutoLiveRunFaultEvidenceWriter:
    '''Atomically publishes immutable fault evidence below ``Run_State``.

    The writer stages a complete package in the destination filesystem and
    renames it into place only after all evidence files have been flushed.  It
    refuses a duplicate fault identifier rather than replacing prior evidence.
    No caller can use this writer to resume or modify a faulted run.
    '''

    FAULT_EVIDENCE_DIRECTORYNAME = 'Fault_Evidence'
    FAULT_SUMMARY_JSON_FILENAME = 'fault_summary.json'
    FAULT_SUMMARY_MARKDOWN_FILENAME = 'fault_summary.md'
    BATCH_CONTEXT_FILENAME = 'batch_context.json'
    PLANNED_PROTOCOL_FILENAME = 'planned_protocol_dataframe.csv'
    EXCEPTION_TRACE_FILENAME = 'exception_trace.txt'

    @classmethod
    def _write_bytes(cls, path, payload_bytes):
        try:
            with open(path, 'wb') as output_file:
                output_file.write(payload_bytes)
                output_file.flush()
                os.fsync(output_file.fileno())
        except OSError as exc:
            raise AutoLiveRunFaultError(
                'Could not write fault evidence file {}: {}.'.format(path, exc)
            )

    @classmethod
    def _write_csv(cls, path, columns, rows):
        if not isinstance(columns, (list, tuple)):
            raise AutoLiveRunFaultError('planned protocol columns must be a list.')
        if not isinstance(rows, (list, tuple)):
            raise AutoLiveRunFaultError('planned protocol rows must be a list.')
        if not all(isinstance(column, str) and column for column in columns):
            raise AutoLiveRunFaultError(
                'planned protocol columns must contain nonempty strings.'
            )
        if len(set(columns)) != len(columns):
            raise AutoLiveRunFaultError('planned protocol columns must be unique.')

        try:
            with open(path, 'w', encoding='utf-8', newline='') as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=list(columns),
                    extrasaction='raise'
                )
                writer.writeheader()
                for row in rows:
                    if not isinstance(row, dict):
                        raise AutoLiveRunFaultError(
                            'planned protocol rows must be dictionaries.'
                        )
                    unknown_fields = set(row.keys()) - set(columns)
                    if unknown_fields:
                        raise AutoLiveRunFaultError(
                            'planned protocol row has unexpected columns: {}.'.format(
                                ', '.join(sorted(unknown_fields))
                            )
                        )
                    writer.writerow(row)
                output_file.flush()
                os.fsync(output_file.fileno())
        except OSError as exc:
            raise AutoLiveRunFaultError(
                'Could not write planned protocol evidence {}: {}.'.format(
                    path,
                    exc
                )
            )

    @staticmethod
    def _summary_markdown(record):
        active_batch = record['active_batch_number']
        if active_batch is None:
            active_batch = 'not applicable (preparation fault)'
        return (
            '# Auto live-run fault evidence\n\n'
            'This package is immutable evidence for human disposition. '
            'It does not authorize retry, resume, model training, or report '
            'finalization.\n\n'
            '- Fault ID: `{fault_id}`\n'
            '- Run ID: `{run_id}`\n'
            '- Recorded (UTC): `{recorded_at_utc}`\n'
            '- Scope: `{fault_scope}`\n'
            '- Physical outcome certainty: `{certainty}`\n'
            '- Preceding lifecycle state: `{preceding_lifecycle_state}`\n'
            '- Lifecycle state: `{lifecycle_state}`\n'
            '- Required disposition: `{disposition}`\n'
            '- Active batch: `{active_batch}`\n'
            '- Last durable event sequence: `{last_event_sequence}`\n'
            '- Exception: `{exception_class}: {exception_message}`\n\n'
            'See `fault_summary.json`, `batch_context.json`, and any '
            'available `planned_protocol_dataframe.csv` for structured audit '
            'details.\n'
        ).format(active_batch=active_batch, **record).encode('utf-8')

    @classmethod
    def write(
        cls,
        run_state_directory,
        fault_record,
        exception_trace,
        planned_protocol_columns=None,
        planned_protocol_rows=None
    ):
        '''Publishes a new evidence package and returns its absolute path.

        A protocol CSV is written only when both its columns and rows are
        supplied.  Preparation faults can therefore retain their complete
        evidence without falsely creating a batch-protocol artifact.
        '''
        validate_fault_record(fault_record)
        _require_nonempty_string(exception_trace, 'exception_trace')
        if (planned_protocol_columns is None) != (planned_protocol_rows is None):
            raise AutoLiveRunFaultError(
                'planned protocol columns and rows must be supplied together.'
            )

        run_state_directory = os.path.abspath(run_state_directory)
        evidence_root = os.path.join(
            run_state_directory,
            cls.FAULT_EVIDENCE_DIRECTORYNAME
        )
        final_directory = os.path.join(evidence_root, fault_record['fault_id'])

        try:
            os.makedirs(evidence_root, exist_ok=True)
        except OSError as exc:
            raise AutoLiveRunFaultError(
                'Could not create fault-evidence directory {}: {}.'.format(
                    evidence_root,
                    exc
                )
            )

        if os.path.exists(final_directory):
            raise AutoLiveRunFaultError(
                'Refusing to overwrite existing fault evidence {}.'.format(
                    final_directory
                )
            )

        try:
            staging_directory = tempfile.mkdtemp(
                prefix='.{0}.'.format(fault_record['fault_id']),
                suffix='.tmp',
                dir=evidence_root
            )
        except OSError as exc:
            raise AutoLiveRunFaultError(
                'Could not stage fault evidence in {}: {}.'.format(
                    evidence_root,
                    exc
                )
            )

        cls._write_bytes(
            os.path.join(staging_directory, cls.FAULT_SUMMARY_JSON_FILENAME),
            _canonical_json_bytes(fault_record, 'fault_record')
        )
        cls._write_bytes(
            os.path.join(staging_directory, cls.BATCH_CONTEXT_FILENAME),
            _canonical_json_bytes(
                fault_record['batch_context'],
                'fault_record.batch_context'
            )
        )
        cls._write_bytes(
            os.path.join(staging_directory, cls.FAULT_SUMMARY_MARKDOWN_FILENAME),
            cls._summary_markdown(fault_record)
        )
        cls._write_bytes(
            os.path.join(staging_directory, cls.EXCEPTION_TRACE_FILENAME),
            (exception_trace.rstrip() + '\n').encode('utf-8')
        )
        if planned_protocol_columns is not None:
            cls._write_csv(
                os.path.join(staging_directory, cls.PLANNED_PROTOCOL_FILENAME),
                planned_protocol_columns,
                planned_protocol_rows
            )

        try:
            # ``rename`` is atomic within the same filesystem and, unlike a
            # replacement operation, cannot silently replace prior evidence.
            os.rename(staging_directory, final_directory)
        except OSError as exc:
            raise AutoLiveRunFaultError(
                'Could not publish fault evidence {}: {}. Staged evidence was '
                'left at {} for manual preservation.'.format(
                    final_directory,
                    exc,
                    staging_directory
                )
            )

        return final_directory
