'''Passive, controller-side audit state for Auto stability monitoring.

Stage 12B deliberately records only facts already observable from the normal
Auto execution path.  In particular, a transfer dispatch is *not* treated as
physical completion.  The controller promotes a well to the active set only
after its existing save/FTP barrier returns, which is the first point at which
the controller can honestly establish that the preceding robot work completed.

This module has no robot, plate-reader, pandas, or optimizer dependencies so
its state transitions can be tested without laboratory hardware.
'''

import csv
import datetime
import math
import os
import tempfile


STABILITY_OBSERVER_SCHEMA_VERSION = 1

ACTIVATION_STATUS_PENDING = 'pending_trigger_completion'
ACTIVATION_STATUS_ACTIVE = 'active'


MANIFEST_COLUMNS = (
    'schema_version',
    'event_sequence',
    'event_type',
    'recorded_at_utc',
    'run_id',
    'batch_number',
    'wellname',
    'trigger_reagent',
    'transfer_volume_uL',
    'trigger_command_id',
    'activation_status',
    'trigger_transfer_dispatched_at_utc',
    'trigger_transfer_completion_observed_at_utc',
    'trigger_completion_time_basis',
    'raw_scan_id',
    'raw_scan_basename',
    'raw_scan_relative_path',
    'notes',
)


class AutoStabilityObserverError(RuntimeError):
    '''Raised when an observer transition would make the audit trail invalid.'''


def _utc_now_string():
    '''Return an explicit, timezone-aware UTC timestamp for durable records.'''
    return datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0
    ).isoformat()


def _safe_filename_component(value):
    '''Make a stable human-readable filename component without path traversal.'''
    text = str(value).strip()
    if not text:
        raise AutoStabilityObserverError('Filename component cannot be blank.')
    return ''.join(
        character if character.isalnum() or character in ('-', '_') else '_'
        for character in text
    )


class AutoStabilityObserver:
    '''Append-only stability manifest and active-completed-well registry.

    ``record_trigger_transfer_dispatched`` records a robot command as pending.
    ``confirm_trigger_transfer_completed`` is intentionally separate so a
    controller cannot confuse a socket dispatch timestamp with physical work.
    Stage 12C will use the active registry to schedule scans; Stage 12B merely
    establishes it and reserves deterministic raw-scan names without creating
    or merging any scan files.
    '''

    def __init__(self, pr_data_path, run_id, trigger_reagent, now=None):
        self.pr_data_path = os.path.abspath(str(pr_data_path))
        self.run_id = str(run_id).strip()
        self.trigger_reagent = str(trigger_reagent).strip()
        self._now = now or _utc_now_string
        self._event_sequence = 0
        self._scan_sequence = 0
        self._active_wells = {}
        self._events = []

        if not self.run_id:
            raise AutoStabilityObserverError('run_id cannot be blank.')
        if not self.trigger_reagent:
            raise AutoStabilityObserverError(
                'trigger_reagent cannot be blank.'
            )

        self.stability_path = os.path.join(self.pr_data_path, 'stability')
        self.raw_scan_path = os.path.join(self.stability_path, 'raw_scans')
        self.manifest_path = os.path.join(
            self.pr_data_path,
            'auto_stability_scan_manifest.csv'
        )
        os.makedirs(self.raw_scan_path, exist_ok=True)
        self._append_event(
            'observer_initialized',
            notes=(
                'Stage 12B passive observer initialized; no stability scan, '
                'shake, or scheduler was started.'
            )
        )

    @staticmethod
    def _coerce_positive_volume(volume_uL):
        try:
            value = float(volume_uL)
        except (TypeError, ValueError):
            raise AutoStabilityObserverError(
                'Trigger transfer volume must be numeric.'
            )
        if not math.isfinite(value) or value <= 0:
            raise AutoStabilityObserverError(
                'Trigger transfer volume must be finite and greater than 0 uL.'
            )
        return value

    @staticmethod
    def _require_wellname(wellname):
        value = str(wellname).strip()
        if not value:
            raise AutoStabilityObserverError('wellname cannot be blank.')
        return value

    def _append_event(self, event_type, **values):
        self._event_sequence += 1
        event = {column: '' for column in MANIFEST_COLUMNS}
        event.update({
            'schema_version': STABILITY_OBSERVER_SCHEMA_VERSION,
            'event_sequence': self._event_sequence,
            'event_type': str(event_type),
            'recorded_at_utc': self._now(),
            'run_id': self.run_id,
            'trigger_reagent': self.trigger_reagent,
        })
        event.update(values)
        self._events.append(event)
        self._write_manifest()
        return dict(event)

    def _write_manifest(self):
        '''Atomically replace the CSV so an interruption cannot leave a half row.'''
        file_descriptor = None
        temporary_path = None
        try:
            file_descriptor, temporary_path = tempfile.mkstemp(
                prefix='.auto_stability_scan_manifest.',
                suffix='.tmp',
                dir=self.pr_data_path
            )
            with os.fdopen(file_descriptor, 'w', newline='', encoding='utf-8') as handle:
                file_descriptor = None
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                writer.writerows(self._events)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.manifest_path)
            temporary_path = None
        except OSError as exc:
            raise AutoStabilityObserverError(
                'Could not atomically write stability manifest {}: {}.'.format(
                    self.manifest_path,
                    exc
                )
            )
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            if temporary_path is not None and os.path.exists(temporary_path):
                os.remove(temporary_path)

    def record_trigger_transfer_dispatched(
            self,
            batch_number,
            wellname,
            transfer_volume_uL,
            trigger_command_id):
        '''Record a sent trigger-transfer packet without claiming completion.'''
        wellname = self._require_wellname(wellname)
        if wellname in self._active_wells:
            raise AutoStabilityObserverError(
                'A stability trigger was already recorded for well {}.'.format(
                    wellname
                )
            )

        transfer_volume_uL = self._coerce_positive_volume(transfer_volume_uL)
        dispatched_at_utc = self._now()
        self._active_wells[wellname] = {
            'batch_number': int(batch_number),
            'wellname': wellname,
            'trigger_reagent': self.trigger_reagent,
            'transfer_volume_uL': transfer_volume_uL,
            'trigger_command_id': str(trigger_command_id),
            'activation_status': ACTIVATION_STATUS_PENDING,
            'trigger_transfer_dispatched_at_utc': dispatched_at_utc,
            'trigger_transfer_completion_observed_at_utc': None,
            'trigger_completion_time_basis': 'unconfirmed_dispatch',
        }
        return self._append_event(
            'trigger_transfer_dispatched',
            batch_number=int(batch_number),
            wellname=wellname,
            transfer_volume_uL=transfer_volume_uL,
            trigger_command_id=str(trigger_command_id),
            activation_status=ACTIVATION_STATUS_PENDING,
            trigger_transfer_dispatched_at_utc=dispatched_at_utc,
            trigger_completion_time_basis='unconfirmed_dispatch',
            notes=(
                'Command was dispatched only. The well is not active until '
                'the controller observes its existing save/FTP barrier.'
            )
        )

    def confirm_trigger_transfer_completed(self, wellname):
        '''Promote a pending well after an existing controller completion barrier.'''
        wellname = self._require_wellname(wellname)
        record = self._active_wells.get(wellname)
        if record is None:
            raise AutoStabilityObserverError(
                'No stability trigger is pending for well {}.'.format(wellname)
            )
        if record['activation_status'] != ACTIVATION_STATUS_PENDING:
            raise AutoStabilityObserverError(
                'Well {} is not awaiting trigger completion.'.format(wellname)
            )

        completed_at_utc = self._now()
        record['activation_status'] = ACTIVATION_STATUS_ACTIVE
        record['trigger_transfer_completion_observed_at_utc'] = completed_at_utc
        record['trigger_completion_time_basis'] = (
            'controller_observed_save_ftp_barrier'
        )
        return self._append_event(
            'trigger_transfer_completed',
            batch_number=record['batch_number'],
            wellname=wellname,
            transfer_volume_uL=record['transfer_volume_uL'],
            trigger_command_id=record['trigger_command_id'],
            activation_status=ACTIVATION_STATUS_ACTIVE,
            trigger_transfer_dispatched_at_utc=(
                record['trigger_transfer_dispatched_at_utc']
            ),
            trigger_transfer_completion_observed_at_utc=completed_at_utc,
            trigger_completion_time_basis=(
                'controller_observed_save_ftp_barrier'
            ),
            notes=(
                'The pre-existing controller save/FTP barrier returned after '
                'the trigger command. This establishes completion before '
                'Stage 12C schedules any active-well observation.'
            )
        )

    def get_active_wells(self):
        '''Return a stable, serializable active-well snapshot for a later scheduler.'''
        return [
            dict(record)
            for _, record in sorted(self._active_wells.items())
            if record['activation_status'] == ACTIVATION_STATUS_ACTIVE
        ]

    def reserve_raw_scan(self, batch_number, wellname):
        '''Reserve a unique, unmerged future raw-scan path for an active well.

        This method only writes a manifest entry. It does not create a scan
        file, invoke the reader, or alter ordinary reader callbacks.
        '''
        wellname = self._require_wellname(wellname)
        record = self._active_wells.get(wellname)
        if record is None or record['activation_status'] != ACTIVATION_STATUS_ACTIVE:
            raise AutoStabilityObserverError(
                'Raw stability scans can be reserved only for an active well: {}.'
                .format(wellname)
            )
        if int(batch_number) != record['batch_number']:
            raise AutoStabilityObserverError(
                'Raw-scan batch {} does not match active well {} batch {}.'.format(
                    batch_number,
                    wellname,
                    record['batch_number']
                )
            )

        self._scan_sequence += 1
        scan_id = '{:04d}'.format(self._scan_sequence)
        basename = 'stability_batch_{:03d}_{}_scan_{}'.format(
            int(batch_number),
            _safe_filename_component(wellname),
            scan_id
        )
        relative_path = os.path.join(
            'stability',
            'raw_scans',
            basename + '.csv'
        )
        return self._append_event(
            'raw_scan_reserved',
            batch_number=int(batch_number),
            wellname=wellname,
            transfer_volume_uL=record['transfer_volume_uL'],
            trigger_command_id=record['trigger_command_id'],
            activation_status=ACTIVATION_STATUS_ACTIVE,
            trigger_transfer_dispatched_at_utc=(
                record['trigger_transfer_dispatched_at_utc']
            ),
            trigger_transfer_completion_observed_at_utc=(
                record['trigger_transfer_completion_observed_at_utc']
            ),
            trigger_completion_time_basis=(
                record['trigger_completion_time_basis']
            ),
            raw_scan_id=scan_id,
            raw_scan_basename=basename,
            raw_scan_relative_path=relative_path,
            notes=(
                'Reserved for a later Stage 12C unmerged raw scan. No reader '
                'operation occurred during this Stage 12B reservation.'
            )
        )
