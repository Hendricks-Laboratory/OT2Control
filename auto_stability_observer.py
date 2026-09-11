'''Controller-side audit state for Auto stability monitoring.

Stage 12B deliberately records only facts already observable from the normal
Auto execution path.  In particular, a transfer dispatch is *not* treated as
physical completion. The controller promotes a well to the active set only
after an explicitly recorded controller completion point. Stage 12C uses the
Pi's existing transfer ``ready`` acknowledgement for a per-well
controller-observed timing reference; the manifest records that limited but
auditable time basis rather than claiming a direct dispense timestamp.

This module has no robot, plate-reader, pandas, or optimizer dependencies so
its state transitions can be tested without laboratory hardware.
'''

import csv
import datetime
import math
import os
import tempfile
import time


# Version 3 adds the logical-well -> physical-reader-location mapping used by
# Stage 12D to reload each immutable raw scan without guessing from a later
# plate state. Every manifest is run-local, so no in-place migration is needed.
STABILITY_OBSERVER_SCHEMA_VERSION = 3

ACTIVATION_STATUS_PENDING = 'pending_trigger_completion'
ACTIVATION_STATUS_ACTIVE = 'active'
ACTIVATION_STATUS_WINDOW_COMPLETE = 'observation_window_complete'


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
    'active_wellnames',
    'active_well_locations',
    'observation_reason',
    'mixing_mode',
    'shake_duration_s',
    'shake_started_at_utc',
    'shake_completed_at_utc',
    'scan_started_at_utc',
    'scan_completed_at_utc',
    'scan_time_basis',
    'notes',
)


class AutoStabilityObserverError(RuntimeError):
    '''Raised when an observer transition would make the audit trail invalid.'''


def _utc_now_string():
    '''Return an explicit, timezone-aware UTC timestamp for durable records.'''
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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
    Stage 12C uses the same registry to schedule scans. Its raw scans are
    whole-active-set files rather than per-well files: one plate-reader file
    can contain many well trajectories, while each well remains separately
    timestamped in the manifest event that references that file.
    '''

    def __init__(
            self,
            pr_data_path,
            run_id,
            trigger_reagent,
            now=None,
            monotonic_clock=None):
        self.pr_data_path = os.path.abspath(str(pr_data_path))
        self.run_id = str(run_id).strip()
        self.trigger_reagent = str(trigger_reagent).strip()
        self._now = now or _utc_now_string
        self._monotonic_clock = monotonic_clock or time.monotonic
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
                'Stage-12 stability observer initialized. A valid active '
                'trigger completion may subsequently schedule a Stage-12C '
                'reader observation.'
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
            # Monotonic timing is intentionally in-memory only. Stage 10's
            # fail-closed interruption behavior does not resume a partially
            # observed trajectory, so it never needs to survive a restart.
            'activation_monotonic_s': None,
            'last_observation_monotonic_s': None,
            'last_scan_completed_monotonic_s': None,
            'observation_count': 0,
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
                'the controller records an explicit completion point.'
            )
        )

    def confirm_trigger_transfer_completed(
            self, wellname,
            completion_time_basis='controller_observed_save_ftp_barrier'):
        '''Promote a pending well after an explicit controller completion point.'''
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

        if completion_time_basis not in (
                'controller_observed_save_ftp_barrier',
                'controller_observed_transfer_ready'):
            raise AutoStabilityObserverError(
                'Unsupported trigger completion time basis: {}.'.format(
                    completion_time_basis
                )
            )

        completed_at_utc = self._now()
        record['activation_status'] = ACTIVATION_STATUS_ACTIVE
        record['trigger_transfer_completion_observed_at_utc'] = completed_at_utc
        record['trigger_completion_time_basis'] = completion_time_basis
        record['activation_monotonic_s'] = self._monotonic_clock()
        record['last_observation_monotonic_s'] = None
        record['last_scan_completed_monotonic_s'] = None
        record['observation_count'] = 0
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
            trigger_completion_time_basis=completion_time_basis,
            notes=(
                'The controller observed the configured completion point '
                'after the trigger command. This establishes a per-well '
                'timing reference before Stage 12C schedules any active-well '
                'observation.'
            )
        )

    def get_active_wells(self):
        '''Return the currently observable wells in trigger-completion order.'''
        return [
            dict(record)
            for record in self._active_wells.values()
            if record['activation_status'] == ACTIVATION_STATUS_ACTIVE
        ]

    def get_active_wells_within_observation_window(
            self, observation_window_s, now_monotonic_s=None):
        '''Return active wells whose window remains open at reader-scan start.

        A plate-reader acquisition has a measurable interval, not an invented
        per-well instantaneous timestamp.  Stage 12C therefore uses the
        controller's ``run_protocol`` start boundary as the eligibility
        decision and retains both reader interval endpoints in the manifest.
        This helper is intentionally non-mutating: the scheduler separately
        records expired windows after reader access has been released.
        '''
        try:
            observation_window_s = float(observation_window_s)
        except (TypeError, ValueError):
            raise AutoStabilityObserverError(
                'Observation window must be numeric.'
            )
        if not math.isfinite(observation_window_s) or observation_window_s <= 0:
            raise AutoStabilityObserverError(
                'Observation window must be finite and positive.'
            )
        if now_monotonic_s is None:
            now_monotonic_s = self._monotonic_clock()
        try:
            now_monotonic_s = float(now_monotonic_s)
        except (TypeError, ValueError):
            raise AutoStabilityObserverError(
                'Observation time must be numeric.'
            )
        if not math.isfinite(now_monotonic_s):
            raise AutoStabilityObserverError(
                'Observation time must be finite.'
            )

        eligible_records = []
        for record in self.get_active_wells():
            activation_time = record['activation_monotonic_s']
            if activation_time is None:
                raise AutoStabilityObserverError(
                    'Active well {} has no monotonic activation time.'.format(
                        record['wellname']
                    )
                )
            if now_monotonic_s < activation_time + observation_window_s:
                eligible_records.append(record)
        return eligible_records

    def record_raw_scan_skipped_expired(
            self,
            batch_number,
            wellnames,
            observation_reason,
            observation_window_s,
            now_monotonic_s):
        '''Record why reader staging produced no in-window raw measurement.'''
        wellnames = self._normalize_active_wellnames(wellnames)
        batch_number = int(batch_number)
        for wellname in wellnames:
            if self._active_wells[wellname]['batch_number'] != batch_number:
                raise AutoStabilityObserverError(
                    'Skipped raw-scan batch {} does not match active well {}.'
                    .format(batch_number, wellname)
                )
        observation_reason = str(observation_reason).strip()
        if not observation_reason:
            raise AutoStabilityObserverError(
                'observation_reason cannot be blank.'
            )
        remaining_records = self.get_active_wells_within_observation_window(
            observation_window_s,
            now_monotonic_s=now_monotonic_s
        )
        if remaining_records:
            raise AutoStabilityObserverError(
                'A raw stability scan cannot be marked skipped as expired '
                'while an active well remains in-window.'
            )
        return self._append_event(
            'raw_scan_skipped_expired',
            batch_number=batch_number,
            wellname='__active_set__',
            activation_status=ACTIVATION_STATUS_ACTIVE,
            active_wellnames=';'.join(wellnames),
            observation_reason=observation_reason,
            scan_time_basis='reader_scan_start_window_eligibility',
            notes=(
                'No raw stability scan was reserved or run because reader '
                'staging reached the scan-start boundary after every '
                'candidate well\'s configured observation window had closed.'
            )
        )

    def _normalize_active_wellnames(self, wellnames):
        if isinstance(wellnames, str):
            wellnames = [wellnames]
        try:
            normalized = [self._require_wellname(value) for value in wellnames]
        except TypeError:
            raise AutoStabilityObserverError(
                'Active wellnames must be a wellname or an iterable of wellnames.'
            )
        if not normalized:
            raise AutoStabilityObserverError(
                'At least one active well is required for a stability scan.'
            )
        if len(set(normalized)) != len(normalized):
            raise AutoStabilityObserverError(
                'A stability scan cannot list the same well more than once.'
            )
        for wellname in normalized:
            record = self._active_wells.get(wellname)
            if record is None or record['activation_status'] != ACTIVATION_STATUS_ACTIVE:
                raise AutoStabilityObserverError(
                    'Stability scans can include only active wells: {}.'.format(
                        wellname
                    )
                )
        return normalized

    @staticmethod
    def _normalize_active_well_locations(wellnames, reader_locations):
        '''Serialize one explicit logical-well -> reader-location mapping.

        The Stage-12C scan layout is positional.  Persisting the resolved
        physical locations at reservation time prevents Stage 12D from
        attempting to reconstruct a historical layout after a plate swap.
        Older direct callers may omit the optional mapping; that remains
        backward compatible but is deliberately blank/auditable rather than
        guessed later.
        '''
        if reader_locations is None:
            return ''
        if isinstance(reader_locations, str):
            reader_locations = [reader_locations]
        try:
            locations = [str(value).strip() for value in reader_locations]
        except TypeError:
            raise AutoStabilityObserverError(
                'Reader locations must be an iterable matching active wells.'
            )
        if len(locations) != len(wellnames) or any(not value for value in locations):
            raise AutoStabilityObserverError(
                'Reader locations must contain one nonblank coordinate for '
                'each active well.'
            )
        if len(set(locations)) != len(locations):
            raise AutoStabilityObserverError(
                'A stability scan cannot map two active wells to one reader '
                'location.'
            )
        return ';'.join(
            '{}={}'.format(wellname, location)
            for wellname, location in zip(wellnames, locations)
        )

    def reserve_raw_scan(self, batch_number, wellnames, reader_locations=None):
        '''Reserve one unique, unmerged raw path for an active-well scan.

        This method only writes a manifest entry. It does not create a scan
        file, invoke the reader, or alter ordinary reader callbacks.  The
        later Stage 12C reader call writes one file for the full active set.
        '''
        wellnames = self._normalize_active_wellnames(wellnames)
        active_well_locations = self._normalize_active_well_locations(
            wellnames, reader_locations
        )
        batch_number = int(batch_number)
        records = [self._active_wells[wellname] for wellname in wellnames]
        for record in records:
            if batch_number != record['batch_number']:
                raise AutoStabilityObserverError(
                    'Raw-scan batch {} does not match active well {} batch {}.'.format(
                        batch_number,
                        record['wellname'],
                        record['batch_number']
                    )
                )

        self._scan_sequence += 1
        scan_id = '{:04d}'.format(self._scan_sequence)
        basename = 'stability_batch_{:03d}_active_set_scan_{}'.format(
            batch_number,
            scan_id
        )
        relative_path = os.path.join(
            'stability',
            'raw_scans',
            basename + '.csv'
        )
        return self._append_event(
            'raw_scan_reserved',
            batch_number=batch_number,
            wellname='__active_set__',
            activation_status=ACTIVATION_STATUS_ACTIVE,
            raw_scan_id=scan_id,
            raw_scan_basename=basename,
            raw_scan_relative_path=relative_path,
            active_wellnames=';'.join(wellnames),
            active_well_locations=active_well_locations,
            notes=(
                'Reserved for one unmerged active-well raw scan. The reader '
                'has not yet been invoked.'
            )
        )

    def record_raw_scan_completed(
            self,
            reservation,
            scan_started_at_utc,
            scan_completed_at_utc,
            scan_started_monotonic_s,
            scan_completed_monotonic_s=None,
            observation_reason=None,
            mixing_mode='none',
            shake_duration_s=0.0,
            shake_started_at_utc=None,
            shake_completed_at_utc=None):
        '''Record one durable active-set observation after its raw file exists.'''
        if not isinstance(reservation, dict):
            raise AutoStabilityObserverError(
                'Raw-scan reservation must be the manifest event dictionary.'
            )
        if reservation.get('event_type') != 'raw_scan_reserved':
            raise AutoStabilityObserverError(
                'Only a raw_scan_reserved event can be completed.'
            )
        wellnames = self._normalize_active_wellnames(
            reservation.get('active_wellnames', '').split(';')
        )
        try:
            scan_started_monotonic_s = float(scan_started_monotonic_s)
        except (TypeError, ValueError):
            raise AutoStabilityObserverError(
                'scan_started_monotonic_s must be numeric.'
            )
        if not math.isfinite(scan_started_monotonic_s):
            raise AutoStabilityObserverError(
                'scan_started_monotonic_s must be finite.'
            )
        if scan_completed_monotonic_s is None:
            scan_completed_monotonic_s = scan_started_monotonic_s
        try:
            scan_completed_monotonic_s = float(scan_completed_monotonic_s)
        except (TypeError, ValueError):
            raise AutoStabilityObserverError(
                'scan_completed_monotonic_s must be numeric.'
            )
        if (
                not math.isfinite(scan_completed_monotonic_s)
                or scan_completed_monotonic_s < scan_started_monotonic_s):
            raise AutoStabilityObserverError(
                'scan_completed_monotonic_s must be finite and no earlier '
                'than scan_started_monotonic_s.'
            )
        try:
            shake_duration_s = float(shake_duration_s)
        except (TypeError, ValueError):
            raise AutoStabilityObserverError(
                'shake_duration_s must be numeric.'
            )
        if not math.isfinite(shake_duration_s) or shake_duration_s < 0:
            raise AutoStabilityObserverError(
                'shake_duration_s must be finite and nonnegative.'
            )
        mixing_mode = str(mixing_mode).strip() or 'none'
        observation_reason = str(observation_reason).strip()
        if not observation_reason:
            raise AutoStabilityObserverError(
                'observation_reason cannot be blank.'
            )
        if shake_duration_s > 0 and mixing_mode == 'none':
            raise AutoStabilityObserverError(
                'A positive shake duration requires a mixing mode.'
            )

        for wellname in wellnames:
            record = self._active_wells[wellname]
            record['last_observation_monotonic_s'] = scan_started_monotonic_s
            record['last_scan_completed_monotonic_s'] = (
                scan_completed_monotonic_s
            )
            record['observation_count'] += 1

        return self._append_event(
            'raw_scan_completed',
            batch_number=reservation['batch_number'],
            wellname='__active_set__',
            activation_status=ACTIVATION_STATUS_ACTIVE,
            raw_scan_id=reservation['raw_scan_id'],
            raw_scan_basename=reservation['raw_scan_basename'],
            raw_scan_relative_path=reservation['raw_scan_relative_path'],
            active_wellnames=';'.join(wellnames),
            active_well_locations=reservation.get('active_well_locations', ''),
            observation_reason=observation_reason,
            mixing_mode=mixing_mode,
            shake_duration_s=shake_duration_s,
            shake_started_at_utc=(
                '' if shake_started_at_utc is None else str(shake_started_at_utc)
            ),
            shake_completed_at_utc=(
                '' if shake_completed_at_utc is None else str(shake_completed_at_utc)
            ),
            scan_started_at_utc=str(scan_started_at_utc),
            scan_completed_at_utc=str(scan_completed_at_utc),
            scan_time_basis=(
                'controller_plate_reader_run_protocol_interval'
            ),
            notes=(
                'Unmerged active-well scan completed and its raw file was '
                'moved into the stability raw-scan directory. Mixing and '
                'reader timing fields describe this exact observation.'
            )
        )

    def get_next_cadence_deadline(
            self, observation_window_s, scan_interval_s):
        '''Return the next in-window cadence deadline, or ``None`` if absent.'''
        try:
            observation_window_s = float(observation_window_s)
            scan_interval_s = float(scan_interval_s)
        except (TypeError, ValueError):
            raise AutoStabilityObserverError(
                'Observation window and cadence interval must be numeric.'
            )
        if observation_window_s <= 0 or scan_interval_s <= 0:
            raise AutoStabilityObserverError(
                'Observation window and cadence interval must be positive.'
            )

        deadlines = []
        for record in self.get_active_wells():
            activation_time = record['activation_monotonic_s']
            last_observation = record['last_observation_monotonic_s']
            if activation_time is None:
                raise AutoStabilityObserverError(
                    'Active well {} has no monotonic activation time.'.format(
                        record['wellname']
                    )
                )
            window_end = activation_time + observation_window_s
            if last_observation is None:
                deadlines.append(activation_time)
                continue
            # Cadence begins after reader completion, not reader start. A
            # slow shake/read therefore cannot cause an immediate catch-up
            # scan that would repeatedly perturb the plate without spacing.
            last_completed = record['last_scan_completed_monotonic_s']
            if last_completed is None:
                last_completed = last_observation
            next_deadline = last_completed + scan_interval_s
            # Do not start a scan after the bounded window solely to meet a
            # cadence. A scan triggered by another currently active well may
            # still add an earlier valid observation for this well.
            if next_deadline < window_end:
                deadlines.append(next_deadline)
        return min(deadlines) if deadlines else None

    def get_next_window_expiration_deadline(self, observation_window_s):
        '''Return the earliest remaining active observation-window endpoint.'''
        try:
            observation_window_s = float(observation_window_s)
        except (TypeError, ValueError):
            raise AutoStabilityObserverError(
                'Observation window must be numeric.'
            )
        if observation_window_s <= 0:
            raise AutoStabilityObserverError(
                'Observation window must be positive.'
            )
        deadlines = []
        for record in self.get_active_wells():
            activation_time = record['activation_monotonic_s']
            if activation_time is None:
                raise AutoStabilityObserverError(
                    'Active well {} has no monotonic activation time.'.format(
                        record['wellname']
                    )
                )
            deadlines.append(activation_time + observation_window_s)
        return min(deadlines) if deadlines else None

    def complete_expired_observation_windows(
            self, observation_window_s, now_monotonic_s=None):
        '''Mark bounded windows complete without pretending an extra scan occurred.'''
        try:
            observation_window_s = float(observation_window_s)
        except (TypeError, ValueError):
            raise AutoStabilityObserverError(
                'Observation window must be numeric.'
            )
        if observation_window_s <= 0:
            raise AutoStabilityObserverError(
                'Observation window must be positive.'
            )
        if now_monotonic_s is None:
            now_monotonic_s = self._monotonic_clock()
        completed = []
        for record in list(self._active_wells.values()):
            if record['activation_status'] != ACTIVATION_STATUS_ACTIVE:
                continue
            if now_monotonic_s < (
                    record['activation_monotonic_s'] + observation_window_s):
                continue
            record['activation_status'] = ACTIVATION_STATUS_WINDOW_COMPLETE
            completed.append(record['wellname'])
            self._append_event(
                'observation_window_completed',
                batch_number=record['batch_number'],
                wellname=record['wellname'],
                transfer_volume_uL=record['transfer_volume_uL'],
                trigger_command_id=record['trigger_command_id'],
                activation_status=ACTIVATION_STATUS_WINDOW_COMPLETE,
                trigger_transfer_dispatched_at_utc=(
                    record['trigger_transfer_dispatched_at_utc']
                ),
                trigger_transfer_completion_observed_at_utc=(
                    record['trigger_transfer_completion_observed_at_utc']
                ),
                trigger_completion_time_basis=(
                    record['trigger_completion_time_basis']
                ),
                notes=(
                    'Configured stability observation window elapsed after {} '
                    'recorded stability scan(s). No additional scan was '
                    'invented at the window boundary.'.format(
                        record['observation_count']
                    )
                )
            )
        return completed
