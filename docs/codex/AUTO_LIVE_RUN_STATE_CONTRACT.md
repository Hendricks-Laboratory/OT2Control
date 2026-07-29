# Auto Live-Run State Contract (Schema Version 1)

**Status:** Stages 0–1. The contract is validated by a pure Python module and
the controller now writes local Stage 1 records at normal Auto lifecycle
boundaries. It does not contact Google Drive, communicate with the Raspberry
Pi, alter recipe selection, or provide a recovery action.

## Scope and authority

The local controller state will be authoritative. A future cloud workbook is
only a delayed, human-facing mirror. A network failure must never by itself
stop a healthy run.

The contract applies only to new Auto recovery records. It does not alter the
existing Auto performance CSV, model checkpoints, raw scan output, QC policy,
or scientific recipe selection.

## Records

Stage 1 writes the following records below one run directory:

```text
Run_State/input_snapshot.json      immutable parsed input-template snapshot
Run_State/header_snapshot.json     immutable Header worksheet snapshot
Run_State/runtime_baseline.json    immutable interpreted controller baseline
Run_State/run_manifest.json
Run_State/current_state.json
Run_State/events.jsonl
```

`run_manifest.json` and the three snapshots are immutable.
`current_state.json` is the latest durable state. `events.jsonl` is
append-only: one complete JSON object per line.
Stage 2 now writes `pending_cloud_sync.jsonl` as a local FIFO queue of
checksum-identified Live-workbook snapshots. The queue stores immutable copies
under `Live_Run/pending_sync_snapshots/` so a later workbook refresh cannot
alter an earlier queued revision. It is local-only until a separately reviewed
Drive adapter is configured; its presence never requires network access during
a normal Auto run.

All records use `schema_version: 1`. Unknown top-level fields are invalid
until a later explicit schema-version change.

### Immutable run manifest

```json
{
  "schema_version": 1,
  "record_type": "auto_live_run_manifest",
  "run_id": "nonempty stable run identity",
  "created_at_utc": "2026-07-29T12:34:56+00:00",
  "input_snapshot_sha256": "sha256 of immutable input snapshot",
  "header_snapshot_sha256": "sha256 of Header snapshot",
  "runtime_baseline_sha256": "sha256 of run-relevant baseline settings",
  "git_branch": "Auto-RTG",
  "git_commit": "committed source identity"
}
```

The manifest identifies what was requested at run start. It never represents a
subsequent reagent replacement, plate replacement, or operator action.

### Mutable current state

```json
{
  "schema_version": 1,
  "record_type": "auto_live_run_current_state",
  "run_id": "same run identity as manifest",
  "revision": 0,
  "lifecycle_state": "created",
  "active_batch_number": null,
  "last_event_sequence": 0,
  "hold_action_id": null,
  "fault_id": null
}
```

Every successful state change increments `revision` by exactly one. A state
cannot change its `run_id` or decrease `last_event_sequence`.

A durable event that does not change lifecycle phase still increments the
revision and advances `last_event_sequence`; this is a same-state transition.
It records an audited milestone but never bypasses the lifecycle matrix.

### Append-only event

```json
{
  "schema_version": 1,
  "record_type": "auto_live_run_event",
  "run_id": "same run identity as manifest",
  "sequence": 1,
  "timestamp_utc": "2026-07-29T12:34:56+00:00",
  "event_type": "run_initialized",
  "state_revision": 0,
  "payload": {}
}
```

`sequence` starts at one and is monotonic. `payload` is always a JSON object.
Stage 1 records immutable input/header/runtime snapshots, Git branch/commit
identity, batch preflight/execution/measurement/processing boundaries, and
normal finalization. It neither accepts nor executes operator actions.

## Lifecycle states and allowed transitions

| State | Meaning | Permitted next state(s) |
|---|---|---|
| `created` | Contract records exist but no batch is prepared. | `ready_for_batch` |
| `ready_for_batch` | A completed batch is durable; a new batch may be preflighted. | `preflighting_batch`, `finalized` |
| `preflighting_batch` | The next unexecuted batch is being checked. | `executing_batch`, `held_for_operator`, `finalized` |
| `executing_batch` | Robot transfers may be occurring. | `measuring_batch`, `faulted_partial_batch` |
| `measuring_batch` | Completed batch is being scanned. | `processing_batch`, `faulted_partial_batch` |
| `processing_batch` | QC, model update, and outputs are being finalized. | `ready_for_batch`, `finalized`, `faulted_partial_batch` |
| `held_for_operator` | A pre-batch resource hold is awaiting one validated action. | `preflighting_batch`, `finalized` |
| `faulted_partial_batch` | Unexpected interruption after a batch began. | `finalized` only |
| `finalized` | No further action is permitted. | none |

This matrix deliberately prohibits automatic resumption from
`faulted_partial_batch`. A batch may be held only before its execution begins;
there is no automatic mid-batch recovery.

## Event vocabulary

The only valid event types in schema version 1 are:

```text
run_initialized
input_snapshot_created
batch_preflight_requested
batch_preflight_validated
batch_preflight_rejected
batch_execution_started
batch_transfer_completed
batch_measurement_completed
batch_model_update_completed
batch_completed
hold_entered
operator_action_requested
operator_action_rejected
operator_action_applied
fault_recorded
cloud_sync_queued
cloud_sync_completed
cloud_sync_failed
run_finalized
```

Cloud events remain reserved for a later configured remote adapter. The local
Stage 2 queue itself does not append those events because doing so would create
another state revision and recursively queue a new mirror update. Its durable
queue records remain auditable independently of remote connectivity.

## Future operator action envelope

Stages 6–8 will use a versioned action envelope containing a unique action ID,
expected state revision, action type, and exact replacement details. The
allowed action types are defined now to prevent unreviewed expansion:

```text
replace_source
register_backup_source
replace_tip_rack
replace_wellplate
retry_cloud_sync
```

Stage 0 accepts **no** operator action. A future action must be rejected when
the run is not in `held_for_operator`, the requested revision is stale, the
action ID has already been applied, or the data are incomplete or incompatible.

## Failure and durability rules for later stages

Stage 1 must write a temporary file in the destination directory, flush it,
then atomically replace `current_state.json`. It must append one complete event
line before treating the associated transition as durable. A failed write must
leave the previous valid state intact and must fail closed rather than execute
a new batch from uncertain state.

No future Stage may use this contract to modify acquisition settings, targets,
GP history, source identity/stock concentration without validation, or
plate-reader configuration in the middle of a run.
