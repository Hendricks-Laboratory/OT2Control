# Auto Live-Run Recovery and Performance Plan

**Status:** Stages 0–4 implemented and hardware-free validated. Stages 1–4
have passed their respective controlled dry-debug checkpoints. Stage 5 is the
next implementation stage.
**Working branch:** `Auto-RTG` on the lab computer.  
**Pi deployment policy:** use a separate `Auto-main` checkout/branch on the Raspberry Pi. Do not modify the protected Pi `main` checkout.

## Objective

Add an offline-resilient, auditable way for Auto runs to record live state, mirror that state to a human-facing workbook when network connectivity is available, and recover from selected resource shortages only at safe batch boundaries. The work also improves high-dimensional conditional-slice generation so it remains practical on the lab PC.

The controller's local state is authoritative. A Google Drive workbook is a best-effort human-facing mirror and action-request surface; a network failure must not stop an otherwise healthy run.

## Non-negotiable boundaries

- No hardware execution is performed during development or automated tests.
- A planned hold occurs only before a new batch begins. An unexpected interruption during a batch requires manual disposition; it is never automatically resumed.
- Early recovery actions replace like with like: same reagent identity and stock concentration, compatible container, and a validated deck location. They do not change experimental variables, acquisition settings, targets, GP history, or plate-reader configuration mid-run.
- The original input is preserved as an immutable snapshot. Runtime state is recorded separately and never silently rewrites experimental history.
- All changes remain on `Auto-RTG` until separately reviewed. Pi work is developed in `Auto-main`, never in the protected `main` checkout.

## State and synchronization design

Each run will have these local records:

```text
Run_State/run_manifest.json        immutable run identity and baseline hashes
Run_State/current_state.json       current durable controller state
Run_State/events.jsonl             append-only event/audit journal
Run_State/pending_cloud_sync.jsonl retry queue for mirror updates
Live_Run/<run>_LIVE.xlsx           local rendered copy of the live workbook
```

The remote workbook contains a read-only status mirror plus a deliberately small `Operator_Action_Request` surface. Each request includes `run_id`, an action ID, expected state revision, and replacement details. The controller rejects duplicate, stale, incomplete, or incompatible requests.

For every state-changing operation: persist a local request event; validate it against controller and Pi state; apply it and receive an acknowledgement/snapshot; persist acknowledged state and increment its revision; queue (rather than require) remote synchronization; then rebuild and preflight the unchanged next batch.

## Implementation stages

### Stage 0 — protocol and data-contract specification

Document exact JSON schemas, event types, state revisions, allowed actions, failure responses, and controller/Pi command payloads before either side is edited.

**Validation:** schema fixtures, state-transition tests, and review only. No dry run is useful yet.

### Stage 1 — controller-local durable run journal

On `Auto-RTG`, add immutable input snapshots; atomic `current_state.json` writes; append-only events; local checksums; and a run manifest. Integrate normal batch lifecycle updates without changing recipe selection, transfers, or model training.

**Validation:** Python 3.9 compilation; fake-controller lifecycle tests; atomic-write/failure-path tests; verify a clean legacy run has unchanged scientific outputs.

**Dry-debug checkpoint 1:** a very small normal Auto dry run. Confirm that local records match terminal events, condition numbers, selected recipes, and completed batches.

**Implementation status:** implemented on `Auto-RTG`. The controller writes
immutable parsed input/Header/runtime snapshots, a SHA-256 manifest, atomic
current-state replacements, and append-only JSONL lifecycle events before
connection, around each existing batch boundary, and at normal finalization.
It fails closed if a journal write fails. It does not yet create a cloud
workbook, queue synchronization, contact the Pi, or offer recovery actions.

**Output-directory isolation:** before an Auto output folder is created, a
previously used `data_dir` is detected. The controller proposes the first
available sibling name (`<data_dir>_1`, `<data_dir>_2`, and so on) and requires
an interactive operator to type `yes`; a noninteractive collision fails closed.
The requested and effective names are retained in the local run-state baseline
and final report. This prevents unrelated artifacts from being mixed into an
older run directory while preserving the existing workbook setting.

**Early terminal transcript:** Auto begins a short-lived, in-memory transcript
after the Header is downloaded and before its values are parsed. Once the
output directory is approved and the normal terminal log can be opened, that
transcript is prepended to `Debug/terminal_output.txt`. It therefore preserves
Header normalization, safety warnings, and output-directory collision
decisions without capturing credential/bootstrap output that occurs before the
Header exists. If setup stops before the log can be opened, the transcript is
discarded and normal streams are restored.

### Stage 2 — optional Live workbook mirror and offline queue

Generate a local Live workbook from the journal and asynchronously mirror it to the configured Drive location when available. Add a pending-sync queue and retry logic. The run continues if ordinary cloud updates fail.

**Implementation status (local portion):** Auto now atomically rebuilds
`Live_Run/<effective-data-dir>_LIVE.xlsx` after each durable Stage 1 journal
event. The derived workbook contains a status summary, current state,
append-only event view, immutable baseline view, and a clearly inactive
future-action sheet. Every state revision is also copied immutably into
`Live_Run/pending_sync_snapshots/` before its queue record is appended to
`Run_State/pending_cloud_sync.jsonl`; therefore a later status refresh cannot
silently change the content associated with an earlier queued revision.

The queue has an injected-adapter replay boundary with strict FIFO ordering,
checksum verification, duplicate-revision prevention, and stop-on-first-
failure behavior. No Drive adapter, credential access, or remote request is
performed by the controller yet. A concrete Drive endpoint and authentication
mechanism require a separate explicit integration decision; until then, the
queue is durable local audit data and normal Auto execution remains entirely
offline-capable.

**Validation:** fake-adapter tests cover initial queueing, transient failure,
replay ordering, duplicate prevention, immutable snapshot preservation, and
offline local updates. The standard-library XLSX package is structurally
validated without adding a new production dependency.

**Dry-debug checkpoint 2a:** run a small workflow and confirm the local Live
workbook, snapshot queue, and journal revisions agree. The later connectivity
loss/catch-up check occurs only after a reviewed Drive adapter is configured.
Do not test recovery actions in either checkpoint.

### Mandatory stop — Pi `Auto-main` onboarding and divergence audit

Stop implementation after Stage 2 and before any Pi-side edit. The project
owner and agent first perform a read-only audit of the Pi's existing deployed
checkout. Record its branch, exact `HEAD`, remotes, recent history, worktree
state, configured runtime path, and a source-only diff against the lab PC's
`main` and `Auto-RTG` commits. Never inspect credentials, change the deployed
checkout, fetch, pull, or reconfigure the currently running Pi repository as
part of this audit.

Only if the Pi checkout is clean and its state is understood, clone that exact
local Git repository into a sibling directory and create an `Auto-main`
branch there. A normal Git clone preserves committed Pi history but not
uncommitted edits; therefore an unclean Pi worktree is a stop condition, not
a reason to copy files. The original Pi checkout remains untouched and remains
the only runtime checkout until `Auto-main` is independently validated and
explicitly selected for a controlled debug.

**Validation:** confirm the new `Auto-main` checkout initially has the same
committed `HEAD` as the original Pi checkout, has a clean worktree, and retains
the original remote configuration for reference. Produce a written
Pi-versus-lab divergence audit before modifying `ot2_robot.py`.

**No physical debug occurs at this stop.** It is a preservation and audit gate,
not a deployment.

### Stage 3 — Pi `Auto-main` foundation and tare handshake

Create the separate Pi `Auto-main` checkout. Correct Pi tube tare constants there, add a versioned tare-calibration identifier, and implement read-only `get_robot_state_snapshot` plus a controller/Pi compatibility handshake. The controller rejects a nonzero legacy tare payload offset when the corrected Pi reports its new calibration version.

**Validation:** Pi-side unit/stub tests using known tube-plus-water weights; controller fake-portal protocol tests; Python 3.9 compatibility checks.

**Dry-debug checkpoint 3:** a supervised, non-transfer connection check against the Pi confirms the state snapshot and tare version are reported and recorded correctly.

**Implementation status:** implemented on `Auto-RTG` and the separately
deployed Pi `Auto-main` branch. The controller requires a versioned,
read-only robot-state snapshot before Auto execution proceeds, and records the
accepted compatibility metadata in its durable local state. The Pi `main`
checkout remains outside this feature's deployment path. The Stage 3
handshake was confirmed by a controlled dry debug; it does not yet alter
transfers, source allocation, or recovery behavior.

### Stage 4 — exact next-batch resource preflight

Keep the controller's existing aggregate source-volume check as an early audit. Add Pi-side `preflight_transfer_plan` as the physical authority for the next batch: it simulates individual sources, dead volumes, predeclared same-reagent backups, and required pipette tips. It returns either an exact feasible plan or a structured deficit before transfers begin.

**Validation:** equivalence with the established whole-aspiration backup and
pipette-selection rules; backup-switch cases; insufficient-source,
insufficient-tip, and reserve-boundary cases; no state mutation during failed
preflight.

**Dry-debug checkpoint 4:** use several same-name source containers and a deliberately insufficient primary container. Verify the plan chooses the valid backup before any liquid handling. Separately induce an insufficient-tip case.

**Implementation status:** implemented on `Auto-RTG` and `Auto-main`; the
controlled Stage 4 dry-debug checkpoint passed. Immediately before an Auto
batch whose `auto_source_volume_check` is `required`, the controller keeps its
existing aggregate inventory audit and also sends the exact planned source and
destination transfer sequence to the Pi. The Pi simulates its current
per-container usable volumes, declared same-name backup order, transfer-size
dependent pipette selection, and fresh-tip availability without mutating robot
state. It returns a versioned feasible allocation or structured deficit; the
controller fails closed before the batch is marked executable. Controlled
dry-debug evidence covered a valid same-name backup allocation, aggregate and
per-container source-shortage rejection, and a one-tip `H12` rack that rejected
the next batch before liquid handling. This stage does not pause, replace
sources, switch wellplates, or retry a rejected batch; those operator-recovery
actions remain later stages.

### Stage 5 — conditional-slice performance and progress reporting

Profile the existing high-dimensional slice path using saved data. The current implementation processes each displayed reagent pair serially, including GP prediction, feasibility evaluation, and plot production. This was adequate for 2D/3D but creates an unacceptable pause for 5D and above (for example, ten displayed pairs). Replace scalar per-grid-point feasibility loops with a vectorized implementation whose results are proven identical to the established physical rules. Keep GP prediction batched. Add clear progress messages for panel preparation and rendering.

After correctness is established, parallelize independent numerical panel preparation with a bounded worker pool, then keep Matplotlib rendering and output writes deterministic and controlled. Leave one or two CPU cores free for the operating system, controller, and robot communication; do not parallelize model mutation, hardware commands, or shared state writes. The resulting figures must be scientifically identical to serial output, apart from timing.

Add an `auto_conditional_slice_profile` policy only if needed after profiling:

- `detailed`: current per-batch atlases and individual slices;
- `standard`: per-batch atlases, with individual slices at finalization;
- `final_only`: final outputs only.

The legacy default must preserve current output behavior.

**Validation:** physical-feasibility equivalence across all-ON and true-zero masks; 2D through 5D saved-data timing comparisons; deterministic image-path tests; bounded-worker safety tests.

**Dry-debug checkpoint 5:** minimal 5D dry run with standard plotting. Record wall time for panel construction, prediction, and PNG saving against the pre-change baseline. Review figures and progress messages.

### Stage 6 — controlled operator holds and offline terminal recovery

When the next-batch preflight reports a recoverable resource shortage, enter a durable pre-batch hold. Record the deficit and offer two equivalent action paths: a validated terminal wizard that works offline, and a synced workbook action request when available. No batch is regenerated or executed while held.

**Validation:** stale revision, duplicate action, malformed action, cloud unavailable, terminal-only, acknowledgement timeout, and audit-trail tests.

**Dry-debug checkpoint 6:** intentionally create a recoverable source shortage. Use the terminal path to resolve it, verify the batch identity is unchanged, and confirm it executes once after fresh preflight.

### Stage 7 — source replacement and backup registration

Implement `replace_source` and `register_backup_source` through the versioned Pi protocol. Validate reagent identity, stock concentration, container type, deck position, tare version, and aspiratable volume. Refresh the Pi snapshot, persist the event, then rerun exact preflight.

**Validation:** idempotent retry; duplicate backup rejection; wrong reagent, wrong stock, and insufficient replacement rejection; correct switching order.

**Dry-debug checkpoint 7:** intentionally exhaust a source at a batch boundary, register an equivalent backup, and confirm the Pi uses it only for subsequent transfers.

### Stage 8 — tip-rack and wellplate replacement

Add compatible-tip-rack replacement and new-plate registration. A plate replacement creates a new `plate_generation`; condition identity, CSV rows, scan paths, and reports remain globally unique even if the new plate begins at `A1`. The operator explicitly selects the new starting well.

**Validation:** plate-generation naming; report/export separation; invalid start-well rejection; tip compatibility; no overwrite of prior scan paths.

**Dry-debug checkpoint 8:** perform separate tests for tip replacement and plate replacement. Do not first test them together.

### Stage 9 — one-time preparation/dilution workflow

Implement a separate, explicitly invoked pre-Auto preparation plan for stock-to-working-solution preparation. It must be idempotent, fully logged, and complete before the Auto batch loop begins. It must never rerun each Auto iteration.

**Validation:** dry material/stock calculation fixtures, one-time execution guards, failure-before-Auto behavior, and source-location reconciliation.

**Dry-debug checkpoint 9:** water-only preparation run followed by a small normal Auto run using the prepared working-source map.

### Stage 10 — unexpected interrupted-batch disposition

Define and implement the conservative fault path for a transfer interruption, robot disconnect, or unacknowledged partial batch. It must freeze the run, export evidence, and require human disposition rather than attempting automatic continuation.

**Validation:** injected fake-portal faults and state recovery tests.

**Dry-debug checkpoint 10:** fault-injection/simulation only at first. Any physical interruption test requires separate human-supervised approval.

## Release gates

No stage is promoted solely on compilation. Each requires relevant Python 3.9 checks, source-level/AST review, isolated fake-portal tests, backward-compatibility review, and a clean diff. A controlled dry debug follows only after its software tests pass. Real chemistry is considered only after all required preceding dry-debug checkpoints have been reviewed.

## Explicit deferrals

- Changing acquisition scores to account for stock availability.
- Automatic continuation after a mid-batch fault.
- Unattended recovery of any hardware fault.
- Changing experimental variables, target, GP model, or acquisition policy during a held run.
- Combining first-time source, tip, and plate recovery tests into one run.
