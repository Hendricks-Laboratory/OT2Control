# Auto-main Validation Notes

**Repository:** Hendricks-Laboratory / OT2Control  
**Branch context:** Raspberry Pi Auto runtime  
**Protected Pi baseline:** `main`  
**Active Pi development branch:** `Auto-main`  
**Updated through:** 2026-10-16

---

## Purpose

This is the Pi-side companion to `Auto-RTG` validation records. It documents
only the version-controlled Raspberry Pi service, its controller-facing packet
contracts, and its controlled deployment boundary. It is not a record of the
laboratory controller, Gaussian-process model, plate-reader analysis, or
spreadsheet workflow.

## Branch and deployment boundary

| Item | Current record |
|---|---|
| Protected Pi rollback checkout | `main`; never modify or deploy Auto changes into it |
| Editable Auto Pi branch | `Auto-main` |
| Current reviewed `Auto-main` revision | `65aad07` (`Validate temperature-controlled Auto preparation water on the Pi`) |
| Paired Lab-PC checkpoint | `Auto-RTG-v1.2.0` at `4fd80dd` (`Document the v1.2.0 pre-stability checkpoint`) |
| Planned paired Pi checkpoint | `Auto-main-v1.2.0`, created from `65aad07` after this record is committed |

The physical Pi keeps `/root/OT2Control` on protected `main`. Reviewed
Auto-specific runtime code is deployed only in the separate
`/root/OT2Control-Auto-main` worktree through a verified, fast-forward Git
bundle. A Git branch or validation record is not itself a Pi deployment and is
not physical-run clearance.

### Laboratory-PC checkout policy — 2026-10-16

Use `/Users/ryangrossman/Documents/GitHub/OT2Control` as the **only canonical
GitHub Desktop repository** for both Lab-PC `Auto-RTG` work and Pi-side
`Auto-main` work. Select the intended branch within that one repository before
reviewing, staging, or committing:

- `Auto-RTG` for controller, optimizer, spreadsheet, and reporting work;
- `Auto-main` only for reviewed Pi packet and `ot2_robot.py` support work.

`/Users/ryangrossman/Documents/GitHub/OT2Control-Auto-main` is a retained,
independent safety-copy clone from the checkout-recovery incident. It is **not
a working repository**: do not open it in GitHub Desktop, edit files there,
stage from it, or deploy from it. It remains untouched only until the current
canonical `Auto-main` work has been independently confirmed and archived by a
deliberate user decision.

Before any implementation or commit checkpoint, confirm the repository root
is the canonical `.../GitHub/OT2Control` path and inspect branch plus status.
Do not create another full clone or use a second local checkout for a branch;
that is what previously caused changes to be invisible in the GitHub Desktop
repository the operator was viewing.

## Implemented Auto-main support at this checkpoint

The current Pi protocol provides the reviewed support needed by the paired
Auto-RTG baseline, including:

- calibrated Pi-side tube-tare defaults and container inventory handling;
- read-only robot-state compatibility snapshots (`auto-main-state-v10`);
- source-volume preflight and source-mass refresh responses;
- pipette-tip-rack reset and plate-generation registration responses;
- grouped working-solution preparation reservation and one-time execution;
- temperature-controlled versus ambient preparation-water validation; and
- versioned Armchair request/response packet registrations for these features.

Each controller-facing feature remains subject to controller final validation.
The Pi does not independently choose chemistry, select recipes, train models,
or make a recovery decision.

## Stage 12 optical-stability compatibility boundary

At the `Auto-main-v1.2.0` checkpoint, **no optical-stability feature is
implemented on the Pi**. In particular, the Pi does not add stability packets,
targeted well mixing, stability scan scheduling, kinetic data storage, or
stability-directed selection.

The initial Stage 12 implementation uses the existing controller-managed
plate-reader whole-plate shake and selected-well scan pathways. Stages 12A
through 12F are therefore expected to be controller-side unless a review finds
an existing packet contract insufficient. Any future `pipette_mix` option for
individual plate wells requires a new, separately designed Auto-main/Pi packet
and liquid-handling implementation, static validation, packet compatibility
review, and a controlled dry debug. It must not be folded into the initial
plate-shake stability release.

## Stage 13B targeted completed-well mixing capability — 2026-09-22

Stage 13B adds the narrowly scoped Pi capability required before the Lab-PC
can implement the future per-trigger stability scheduler. It is implemented in
`Auto-main` only and has **not** been deployed to the Pi or cleared by a
physical dry debug.

- Appended Armchair packet identifiers preserve every existing packet value:
  `mix_auto_completed_well` (`0x1F`) and
  `auto_completed_well_mixed` (`0x20`). Both are ghost request/response
  packets, so they do not alter ordinary transfer-ready buffering.
- The read-only compatibility snapshot reports `auto-main-state-v10`, the
  first version that guarantees the schema-v2 targeted-mix tip preflight used
  by the bounded Stage 13D controller path, and advertises the optional
  targeted-mix command. The Lab-PC rejects older snapshot versions before a
  batch begins instead of discovering missing preflight support at execution.
- The request names exactly one logical `autowell...` product and repeats its
  expected plate-reader deck position, well location, plate-mapping revision,
  and plate generation. The Pi rejects any stale/mismatched identity, generic
  reagent/container, non-reader plate, non-`Well96` target, target not bound
  to its registered custom plate-reader labware, or target whose final recorded
  material transfer is not the named trigger reagent.
- The Pi validates a finite mix volume, a bounded cycle count, exactly one
  configured and loaded P300 (300 uL) instrument, its actual minimum and
  maximum volumes, configured-versus-loaded capacity agreement, and a
  conservative maximum of 50% of the robot-tracked completed-well volume.
  Ordinary reagent-transfer selection remains unchanged: a targeted mix is
  not allowed to fall back to the P20 or P1000 when deck hardware differs.
  The 50% limit is a Pi execution safety bound, not a chemistry optimization
  setting; future controller work must choose a requested volume inside it.
- Mixing uses a dedicated clean tip for the single verified target well and
  discards it immediately afterwards **only after a completed mix**. A retained
  clean tip may serve as that dedicated tip; a contaminated or absent tip is
  replaced before mixing. The Pi reserves enough unused tips to restore a clean
  tip after a completed operation. If the mix faults, it issues no automatic
  post-fault disposal or replacement motion; the existing error boundary stops
  in that indeterminate physical state for human review.
- The only new liquid-handling primitive is `Well96.mix_targeted`. It repeats
  the established repository `pipette.mix(1, ...)` primitive at a conservative
  96-well 2 mm aspiration and dispense clearance for its first supervised
  commissioning run, restores prior pipette
  clearances even on failure, and deliberately omits legacy blow-out and
  touch-tip cleanup so this route adds no extra contact motion. Its rate is
  the documented `1.0` baseline, not the historical tube-only `100.0`
  multiplier; its requested mix volume is bounded by verified well volume
  rather than defaulting to the P300's full 300 uL capacity.
  It does not call or alter the legacy generic `mix` handler, source inventory,
  reaction-well volume accounting, reader, shaking, scheduler, or model.
- A validation failure sends an explicit rejected acknowledgement without
  motion. If physical execution begins and then raises, the Pi deliberately
  emits no success-like acknowledgement or automatic post-fault tip motion;
  the existing error boundary records the fault and the future controller must
  fail closed rather than replaying the mix. The command audit records a mix
  as `started` before the liquid operation and as `completed` only after the
  mix and clean-tip restoration both succeed.

Hardware-free validation on the laboratory PC used the system Python
`3.9.6`: compilation of the changed Pi files and focused packet/snapshot,
source-preflight, and targeted-mix stub tests passed (34 tests). The stubs
verify append-only packet identifiers, ghost registration, request schema,
stale/mismatched target rejection before motion, final-trigger validation,
tip-shortage rejection without state mutation, registered-plate identity,
live-instrument range and capacity-agreement rejection, one-target mix
execution, dedicated-tip discard/replacement after success only, no post-fault
tip motion after a synthetic target-well mix fault, unchanged target volume,
exact 2 mm targeted-mix geometry, one-cycle repetition, and clearance
restoration after both success and a synthetic pipette fault. This is static
and synthetic evidence only. Before Stage 13C can
use the capability, the
exact Auto-main commit needs its established bundle deployment and a human
supervised, chemistry-free Pi mixing dry debug that confirms the target well,
tip sequence, acknowledgement, and no change to ordinary Auto behavior.

## Stage 13G C2d calibration binding/resource preflight — 2026-10-16

Stage 13G C2d extends the supervised calibration contract from v1 to
`auto-mix-calibration-v2`. It is Pi-side accounting support only and has not
been deployed or physically debugged.

- Four append-only ghost packets add a no-motion well binding request/result
  (`0x25`/`0x26`) and a no-motion resource-preflight request/result
  (`0x27`/`0x28`). Existing packet identifiers are unchanged.
- Binding claims exactly one requested, addressable vacant `Well96` at the
  registered plate-reader deck/location. It rejects stale plate revisions,
  logical-name collisions, or any already occupied physical well. An accepted
  binding creates only an in-memory container record; it does not home, move,
  scan, transfer, consume a tip, change the plate cursor, or change inventory.
- Resource preflight accepts only the paired Lab-PC's canonical reviewed assay
  profiles: `water_safety_20ul_first` (`20` then `180` uL) or
  `qc_two_dye_5ul_red` (`5` then `195` uL). For each named source,
  it invokes the existing Pi `_simulate_preflight_source` path, preserving the
  live container volume, calibrated dead volume, same-name backup selection,
  and installed-pipette substep routing without changing source state.
- It independently checks one P20 transfer tip, one P300 transfer tip, and
  eleven dedicated P300 calibration-cycle/restoration tips. The P300 count is
  checked as the combined twelve-tip requirement. A rejection returns empty
  evidence lists so the controller cannot mistake a partial plan for approval.
- The snapshot now advertises v2 and both optional C2d commands. Ordinary Auto
  transfers, standard stability mixing, preparation, plate replacement, and
  existing calibration v1 mix execution remain unchanged.

Hardware-free Python 3.9 tests cover binding success/collision rejection,
both transfer profiles, native dead-volume/source exhaustion behavior,
combined P300 shortage rejection, no mutation during resource checks, packet
registration, and explicit no-motion handler structure. This is static and
synthetic evidence only; C2d itself adds no runnable controller path.

## Stage 13G C2eD v3 calibration-profile history guard — 2026-10-16

The initial v2 Pi calibration guard retained stale fixed `180` then `20` uL
history assumptions. That was incompatible with the reviewed
`water_safety_20ul_first` profile, which transfers `20` then `180` uL, and
would also have rejected the reviewed `qc_two_dye_5ul_red` `5` then `195` uL
profile. The paired v3 contract now transports the immutable assay profile and
ordered first/second transfer volumes in every preflight and action request.
The Pi validates both against its local reviewed profile map before any
calibration action, then validates the actual destination-well history against
those same values.

During the first v3 water commissioning attempt, the Pi correctly completed
the two reviewed water transfers and the controller completed baseline scan
00, but the first P300 mix action was rejected before motion. The preflight
handler had not retained the new profile/volume fields in its in-memory
authorization record, so the action identity could not match it. Commit
`740b78f` adds those fields to the persisted identity and a handler-level
regression test that executes the real preflight handler and verifies its
stored record contains every later action-identity field.

The correction changes no calibration geometry, pipette selection, 150 uL
P300 mix volume, 2 mm clearance, transfer order, reader mapping, tip policy,
ordinary Auto transfer behavior, targeted stability mixing, or preparation
behavior. Python 3.9.6 compilation, the 59-test Auto-main suite, and the
50-test paired Lab-PC/Pi calibration contract suite passed. This is static and
synthetic evidence only. The `740b78f` Auto-main commit must be deployed with
the paired v3 Lab-PC controller before a fresh-well, supervised water retry;
the previously used well must not be reused because it contains the two
already-executed water transfers.

## Validation and deployment limits

The historical evidence for this branch consists of static/source-level
reviews, isolated contract tests, and human-supervised controlled-debug work
recorded with the paired controller changes. It does not establish real
chemistry clearance, unattended-run clearance, or correctness of future
stability functionality.

Before any later Auto-main code change is bundled to the Pi:

1. verify controller and Pi packet versions and request/response schemas;
2. run Python 3.9-compatible static and isolated validation on both sides;
3. review the exact Pi deployment commit and confirm the Pi Auto-main worktree
   is clean;
4. transfer only the reviewed commit as a verified fast-forward bundle; and
5. record the deployed commit and controlled-debug result in the associated
   run record.

`main` remains untouched throughout this process.
