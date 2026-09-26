# Auto-main Validation Notes

**Repository:** Hendricks-Laboratory / OT2Control  
**Branch context:** Raspberry Pi Auto runtime  
**Protected Pi baseline:** `main`  
**Active Pi development branch:** `Auto-main`  
**Updated through:** 2026-09-24

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

## Implemented Auto-main support at this checkpoint

The current Pi protocol provides the reviewed support needed by the paired
Auto-RTG baseline, including:

- calibrated Pi-side tube-tare defaults and container inventory handling;
- read-only robot-state compatibility snapshots (`auto-main-state-v9`);
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
- The read-only compatibility snapshot continues to report
  `auto-main-state-v9` and now advertises the optional targeted-mix command.
  Existing Lab-PC validation checks required command membership and therefore
  remains compatible with this additional advertised capability. The Lab-PC
  does not call the command until its dedicated Stage 13C work.
- The request names exactly one logical `autowell...` product and repeats its
  expected plate-reader deck position, well location, plate-mapping revision,
  and plate generation. The Pi rejects any stale/mismatched identity, generic
  reagent/container, non-reader plate, non-`Well96` target, target not bound
  to its registered custom plate-reader labware, or target whose final recorded
  material transfer is not the named trigger reagent.
- The Pi validates a finite mix volume, a bounded cycle count, the selected
  loaded instrument's actual minimum and maximum volumes, configured-versus-
  loaded pipette-capacity agreement, and a conservative maximum of 50% of the
  robot-tracked completed-well volume. Targeted well mixing deliberately uses
  the same larger-pipette selection policy as legacy preparation mixing;
  ordinary reagent-transfer selection remains unchanged. The 50% limit is a
  Pi execution safety bound, not a chemistry optimization setting; future
  controller work must choose a requested volume inside it.
- Mixing uses a dedicated clean tip for the single verified target well and
  discards it immediately afterwards. A retained clean tip may serve as that
  dedicated tip; a contaminated or absent tip is replaced before mixing. The
  Pi reserves enough unused tips to restore a clean tip after the operation.
- The only new liquid-handling primitive is `Well96.mix_targeted`. It repeats
  the established repository `pipette.mix(1, ...)` primitive at the existing
  96-well 1 mm aspiration and dispense clearance, restores prior pipette
  clearances even on failure, and deliberately omits legacy blow-out and
  touch-tip cleanup so this route adds no extra contact motion. Its rate is
  the documented `1.0` baseline, not the historical tube-only `100.0`
  multiplier; its requested mix volume is bounded by verified well volume
  rather than defaulting to the P300's full 300 uL capacity.
  It does not call or alter the legacy generic `mix` handler, source inventory,
  reaction-well volume accounting, reader, shaking, scheduler, or model.
- A validation failure sends an explicit rejected acknowledgement without
  motion. If physical execution begins and then raises, the Pi deliberately
  emits no success-like acknowledgement; the existing error boundary records
  the fault and the future controller must fail closed rather than replaying
  the mix.

Hardware-free validation on the laboratory PC used the system Python
`3.9.6`: compilation of the changed Pi files and focused packet/snapshot,
source-preflight, and targeted-mix stub tests passed (33 tests). The stubs
verify append-only packet identifiers, ghost registration, request schema,
stale/mismatched target rejection before motion, final-trigger validation,
tip-shortage rejection without state mutation, registered-plate identity,
live-instrument range and capacity-agreement rejection, one-target mix
execution, dedicated-tip discard/replacement, unchanged target volume, exact
1 mm targeted-mix geometry, one-cycle repetition, and clearance restoration
after both success and a synthetic pipette fault. This is static
and synthetic evidence only. Before Stage 13C can
use the capability, the
exact Auto-main commit needs its established bundle deployment and a human
supervised, chemistry-free Pi mixing dry debug that confirms the target well,
tip sequence, acknowledgement, and no change to ordinary Auto behavior.

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
