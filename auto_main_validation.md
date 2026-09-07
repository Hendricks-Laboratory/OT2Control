# Auto-main Validation Notes

**Repository:** Hendricks-Laboratory / OT2Control  
**Branch context:** Raspberry Pi Auto runtime  
**Protected Pi baseline:** `main`  
**Active Pi development branch:** `Auto-main`  
**Updated through:** 2026-09-07

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
