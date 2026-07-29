# AGENTS.md — OT2Control Pi Auto-main

## Purpose and scope

`Auto-main` is the version-controlled Raspberry Pi runtime branch for the
Auto-RTG system. It runs the robot-side service only. The laboratory computer
runs the Auto controller and optimizer from `Auto-RTG`.

## Branch policy

```text
main         = protected deployed Pi baseline and immediate rollback checkout
Auto-RTG     = laboratory-computer Auto controller, optimizer, analysis, and reports
Auto-main    = editable Pi runtime branch for reviewed Auto-specific robot support
```

Never modify `main` or `Auto-RTG-v1`. Do not merge `Auto-RTG` into
`Auto-main`: the branches intentionally have different responsibilities.

`Auto-main` may contain only the Pi-side changes needed to support a reviewed
Auto workflow, such as `ot2_robot.py`, `robot_script.sh`, Armchair protocol
support, Pi-specific tests, and Pi deployment documentation. It must not
become a second copy of the laboratory controller or optimizer.

## Deployment policy

- Keep `/root/OT2Control` on `main` unchanged at all times.
- Run the Pi service from `/root/OT2Control-Auto-main` only after an approved,
  controlled validation.
- The Pi currently has no DNS access to GitHub. Develop, test, commit, and
  push `Auto-main` from the laboratory computer; deploy reviewed commits to
  the Pi as verified Git bundles over the existing secure SSH connection.
- Before applying a bundle or any deployment update, confirm the Pi
  `Auto-main` worktree is clean. Use fast-forward-only integration; never
  force-push, reset, rebase, or silently merge on the Pi.
- Record the deployed commit in the run record before a controlled debug.

## Safety rules

1. Never run the robot, plate reader, serial connection, Eve server, or live
   protocol as part of code review or automated validation.
2. Never access, print, copy, or commit credentials, SSH keys, or tokens.
3. Preserve physical safeguards: executable transfer bounds, water and
   overflow checks, source-volume limits, pipette-tip limits, and controller
   final authority.
4. Keep Python 3.9 compatibility and the existing deployed dependency set.
5. Validate Pi changes with static checks, source-level review, and isolated
   stubs before a supervised, non-transfer connectivity check.
6. Do not claim a physical robot is ready based only on static or synthetic
   validation.

## Tare-calibration policy

Pi tube tare constants are safety-relevant because they determine calculated
source volume and aspiration depth. Any tare update must identify the physical
tube class, measured calibration basis, calibration version, affected
container classes, and a unit/stub validation using known tube-plus-liquid
weights. The nominal `Tube20000uL` class is used with the laboratory's 15 mL
tubes; document that mapping whenever it is relevant.

## Required review before editing

Before changing Pi runtime code, inspect the current branch, worktree state,
all relevant robot-side call sites, and the controller/Pi interface. Keep each
change narrowly scoped, document intentionally unchanged behavior, and do not
commit or deploy unless the project owner explicitly requests it.
