# First Codex Prompt — OT2Control Acquisition Modes

Paste this into Codex after opening the repository on `Auto-RTG`.

```text
Read AGENTS.md and docs/codex/OT2CONTROL_AGENT_PLAYBOOK.md in full.

This is a read-only planning task. Do not edit files, create files, install
dependencies, run hardware-connected code, commit, push, or switch branches.

Confirm the current Git branch and working-tree status. Proceed only if the branch is
Auto-RTG. Stop if the branch is Auto-RTG-v1, main, or another protected branch.

Then inspect the entire repository and trace the acquisition path from:

1. Header worksheet parsing;
2. launch_auto and OptimizationModel construction;
3. GP initialization and updates;
4. replicate-QC/model-training data;
5. mixed ON/OFF mask generation;
6. active-dimension bounds;
7. physical volume-feasibility scoring;
8. target-distance optimization;
9. getNextReaction;
10. performance logging;
11. plots and auto_run_report.md.

Identify every definition and call site that would be affected by adding these
canonical spreadsheet modes:

- exploit
- explore
- balanced
- target_ei

Requirements:

- old spreadsheets default to exploit;
- exploit preserves stable-v1 behavior;
- every mode remains target-aware;
- do not directly use generic EI/MPI/LCB on raw lambda max;
- preserve exact-zero masks, >=5 uL executable transfer bounds, water top-off,
  overflow rejection, cumulative GP history, replicate QC, controller stopping,
  plotting orientation, and controller final authority;
- target_ei must use the best QC-approved condition-level target error;
- no physical robot, plate-reader, Google Sheets, credentials, or live protocol
  may be accessed.

Return:

A. current branch and worktree status;
B. repository files and methods involved;
C. current exact acquisition behavior;
D. risks and backward-compatibility requirements;
E. a staged implementation plan with one logical change per stage;
F. a synthetic test matrix for all four modes;
G. the exact first implementation stage only.

Do not make any edits.
```
