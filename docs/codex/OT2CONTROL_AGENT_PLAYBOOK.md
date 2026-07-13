# OT2Control Codex Agent Playbook

## 1. Purpose

This playbook gives a coding agent the deeper project context behind the root `AGENTS.md`.

OT2Control Auto-RTG is a laboratory automation system that combines:

- spreadsheet-defined protocol configuration;
- Opentrons liquid handling;
- plate-reader acquisition;
- λmax extraction;
- replicate quality control;
- Gaussian-process modeling;
- autonomous reaction selection;
- physically feasible recipe generation;
- scientific plots and Markdown reporting.

A code change can be logically valid while still being physically unsafe, scientifically mislabeled, backward-incompatible, or impossible for the robot to execute. The required workflow is therefore conservative and evidence-driven.

## 2. Current branch model

### Protected upstream branch

```text
main
```

Do not switch to, edit, commit on, merge into, or push to `main` as part of this
project workflow.

### Stable rollback point

```text
Auto-RTG-v1
```

This branch is the frozen validated backup of Auto-RTG. Do not edit it while new changes are being developed.

### Active working branch

```text
Auto-RTG
```

All acquisition-mode work and other new updates belong on `Auto-RTG`. Changes remain there until they are reviewed and validated against the stable baseline.

The project owner may later promote validated work to `Auto-RTG-v1` through a
separate controlled process. Codex must not perform that promotion or modify
the stable branch under the standing policy.

There is currently no separate integration or feature branch requirement. Do not create `Auto-RTG` unless the user later requests a different branching strategy.

## 3. Stable-v1 behavior that must be preserved

Stable v1 includes:

- feasible maximin seed generation;
- independent reagent concentration bounds;
- true-zero mixed masks;
- exact-zero OFF reagents;
- executable ON-reagent bounds;
- fixed-volume accounting;
- water top-off feasibility;
- controller final feasibility hard-stop;
- condition-level replicate QC;
- QC-aware model-training data;
- cumulative GP/optimizer history;
- condition-level target stopping;
- per-batch and final progress/replicate plots;
- post-update GP prediction/uncertainty heatmaps;
- correct 2D axis orientation;
- seed-only and full exploration design-space plots;
- terminal capture;
- condition-level CSV logging;
- run report generation;
- defensive plotting that does not block shutdown.

## 4. Why the workflow is strict

Past validation found defects that looked visually or structurally harmless but changed scientific behavior:

### Cumulative history loss

The GP model could update while `optimizer.X/Y` remained stale. A later controller update could then rebuild history from seeds plus only the newest batch, forgetting earlier optimizer data.

### Heatmap transpose

A square matrix remained visually plausible even though a final `.T` swapped reagent effects relative to axis labels.

### Lifecycle timing

A full GP grid could be generated while choosing the next reaction, before the just-completed batch entered the model. A plot labeled after a batch could therefore show the prior model.

### NumPy truth-value error

A bounds helper used `array or []`, causing seed and exploration plots to fail only at finalization.

These examples justify repository-wide search, synthetic tests, saved-run regression, and explicit review after every stage.

## 5. User workflow preferences

The project owner prefers:

- exact, novice-readable steps;
- one implementation stage at a time;
- complete replacement functions when manual editing is required;
- no repeated partial edits to one function;
- explicit save/upload checkpoints;
- line-by-line validation of uploaded code;
- Python 3.9 compile checks;
- concrete synthetic and debug-run tests;
- GitHub Desktop rather than command-line Git instructions;
- a concise commit title plus a detailed human-readable commit message at every save step;
- a living validation document recording implementation, validation, errors, fixes, and rationale;
- stable branches before major feature work;
- honest distinction between static validation and physical-run evidence.

Codex should adapt to this process rather than attempting a large autonomous rewrite.

## 6. Recommended task lifecycle

### Stage A — read-only reconnaissance

The first task for any major update on `Auto-RTG` should be read-only.

Required output:

- current branch and worktree state;
- files and methods involved;
- complete data-flow trace;
- current behavior;
- backward-compatibility constraints;
- protected invariants;
- proposed stages;
- test strategy;
- unresolved design questions.

No edits.

### Stage B — isolated interface change

For acquisition modes, the first stage should only parse and normalize `acquisition_mode` from the Header.

It must not change optimizer behavior.

Validate:

- missing value defaults to `exploit`;
- canonical values accepted;
- aliases normalized;
- invalid values fail clearly;
- existing `auto_plot_profile` behavior remains intact.

### Stage C — propagation

Pass the canonical mode from controller setup into `OptimizationModel`.

Do not change recipe selection yet.

Validate constructor compatibility and old call sites.

### Stage D — acquisition scoring API

Create a single target-aware scoring interface that accepts GP mean and standard deviation and dispatches by canonical mode.

Keep physical feasibility and mask enumeration separate from statistical scoring.

Validate formulas with deterministic synthetic values.

### Stage E — mixed-mask integration

Use the selected score inside the existing masked optimizer pathway.

Preserve:

- all masks;
- active dimensions;
- SciPy multistart behavior;
- penalty handling;
- full-recipe reconstruction;
- selected-mask comparison;
- controller return shape.

### Stage F — incumbent management for target EI

Define the incumbent as the best QC-approved condition-level target error used by the model.

Avoid raw replicate incumbents.

Specify behavior before any optimizer batch has been measured.

### Stage G — logging and reporting

Add canonical mode and score to terminal output, CSV, report, and diagnostics.

Keep older logs readable where feasible.

### Stage H — workbook update

Only after code validation:

- add `acquisition_mode` to the Header;
- add a dropdown containing canonical names only;
- retain style, sheet order, formulas, and existing validations;
- do not alter unrelated protocol rows.

### Stage I — controlled debug run

Use a small, bounded test configuration.

Audit:

- selected recipes;
- masks;
- acquisition scores;
- cumulative history;
- GP predictions and uncertainty;
- physical volume balance;
- QC decisions;
- stop reason;
- plots;
- CSVs;
- report;
- terminal log.

## 7. Acquisition science

### 7.1 Stable exploitation

Stable v1 minimizes predicted squared target error:

```text
(predicted λmax - target λmax)^2
```

The new `exploit` mode should reproduce this behavior or an explicitly equivalent minimizer.

### 7.2 Pure exploration

A pure exploration score should prioritize predictive standard deviation.

Because the surrounding optimizer minimizes an objective, a typical score may negate uncertainty, but the sign convention must be documented and tested.

Do not confuse:

- GP variance;
- GP standard deviation;
- replicate SD;
- replicate SEM.

### 7.3 Balanced target-aware acquisition

A balanced mode should reward candidates that are near the target and/or uncertain enough to be informative.

The design must explicitly address scale. Both terms can be represented in nanometers, or each can be normalized using a documented scale.

Do not introduce a magic weight without:

- a named parameter;
- a default;
- documentation;
- synthetic sensitivity tests;
- logging.

A target-straddle concept may be appropriate, but implementation must be tied to this project’s target-seeking objective rather than copied from raw maximization.

### 7.4 Target expected improvement

Define current best target error:

```text
d_best = min |observed QC-approved condition mean - target|
```

For a candidate whose GP posterior is Normal with mean `μ` and standard deviation `σ`, target EI concerns improvement in:

```text
|Y - target|
```

not improvement in raw `Y`.

The exact analytic or numerical method must be documented. Tests must cover:

- `σ = 0`;
- mean exactly on target;
- mean far from target;
- candidate worse in mean but high uncertainty;
- no incumbent yet;
- finite and nonnegative EI;
- stable behavior for very small `σ`.

Do not label ordinary GPyOpt EI as `target_ei`.

## 8. Physical-feasibility architecture

Statistical acquisition scoring must not bypass physical feasibility.

Preferred conceptual order:

```text
generate mask
→ define executable active bounds
→ optimize acquisition score in active dimensions
→ expand to full normalized recipe
→ evaluate fixed/variable/water volumes
→ reject or penalize infeasible candidate
→ compare best feasible result across masks
→ controller performs final validation
```

The controller remains authoritative.

## 9. Model-update architecture

The update sequence must remain:

```text
build cumulative arrays
→ update GP model
→ only after success, synchronize optimizer X/Y
→ increment iteration
→ update quit state
→ refresh plotting grid through controller lifecycle
```

Never mutate history before confirming the GP update succeeded.

## 10. Replicate architecture

The workflow distinguishes:

- raw physical-well observations;
- QC-included replicate observations;
- QC-excluded observations;
- condition-level mean, SD, and SEM;
- model-training inclusion;
- controller target stopping.

Acquisition incumbents and training data must use the approved condition-level/QC pathway.

## 11. Plot architecture

### Scan-derived plot

UV overlays are based on plate-reader scan data and may be spreadsheet-triggered.

### Auto lifecycle plots

GP, progress, replicate, design-space, and report plots are coordinated through lifecycle stages.

### Heatmap orientation

For matrix `Z` passed with x-values and y-values:

```text
Z[row, column] = Z[y, x]
```

Keep first reagent on x and second reagent on y.

### Comparative color scales

Changing color scaling is a scientific presentation choice and must be treated separately from acquisition behavior.

## 12. Testing recipes

### Header parser tests

Construct minimal synthetic Header rows and validate:

- each canonical mode;
- uppercase and spacing aliases;
- missing value;
- empty value;
- invalid value;
- unrelated plot-profile parsing.

### Acquisition score tests

Use a fake GP posterior with known mean and SD.

Create candidates where:

- one is closest to target but certain;
- one is farther but uncertain;
- one has maximum uncertainty;
- one is physically infeasible.

Verify each mode chooses the intended candidate.

### Axis test

Use an asymmetric surface such as:

```text
f(x, y) = x + 10y
```

After reshape, columns must vary with x and rows with y.

### Cumulative history test

Simulate:

```text
seed
seed + B1
seed + B1 + B2
```

Assert `optimizer.X/Y` contain all rows after each successful update.

Force a GP update exception and assert no partial mutation.

### Saved-run regression

Use saved `auto_model_performance_log.csv` or isolated run artifacts rather than live hardware.

Confirm stable-v1 exploit reproduces the expected selected point under the same deterministic fake model.

## 13. Validation language

Use precise status labels:

### “Static validation passed”

Means parsing, compilation, structure, and call-site checks passed.

### “Synthetic behavior validation passed”

Means deterministic isolated tests passed.

### “Combined controller/optimizer validation passed”

Means both current exact files were tested together through stubs or saved data.

### “Ready for controlled debug run”

Means code is ready for a human-supervised non-chemistry or controlled debug protocol.

### “Ready for real robot run”

This requires human review plus the project owner’s explicit decision. An agent should not grant this status independently.

## 14. Commit-message standard

At each save checkpoint, provide:

### Title

- imperative or clear present-tense summary;
- generally under 72 characters;
- names the logical change.

### Message

Explain:

- what changed;
- why it was needed;
- the prior defect or missing behavior;
- how backward compatibility is preserved;
- what was deliberately not changed;
- validation completed;
- any remaining limitation.

Do not merely repeat the title.

## 15. Documentation standard

The validation Markdown is a living SOP and audit record.

When updating it:

- preserve history;
- add dates;
- distinguish implemented from validated;
- record exact debug configuration;
- record errors and fixes;
- record scientific caveats;
- record filenames and outputs;
- state branch and stable checkpoint;
- avoid retroactively erasing failed approaches.

## 16. When to stop and ask

Stop rather than guessing when:

- current branch is protected;
- uncommitted unrelated edits exist;
- requested behavior conflicts with a stable invariant;
- a formula is scientifically ambiguous;
- a test would require hardware or credentials;
- Python 3.9 is unavailable;
- dependency behavior cannot be reproduced;
- a file differs substantially from the documented stable version;
- spreadsheet schema is unclear;
- a requested cleanup would broaden scope;
- the correct incumbent for target EI is not available.

Provide the exact blocking question and the safest next action.
