# AGENTS.md — OT2Control Auto-RTG

## Mission

This repository controls laboratory automation and a Bayesian-optimization workflow. Changes can affect:

- physical liquid handling,
- reagent-volume feasibility,
- plate-reader execution,
- replicate quality control,
- Gaussian-process training,
- autonomous reaction selection,
- scientific plots and reports.

Treat every change as safety-relevant and scientifically consequential.

The validated rollback branch is:

```text
Auto-RTG-v1
```

The active working branch for acquisition-function development and other verified updates is:

```text
Auto-RTG
```

`Auto-RTG-v1` is the frozen stable backup of the validated Auto-RTG implementation. `Auto-RTG` is the only branch where new changes should be made until those changes have been reviewed and validated.

Read `auto_mode_validation.md` before planning Auto-mode changes. For detailed workflow rules, also read:

```text
docs/codex/OT2CONTROL_AGENT_PLAYBOOK.md
```

## Non-negotiable safety rules

1. Never run a physical robot, plate reader, serial connection, Eve server, Google Sheets workflow, or live Auto protocol.
2. Never execute `controller.py` through its main launcher or call `launch_auto()`, `launch_protocol_exec()`, `run_protocol()`, or hardware initialization methods.
3. Use static analysis, isolated unit tests, synthetic stubs, and saved debug data only.
4. Never read, print, modify, copy, or commit credentials, keys, tokens, or secrets.
5. Never access or expose files such as:
   - `Credentials/`
   - `ssh_key`
   - `PersonalAccesstoken.txt`
   - `client_secret*`
   - service-account JSON
   - API tokens or private keys
6. Never modify `Auto-RTG-v1` or `main`.
7. Stop immediately if the current branch is `Auto-RTG-v1`, `main`, or another protected baseline and the task requests edits.
8. Do not commit, push, merge, rebase, tag, publish, or open a pull request unless the user explicitly asks.
9. Do not install, upgrade, or remove dependencies without explicit approval.
10. Do not claim physical robot clearance based only on compilation, static review, or synthetic tests.

## Compatibility requirements

- Preserve Python 3.9 compatibility.
- Preserve the validated legacy dependency environment unless a separate modernization task explicitly changes it.
- Do not introduce Python syntax requiring 3.10 or newer.
- Do not upgrade Opentrons, GPy, GPyOpt, pandas, NumPy, SciPy, or other production dependencies as part of an unrelated feature.
- Avoid broad formatting or refactoring in large legacy files.
- Preserve public method signatures and controller-facing return structures unless the approved plan explicitly changes them.
- Preserve backward compatibility for older spreadsheets whenever reasonably possible.

## Protected scientific and physical invariants

### Final reaction volume

For each generated recipe:

```text
fixed reagent volume
+ variable reagent volume
+ water top-off
= configured total reaction volume
```

In validated RTG workflows, the total is commonly `200 µL`, but never hard-code that value.

### Variable-reagent transfer executability

Each variable reagent must be:

```text
0 µL exactly
or
>= 5 µL
```

The interval:

```text
0 < transfer < 5 µL
```

is invalid and must not be treated as an ordinary continuous chemistry region.

### Water top-off

Water must be:

```text
0 µL exactly
or
>= 5 µL
```

A required water transfer between `0` and `5 µL` is invalid because skipping it changes the final volume and concentrations.

### Overflow

A recipe is invalid when:

```text
fixed volume + variable volume > total reaction volume
```

The optimizer should avoid invalid candidates, and the controller must remain the final hard-stop before execution.

### True-zero masks

When `allow_true_zero` is enabled:

- OFF reagents are exactly zero.
- ON reagents are optimized only in executable ranges.
- all non-empty masks may be considered.
- the all-off mask remains excluded by default.

When `allow_true_zero` is disabled:

- use only the all-ON mask.

### Cumulative model history

After every successful model update, later optimization must retain:

```text
seed data + every prior QC-approved optimizer batch
```

Never regress to:

```text
seed data + only the newest optimizer batch
```

`self.optimizer.X` and `self.optimizer.Y` must remain synchronized with the complete cumulative GP training arrays after successful updates.

A failed GP update must not partially mutate optimizer history, iteration count, or stop state.

### Replicate QC and training

- Preserve raw replicate values for auditability.
- Preserve QC-included and QC-excluded status.
- Train on the controller-approved condition-level data.
- Do not stop on one lucky replicate.
- Target stopping remains condition-level and controller-owned.
- Do not silently discard ambiguous replicate groups.
- Preserve model-training reason/status metadata.

### GP units

The model output normalization is:

```python
lambda_nm = normalized_mean * 600.0 + 300.0
```

Predictive standard deviation conversion is:

```python
std_nm = normalized_std * 600.0
```

The returned GP uncertainty is already a standard deviation. Do not take another square root.

### 2D heatmap orientation

For exactly two variable reagents:

```text
columns = first variable reagent = x-axis
rows    = second variable reagent = y-axis
```

With `meshgrid(indexing="xy")` and C-order flattening/reshaping, do not add an erroneous final transpose.

Prediction and uncertainty grids must use identical orientation and shape.

### GP plot timing

Full prediction and uncertainty grids must be refreshed after the completed batch has been incorporated into the fitted model.

Do not regenerate the full plotting grid inside `getNextReaction()` before the proposed experiment has been measured.

A plot labeled “After Batch N” must be based on a model that includes Batch N.

### Plot profiles

Preserve:

```text
standard
final_only
off
```

Older spreadsheets without `auto_plot_profile` default to `standard`.

## Branch and Git workflow

Current project branch policy:

```text
main         = protected upstream branch; never edit
Auto-RTG-v1  = frozen stable backup; do not edit
Auto-RTG     = active working branch; make and validate changes here
```

The project owner may later promote validated work from `Auto-RTG` to
`Auto-RTG-v1` through a separate, explicitly controlled process. Codex must not
perform that promotion, modify the stable branch, or modify `main` under this
policy.

Do not create or require a separate `Auto-RTG` branch unless the user later changes this policy.


Before editing, run or inspect the equivalent of:

```text
git status --short --branch
git diff --stat
```

Then report:

- current branch,
- modified/untracked files,
- whether the working tree is clean,
- whether the requested work is safe to begin.

Rules:

- Work only on the approved working branch: `Auto-RTG`.
- Never switch away from `Auto-RTG` without explicit permission.
- Never overwrite unrelated uncommitted changes.
- Never use destructive Git commands.
- Never use `git reset --hard`, `git clean -fd`, force push, or history rewriting.
- Keep each stage reviewable and narrowly scoped.
- Do not commit automatically.
- At every save/review checkpoint, provide:
  - a concise GitHub Desktop commit title;
  - an in-depth but human-readable commit message.
- Do not provide shell Git commands to the user unless explicitly requested.

## Required implementation workflow

### 1. Inspect before editing

Before changing code:

1. Read this file and the stable validation document.
2. Inspect all definitions and call sites related to the request.
3. Trace data flow across controller, optimizer, spreadsheet parsing, logging, plotting, and report generation.
4. Identify backward-compatibility requirements.
5. Identify scientific and physical invariants at risk.
6. State the proposed staged plan.
7. Do not edit until the requested stage is clear.

### 2. One logical stage at a time

- Implement only the approved stage.
- Do not opportunistically implement later stages.
- Do not combine unrelated cleanup.
- If several edits are required inside one function, make the complete coherent function change once rather than repeatedly rewriting the same function across stages.
- Keep large-file diffs minimal.
- Preserve existing comments and style unless they are incorrect.

### 3. Validate each stage

At minimum, after Python changes:

```text
python3.9 -m py_compile controller.py optimizers.py
```

Use the active Python 3.9 environment. If `python3.9` is unavailable, report that clearly rather than silently validating with an incompatible interpreter.

Also perform relevant checks:

- AST parse and duplicate-method detection.
- Method placement and class indentation.
- Public interface compatibility.
- Search for all call sites.
- Isolated synthetic behavior tests.
- Failure-path and atomicity tests.
- Backward-compatibility tests.
- Diff review for unrelated changes.

Do not import hardware-dependent modules merely to test a small method when a source-level extraction, stub object, or dependency mock is safer.

### 4. Report results honestly

After each stage, report:

- files changed;
- functions changed;
- behavior added;
- behavior intentionally unchanged;
- exact commands/tests run;
- pass/fail result;
- unresolved risks;
- whether the code is:
  - statically validated,
  - synthetically validated,
  - ready for combined-file review,
  - ready for a controlled debug run,
  - or not ready.

Do not use “flawless,” “fully safe,” or “robot-ready” without evidence supporting that exact claim.

## Acquisition-mode feature requirements

The planned spreadsheet values are:

```text
exploit
explore
balanced
target_ei
```

### Backward compatibility

- Missing `acquisition_mode` must default to `exploit`.
- `exploit` must reproduce stable-v1 selection behavior.
- Old spreadsheets must continue to run without modification.
- Normalize reasonable aliases, but store and log only canonical names.

### Target-aware design

This project seeks a response near a target λmax. It is not ordinary raw-output minimization or maximization.

Do not directly substitute generic GPyOpt EI, MPI, or LCB on raw λmax.

Every acquisition mode must remain target-aware and must continue to use the custom mixed-mask, physically feasible candidate pathway.

### Mode semantics

#### `exploit`

Select the feasible candidate whose GP predicted mean is closest to the target.

Stable-v1 behavior must be preserved exactly enough for regression testing.

#### `explore`

Select the feasible candidate with the greatest GP predictive standard deviation.

Exploration must still obey:

- mask rules,
- executable transfer bounds,
- volume feasibility,
- water feasibility,
- all-off exclusion,
- controller validation.

#### `balanced`

Use a scientifically documented target-aware hybrid of:

- target proximity;
- predictive uncertainty.

The score must be dimensionally coherent in nanometers or explicitly normalized. Do not combine quantities with arbitrary scales without documenting and testing the weighting.

#### `target_ei`

Implement expected improvement in target error, not generic improvement in raw λmax.

The incumbent must be based on the best QC-approved condition-level target error available to model training, not one individual replicate.

Document the formula, numerical stability handling, and zero-uncertainty limit.

### Acquisition metadata

For optimizer-selected conditions, preserve or add audit fields for:

- canonical acquisition mode;
- acquisition score;
- predicted λmax mean;
- predicted GP standard deviation;
- predicted target error;
- selected mask;
- normalized recipe;
- physical concentrations;
- volume balance;
- incumbent target error when applicable;
- any mode-specific parameter or weight.

Metadata should flow to:

- terminal output;
- `auto_model_performance_log.csv`;
- `auto_run_report.md`;
- optimizer mask-result diagnostics when added.

### Acquisition tests

Synthetic tests must demonstrate distinct expected behavior:

- `exploit` chooses the closest predicted mean to target.
- `explore` chooses the maximum uncertainty.
- `balanced` changes selection appropriately as proximity and uncertainty trade off.
- `target_ei` chooses the greatest expected reduction in target error.
- all modes reject infeasible candidates.
- all modes preserve mask semantics.
- all modes preserve cumulative GP history.
- `exploit` regresses against stable-v1 behavior.
- invalid mode names fail clearly.
- missing Header value defaults to `exploit`.
- logging records the canonical mode and score.

## Key files and responsibilities

### `controller.py`

Important responsibilities include:

- spreadsheet/Header parsing;
- launching and configuring `OptimizationModel`;
- robot and plate-reader lifecycle;
- recipe conversion and final feasibility validation;
- replicate QC and model-training decisions;
- condition-level target stopping;
- CSV logging;
- plot coordination;
- final report generation.

Relevant areas commonly include:

- `launch_auto()`
- `_init_robo_header_params()`
- `AutoContr._run()`
- `_generate_auto_plot_suite()`
- Auto performance-log helpers
- Auto report helpers

### `optimizers.py`

Important responsibilities include:

- initial feasible maximin design;
- true-zero mask generation;
- active-dimension bounds;
- masked candidate expansion;
- physical-feasibility scoring;
- GP initialization and updates;
- cumulative history synchronization;
- next-reaction selection;
- prediction/uncertainty grid refresh.

Relevant areas commonly include:

- `OptimizationModel.__init__()`
- `_masked_target_distance_objective()`
- `_optimize_single_mask()`
- `_optimize_target_distance_with_masks()`
- `initialize_optimizer()`
- `getNextReaction()`
- `update_experiment_data()`
- `refresh_prediction_grid_for_plotting()`

### `ot2_robot.py`

Treat as physical-execution code. Do not modify it during acquisition work unless the user explicitly expands scope.

### Validation documentation

Use:

```text
auto_mode_validation.md
```

as the authoritative stable baseline. Update validation notes only after validated behavior changes.

## Files and directories not to commit

Review `.gitignore` and avoid committing generated or sensitive artifacts, including:

- credentials and secrets;
- local caches;
- `.DS_Store`;
- Python bytecode and `__pycache__/`;
- protocol output folders;
- plate-reader output;
- large generated CSV/PNG archives;
- temporary extracted zip folders;
- local environment files;
- debug outputs unless deliberately added as small test fixtures.

Do not delete user data merely because it should not be committed.

## Testing strategy

Preferred order:

1. syntax and Python 3.9 compilation;
2. AST/static structure checks;
3. pure-function tests;
4. synthetic GP mean/SD stubs;
5. fake optimizer/controller integration;
6. saved-run regression data;
7. combined `controller.py`/`optimizers.py` review;
8. human-controlled debug protocol;
9. human review of plots, logs, reports, and recipe CSVs;
10. only then consider a real chemistry run.

Never skip directly from code edit to physical execution.

## Review checklist

Before declaring a stage complete, verify:

- `Auto-RTG-v1` was not modified;
- no secret was accessed;
- no hardware command was run;
- Python 3.9 compilation passed;
- only intended files changed;
- no duplicate/misplaced methods;
- call sites match new signatures;
- old spreadsheets remain compatible;
- physical volume invariants remain enforced;
- true-zero behavior remains correct;
- cumulative GP history remains correct;
- QC and stop behavior remain correct;
- prediction and uncertainty units remain correct;
- plotting orientation/timing remain correct;
- output metadata is auditable;
- tests distinguish the new behavior from old behavior;
- limitations are stated honestly;
- GitHub Desktop title and message are provided at the save checkpoint.

## Response style for this repository

- Be exact and operational.
- Prefer a staged plan over a large unreviewed rewrite.
- Explain scientific consequences in plain language.
- Separate blocking defects from nonblocking improvements.
- Do not hide warnings or failed tests.
- Do not ask the user to manually copy many scattered edits when direct repository editing is available.
- When presenting a manual replacement, provide the complete final function with correct four-space class indentation.
- Do not split multiple changes to the same function across several steps.
- Include a human-readable Git title and detailed message whenever the next action is to save or upload changed code.
