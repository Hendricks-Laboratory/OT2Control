# Auto Mode Validation Notes

**Repository:** Hendricks-Laboratory / OT2Control  
**Branch context:** Auto mode development branch  
**Prepared for:** Branch-local documentation / validation notes  
**Originally prepared:** 2026-06-08  
**Updated through:** 2026-06-15  

---

## Purpose

This note documents the current **Auto mode optimizer implementation and validation status** based on the recent branch history, dry/debug output, water-only testing, RTG_008 real-reagent validation, later RTG_009/DEBUGRTG-style output validation, and the latest replicate-QC/plot-layout validation work.

It is intended as a branch-specific record of:

- what has been implemented,
- what has been validated,
- what design decisions were made,
- what remains deferred,
- and what the next development steps should be.

> [!NOTE]  
> This note is **not** intended to describe the whole shared `OT2Control` repository.  
> It specifically documents Auto mode branch work around autonomous reaction selection, true-zero reagent handling, volume feasibility, mixed mask optimization, model-performance logging, plotting, debug exports, terminal-output capture, and protocol validation.

---

# High-Level Status

The current Auto mode branch has reached a substantially more mature validation milestone than the original June 8 notes. The branch now supports:

- volume-safe Auto recipe generation,
- true-zero variable reagent handling,
- mixed discrete/continuous mask optimization,
- condition-level model-performance logging,
- condition-level stopping logic,
- lambda-max progress plotting,
- final Auto progress summary plotting,
- optional display-capped error bars for readability,
- 2D GP prediction heatmaps,
- 2D GP uncertainty heatmaps,
- organized debug output folders,
- terminal-output capture to the run-specific Debug folder,
- and a cleaner division between primary results and debug/audit artifacts.

The latest validated debug run confirmed that the terminal-output capture and output-folder reorganization worked as intended.

| Area | Current status |
|---|---|
| Normalized recipe generation | Implemented |
| Concentration-to-volume conversion | Implemented |
| Independent reagent bounds | Implemented |
| True-zero variable reagent support | Implemented |
| Variable reagent executable-volume handling | Implemented |
| Water top-off feasibility | Implemented |
| Fixed reagent parsing from original template | Implemented |
| 200 µL well-volume preservation | Validated |
| Multi-fixed-reagent sheets | Validated |
| Mixed discrete/continuous mask optimizer | Implemented |
| Controller final hard-stop | Preserved |
| Condition-level Auto model-performance log | Implemented and validated |
| Condition-level stopping rule | Implemented and validated |
| Lambda-max progress plots | Implemented and validated |
| Final lambda-max summary plot | Implemented and validated |
| Replicate-level lambda diagnostic plots | Implemented and validated |
| Replicate QC metadata and model-training flags | Implemented and validated |
| Optional error-bar display cap | Implemented and validated |
| 2D GP prediction heatmap | Implemented and validated |
| 2D GP uncertainty heatmap | Implemented and validated |
| Terminal output saved to Debug folder | Implemented and validated |
| Auto recipe-design exports moved to Debug subfolder | Implemented and validated |
| Optimizer duplicate max-iteration print | Removed / controller owns user-facing stop print |
| Water-used tip handling | Deferred |
| Source/deck volume hard-stop | Deferred |
| Uncertainty-aware acquisition | Deferred |

Validated behavior includes:

- Auto recipes are generated in normalized model space and converted to physical transfer volumes.
- Variable reagents can be independently bounded by their own concentration/volume limits rather than forced into an old equal-split/simplex-like search region.
- True-zero behavior is supported for variable reagents when enabled in the Header.
- Variable reagent transfers are treated as physically executable only when they are exactly `0 µL` or at least `5 µL`.
- Water top-off is handled separately from true-zero reagent repair.
- Water top-off must be exactly `0 µL` or at least `5 µL`; `0 < water < 5 µL` is invalid.
- Fixed reagent volumes are parsed from the original input template rather than from mutated live recipe dataframes.
- Total well volume remains constrained to the intended total volume, currently `200 µL` in tested RTG workflows.
- More complex sheets containing multiple fixed reagents were handled successfully.
- The mixed discrete/continuous mask optimizer can choose which variable reagents are OFF or ON.
- OFF variable reagents are forced to exact zero.
- ON variable reagents are optimized continuously within executable transfer bounds.
- The controller remains the final hard-stop layer before protocol execution.
- Debug exports now include enough recipe-level information to verify concentration repair, transfer volumes, water top-off, and volume feasibility.
- Auto performance logging now records condition-level model predictions and condition-level measured outcomes.
- The lambda-progress plots show observed mean ± SEM and GP prediction ± SD across condition number.
- The 2D GP uncertainty plot shows the GP model's predictive standard deviation in nanometers, not replicate variability.
- Terminal output is mirrored to `Debug/terminal_output.txt` while preserving normal terminal printing.

> [!IMPORTANT]  
> The current implementation has passed software/protocol validation, water-only physical validation, debug-output validation, and at least one small real-reagent closed-loop validation. Before larger autonomous chemistry runs, the remaining operational hardening tasks are source/deck-volume validation, tip-contamination audit behavior, broader real-chemistry reproducibility testing, and eventual uncertainty-aware acquisition.

---

# Commit History and Development Summary

The Auto mode branch history shows a progression from low-level dataframe and transfer debugging into volume-aware recipe validation, true-zero behavior, continuous optimization, mixed mask optimization, model-performance logging, plotting, terminal-output capture, and debug-output organization.

---

## May 26, 2026 — Debug Export Foundation

**Relevant commits:**

- Add debug output for concentration-to-volume conversion
- Save concentration-to-volume debug CSV to Desktop
- Add debug exports for initial reaction and recipe dataframes

### Purpose

These commits established the initial debug-export foundation needed to inspect how input recipe concentrations were translated into physical transfer volumes. Auto mode depends on safely converting model-space recipes into robot-executable liquid handling commands, so early CSV visibility was essential.

The debug files made it possible to compare:

- initial reaction dataframe rows,
- recipe dataframe rows,
- concentration-to-volume conversion output,
- expected transfer volumes,
- generated robot-facing recipe instructions.

### Outcome

These exports became the basis for later diagnosing volume artifacts, zero-transfer behavior, recipe feasibility, and water top-off logic.

---

## May 27, 2026 — Transfer Normalization and Command Logging

**Relevant commits:**

- Save debug dataframes to Desktop folder
- Add transfer command debug logging
- Normalize transfer volumes before robot commands
- Retitled normalize function to be consistent with linguistic logic of rest of branch

### Purpose

This group improved visibility into the actual transfer commands being sent toward the robot-control layer and introduced safer normalization of transfer volumes prior to execution.

The practical reason for this work was to avoid robot commands being generated from raw dataframe artifacts or unclean floating-point values. It also made transfer command generation easier to inspect when a protocol did not behave as expected.

### Key behavior

- Transfer volumes were normalized before robot commands.
- Transfer-command debug information became available.
- Debug dataframes could be saved for inspection.
- Function names were cleaned up for consistency.

---

## May 29, 2026 — Experiment Data Update Debugging

**Relevant commits:**

- Add debug logging for experiment data updates
- Fix experiment data update dataframe construction
- small temp print statement adjustment
- printing experiment data DF
- removing print statement clutter

### Purpose

This group focused on the experiment-data update path. Auto mode needs to append new observed outcomes after each batch so the optimizer can train/update on the most recent results.

### Outcome

The branch moved closer to a complete closed-loop Auto cycle:

```text
1. generate recipes
2. execute or simulate recipes
3. collect scan data
4. update experiment_data
5. retrain/update model
6. suggest next recipe
```

---

## June 1, 2026 — Transfer Artifact Handling and Clearer Progress Output

**Relevant commits:**

- adding documentation to update experiment data
- Fix Auto experiment data logging
- Handle floating-point volume artifacts more safely
- small compatibility fix
- Improve float tolerance in transfer volume handling
- Handle NaN product volumes before transfer commands
- Add clearer Auto model progress output
- print statement terminal output adjustments
- adjusting prints again
- Clean up regex escape warnings
- add debug prints to test execute scan changes
- small debug change

### Purpose

This group improved robustness around transfer values, missing values, and terminal visibility.

### Important issues addressed

- Floating-point volume artifacts could create misleading tiny transfer values.
- NaN product volumes needed to be handled before transfer commands.
- Terminal output needed to clearly indicate model/protocol progress.
- Experiment-data logging needed to be consistent enough for debugging.

### Outcome

The Auto mode run became more inspectable from the terminal, and small numerical artifacts became less likely to propagate into robot transfer commands.

---

## June 2, 2026 — Capacity and Filename Prechecks

**Relevant commits:**

- Add Auto well capacity precheck
- Tag Auto scan output filenames with experiment name
- Use reaction sheet name in Auto scan filenames
- cleaning up precheck statements
- Add Auto pipette tip readiness prompts
- Add Auto pipette tip capacity precheck
- removing per-batch pipette prompt
- Additional tweaks to pipette tip estimation precheck
- small tweaks to pipette estimator

### Purpose

This group added pre-run safety and usability checks.

### Key behavior

- Auto mode can check whether the run has enough destination wells.
- Auto mode can estimate whether enough pipette tips are available.
- Scan output filenames include the experiment/reaction sheet name, making generated files easier to associate with a run.
- Some manual prompts were cleaned up or removed to make runs smoother.

### Outcome

The branch gained basic preflight safety around well capacity and tip capacity.

> [!NOTE]  
> These checks are useful but separate from later physical recipe-feasibility checks. Well capacity answers whether there are enough wells for planned samples. Tip capacity estimates whether the protocol has enough tips. Later work added checks for whether each recipe physically fits in its well volume.

---

## June 3, 2026 — Plotter, Maximin Seeding, and True-Zero Repair

**Relevant commits:**

- Update .gitignore
- Fixing the 2D plotter
- Adjusting plot limits
- Stop tracking .DS_Store
- Fix Auto GPR plotting model access
- Use maximin LHS for Auto initial design
- Apply true-zero repair to Auto recipes
- Small fix for true-zero repair
- Another patch for true zero repair
- Adjusting true zero repair functions

### Purpose

This group moved the optimizer toward better initial design behavior and introduced true-zero repair.

### Maximin initial design

The initial Auto seed design was moved toward a maximin Latin-hypercube-style approach to improve spread across the variable-reagent search space. The goal was to avoid poorly distributed initial experiments and improve model learning from early data.

### True-zero repair

The true-zero repair work addressed a major physical reality of the robot: variable reagents cannot be treated as continuously meaningful between `0` and `5 µL`. That region is chemically/model-wise misleading because the robot cannot execute tiny reagent volumes reliably.

| Variable reagent state | Physical meaning |
|---|---|
| OFF | exactly `0 µL` |
| ON | at least `5 µL` |
| `0 < volume < 5 µL` | invalid / not meaningful chemistry |

### Outcome

True-zero repair began preventing the optimizer/controller from interpreting non-executable tiny reagent volumes as valid chemistry. This became the foundation for the later mixed mask optimizer.

---

## June 4, 2026 — Continuous Optimizer and True-Zero-Aware Optimization

**Relevant commits:**

- Debugging csv for the LHS maximin implementation
- Update debug csv to include conc repair column
- Debugging the debugging csv export
- Read Auto replicate count from Header
- Fixing experiment_data csv
- Replace grid search with continuous optimizer
- Slight tweak of last push for code consistency
- Adjusting new optimizer to accept true zero bounds
- Allow controller to pass true zero optionally
- Make optimizer fallback more robust
- Improving controller handling of 0 volume transfers

### Purpose

This group replaced the old 2D brute-force grid search with a more dimension-general continuous optimizer and connected true-zero behavior more deeply into the optimization path.

### Major change: replacing brute-force grid search

The prior grid search was effectively tied to a 2D reaction space and used a dense grid approach. This was not scalable or appropriate for future 3D/multi-dimensional reagent spaces.

The new continuous optimizer searched normalized `0–1` reagent space and evaluated candidate recipes according to predicted target-distance behavior. For 2D experiments, the system still maintained the old prediction-grid output for plotting.

### True-zero-aware optimizer behavior

The optimizer was updated so candidates could be repaired according to true-zero rules before being scored and returned. This prevented the optimizer from searching the non-executable `0–5 µL` region as though it were meaningful.

### Replicate count

The Auto replicate count was moved into the Header using:

```text
num_duplicates
```

allowing repeated wells to be generated more flexibly from the input sheet.

### Zero-transfer handling

The controller was improved so `0`-volume transfer rows were skipped rather than generating robot commands for non-transfers.

### Outcome

By the end of this stage, Auto mode had:

- a more dimension-general optimizer,
- true-zero-aware candidate handling,
- Header-controlled duplicate counts,
- better debug CSVs,
- and improved handling of zero-volume transfers.

---

## June 5, 2026 — Volume-Aware Auto Recipe Checks

**Relevant commits:**

- Adding volume helpers for conc limit implementation
- Add volume-aware Auto recipe checks
- Making get fixed reagent volumes more robust
- Slight tweaks to volume helper
- Fix volume-feasible maximin seeding
- Fix fixed-volume parsing and water feasibility
- Update error code and documentation for volume overfill

### Purpose

This group was a major safety and correctness milestone. It added explicit physical volume feasibility logic for Auto-generated recipes.

### Key decisions implemented

#### 1. Independent variable reagent bounds

Variable reagents are no longer artificially constrained by an equal-split assumption. Each variable reagent can have an independent maximum based on the remaining available volume after fixed reagents are accounted for.

High/high combinations may physically overfill the well, so those combinations are rejected by feasibility checks rather than prevented by artificial equal-split bounds.

#### 2. Fixed reagent volume parsing from the original template

A bug was found where fixed reagent parsing could accidentally inspect mutated live dataframes after generated protocol rows had been added. This caused reagents such as Water, silver nitrate, and potassium bromide to be misclassified or confused as fixed reagents after batch 0.

The fix was to parse fixed reagent volumes from the original input template and stored fixed reagent list rather than the live generated `rxn_df`.

#### 3. Water top-off feasibility

Water is not treated like a true-zero reagent. Water is the top-off needed to keep the final well volume at the intended total.

| Water top-off | Status | Reason |
|---:|---|---|
| `0 µL` | Valid | no water needed |
| `>= 5 µL` | Valid | executable transfer |
| `0 < water < 5 µL` | Invalid | required but not executable |
| `< 0 µL` | Invalid | overflow |

This rule is essential because skipping a required `1–4 µL` water top-off would underfill the well and change the actual concentrations.

#### 4. Controller hard-stop

The controller became the final authority for physical feasibility. Even if the optimizer suggests something bad, the controller checks the recipe before execution and hard-stops invalid volume combinations.

#### 5. Volume-aware initial seeding

The initial maximin seed generator was repaired after an early failure mode where a full Latin hypercube design was rejected if any point overfilled.

The improved approach:

```text
1. build a feasible candidate pool
2. reject physically invalid candidates
3. greedily select a maximin subset from feasible candidates
```

### Outcome

The Auto branch could now generate seed recipes and optimizer-suggested recipes that:

- fit within the total well volume,
- preserve fixed reagent volumes,
- top off with executable water volumes,
- skip true-zero variable reagents correctly,
- reject overflow or invalid water-top-off recipes.

---

## June 8, 2026 — Mixed Mask Optimizer Implementation

**Relevant commits:**

- Add mixed mask optimizer scaffolding
- Logic adjustments for last push
- Add Auto mode validation notes

### Purpose

This group added the first mixed discrete/continuous optimizer implementation and captured the branch state in validation notes.

### Core idea

The optimizer now uses binary reagent masks to separate reagent presence/absence from continuous concentration optimization.

| Mask value | Meaning | Optimizer behavior |
|---:|---|---|
| `0` | reagent OFF | forced to exact zero |
| `1` | reagent ON | optimized continuously |

ON reagents have lower bounds corresponding to executable transfer volume, currently `5 µL`-equivalent. OFF reagents are not included in the active continuous search.

The mask optimizer therefore avoids treating the forbidden `0–5 µL` region as a continuous chemical response region.

### `allow_true_zero` behavior

#### When `allow_true_zero = TRUE`

The optimizer considers all non-empty ON/OFF masks.

For a 2D variable-reagent system:

```text
[1, 0]
[0, 1]
[1, 1]
```

The all-off mask is excluded by default because the workflow already uses blank/background subtraction.

#### When `allow_true_zero = FALSE`

The optimizer uses only the all-ON mask.

For a 2D variable-reagent system:

```text
[1, 1]
```

### Implemented helpers

- `_generate_reagent_masks()`
- `_get_active_mask_indices()`
- `_expand_masked_candidate_to_full_recipe()`
- `_get_masked_bounds()`
- `_masked_target_distance_objective()`
- `_get_reagent_masks_for_current_settings()`
- `_generate_feasible_masked_starting_points()`
- `_optimize_single_mask()`
- `_optimize_target_distance_with_masks()`

### Controller integration

- `OptimizationModel` now accepts `allow_true_zero`.
- `launch_auto()` passes `allow_true_zero` from `auto.robo_params`.
- `getNextReaction()` now calls `_optimize_target_distance_with_masks()`.
- `getNextReaction()` still returns `[best_x]`, preserving the controller-facing return format.
- The optimizer stores the selected mask and prediction values for downstream logging.

### Outcome

The mixed mask optimizer now behaves consistently with the intended true-zero chemistry logic and with the controller’s physical-volume safety layer.

---

## June 9, 2026 — Auto Model-Performance Logging and Lambda Progress Plot Foundation

**Relevant commits:**

- Add Auto mode model-performance logging
- Adjusting std math logic
- Add Auto progress plots and true-zero repair warning
- Pushing the previous push but for real
- Polish Auto lambda-progress plots
- Adjusting Auto lambda-progress plots
- Removing unused code
- Adjusting lambda max progress plots
- Remove plotting offsets

### Purpose

This group turned Auto mode from a protocol generator with debug CSVs into a more inspectable autonomous model loop. The key addition was condition-level model-performance logging and a first version of lambda-max progress plotting.

### Condition-level performance logging

The branch gained `auto_model_performance_log.csv`, a condition-level log separate from physical-well `experiment_data.csv`.

The distinction is important:

| File | Level | Purpose |
|---|---|---|
| `experiment_data.csv` | physical well / replicate | raw executed well-level data |
| `auto_model_performance_log.csv` | unique condition | model-vs-outcome performance tracking |

The performance log records one row per unique condition, including seed conditions and optimizer-selected conditions. Duplicate wells are summarized before being logged at the condition level.

### Prediction logging

For optimizer-selected rows, the branch logs:

- GP predicted lambda max mean in nm,
- GP predicted lambda max standard deviation in nm,
- actual measured mean lambda max,
- actual measured sample SD,
- actual measured SEM,
- target error,
- prediction error,
- closest-to-target-so-far information,
- recipe feasibility fields.

A key correction was made to the GP standard deviation conversion. The GP returns a predictive standard deviation in normalized model space, so the conversion to nanometers is:

```python
predicted_lambda_std_nm = normalized_std * 600.0
```

It should not be square-rooted again.

### Lambda progress plots

The first Auto lambda-progress plots were added. These plots show condition number on the x-axis and lambda max on the y-axis, with:

- observed condition mean ± SEM,
- GP prediction ± GP SD,
- target lambda line,
- batch-to-batch progress across Auto conditions.

### Outcome

By the end of this stage, Auto mode could produce a condition-level record of how well the GP model’s predictions matched observed results, and could visualize progress toward the target over time.

---

## June 10, 2026 — Condition-Level Stop Rule, Final Summary Plot, and Plot Readability

**Relevant commits:**

- Add final Auto λmax progress summary plot
- Fix Auto stopping rule to require validated duplicate-level target hit
- Improve Auto λmax progress plot readability with robust y-axis scaling
- Adjusting plot visuals

### Purpose

This group improved both scientific correctness and plot readability.

### Condition-level stopping rule

The stopping rule was moved away from raw physical-replicate logic and into condition-level duplicate summary logic.

The reason was that a single physical replicate can accidentally hit the target even when the duplicate condition is not reliable. In duplicate-based Auto mode, stopping should not be triggered by one lucky or noisy well.

The controller now checks condition-level summary metrics before accepting a target hit, including:

- mean target error,
- duplicate/replicate variability,
- target tolerance,
- replicate SD tolerance.

The optimizer’s internal `update_quit()` still tracks max-iteration state, but target-based stopping is intentionally controlled by the controller using summarized condition-level data.

### Final lambda progress plot

The controller now produces a final summary plot:

```text
lambda_progress_final.png
```

This is exported alongside the batch-wise lambda-progress plots.

### Robust y-axis scaling

The lambda-progress plot moved toward robust y-axis scaling so outlier means or huge uncertainty values would not make the whole plot unreadable.

### Outcome

The Auto mode loop became less likely to terminate on a false-positive noisy replicate, and the final run output became more reviewable.

---

## June 11, 2026 — Error-Bar Display Capping, 2D GP Uncertainty Heatmaps, Plot Polishing, and Debug Logging

**Relevant commits:**

- Adding optional error bar visual capping
- Adjusting clip functionality
- Adjusting wording for scientific clarity
- Warning adjustments
- Implemented 2D GP uncertainty heatmap support for Auto mode.
- Adjusting plot spacing
- Restoring cap annotations
- Tidying up graph axes titles
- Adjust plot margins
- Adjusting plot margin again
- Finalizing lambda plot
- Organize Auto debug outputs and save terminal logs
- Improving auto terminal saving
- Polish Auto debug logging and output organization

### Purpose

This was a large polish and output-validation stage. It focused on making Auto mode outputs scientifically interpretable, visually readable, and easier to debug after a run.

### Optional lambda-progress error-bar display cap

The lambda progress plot gained optional display capping for very large error bars.

This feature is visual only. Raw SEM and GP SD values remain in the CSV logs. The display cap prevents one extremely large SEM or GP predictive SD from dominating the plot and making all other points unreadable.

The terminology was intentionally changed from “clipped” to “display-capped” for scientific clarity.

| Term | Meaning |
|---|---|
| raw SEM / raw GP SD | preserved in CSV/log values |
| display-capped error bar | visually shortened for readability only |

The plot annotation is shown only when a cap is actually applied. The preferred annotation content was restored as:

```text
Display cap: 75 nm | GP SD display-capped at conditions ... | SEM display-capped at conditions ...
```

### Plot axis and spacing polish

The lambda-progress plot was refined through several iterations to reduce whitespace while avoiding overlap between the x-axis label and the display-cap annotation.

The validated settings reached in this stage were:

```python
ax.set_xlabel('Reaction condition number', labelpad=2)
```

and, when the display-cap annotation is present:

```python
fig.text(
    0.5,
    0.018,
    display_cap_note,
    ha='center',
    va='center',
    fontsize=7.3,
    color='0.35'
)

bottom_margin = 0.20
```

A later plot-margin adjustment was explored to provide slightly more spacing between the x-axis title and the annotation. The final margin should be treated as a visual parameter rather than a scientific/logical change.

### 2D GP uncertainty heatmaps

The 2D GPR plotting path was expanded from only predicted lambda max to both:

```text
gpr_predictions_batch_X.png
gpr_uncertainty_batch_X.png
```

The prediction heatmap shows:

```text
GP predicted lambda max in nm
```

The uncertainty heatmap shows:

```text
GP predictive standard deviation of lambda max in nm
```

The uncertainty plot is model uncertainty, not replicate variability.

In code, this comes from the GP prediction call:

```python
normalized_predictions, normalized_prediction_stds = self.gp_model.predict(grid_points)
```

and conversion back to nanometers:

```python
prediction_uncertainty_nm = normalized_prediction_std * 600.0
```

Therefore, an uncertainty heatmap colorbar value of `400` means:

```text
GP predictive SD ≈ 400 nm
```

It does not mean the predicted lambda max is 400 nm. It means the model is extremely uncertain there. Large uncertainty values are especially expected early in sparse runs.

### 2D-only behavior

The 2D heatmaps are intentionally only generated when exactly two variable reagents are active in the plotting context. For non-2D cases, the plotter skips cleanly rather than trying to visualize higher-dimensional surfaces incorrectly.

### Terminal-output capture

Auto mode now mirrors terminal output to:

```text
Debug/terminal_output.txt
```

while preserving normal terminal printing.

The implementation uses:

- `TeeTerminalOutput` to write each stdout/stderr message to both the original stream and a log file,
- `_start_terminal_output_capture()` to start capture once after output folders exist,
- `_stop_terminal_output_capture()` to restore stdout/stderr and close the file safely,
- `terminal_output_capture_guard()` to finalize capture after `_run()` succeeds or fails,
- and a `launch_auto()` outer cleanup guard to close capture if the run exits before `_run()` begins.

The terminal capture is process-local. It wraps `sys.stdout` and `sys.stderr` only inside the running Python process. It does not alter the macOS Terminal app, other tabs, shell environment, PATH, or other processes.

### Debug-output organization

Auto recipe-design CSVs were moved out of `pr_data` and into:

```text
Debug/auto_recipe_design/
```

This separates debug/audit artifacts from primary plate-reader/model outputs.

The intended output layout became:

```text
DEBUGRTG_007/
  Plots/
    lambda_progress_after_batch_*.png
    lambda_progress_final.png
    gpr_predictions_batch_*.png
    gpr_uncertainty_batch_*.png

  pr_data/
    experiment_data.csv
    auto_model_performance_log.csv
    *_auto_scan-*.csv
    *full_df.csv

  Debug/
    terminal_output.txt
    auto_recipe_design/
      auto_recipe_design_batch_*.csv

  Eve_Files/
    protocol_record.txt
    wellmap.tsv
    well_history.tsv
    translated_wellmap.tsv
```

### Validation performed

The terminal-output/debug-saves run validated that:

- `Debug/terminal_output.txt` was created,
- terminal output was captured from capture start through shutdown,
- terminal output capture closed cleanly,
- `auto_recipe_design_*.csv` files were written to `Debug/auto_recipe_design/`,
- recipe-design files were no longer loose in `pr_data`,
- lambda-progress plots were generated,
- final lambda-progress plot was generated,
- 2D GP prediction heatmaps were generated,
- 2D GP uncertainty heatmaps were generated,
- `auto_model_performance_log.csv` contained condition-level rows,
- and the run completed successfully.

### Outcome

By the end of June 11, Auto mode had much cleaner output artifacts and much better post-run debuggability.

---

## June 12, 2026 — Plot Margin Fix and Optimizer Internal Quit Logic Cleanup

**Relevant commits:**

- Plot margin fix
- Adjusting optimizer internal quit logic

### Purpose

This stage cleaned up the remaining visual and terminal-output polish issues noticed during debug-output validation.

### Plot margin fix

The lambda-progress annotation spacing was revisited after real debug images showed the display-cap annotation and x-axis label were close. The current working interpretation is that the visual margin is a plot-polish parameter and should be tuned by inspecting real generated output.

The validated settings at one point were:

```python
labelpad=2
annotation y=0.018
fontsize=7.3
bottom_margin=0.20
```

A later margin adjustment can increase bottom spacing slightly if needed, without affecting the data, model, or optimization logic.

### Optimizer internal quit logic

During terminal-log validation, the max-iteration exit message appeared twice:

```text
Exit due to max_iters
<<controller>> Exit due to max_iters
```

The source was two print locations:

1. `OptimizationModel.update_quit()` printed:

```python
print("Exit due to max_iters")
```

2. The controller printed:

```python
print("<<controller>> Exit due to max_iters")
```

The cleaner design is for the optimizer to silently maintain its internal `quit` flag and for the controller to own user-facing stop messages. This is also consistent with the current architecture where condition-level stopping and duplicate-aware validation are controller responsibilities.

The optimizer `update_quit()` should therefore set:

```python
self.quit = True
```

when `curr_iter >= max_iters`, but should not print the user-facing max-iteration message.

### Outcome

Terminal output becomes less redundant, and stop-message ownership is cleaner:

| Layer | Responsibility |
|---|---|
| optimizer | maintain internal `quit` state |
| controller | report user-facing stop reason |

---

## June 13–15, 2026 — Real-Reagent Baseline, Replicate QC, Model-Training Metadata, and Replicate Diagnostic Plots

**Relevant development work:**

- Validated RTG_008 as the first real-reagent Auto mode closed-loop baseline.
- Added condition-level replicate QC fields to `auto_model_performance_log.csv`.
- Added raw and QC-cleaned lambda summary columns.
- Added model-training decision metadata so flagged or QC-excluded conditions remain auditable.
- Integrated QC-filtered replicate values into GP model training while preserving raw well-level values.
- Added companion replicate-level lambda diagnostic plots.
- Added spreadsheet-triggered UV-vis overlay support alongside 2D GPR plots.
- Iteratively refined replicate diagnostic plot layout through `controller(141).py` to `controller(147).py`.

### RTG_008 real-reagent validation baseline

RTG_008 should be treated as a successful small real-reagent validation of the Auto mode pipeline.

Observed behavior:

- Auto mode generated initial seed recipes.
- Real reagents were transferred.
- Real spectra were collected.
- Lambda max values were extracted.
- `experiment_data.csv` was updated.
- The optimizer selected a subsequent recipe.
- True-zero / mask logic worked.
- Volume accounting worked.
- Water top-off worked.
- Total well volume remained `200 µL`.
- The run reached the normal success message:

```text
Success!!!
```

A post-success Eve/serial teardown error occurred after the success message. This is interpreted as a shutdown/communication teardown issue, not an optimizer, recipe, volume, scan, or model-update failure.

Important RTG_008 result:

| Metric | Value |
|---|---:|
| Target lambda max | `650 nm` |
| Optimizer-selected replicate 1 | `689 nm` |
| Optimizer-selected replicate 2 | `648 nm` |
| Duplicate mean | `668.5 nm` |
| Mean error from target | `18.5 nm` |
| Closest individual replicate | `648 nm` |

Interpretation:

- The real-reagent loop worked end-to-end.
- One duplicate landed extremely close to the 650 nm target.
- The condition-level mean was still outside a strict 10 nm target window, supporting the later duplicate/replicate-aware stopping-rule design.

### RTG_008 volume validation

The fixed reagent total in the real validation was `90 µL`:

| Fixed reagent | Volume |
|---|---:|
| trisodium_citrate | `20 µL` |
| hydrogen_peroxide | `50 µL` |
| sodium_borohydride | `20 µL` |
| **Total fixed volume** | **`90 µL`** |

Example optimizer-selected batch recipe:

| Volume component | Volume |
|---|---:|
| Fixed volume | `90.000000 µL` |
| Variable volume | `88.891374 µL` |
| Water | `21.108626 µL` |
| **Total** | **`200.000000 µL`** |
| Volume feasible | True |
| Water transfer executable | True |

This confirmed that the water top-off rule and 200 µL final-volume invariant held during real-reagent execution.

### Replicate QC policy

Replicate QC was added to distinguish raw well-level observations from QC-cleaned condition summaries.

Important functions added or refined:

- `_get_auto_replicate_outlier_threshold_nm()`
- `_run_lambda_replicate_qc(lambda_values)`
- `_get_auto_model_training_decision_from_replicate_qc(replicate_qc)`
- `_build_auto_qc_model_training_data(unique_recipes, lambda_max_values)`

Default replicate outlier threshold:

```text
50 nm
```

Triplicate behavior:

1. Find the closest pair of valid replicate lambda values.
2. Treat that pair as agreement only if the closest-pair distance is `<= replicate_outlier_threshold_nm`.
3. If the third value is more than the threshold away from the closest-pair mean, exclude the third value.
4. If no tight pair exists, flag the condition as `flagged_not_excluded` and preserve all valid replicates.

Examples:

| Replicates | QC result | Training behavior |
|---|---|---|
| `650, 653, 980` | `excluded_replicate` | train on `650, 653` |
| `650, 720, 790` | `flagged_not_excluded` | train on all valid replicates, but tag condition |
| `650, 660, 670` | `passed` | train on all valid replicates |
| `650, 650, 900, 900` | `flagged_not_excluded` | train on all valid replicates, no automatic exclusion |

The guiding design choice was conservative data preservation. A suspicious replicate is excluded only when there is a clear internally consistent pair and one outlier. Ambiguous spread is flagged but not discarded.

### Model-training metadata

The performance log now separates replicate QC status from model-training decisions.

Model-training policy:

| QC status | Model-training behavior |
|---|---|
| `passed` | use all valid replicates |
| `excluded_replicate` | use QC-included replicates only |
| `flagged_not_excluded` | still use all valid replicates, tagged as `used_flagged_condition` |
| no valid usable values | skip condition / fail cleanly if no training data remain |

Important model-training metadata columns:

- `use_for_model_training`
- `model_training_status`
- `n_replicates_used_for_model_training`
- `model_training_reason`

This allows the GP model to learn from as much data as possible while keeping questionable or QC-altered conditions auditable.

### Expanded `auto_model_performance_log.csv`

The performance log now includes raw replicate values, QC-cleaned values, replicate inclusion flags, QC status, and model-training metadata.

Important added columns include:

- `actual_lambda_values_raw_nm`
- `actual_lambda_mean_raw_nm`
- `actual_lambda_sd_raw_nm`
- `actual_lambda_sem_raw_nm`
- `actual_lambda_values_qc_nm`
- `actual_lambda_mean_qc_nm`
- `actual_lambda_sd_qc_nm`
- `actual_lambda_sem_qc_nm`
- `actual_lambda_rep_1_nm`
- `actual_lambda_rep_2_nm`
- `actual_lambda_rep_3_nm`
- `actual_lambda_rep_1_included_in_qc`
- `actual_lambda_rep_2_included_in_qc`
- `actual_lambda_rep_3_included_in_qc`
- `n_replicates_total`
- `n_replicates_valid`
- `n_replicates_used`
- `n_replicates_excluded`
- `excluded_replicate_indices`
- `excluded_lambda_values_nm`
- `replicate_qc_status`
- `replicate_qc_reason`
- `replicate_outlier_threshold_nm`
- `use_for_model_training`
- `model_training_status`
- `model_training_reason`

Backward-compatible columns remain:

- `actual_lambda_mean_nm`
- `actual_lambda_sd_nm`
- `actual_lambda_sem_nm`
- `target_error_nm`
- `prediction_error_nm`

These backward-compatible fields now represent QC-cleaned condition summaries.

### Replicate diagnostic plots

A new companion plotting function was added:

```text
_plot_lambda_replicate_progress_after_batch()
```

Generated outputs:

```text
lambda_replicates_after_batch_X.png
lambda_replicates_final.png
```

These plots are companion diagnostic plots and do not replace the main condition-level lambda-progress plots.

The replicate plots show:

- QC-included replicate lambda values as blue open circles.
- QC-excluded replicate lambda values as red x markers.
- GP prediction ± GP SD as orange square/errorbar.
- Target lambda max as a dashed gray horizontal line.
- Bottom annotation notes for display-capped GP SD, QC exclusions, and flagged-not-excluded conditions.

Important display decisions:

- Raw replicate values are not display-capped.
- GP SD error bars are display-capped for readability.
- Replicates are plotted at the exact same x-position for the same reaction condition.
- No horizontal jitter is used because the user preferred exact condition alignment.
- Identical replicate values therefore overplot exactly and may visually look like fewer points.

Possible future refinement:

- Add multiplicity annotations such as `×2` or `×3` for exactly overlapping replicate markers.
- Do not add jitter unless the user explicitly requests it.

### Replicate diagnostic plot layout status

The replicate diagnostic plot underwent several layout refinements to fit the title, legend/key, plot body, x-axis title, and bottom annotations without overlap.

Latest validated file in the prior chat:

```text
controller(147).py
```

Validated settings in `controller(147).py`:

```python
bbox_to_anchor=(0.5, 1.015)
top=0.86
bottom_margin = 0.18   # when annotation note is present
bottom_margin = 0.11   # when no annotation note is present
```

A fake-data plot was generated from the verbatim uploaded function and looked substantially improved.

Latest requested small adjustment, not yet validated in a new uploaded file at the time of this note:

```python
ax.set_xlabel('Reaction condition number', labelpad=5)
```

and:

```python
bottom_margin = 0.195  # when annotation note is present
bottom_margin = 0.12   # when no annotation note is present
```

Keep unchanged:

```python
bbox_to_anchor=(0.5, 1.015)
top=0.86
```

This adjustment is only intended to add slight breathing room between:

- x-axis ticks/axis and the x-axis title,
- and the x-axis title and the bottom annotations.

It should not alter optimization, QC, model training, or scientific data values.

### Spreadsheet-triggered plots

The spreadsheet protocol now supports both GP model plots and UV-vis overlay plots.

Recommended plot rows:

```text
operation = plot
scan filename = auto_scan
plot filename = auto_plot
plot protocol = 2d_gpr
Template = 1
```

and:

```text
operation = plot
scan filename = auto_scan
plot filename = auto_uv_overlay
plot protocol = OVERLAY
Template = 1
```

The `2d_gpr` plot generates:

- `gpr_predictions_batch_X.png`
- `gpr_uncertainty_batch_X.png`

The `OVERLAY` plot generates raw UV-vis overlay plots and includes all wells, including QC-excluded wells, for auditability.

### Key debug-output observations from later runs

A later replicate/debug output confirmed that QC and model-training metadata behaved as intended.

Example condition outcomes:

| Condition | Replicates | QC status | Model-training behavior |
|---:|---|---|---|
| 0 | `689, 683, 683` | `passed` | use all |
| 1 | `689, 689, 689` | `passed` | use all |
| 2 | `684, 824, 630` | `flagged_not_excluded` | use all, tagged |
| 3 | `824, 683, 824` | `excluded_replicate` | use `824, 824`; exclude `683` |

Another debug output showed that apparent missing replicate points were actually overplotted identical values because replicate x-offsets are intentionally zero.

Example:

| Condition | Values | Visual result |
|---:|---|---|
| 0 | `689, 824, 824` | red x at 689; two blue circles exactly overlap at 824 |
| 1 | `824, 824, 630` | two blue circles exactly overlap at 824; red x at 630 |
| 3 | `824, 824, 824` | three blue circles exactly overlap at 824 |

This is expected with no jitter.


---

# Current Technical Architecture

## 1. Optimizer model and masks

The optimizer uses one GP model over full normalized recipe vectors. True-zero behavior is handled by candidate generation/search masks, not by creating separate model structures.

- OFF reagents are exact zero.
- ON reagents are optimized continuously.
- Non-executable `0–5 µL` reagent regions are avoided.
- The all-off mask is excluded when true-zero is enabled.
- If true-zero is disabled, only the all-ON mask is used.

## 2. Acquisition behavior

The current acquisition remains target-distance exploitation:

```text
choose the candidate whose GP mean lambda max is closest to the target
```

The optimizer stores:

- selected mask,
- predicted lambda mean in nm,
- predicted lambda SD in nm,
- suggested recipe,
- and volume balance information.

Uncertainty-aware acquisition has not yet been implemented.

## 3. Prediction and uncertainty units

The model operates in normalized output space and converts back to nanometers.

Predicted mean:

```python
predicted_lambda_mean_nm = normalized_mean * 600.0 + 300.0
```

Predicted standard deviation:

```python
predicted_lambda_std_nm = normalized_std * 600.0
```

The standard deviation is already a standard deviation and should not be square-rooted again.

## 4. Condition-level logging

The branch now keeps physical-well raw data and condition-level model-performance data separate.

`experiment_data.csv` remains physical-well level.

`auto_model_performance_log.csv` is condition-level and contains summarized duplicate statistics and model predictions.

## 5. Plotting outputs

The plot outputs are:

| File | Meaning |
|---|---|
| `lambda_progress_after_batch_X.png` | cumulative lambda progress through batch X |
| `lambda_progress_final.png` | final full-run lambda progress summary |
| `gpr_predictions_batch_X.png` | 2D GP predicted lambda max surface |
| `gpr_uncertainty_batch_X.png` | 2D GP predictive SD surface |

## 6. Debug output organization

Current intended structure:

```text
DEBUGRTG_###/
  Plots/
    lambda_progress_after_batch_*.png
    lambda_progress_final.png
    gpr_predictions_batch_*.png
    gpr_uncertainty_batch_*.png

  pr_data/
    experiment_data.csv
    auto_model_performance_log.csv
    *_auto_scan-*.csv
    *full_df.csv

  Debug/
    terminal_output.txt
    auto_recipe_design/
      auto_recipe_design_*.csv

  Eve_Files/
    protocol_record.txt
    wellmap.tsv
    well_history.tsv
    translated_wellmap.tsv
```

---

# Validation Runs Completed

## 1. Dry Debug Protocol Validation

A dry/debug protocol run was performed with small conservative settings:

| Setting | Value |
|---|---:|
| `initial_data` | `5` |
| `max_iterations` | `1` |
| `num_duplicates` | `2` |
| `allow_true_zero` | `TRUE` |

The run exercised the controller/optimizer/protocol generation path.

### Observed validation results

- The protocol completed with `Success!!!`.
- The optimizer generated a volume-feasible maximin initial design.
- The mixed mask optimizer selected a non-empty mask.
- One observed selected mask was `[0, 1]`.
- OFF reagent transfer was skipped correctly.
- ON reagent transfer was retained at an executable `5 µL` boundary.
- Water top-off adjusted to preserve `200 µL` total well volume.
- Debug CSV rows showed volume feasibility.
- Water top-off values were either `0` or `>= 5 µL`.
- Variable reagent transfers were either `0` or `>= 5 µL`.

### Interpretation

The dry debug run passed the software/protocol-generation validation.

> [!NOTE]  
> Spectral/lambda-max values from this run were not considered chemically meaningful because the spectrometer was being run without a meaningful sample, so noise/edge maxima were expected.

---

## 2. Water-Only Physical Validation

A water-only physical validation was performed using water in place of the actual chemical reagents while preserving the input sheet reagent names, concentrations, and source mapping.

This was intentionally **not** a chemistry validation. It was a liquid-handling and protocol validation.

> [!WARNING]  
> One output folder name did not exactly match the input sheet name because the working directory/experiment directory was accidentally left as `RTG_007`. The run should be interpreted according to the provided water-only input and output files, not the mismatched folder label alone.

### Observed validation results

- The run used a more complex input sheet than the earlier simple debug setup.
- The sheet included multiple fixed reagents.
- Fixed reagent volume totaled `90 µL` in the validated run.

| Fixed reagent | Volume |
|---|---:|
| trisodium_citrate | `20 µL` |
| hydrogen_peroxide | `50 µL` |
| sodium_borohydride | `20 µL` |
| **Total fixed volume** | **`90 µL`** |

One selected optimizer recipe had:

| Volume component | Volume |
|---|---:|
| Fixed volume | `90.0000 µL` |
| Variable volume | `95.5581 µL` |
| Water | `14.4419 µL` |
| **Total** | **`200.0000 µL`** |
| Volume feasible | True |

Water top-off was valid because:

```text
14.4419 µL >= 5 µL
```

Recipe debug outputs showed:

- `volume_feasible = True`
- `volume_does_not_overflow = True`
- `water_transfer_executable = True`

### Interpretation

The water-only physical validation confirmed that the volume logic scales beyond the simplest citrate/silver/bromide setup.

The controller correctly accounted for multiple fixed reagents and still adjusted water to maintain `200 µL` total well volume.

---

## 3. Multi-Iteration Debug Output Validation

A later debug run validated the newer logging/plotting/output organization.

### Observed output structure

The expected files were generated:

```text
DEBUGRTG_007/
  Debug/
    terminal_output.txt
    auto_recipe_design/
      auto_recipe_design_batch_0.csv
      auto_recipe_design_batch_1.csv
      auto_recipe_design_batch_2.csv
      auto_recipe_design_batch_3.csv
      auto_recipe_design_batch_4.csv

  Plots/
    lambda_progress_after_batch_0.png
    lambda_progress_after_batch_1.png
    lambda_progress_after_batch_2.png
    lambda_progress_after_batch_3.png
    lambda_progress_after_batch_4.png
    lambda_progress_final.png
    gpr_predictions_batch_1.png
    gpr_predictions_batch_2.png
    gpr_predictions_batch_3.png
    gpr_predictions_batch_4.png
    gpr_uncertainty_batch_1.png
    gpr_uncertainty_batch_2.png
    gpr_uncertainty_batch_3.png
    gpr_uncertainty_batch_4.png

  pr_data/
    experiment_data.csv
    auto_model_performance_log.csv
    DEBUGRTG_007_auto_scan-0.csv
    DEBUGRTG_007_auto_scan-1.csv
    DEBUGRTG_007_auto_scan-2.csv
    DEBUGRTG_007_auto_scan-3.csv
    DEBUGRTG_007_auto_scan-4.csv
    DEBUGRTG_007full_df.csv
```

### Terminal log validation

`Debug/terminal_output.txt` was created and captured the run from terminal-capture start through shutdown.

It included the expected start marker:

```text
<<controller>> saving terminal output to .../DEBUGRTG_007/Debug/terminal_output.txt
```

It included the successful end of the run:

```text
Success!!!
<<controller>> shutting down
<<Reader>> executing: ... Terminate
<<controller>> terminal output capture complete
```

No traceback or error markers were found in the validated successful run.

### Performance log validation

`auto_model_performance_log.csv` contained condition-level rows consistent with the run structure:

- seed conditions,
- optimizer-selected conditions,
- prediction fields,
- observed duplicate summary fields,
- volume feasibility fields,
- and closest-to-target tracking.

The uncertainty values in early batches were large, which is expected because the GP model had sparse early data. Later batches showed lower GP predictive SD.

### Interpretation

This run validated the latest output organization and terminal-saving behavior.

---

# Current Implementation Invariants

The current Auto mode branch should preserve these invariants.

## 1. Final Well Volume Invariant

For every generated recipe:

```text
fixed reagent volume + variable reagent volume + water top-off = total well volume
```

In tested workflows:

```text
total well volume = 200 µL
```

---

## 2. Variable Reagent Transfer Invariant

Each variable reagent transfer should be either:

| Transfer volume | Meaning |
|---:|---|
| `0 µL` | OFF / true-zero |
| `>= 5 µL` | ON / present |
| `0 < volume < 5 µL` | invalid / non-executable |

The optimizer should not intentionally search or return variable reagent volumes in the non-executable `0–5 µL` region.

---

## 3. Water Top-Off Invariant

Water top-off should be either:

| Water top-off | Status |
|---:|---|
| `0 µL` | Valid |
| `>= 5 µL` | Valid |
| `0 < water < 5 µL` | Invalid |

Water top-off must not be in the range:

```text
0 < water < 5 µL
```

because skipping required water would underfill the well and alter concentrations.

---

## 4. Overflow Invariant

A recipe is invalid if:

```text
fixed reagent volume + variable reagent volume > total well volume
```

The optimizer should avoid this and the controller should hard-stop it.

---

## 5. Mask Invariant

When true-zero is enabled:

- non-empty masks are allowed,
- all-off mask is excluded by default,
- OFF reagents are exactly zero,
- ON reagents are bounded to executable transfer regions.

When true-zero is disabled:

- only all-ON mask is used.

---

## 6. Controller Authority Invariant

Even if the optimizer suggests a bad candidate, the controller remains the final execution guard. The controller must validate volume feasibility before robot execution.

---

## 7. Condition-Level Stop Invariant

Auto mode should not stop based on one lucky replicate well. A target hit should be accepted only after duplicate-level/condition-level summary criteria are satisfied.

The controller owns this user-facing stop decision.

---

## 8. Terminal Capture Invariant

Terminal capture should mirror output, not hijack it.

The implementation should:

- preserve normal terminal printing,
- write a copy to `Debug/terminal_output.txt`,
- restore stdout/stderr after success or error,
- close the log if setup exits before `_run()` begins,
- and avoid affecting other Terminal tabs or shell processes.

---

# Current Known Deferred Issues

## 1. Water-used tips are currently treated as clean enough

Water-used tips are currently treated as clean enough in `ot2_robot.py`.

### Observed behavior

The water-only physical run suggested that the robot/protocol may reuse water-used tips when switching from water to another reagent source. The protocol record indicated transitions where no visible drop/pickup occurred between `WaterC1.0` and `trisodium_citrateC12.5`.

### Relevant code pattern

`ot2_robot.py` appears to treat `WaterC1.0` as an acceptable/clean tip state in lists such as:

```python
['WaterC1.0', 'clean', src]
```

and:

```python
['clean', 'WaterC1.0']
```

### Why this matters

For real chemistry, a water-used tip is not the same thing as a fresh tip. Even if water is chemically harmless, the tip may have contacted product wells, carried droplets, or overwritten prior history in `last_used`.

The safer rule is:

| Tip state | Should count as clean? |
|---|---|
| fresh / clean tip | Yes |
| same source reagent, same transfer group | Usually reusable |
| water-used tip | Not automatically clean |
| different source reagent | Fresh tip required |

### Current decision

This issue is deferred for now and should be revisited before higher-risk or larger real chemistry runs.

It does **not** invalidate the volume/optimizer validation, but it is relevant for contamination-sensitive chemistry.

### Recommended later fix

Remove `WaterC1.0` from clean-enough tip lists in `ot2_robot.py`, including logic in:

- `_exec_transfer()`
- `_get_clean_tips()`
- `_liquid_transfer()`
- `_mix()`

### Recommended later debug enhancement

Add a transfer/tip audit CSV at the robot-control layer logging:

- transfer group starts,
- source reagent name,
- destination,
- volume,
- selected arm/pipette,
- `last_used` before transfer,
- `last_used` after transfer,
- tip pickup events,
- tip drop events.

This would definitively distinguish between reusing the same pipette body and reusing the same physical tip.

---

## 2. Source/deck volume validation is not yet a hard-stop

The controller currently checks whether each recipe fits into a well. It does not yet fully hard-stop a batch that demands too much total source volume from a reagent tube or reservoir.

This should be added before scaling to larger autonomous runs.

---

## 3. Uncertainty-aware acquisition is not yet implemented

The current optimizer uses target-distance exploitation based on GP mean lambda max.

The new GP uncertainty logging and heatmaps are diagnostic outputs. They do not yet drive the acquisition policy.

A later acquisition function could use model uncertainty directly.

A likely first uncertainty-aware acquisition would be:

```text
maximize P(|lambda_max - target| <= tolerance)
```

or a related target-probability criterion using GP mean and SD.

---

# Recommended Next Steps

## 1. Continued Small Real-Chemistry Validation / Scale-Up

The first small real-reagent validation has passed. The next scientific validation should remain conservative and should test repeatability, reporting outputs, replicate QC behavior, and modestly larger Auto loops rather than jumping directly to a large autonomous campaign.

| Setting | Value |
|---|---:|
| `initial_data` | `5` |
| `max_iterations` | `1` |
| `num_duplicates` | `2` |
| `allow_true_zero` | `TRUE` |

### Purpose

- Confirm that real spectra produce meaningful lambda-max values.
- Confirm replicates are not wildly inconsistent.
- Confirm the optimizer update path behaves with real data.
- Confirm selected masks make chemical sense.
- Verify debug CSVs still show valid volume behavior.
- Verify terminal-output capture works during a real run as it did in debug validation.

This run should be treated as a small validation run, not a full autonomous optimization campaign.

---

## 2. Add Optimizer Mask-Result Export

A useful next code improvement is an optimizer-side CSV export summarizing the mask search.

For each allowed mask, export:

- mask,
- success/failure,
- optimizer message,
- objective value,
- predicted lambda max,
- predicted GP SD,
- selected/not selected,
- normalized full recipe,
- active-only candidate,
- fixed volume,
- variable volume,
- water volume,
- volume feasibility.

### Purpose

This would make optimizer decisions transparent. It would answer questions such as:

- Why did the optimizer choose `[0, 1]` instead of `[1, 0]`?
- Was another mask infeasible?
- Was the selected point on a boundary?
- Did masks differ mainly by predicted lambda max or by feasibility constraints?

---

## 3. Add Deck/Source-Volume Validation Before Scaling

The next major safety feature before larger runs should be a batch-level deck/source-volume validator.

Before execution, calculate total required source volume for the full batch, including duplicates, for every source reagent:

- Water,
- fixed reagents,
- variable reagents.

Then compare required volume against available deck/source volume with a safety margin.

### Purpose

Independent reagent bounds make high-volume reagent usage more likely than the old equal-split approach. Source depletion is therefore one of the next likely practical risks when scaling.

---

## 4. Consider Uncertainty-Plot Display Scaling

The uncertainty heatmap currently shows raw GP predictive SD in nm. In sparse early batches this can be very large, sometimes hundreds of nm.

This is technically correct, but may make the heatmap visually dominated by huge early uncertainty values.

A later visualization-only improvement could add robust color scaling or a display cap for the uncertainty heatmap, while preserving raw uncertainty values in logs.

---

## 5. Gradual Scale-Up Plan

After small real-chemistry validation passes, scale gradually.

### Stage A

| Setting | Value |
|---|---:|
| `initial_data` | `5` |
| `max_iterations` | `1` |
| `num_duplicates` | `2` |

### Stage B

| Setting | Value |
|---|---:|
| `initial_data` | `8` |
| `max_iterations` | `2` |
| `num_duplicates` | `2` |

### Stage C

| Setting | Value |
|---|---:|
| `initial_data` | `10` |
| `max_iterations` | `3` |
| `num_duplicates` | `2` |

> [!CAUTION]  
> Do not jump directly to a large autonomous run until deck/source-volume validation and spectral behavior are better characterized.

---

## 6. Later Model/Chemistry Improvements

Later improvements may include:

- duplicate-aware model aggregation,
- better replicate handling,
- improved lambda-max extraction,
- edge-maximum detection and rejection,
- spectral smoothing or preprocessing,
- acquisition-function alternatives beyond pure target-distance,
- uncertainty-aware acquisition,
- batch candidate selection,
- deck-volume-aware optimizer penalties,
- transfer/tip audit logging.

---

# Validation Summary

## Current status

The Auto mode branch has passed the main optimizer/protocol validation milestone for mixed mask optimization and volume-safe recipe generation. It has also passed a major output-validation milestone for condition-level logging, lambda progress plotting, GP heatmaps, terminal-output capture, and debug-folder organization.

## Validated

- true-zero variable reagent handling,
- mixed mask optimizer path,
- independent reagent bounds,
- multi-fixed-reagent volume handling,
- water top-off logic,
- `200 µL` well-volume preservation,
- dry/debug protocol execution,
- water-only physical liquid-handling volume validation,
- condition-level Auto model-performance logging,
- condition-level stopping logic,
- final lambda progress plot export,
- display-capped error-bar annotation behavior,
- 2D GP prediction heatmap generation,
- 2D GP uncertainty heatmap generation,
- terminal-output capture to `Debug/terminal_output.txt`,
- recipe-design debug export relocation to `Debug/auto_recipe_design/`,
- and optimizer/controller stop-message cleanup.

## Deferred

- water-used tip behavior in `ot2_robot.py`,
- transfer/tip audit CSV,
- deck/source-volume hard-stop,
- broader real-chemistry reproducibility / scale-up validation,
- uncertainty-aware acquisition,
- larger autonomous scale-up.

## Recommended immediate next step

Continue with small, cautious real-chemistry validation/scale-up using conservative settings, then review:

- spectra,
- replicate consistency,
- mask selection,
- debug CSV volume fields,
- `auto_model_performance_log.csv`,
- lambda progress plots,
- GP prediction heatmaps,
- GP uncertainty heatmaps,
- and `Debug/terminal_output.txt`.

Only after that should the system be scaled to larger autonomous optimization runs.

---

# End of Auto Mode Validation Notes