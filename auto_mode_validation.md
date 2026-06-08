# Auto Mode Validation Notes

**Repository:** Hendricks-Laboratory / OT2Control  
**Branch context:** Auto mode development branch  
**Prepared for:** Branch-local documentation / validation notes  
**Date:** 2026-06-08  

---

## 🎯 Purpose

This note documents the current **Auto mode optimizer implementation and validation status** based on recent branch history and dry/water-only testing. It is intended as a branch-specific record of:

- what has been implemented,
- what has been validated,
- what remains deferred,
- and what the next development steps should be.

> [!NOTE]  
> This note is **not** intended to describe the whole shared `OT2Control` repository.  
> It specifically documents the Auto mode branch work around autonomous reaction selection, true-zero reagent handling, volume feasibility, mixed mask optimization, debug exports, and protocol validation.

---

# ✅ High-Level Status

The current Auto mode branch has reached a stable milestone for the new optimizer implementation.

The following behavior has been implemented and validated through dry-debug and water-only physical validation runs:

| Area | Current status |
|---|---|
| Normalized recipe generation | ✅ Implemented |
| Concentration-to-volume conversion | ✅ Implemented |
| Independent reagent bounds | ✅ Implemented |
| True-zero variable reagent support | ✅ Implemented |
| Variable reagent executable-volume handling | ✅ Implemented |
| Water top-off feasibility | ✅ Implemented |
| Fixed reagent parsing from original template | ✅ Implemented |
| 200 µL well-volume preservation | ✅ Validated |
| Multi-fixed-reagent sheets | ✅ Validated |
| Mixed discrete/continuous mask optimizer | ✅ Implemented |
| Controller final hard-stop | ✅ Preserved |
| Recipe-level debug exports | ✅ Expanded |
| Water-used tip handling | ⚠️ Deferred |

Validated behavior includes:

- Auto recipes are generated in normalized model space and converted to physical transfer volumes.
- Variable reagents can be independently bounded by their own concentration/volume limits rather than being forced into an old equal-split/simplex-like search region.
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

> [!IMPORTANT]  
> The current implementation has passed recipe/protocol validation, but there are still deferred operational hardening tasks before scaling to larger autonomous runs.

---

# 🧾 Commit History Summary

The Auto mode branch history shows a progression from low-level dataframe and transfer debugging into volume-aware recipe validation, true-zero behavior, continuous optimization, and finally mixed mask optimization.

---

## May 26, 2026 — Debug Export Foundation

**Relevant commits:**

- Add debug output for concentration-to-volume conversion
- Save concentration-to-volume debug CSV to Desktop
- Add debug exports for initial reaction and recipe dataframes

### Purpose

These commits established the initial debug-export foundation needed to inspect how input recipe concentrations were translated into physical transfer volumes. This was important because the Auto mode workflow depends on safely converting model-space recipes into robot-executable liquid handling commands.

The debug files made it possible to compare:

- initial reaction dataframe rows,
- recipe dataframe rows,
- concentration-to-volume conversion output,
- expected transfer volumes,
- generated robot-facing recipe instructions.

### Outcome

These exports became the basis for later diagnosing volume artifacts, zero-transfer behavior, and recipe feasibility.

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

This group focused on the experiment-data update path. The Auto mode model needs to append new observed outcomes after each batch so the optimizer can train/update on the most recent results.

The commits improved construction of experiment-data dataframes and added or removed temporary prints as needed to debug the data update process.

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
> These checks are useful but separate from the later volume-feasibility checks.  
> Well capacity answers whether there are enough wells for the planned number of samples.  
> Tip capacity estimates whether the protocol has enough tips.  
> Later work added checks for whether each recipe physically fits in its well volume.

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

The intended rule became:

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

## Key decisions implemented

### 1. Independent variable reagent bounds

Variable reagents are no longer artificially constrained by an equal-split assumption. Each variable reagent can have an independent maximum based on the remaining available volume after fixed reagents are accounted for.

For example:

| Component | Volume |
|---|---:|
| Total volume | `200 µL` |
| Fixed volume | `20 µL` |
| Remaining volume | `180 µL` |

Each variable reagent can independently have a maximum corresponding to the full `180 µL` remaining volume. High/high combinations may physically overfill the well, so those combinations are rejected by feasibility checks rather than being prevented by artificial equal-split bounds.

### 2. Fixed reagent volume parsing from the original template

A bug was found where fixed reagent parsing could accidentally inspect mutated live dataframes after generated protocol rows had been added. This caused reagents such as Water, silver nitrate, and potassium bromide to be misclassified or confused as fixed reagents after batch 0.

The fix was to parse fixed reagent volumes from the original input template and stored fixed reagent list rather than the live generated `rxn_df`.

### 3. Water top-off feasibility

Water is not treated like a true-zero reagent. Water is the top-off needed to keep the final well volume at the intended total.

| Water top-off | Status | Reason |
|---:|---|---|
| `0 µL` | ✅ Valid | no water needed |
| `>= 5 µL` | ✅ Valid | executable transfer |
| `0 < water < 5 µL` | ❌ Invalid | required but not executable |
| `< 0 µL` | ❌ Invalid | overflow |

This rule is essential because skipping a required `1–4 µL` water top-off would underfill the well and change the actual concentrations.

### 4. Controller hard-stop

The controller became the final authority for physical feasibility. Even if the optimizer suggests something bad, the controller checks the recipe before execution and hard-stops invalid volume combinations.

### 5. Volume-aware initial seeding

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

### Purpose

This group added the first mixed discrete/continuous optimizer implementation.

## Core idea

The optimizer now uses binary reagent masks to separate reagent presence/absence from continuous concentration optimization.

| Mask value | Meaning | Optimizer behavior |
|---:|---|---|
| `0` | reagent OFF | forced to exact zero |
| `1` | reagent ON | optimized continuously |

ON reagents have lower bounds corresponding to executable transfer volume, currently `5 µL`-equivalent. OFF reagents are not included in the active continuous search.

The mask optimizer therefore avoids treating the forbidden `0–5 µL` region as a continuous chemical response region.

## `allow_true_zero` behavior

### When `allow_true_zero = TRUE`

The optimizer considers all non-empty ON/OFF masks.

For a 2D variable-reagent system:

```text
[1, 0]
[0, 1]
[1, 1]
```

The all-off mask is excluded by default because the workflow already uses blank/background subtraction.

### When `allow_true_zero = FALSE`

The optimizer uses only the all-ON mask.

For a 2D variable-reagent system:

```text
[1, 1]
```

## Implemented helpers

- `_generate_reagent_masks()`
- `_get_active_mask_indices()`
- `_expand_masked_candidate_to_full_recipe()`
- `_get_masked_bounds()`
- `_masked_target_distance_objective()`
- `_get_reagent_masks_for_current_settings()`
- `_generate_feasible_masked_starting_points()`
- `_optimize_single_mask()`
- `_optimize_target_distance_with_masks()`

## Controller integration

- `OptimizationModel` now accepts `allow_true_zero`.
- `launch_auto()` passes `allow_true_zero` from `auto.robo_params`.
- `getNextReaction()` now calls `_optimize_target_distance_with_masks()`.
- `getNextReaction()` still returns `[best_x]`, preserving the controller-facing return format.
- The optimizer prints the selected reagent mask and suggested recipe volume balance.

## Logic adjustments after review

- Fixed controller handoff so `allow_true_zero` is read from:

```python
auto.robo_params.get('allow_true_zero', False)
```

rather than a nonexistent `auto.allow_true_zero` attribute.

- Corrected masked-bound validation so the `lower_bound > 1.0` check occurs before clipping.
- Updated initial design generation to exclude all-off repaired seed recipes when true-zero is enabled, while still allowing partial-off recipes.

### Outcome

The mixed mask optimizer now behaves consistently with the intended true-zero chemistry logic and with the controller’s physical-volume safety layer.

---

# 🧪 Validation Runs Completed

---

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
> The output folder name did not exactly match the input sheet name because the working directory/experiment directory was accidentally left as `RTG_007`.  
> The run should be interpreted according to the provided water-only input and output files, not the mismatched folder label alone.

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

The optimizer selected a mask such as:

```text
[1, 1]
```

One selected optimizer recipe had:

| Volume component | Volume |
|---|---:|
| Fixed volume | `90.0000 µL` |
| Variable volume | `95.5581 µL` |
| Water | `14.4419 µL` |
| **Total** | **`200.0000 µL`** |
| Volume feasible | ✅ True |

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

> [!IMPORTANT]  
> This is an important milestone because the logic did not assume only one fixed reagent or only the earlier simple test case.

---

# ⚠️ Current Known Deferred Issue

Water-used tips are currently treated as clean enough in `ot2_robot.py`.

## Observed behavior

The water-only physical run suggested that the robot/protocol may reuse water-used tips when switching from water to another reagent source. The protocol record indicated transitions where no visible drop/pickup occurred between `WaterC1.0` and `trisodium_citrateC12.5`.

## Relevant code pattern

`ot2_robot.py` appears to treat `WaterC1.0` as an acceptable/clean tip state in lists such as:

```python
['WaterC1.0', 'clean', src]
```

and:

```python
['clean', 'WaterC1.0']
```

## Why this matters

For real chemistry, a water-used tip is not the same thing as a fresh tip. Even if water is chemically harmless, the tip may have contacted product wells, carried droplets, or overwritten prior history in `last_used`.

The safer rule is:

| Tip state | Should count as clean? |
|---|---|
| fresh / clean tip | ✅ Yes |
| same source reagent, same transfer group | ✅ Reusable |
| water-used tip | ❌ Not automatically clean |
| different source reagent | ❌ Fresh tip required |

## Current decision

This issue is deferred for now and should be revisited before higher-risk or larger real chemistry runs.

It does **not** invalidate the volume/optimizer validation, but it is relevant for contamination-sensitive chemistry.

## Recommended later fix

Remove `WaterC1.0` from clean-enough tip lists in `ot2_robot.py`, including logic in:

- `_exec_transfer()`
- `_get_clean_tips()`
- `_liquid_transfer()`
- `_mix()`

## Recommended later debug enhancement

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

# 📐 Current Implementation Invariants

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
| `0 µL` | ✅ Valid |
| `>= 5 µL` | ✅ Valid |
| `0 < water < 5 µL` | ❌ Invalid |

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

# 🚀 Suggested Next Steps

---

## 1. Small Real-Chemistry Validation

The next scientific validation should be a small real-chemistry run using conservative settings.

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

The controller currently checks whether each recipe fits into a well. It does not yet fully hard-stop a batch that demands too much source volume from a reagent tube/reservoir.

### Recommended behavior

Before execution, calculate total required source volume for the full batch, including duplicates, for every source reagent:

- Water,
- fixed reagents,
- variable reagents.

Then compare required volume against available deck/source volume with a safety margin.

### Purpose

Independent reagent bounds make high-volume reagent usage more likely than the old equal-split approach. Source depletion is therefore one of the next likely practical risks when scaling.

---

## 4. Gradual Scale-Up Plan

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

## 5. Later Model/Chemistry Improvements

Later improvements may include:

- duplicate-aware model aggregation,
- better replicate handling,
- improved lambda-max extraction,
- edge-maximum detection and rejection,
- spectral smoothing or preprocessing,
- acquisition-function alternatives beyond pure target-distance,
- batch candidate selection,
- deck-volume-aware optimizer penalties,
- transfer/tip audit logging.

---

# 🧭 Validation Summary

## Current status

The Auto mode branch has passed the main optimizer/protocol validation milestone for mixed mask optimization and volume-safe recipe generation.

## Validated

- true-zero variable reagent handling,
- mixed mask optimizer path,
- independent reagent bounds,
- multi-fixed-reagent volume handling,
- water top-off logic,
- `200 µL` well-volume preservation,
- dry/debug protocol execution,
- water-only physical liquid-handling volume validation.

## Deferred

- water-used tip behavior in `ot2_robot.py`,
- transfer/tip audit CSV,
- deck/source-volume hard-stop,
- real-chemistry spectral validation,
- larger autonomous scale-up.

## Recommended immediate next step

Run a small real-chemistry validation using the same conservative settings, then review:

- spectra,
- replicate consistency,
- mask selection,
- debug CSV volume fields,

before scaling.

---

# End of Auto Mode Validation Notes
