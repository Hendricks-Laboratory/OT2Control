# Auto Mode Validation Notes

**Repository:** Hendricks-Laboratory / OT2Control  
**Branch context:** Historical stable baseline plus active Auto-RTG development record
**Stable branch:** `Auto-RTG-v1`
**Active development branch:** `Auto-RTG`
**Prepared for:** Branch-local documentation / validation notes  
**Originally prepared:** 2026-06-08  
**Updated through:** 2026-07-29

> [!NOTE]
> **Authorship tagging.** Entries marked **[Claude Code]** were implemented by
> Claude Code (Opus 5) working in this repository under the project owner's
> direction and review. Untagged entries predate that convention or were
> authored directly by the project owner. The tag records provenance only: a
> tagged change passed the same Python 3.9 compilation, isolated hardware-free
> test, and human-review gates as any other change, and no tagged change has
> physical-run clearance on the strength of this record alone.

---

## Purpose

This note documents the **stable v1 Auto mode implementation and validation status** based on the branch history, dry/debug output, water-only testing, RTG_008 real-reagent validation, later RTG_009/DEBUGRTG output validation, replicate-QC and report work, lifecycle-coordinated plotting, cumulative GP-history repair, corrected 2D heatmap orientation, and the July 13, 2026 stable-checkpoint debug run and post-fix design-space plot validation.

It is intended as a branch-specific record of:

- what has been implemented,
- what has been validated,
- what design decisions were made,
- what remains deferred,
- and what the next development steps should be.

> [!NOTE]  
> This note is **not** intended to describe the whole shared `OT2Control` repository.  
> It specifically documents Auto mode branch work around autonomous reaction selection, true-zero reagent handling, volume feasibility, mixed mask optimization, cumulative GP model history, model-performance logging, lifecycle-aware plotting, debug exports, terminal-output capture, reporting, and protocol validation.

---

# Current Auto-RTG Development Record — Complete Reconciliation, July 21, 2026

> [!IMPORTANT]
> The older sections below preserve historical `Auto-RTG-v1` validation
> evidence. This section is authoritative for active work on `Auto-RTG`.
> `main` and `Auto-RTG-v1` remain protected; development stays on `Auto-RTG`.
> Statements in the historical sections that describe acquisition modes,
> source-volume preflight, or other features as deferred are not current
> `Auto-RTG` status.

## Reconciliation scope and code identity

This current record was reconciled against the complete repository range from
the pre-Auto-RTG baseline branch to the active branch, rather than against one
recent feature commit only.

| Item | Reconciled value |
|---|---|
| Baseline branch and commit | `Auto` at `356971a` (`Last TODO`) |
| Initial reconciliation snapshot | `Auto-RTG` at `b152ed1` (`Update Auto-RTG validation record for current report features`) |
| Latest validated working tree | `Auto-RTG` at `5428191` plus the current uncommitted pairwise-legend, standalone-slice layout, mask-aware true-zero feasibility-overlay refinement, and recipe-concentration-history plotting work |
| Relationship | `Auto` is an ancestor of `Auto-RTG`; the merge base is `356971a` |
| Reviewed range | `Auto...Auto-RTG`, containing 174 commits |
| Files changed in the range | `controller.py`, `optimizers.py`, `ot2_robot.py`, Auto tests, validation/governance documents, `.gitignore`, and removal of an ignored local `.DS_Store` artifact |
| Current validation result | Python 3.9.6 compilation passed; 179 isolated hardware-free tests passed on July 27, 2026 |

The reconciliation inspected current controller, optimizer, robot-container,
plot/report, and test code as well as the accumulated Git history. It did not
run a controller launcher, robot, plate reader, credential workflow, or live
protocol. Therefore it confirms code/documentation coverage and isolated
behavior, not physical-run clearance.

## Current branch purpose

`Auto-RTG` is a target-seeking, physically constrained Bayesian-optimization
workflow. It proposes robot-executable formulations near a requested λmax
target while retaining condition-level replicate QC, cumulative GP history,
true-zero masks, water/overflow feasibility, and controller-owned stopping.

The optimizer and protocol path accept a general number of variable reagents.
Two-variable GP-surface visualizations remain direct physical concentration
maps. The established three-variable conditional slice atlases are retained.
For four or more variable reagents, Auto-RTG now generates every two-reagent
conditional GP slice, freezing all remaining reagents at one shared
QC-aware reference recipe (the best QC-approved observed condition when
available). These are paginated into readable atlases and saved individually,
alongside mean, uncertainty, target-probability, and feasibility-overlay
variants. Higher-dimensional plots are conditional views, projections, or
summaries; they do not claim that a two-dimensional image fully represents the
corresponding chemistry space.

The active branch also supports ordered acquisition portfolios. A `core3`
batch compares `exploit`, `explore`, and `balanced` selections from the same
current QC-approved cumulative GP model; the modes do not maintain isolated
model histories. Each selected condition is then measured with its configured
duplicate count and returns through the usual QC/model-update pathway.

## Implemented active-branch capabilities

| Capability | Current status | Validation scope |
|---|---|---|
| Constraint-aware volume-feasible maximin initial design, cumulative GP history, and post-update plot refresh | Implemented | All-ON seed pools are directly sampled from the executable transfer-volume simplex; true-zero and restrictive custom-bound cases retain the mask-aware feasibility-filtered path. Synthetic five-variable volume, final-guard, cumulative-history, and timing regression tests; controlled debug-output review |
| Physical recipe constraints: overflow, water top-off, 5 uL executable-transfer bounds, mixed masks, and all-off exclusion | Implemented | Isolated feasibility, mask, and controller-handoff tests |
| `exploit`, `explore`, `balanced`, and `target_ei` acquisition modes | Implemented | Synthetic scoring and controller/optimizer integration tests; dry-debug review |
| Selective true-zero variable-reagent masks | Implemented | Header normalization, controller-bound/repair, mask, and feasibility-overlay regression tests |
| Ordered `acquisition_modes` portfolios and `core3` | Implemented | Synthetic portfolio/controller handoff tests |
| QC-approved condition-level target-EI incumbent | Implemented | Synthetic incumbent and stop-eligibility tests |
| Friendly `portfolio_min_distance` inputs plus numeric values | Implemented | Header normalization tests |
| Condition-level replicate QC, GP-training eligibility, and controller-owned target stopping | Implemented | Isolated condition/QC/stop-eligibility regression tests; saved-run review |
| Configurable condition target tolerance and replicate-SD target-decision gate | Implemented | Header and target-decision regression tests |
| Two-variable GP mean/uncertainty heatmaps and matching mask-aware feasibility overlays | Implemented | Static/synthetic orientation, continuous 0–5 uL exclusion, exact-zero edge, all-off exclusion, and rendering-semantics tests |
| Three-variable mean, uncertainty, target-probability, and mask-aware feasibility slice atlases | Implemented | Static/synthetic conditional-slice, true-zero/mask feasibility, and tolerance-propagation tests; controlled debug-output review |
| Four-or-more-variable all-pair conditional GP slice atlases and standalone slices | Implemented | Synthetic four-variable pair/held-recipe, true-zero/mask feasibility, rendering/path validation, compact atlas-layout regression checks, and full hardware-free test suite |
| Dimension-aware design-space plots and portfolio trace with mode-specific markers | Implemented | Isolated plot-classification and marker/error-bar legend tests, plus collision-aware condition-label and compact-header layout checks |
| Categorized Auto plot folders and plot manifest | Implemented in `cf2fcb9` | Python 3.9 compilation and isolated path tests |
| Condition-level recipe-concentration history plots | Implemented in current working tree | Variable-only and complete-recipe grouped bar charts use executed final concentrations in mM, write under `Plots/recipe_history`, obey plot profiles, append to the plot manifest, and appear in the final report; isolated lifecycle, report, data-contract, and rendering tests |
| Controller-side source-volume preflight and reserve volume | Implemented | Isolated fail-closed preflight tests; needs run-specific source-inventory review |
| Corrected tube-tare defaults and Raspberry Pi legacy tare compatibility offset | Implemented | Source review and Header/payload compatibility tests; physical weighing remains human-verified |
| Terminal verbosity and lifecycle progress messages | Implemented | Header normalization and source-level lifecycle review |
| Portable Auto model checkpoint packages (`save` and `import` modes) | Implemented | Hardware-free JSON/NumPy archive round-trip, integrity, manual inbox and prior-run source selection, immutable lineage copy/provenance, compatibility, fresh reconstruction, Header, and save-boundary tests |
| Stage 1 controller-local live-run journal | Implemented; dry debug pending | For real configured Auto models only, writes immutable parsed input/Header/runtime snapshots, a SHA-256 manifest, atomic current state, and append-only JSONL lifecycle events before connection and at normal batch/finalization boundaries. It is local-only and fail-closed; it does not add cloud synchronization, Pi state, recovery prompts, or change scientific/model behavior. Python 3.9 contract, journal durability, failure-path, and static controller-placement tests passed. |
| Auto output-directory collision isolation | Implemented; dry debug pending | Before an Auto output directory is created, an existing Header `data_dir` proposes the first unused `_N` sibling and requires an interactive exact `yes`; noninteractive use fails closed. The selected effective directory is recorded in the Stage 1 runtime baseline and final report. Legacy non-Auto output behavior is unchanged. Isolated resolver and source-placement tests passed. |
| Import-only run-context lineage, tabular snapshots, final cross-run plots, and reporting | Implemented | Existing-output imports flatten and checksum-identify ancestor runs without nesting; they export de-duplicated native condition and well-level replicate CSV snapshots plus source-availability diagnostics. Final import-only plots provide separate current-run-only and cumulative-lineage λmax progress and replicate views from those flat snapshots, with run identity and provenance-aware semantics. The final report describes lineage scope, source availability, condition/replicate row counts, raw-scan deferral, and links the generated cross-run figures. Manual checkpoint imports remain model-only. Raw scan ingestion remains deferred. Python 3.9 isolated lineage, branching, duplicate-conflict, legacy, tabular-history, cross-run rendering, and report tests passed. |
| Shared seed/optimizer λmax extraction | Implemented | The seed batch and every later optimizer or imported-continuation batch use one class-scoped blank-correction and scan-quality helper; isolated regression test prevents a seed-local helper scope failure |
| SciPy boundary-status recovery | Implemented | Deterministic optimizer recovery test |
| Current-controller completion report marker | Implemented | Saved-log regression test; a normal Auto completion is no longer reported as `Unknown` |
| Configured target-stop, acquisition, and portfolio report provenance | Implemented | Hardware-free report regression test with an early-stop condition and physical replicate-well locations |
| Labeled raw well-level Auto export | Implemented | Hardware-free export regression test; exported columns identify final-reaction concentration in mM and measured λmax in nm without changing internal training data |
| Reader-oriented report appendix and plate-reuse guidance | Implemented | Hardware-free report regression tests for appendix ordering, physical well span, remaining sequential capacity, and next-well recommendation |
| Objective UV scan-quality diagnostics | Implemented, warning-only by default | Records blank-corrected peak height and 300/1000 nm boundary maxima; legacy worksheets retain warning-only behavior |
| Boundary-aware exact-λmax routing | Implemented, opt-in | `boundary_aware` excludes only exact 300/1000 nm maxima from exact-λmax QC/training and target decisions while preserving raw outcomes |
| Usable-spectrum probability classifier and conditional/joint maps | Implemented, observational | Cumulative binary GPy classifier learns interior versus exact-boundary outcomes; it does not yet affect acquisition or stopping |
| Imported-continuation reaction and batch numbering | Implemented **[Claude Code]** in `248d95d` | A resumed run continues numbering above its imported history instead of restarting at zero; isolated tests assert no duplicate batch/reaction keys and cover the exact key-name defect that caused the reset |
| Per-row Auto run provenance (`executed_in_current_run`, `origin_run_directory`) | Implemented **[Claude Code]** in `248d95d` | Distinguishes conditions this run physically executed from inherited checkpoint history, and survives multi-generation imports; physical-well reporting is scoped to locally executed rows |
| Inherited seed-design figure and report labeling | Implemented **[Claude Code]** in `248d95d` | An imported run titles seed figures and report headings `Inherited Seed Design (from <run>)`, naming the run that built the seed rather than the immediate import source |
| Auto design-space gridline styling | Implemented **[Claude Code]** in `4ae04b1` | Green dashed gridlines drawn behind plotted points across every design-space dimensionality, including the 3D pane grid, which ignores ordinary Matplotlib grid keyword arguments |
| Pairwise/parallel label and higher-dimensional atlas layout | Implemented | Pairwise and parallel-coordinate condition labels move text only through deterministic non-overlapping positions while plotted recipes remain exact. Shared title/legend headers and higher-dimensional atlas spacing are compacted without changing 300-DPI export or GP/feasibility grid density. Standalone slices retain their centered axis-label/colorbar group and compact annotation band. Source-level layout regression tests and synthetic Matplotlib rendering passed. |

### Current acquisition semantics

Every candidate first passes the existing mask and physical-feasibility route.
The remaining selection score is minimized as follows:

| Mode | Minimized score |
|---|---|
| `exploit` | `(predicted_mean_nm - target_nm)^2` |
| `explore` | `-predicted_standard_deviation_nm` |
| `balanced` | `abs(predicted_mean_nm - target_nm) - weight * predicted_standard_deviation_nm` |
| `target_ei` | negative expected improvement in QC-approved condition-level target error |

`target_ei` is not ordinary expected improvement on raw λmax, and its
incumbent is never set by one favorable replicate. The all-off mask remains
excluded; ON variable reagents must be at least 5 µL; OFF reagents are exactly
zero; water top-off is either zero or at least 5 µL; overflow is infeasible.

When `allow_true_zero` is enabled, the optional Header
`true_zero_reagents` setting selects which variable reagents may be exactly
zero. It accepts comma- or semicolon-separated canonical variable-reagent
names case-insensitively, plus `all` and `none` aliases. Unselected variable
reagents remain required ON and retain their 5 µL-equivalent lower bound.
The all-off recipe remains excluded. A missing list preserves legacy behavior:
all variable reagents are eligible when `allow_true_zero` is true, and none
are eligible when it is false. Unknown, duplicate, fixed-reagent, or
contradictory Header values fail before Auto execution.

### Boundary-aware spectral-response foundation — July 23, 2026

The optional Header `auto_spectral_response_policy` now provides an explicit,
backward-compatible distinction between an exact interior λmax observation and
a scan-boundary-censored outcome:

| Policy | Behavior |
|---|---|
| `audit_only` (default) | Retains historic behavior: exact 300/1000 nm maxima remain finite λmax values for QC, primary-GP training, incumbents, and stopping, while their boundary status is audited. |
| `boundary_aware` | Treats only an extracted maximum exactly at 300 or 1000 nm as censored. It remains in raw well/condition audit output, but is excluded from exact-λmax QC, primary-GP training, target-EI incumbents, and target stopping. No arbitrary low-signal or near-edge cutoff is used. |

The controller records an immutable per-replicate observation type
(`interior_peak`, `lower_scan_censored`, `upper_scan_censored`, or
`unknown_scan_quality`), exact-λmax eligibility and reason, and
future usable-spectrum-model eligibility in `auto_model_performance_log.csv`.
The report summarizes those counts and identifies the active policy.

An interior pair plus one censored replicate can train the primary λmax GP and
can support a target decision if it passes the existing two-replicate and
replicate-SD gates. A fully censored later optimizer batch leaves primary GP
history unchanged but is counted as a completed physical batch, avoiding
history corruption or an unbounded retry loop. A fully censored initial seed
batch fails clearly because no exact λmax GP can be initialized.

Validation on July 23 used `/usr/bin/python3` version 3.9.6: `py_compile`
passed for `controller.py`, `optimizers.py`, and the focused test module;
130 focused isolated tests and 134 total isolated tests passed. The focused
Stage 3 tests also verified fresh cumulative binary classifier reconstruction,
finite probability output, binary-history validation, and controller replay of
interior/censored replicate labels. No controller launcher, robot, plate
reader, credential workflow, or live protocol ran.

The local review environment now contains `GPy 1.13.2` with its compatible
NumPy/SciPy requirements. `GPy.models.GPClassification` returns finite
probabilities in `[0, 1]` when constructed from cumulative binary
observations. Its returned variance is not finite in this environment and is
therefore never used. Additionally, calling `set_XY()` after changing the
observation count fails in GPy's EP inference implementation. Auto-RTG
therefore rebuilds a fresh classifier from the complete cumulative assessed
replicate history after each batch rather than mutating it in place.

Under `boundary_aware`, each objectively assessed replicate contributes one
binary outcome: `interior_peak` is usable and exact 300/1000 nm boundary
maxima are not usable. Unknown scan quality is retained in audit output but
does not become a guessed binary failure. The companion classifier is passive:
it does not change masks, transfer bounds, physical feasibility, primary
lambda-GP training, target-EI incumbents, early stopping, or acquisition.

Existing conditional mean and GP-SD maps are now explicitly interpreted as
`lambda max | interpretable spectrum`. Separate maps show
`P(interpretable interior spectrum)` and the joint quantity
`P(interpretable spectrum) × P(target window | interpretable spectrum)`.
Optical reliability is never shown with the gray physical-infeasibility
overlay, because a recipe can be physically executable yet optically
unreliable. Reliability-aware acquisition remains a separately approved,
future Stage 5 policy decision.

### Imported-continuation numbering, run provenance, and seed labeling — July 24, 2026 **[Claude Code]**

Commits `4ae04b1` (plot styling) and `248d95d` (numbering, provenance, seed
labeling). This work began as a read-only audit of a deliberately paired debug
run set supplied by the project owner: `RTG_debuggingsave2` ran
`auto_model_checkpoint_mode = save`, and `RTG_debuggingimport` then resumed
from that run's `model_final.zip` through the prior-run import route. Only
`data_dir` and the checkpoint mode differed between the two worksheets.

#### What the audit confirmed as already correct

Checkpoint save/import behaved as designed. Packages were written at all three
boundaries, contained only checksummed JSON/NumPy payloads, and the resumed run
skipped its seed design, archived the source immutably, and wrote
`import_provenance.json`. Cumulative GP history was preserved exactly across
the import boundary: ten imported observations plus seven new QC-included
replicate rows produced seventeen, with no regression to seed-plus-newest.
Volume invariants held exactly, with water pinned near its 5 µL floor and the
variable total capped at 195 µL. Acquisition scores reproduced their documented
formulas to full precision.

#### Defect 1 — condition counter read a key the controller never writes

The import restore path computed its starting condition number from
`row.get('condition_number')`, but performance rows use `reaction_number`. The
lookup therefore always found nothing and reset the counter to zero instead of
continuing above the imported history.

This survived review because the corresponding test fixture invented the same
`condition_number` key. Production and test agreed with each other and both
disagreed with the real row schema. The fixture now mirrors the keys the
controller actually writes, and a regression test asserts `condition_number` is
not honored for numbering.

#### Defect 2 — batch counter never advanced past the imported history

`_run()` resets `batch_num` to zero for every protocol, and the seed path's
"batch 0 is the seed" increment sits after the import early-return, so it never
executed on the continuation route. The resumed batch was therefore labeled
batch 0, colliding with the imported seed batch and producing
`model_after_batch_000.zip`.

Both counters now derive from a shared `_get_next_auto_number_after_rows`
helper that ignores missing, non-numeric, and boolean values, so a partially
populated legacy row cannot pull numbering back onto existing keys. Against the
supplied checkpoint the condition counter continues at 4 and the first
continuation batch is 2.

#### Downstream defects repaired by the numbering fix

| Artifact | Prior behavior | Current behavior |
|---|---|---|
| `lambda_progress_final.png` | The `batch_number <= N` window silently dropped three of seven conditions, and two more overlapped at one x-position | All seven conditions render at distinct positions |
| `acquisition_portfolio_trace_final.png` | The inherited seed marker was overplotted by a new selection sharing its condition number | Seed and each portfolio mode occupy their own condition |
| `auto_design_space_exploration_*.png` | Duplicate condition annotations collided and the best-condition star was drawn on two unrelated recipes | Unique annotations and exactly one best-condition marker |
| Checkpoint and GP-surface filenames | A resumed run reused source-run batch numbers | Numbering continues across the lineage |

#### Per-row run provenance

Performance rows now carry two audit fields. `executed_in_current_run`
separates conditions this run physically executed from inherited checkpoint
history. `origin_run_directory` records the output directory of the run that
produced the condition; the run directory is used rather than the experiment
name because a save/import pair commonly shares one worksheet name.

On import the executed marker is forced to false rather than trusted, so a
second-generation import cannot inherit a stale true, while
`origin_run_directory` is preserved rather than overwritten. A row lacking the
stamp is backfilled from the immediate source. Rows written before these fields
existed are treated as locally executed, so older runs and older checkpoint
packages report unchanged.

Physical-well reporting is now scoped to locally executed rows. Previously the
report unioned the source run's well locations with the resumed run's, claiming
twelve wells spanning A1 through D2 for a run that used nine wells spanning A1
through A2, and advancing the same-plate reuse recommendation to `E2` instead
of `B2`. An imported continuation always starts a fresh plate.

#### Inherited seed-design labeling

An imported run performs no seed design, yet emitted figures titled
`Initial Maximin Seed Design` containing the source run's seed condition. Seed
figures and their report headings now read
`Inherited Seed Design (from <run>)`.

The named run is resolved from the seed row's own `origin_run_directory` stamp
rather than from this run's immediate import source, so a lineage of A imported
into B imported into C still attributes the seed to A. Fallbacks cover
pre-provenance packages (immediate source folder), the manual inbox route
(source run id), and finally a generic label. Titles route through one helper
for figures and one for report headings, so the default wording remains
declared at each call site. Filenames are deliberately unchanged, because plot
path routing and report lookups key on the
`initial_maximin_seed_design_` prefix.

#### Plot styling

Design-space gridlines were previously drawn in Matplotlib's default light grey
at 35 percent alpha and were effectively invisible. They are now green dashed
lines drawn behind plotted data. The three-dimensional case required a separate
mechanism: `Axes3D.grid()` only toggles visibility and silently ignores styling
keyword arguments, so pane grid appearance is set through each axis's
`_axinfo['grid']` entry.

#### Validation

Python 3.9.6 `py_compile` passed for `controller.py` and `optimizers.py`. The
full isolated hardware-free suite passed at 168 tests, up from 150 before this
work, including two new classes covering continuation numbering, well-span
scoping, and inherited seed labeling across single and multi-generation
imports.

Behavior was additionally checked by extracting the real controller plotting
and provenance methods from source and re-rendering the supplied
`RTG_debuggingimport` artifacts under both the as-executed and corrected
numbering, then comparing the outputs. A simulated third-generation import
confirmed that seed attribution survives more than one hop. No controller
launcher, robot, plate reader, credential workflow, or live protocol ran.

#### Reagent-free debug-run caveat

The two supplied runs were executed with no reagents loaded, which the project
owner confirmed was intentional. Every extracted λmax in both runs is therefore
an artifact of subtracting the hardcoded blank reference from an essentially
non-absorbing well: raw absorbance was flat within roughly ±0.003 AU, every
blank-corrected peak height was negative, and the reported maxima track the
minimum of the hardcoded blank array near 683 nm together with a shallow
secondary region near 824 nm. Replicate SD of exactly 0.0000 nm follows from
that determinism rather than from measurement agreement.

This is expected for a plumbing run and is not a defect. It is recorded here so
these two runs are read as mechanism validation only. Their checkpoints must
not be reused as a scientific prior, and the λmax values in their logs and
reports carry no chemical meaning.

#### Deliberately unchanged

Report condition counts still describe the model's full history rather than
this run's physical execution, and the exploration plot does not visually
distinguish inherited from newly executed optimizer conditions. Both now have
the provenance fields available should the project owner want them scoped.
Scan-quality gating remains warning-only, so a condition whose spectrum carries
no real peak can still pass QC; that remains the previously recorded deferred
issue rather than something addressed here.

### Current spreadsheet Header interface

Header rows are read by key, so their physical row order does not matter.
Every setting below is resolved before Auto execution. New optional settings
retain the stated legacy defaults when their Header row is absent.

| Header key | Current behavior and legacy default |
|---|---|
| `using_temp_ctrl`, `temp` | Existing temperature-module settings. `using_temp_ctrl` must be `yes` to enable temperature control; `temp` is then required and constrained to 4–95 °C. |
| `data_dir` | Existing output-directory identifier used by the Auto workflow. |
| `dilution_cont`, `dilution_vol` | Existing dilution-container and dilution-volume settings. |
| `target` | Required λmax target in nm. |
| `initial_data`, `max_iterations` | Required seed-condition count and maximum optimizer-batch count. |
| `num_duplicates` | Physical replicate wells per selected condition; defaults to `3`; must be at least `1`. In a portfolio, this count applies independently to each listed acquisition mode. |
| `target_tolerance_nm` | Maximum absolute condition-mean target error eligible for an early stop; defaults to `10` nm. It does not bypass QC or replicate-agreement requirements. |
| `replicate_sd_tolerance_nm` | Maximum sample SD across QC-included replicates for a condition to establish a target-EI incumbent or an early stop; defaults to `25` nm. It gates target decisions, not normal model-training eligibility. |
| `allow_true_zero` | Enables true-zero search. A missing row remains `false`, preserving legacy all-ON optimization. |
| `true_zero_reagents` | Optional selective true-zero list used only when `allow_true_zero` is enabled. It accepts case-insensitive comma- or semicolon-separated variable-reagent names, plus `all` and `none`. A missing list preserves legacy all-or-none true-zero behavior. |
| `acquisition_mode` | Singular interface: `exploit`, `explore`, `balanced`, or `target_ei`; missing defaults to `exploit`. `target_ei` requires at least two duplicates so a replicate-agreement decision is defined. When a portfolio is active, set this field explicitly to `off`. |
| `acquisition_modes` | Optional ordered semicolon-separated portfolio with no duplicate canonical modes. `core3` expands to `exploit;explore;balanced`. `off` disables the portfolio. The listed order is a preference order only when selections collide within the configured diversity radius; all modes otherwise select from the same pre-batch cumulative GP. A portfolio containing `target_ei` also requires at least two duplicates. |
| `portfolio_min_distance` | Portfolio diversity radius in normalized design space. Friendly values include `none` = `0.00`, `modest` = `0.05`, `strong` = `0.10`, and `very_strong` = `0.15`; finite numeric input from `0` through `1` is also accepted. It is inactive when `acquisition_modes` is off. |
| `auto_plot_profile` | `standard`, `final_only`, or `off`; missing defaults to `standard`. It controls automatic diagnostic/final plots, not model fitting, acquisition, QC, or execution. |
| `auto_terminal_verbosity` | `essential`, `standard`, or `diagnostic`; missing defaults to `standard`. `off` means essential safety/scientific output, not silence; `limited` maps to standard; `all` maps to diagnostic. Persistent CSV/report audit output is unaffected. |
| `auto_model_checkpoint_mode` | `off` (legacy default), `save`, or `import`. `save` exports immutable JSON/NumPy packages after the seed GP fit, every completed optimizer batch, and finalization to `Model_Checkpoints/`. For `import`, before completing the reagent sheet choose either `manual` (place exactly one compatible package in this run's `Model_Checkpoints/Import_Here/`) or `run` (enter an exact prior `Protocol_Outputs` folder name such as `RTG_020`, then select `final`, `seed`, `batch N`, or a listed package filename). Prior-run selection is restricted to direct output-run children and canonical checkpoint files; arbitrary paths are rejected. Auto checksum-validates and archive-copies the source, writes `import_provenance.json`, then rebuilds a fresh model from numeric cumulative history after confirming current chemistry, normalized bounds, and spectral-response policy compatibility. The imported model skips a new seed design; `max_iterations` applies to new batches only. |
| `auto_source_volume_check` | `off` (legacy default) or `required`. `required` performs a fail-closed aggregate source-inventory preflight before each batch. |
| `auto_source_reserve_volume_uL` | Nonnegative additional source reserve beyond the robot's dead-volume calculation; defaults to `0`. It matters only when source-volume checking is required. |
| `pi_legacy_tare_offset_g` | Nonnegative payload-only compatibility offset for a deployed Raspberry Pi that still uses the old tare constants; defaults to `0`. Do not enable after the Pi has the corrected constants. |
| `auto_spectral_response_policy` | `audit_only` (default; preserves legacy finite-boundary treatment) or `boundary_aware` (censors only exact 300/1000 nm maxima from exact-λmax routing). Aliases `audit`, `boundary`, and `censored` are accepted. |

The singular and portfolio acquisition interfaces are deliberately mutually
explicit. A new portfolio worksheet must put `off` in `acquisition_mode`; a
legacy worksheet lacking `acquisition_modes` remains singular and defaults to
`exploit`. This prevents a spreadsheet from silently mixing two selection
interfaces.

### Tare correction and Raspberry Pi compatibility

`Auto-RTG` contains two distinct +0.3 g tare mechanisms that must not be
confused:

1. `ot2_robot.py` corrects the default tare constants used by the repository's
   2 mL, 15 mL,
   and 50 mL tube models: 1.4 → 1.7 g, 6.9731 → 7.2731 g, and 13.3950 →
   13.6950 g, respectively. This is the corrected baseline for a runtime that
   actually receives the current `ot2_robot.py`.
2. `pi_legacy_tare_offset_g` is a controller-side compatibility shim. It
   subtracts an operator-selected positive offset only from the reagent mass
   payload sent to a Raspberry Pi still running old, lower tube tares, while
   preserving the real measured tube-plus-solution mass in controller records.

The first change does not alter a frozen remote Pi by itself. The second is
therefore needed only while that remote runtime remains on the old constants.
Neither mechanism substitutes for physical tare verification.

### Current report provenance

`auto_run_report.md` records the configured target tolerance and replicate-SD
tolerance alongside the condition-level QC stopping rule. When terminal output
records a validated target stop, the report identifies the triggering batch,
condition, QC-cleaned mean, target error, replicate SD, and physical replicate
well locations captured in `auto_model_performance_log.csv`. It also defines
each active acquisition mode and explains the ordered portfolio's normalized
RMS diversity radius. These report fields are audit metadata only; they do not
alter recipe selection, QC, GP training, or robot execution.

The report keeps compact reader-facing results, warnings, and conclusion ahead
of the complete machine-oriented audit appendix. The appendix remains in the
same Markdown file after the conclusion so Markdown-aware viewers can expand
full recipe, mask, optimizer, QC, and volume-balance records, while Google
Drive plain-text preview reaches the scientific narrative first.

Experiment Overview also derives the physical plate-well span from recorded
condition-level replicate-well locations. It reports the first and final wells
in controller plate order, unique physical-well count, remaining sequential
well capacity, and the recommended next plate-reader starting well for
same-plate reuse. This is audit-derived guidance, not a capacity prediction;
the operator must still confirm that no other wells were used before reusing
the plate.

`experiment_data.csv` remains the physical-well-level export. Its Auto
variable-reagent columns are labeled as final-reaction concentrations in mM
and its response column as λmax in nm. These labels clarify exported physical
units without changing internal normalized GP coordinates, raw data, model
training, recipe selection, or execution.

### July 21, 2026 — Selective true-zero and report-usability follow-up

The active branch incorporated selective true-zero masks, explicit raw-data
column labels, and report usability/provenance improvements in:

```text
7567290  Add selective true-zero controls and label Auto experiment exports
2678599  Improve Auto report readability and plate reuse guidance
```

The source-level validation at the time of this follow-up covered Header parsing,
selective mask generation, controller transfer validation, 2D feasibility
overlay semantics, legacy all-or-none true-zero compatibility, labeled raw
export columns, appendix placement after the conclusion, and plate-span/reuse
guidance. That point-in-time hardware-free suite result was 118 passing tests
after compilation using the available local runtime. The superseding current
reconciliation result is 121 passing tests and is recorded above.
No controller launcher, robot, plate reader, credentials, or live protocol
path was invoked. These results validate code behavior and audit output only;
they do not substitute for a supervised physical validation of selective
true-zero chemistry or same-plate reuse.

### Current plotting/output behavior

New controller-generated Auto plots use this categorized layout. Existing
root-level plot locations remain readable through report fallback logic.

```text
Plots/
  progress/
  design_space/
  gp_surfaces/
    2d/
      mean/
        atlases/
      uncertainty/
        atlases/
      {field}/feasibility_overlays/
        atlases/
    3d/
      {mean, uncertainty, target_probability}/
        atlases/
        conditional_slices/
        feasibility_overlays/
          atlases/
          conditional_slices/
    {4d-and-higher}/
      {mean, uncertainty, target_probability}/
        atlases/
        conditional_slices/
        feasibility_overlays/
          atlases/
          conditional_slices/
  auto_plot_manifest.csv
```

All GP-surface artifacts now use the same directory vocabulary. An `atlases`
folder contains the complete surface for that field (a single 2D map, a 3D
three-slice atlas, or a paginated higher-dimensional atlas); a
`conditional_slices` folder contains individual held-recipe views; and
`feasibility_overlays` mirrors the same distinction for physical-constraint
diagnostics. Existing filenames remain stable for report compatibility.

For each three-variable plot stage, the five established multi-panel slice
atlases are preserved. Fifteen standalone conditional slices are added: one
per held reagent for mean, uncertainty, target probability, mean-feasibility,
and uncertainty-feasibility. These are renderings of the same already-computed
conditional GP panel data; they do not change model fitting, acquisition,
recipe generation, or robot execution. They add expected rendering time.

### Dimension-aware plotting scope

| Variable-reagent count | Current Auto plot behavior | Interpretation boundary |
|---:|---|---|
| 0 | No design-space plot; a warning explains that no variable concentration columns were found. | No chemistry-space visualization is possible. |
| 1 | Seed and full 1D design-space strip plots. | Direct physical concentration axis. |
| 2 | Square seed/full design-space plots; GP mean and uncertainty heatmaps; separate feasibility-overlay versions of those heatmaps. | Both axes are physical reagent concentrations. The original and overlay figures remain distinct artifacts. |
| 3 | Pairwise and cubic seed/full design-space plots; conditional mean, uncertainty, target-probability, and feasibility slice atlases plus standalone slices. | Each GP slice holds one reagent at a stated physical concentration while displaying the other two. It is a conditional view, not a full 3D response surface. |
| 4–6 | Complete pairwise-projection and parallel-coordinate seed/full design-space plots, plus every two-reagent conditional GP slice as paginated atlases and standalone views. | Each slice holds all non-displayed reagents at one stated shared QC-aware reference recipe; projections and slices do not show a complete multidimensional response surface. |
| 7+ | Compact pairwise, parallel-coordinate, and two-component PCA seed/full design-space plots, plus paginated all-pair conditional GP slice atlases and standalone views. | PCA axes are display coordinates rather than physical reagent axes. Conditional-slice count grows as D choose 2, so plot generation time and output volume increase with dimension. |

Portfolio runs also produce an acquisition trace alongside the ordinary
progress and replicate plots. The trace uses mode-specific markers for
`exploit`, `explore`, and `balanced`, distinguishes filled QC-included observed
condition means from hollow GP predictions, and labels observed SEM separately
from predictive GP SD. Its data are common-model portfolio diagnostics; it
does not create separate GPs or alter the ordinary cumulative plots.

## Current hardware-free validation evidence

```text
Baseline: Auto at 356971a
Active commit: Auto-RTG at b152ed1
Reviewed range: Auto...Auto-RTG (174 commits)
```

The reconciled current source passed the following hardware-free validation on
Python 3.9.6:

```text
PYTHONPYCACHEPREFIX=/tmp/ot2control_pycache /usr/bin/python3 -m py_compile \
  controller.py optimizers.py ot2_robot.py tests/test_acquisition_scoring.py \
  tests/test_auto_plot_organization.py
/usr/bin/python3 -m unittest -v tests.test_auto_plot_organization \
  tests.test_acquisition_scoring
```

Result: 121 isolated tests passed. Source-level AST inspection also found no
duplicate class-method definitions in `controller.py`, `optimizers.py`, or
`ot2_robot.py`. No controller launcher, robot, plate reader, credentials, or
live protocol path was invoked. These results validate isolated code behavior
and audit output only; they do not establish physical readiness or replace a
human-supervised dry/debug or chemistry review.

`git diff --check Auto...Auto-RTG` still reports accumulated trailing
whitespace in historical legacy code and Markdown hard-break lines. That is a
nonfunctional formatting debt in the full branch range, not a failed
controller/optimizer validation. It should be addressed only in a separate
formatting-only review, not mixed into scientific or robot-control work.

### RTG_014 output-audit follow-up — July 17, 2026

The saved RTG_014 three-variable chemistry output was reviewed after it
stopped on a validated condition-level target hit. The categorized plot files
and manifest were produced correctly, but two presentation/report defects were
identified:

- standalone conditional-slice titles and legends could be clipped because
  their compact canvas did not accommodate long reagent and feasibility labels;
- the Markdown report still hard-coded root-level final progress/replicate
  paths, causing broken embeds and false `not found` entries despite the files
  existing in `Plots/progress/`.

The subsequent active-branch follow-up gives individual slices a wrapped title and
dedicated legend band, routes final progress/replicate report links and file
status through the categorized-path resolver, and records an explicit
condition-level-target-hit exit reason. It does not change chemistry,
liquid-handling, GP fitting, QC, acquisition, or stopping behavior. This
follow-up requires a later controlled output review to validate the rendered
standalone figures in the lab environment.

### Spreadsheet-configurable replicate agreement — July 20, 2026

`replicate_sd_tolerance_nm` is an optional Header setting for the maximum
sample SD across QC-included replicates of a single condition when deciding
whether that condition may establish a target-EI incumbent or stop an Auto
run. A missing value preserves the established 25 nm default. This is a
replicate-SD threshold, not an SEM threshold: more replicates can lower SEM
without making their individual λmax values agree more closely.

The setting does not change replicate outlier exclusion, model-training
eligibility, GP fitting, acquisition scoring, or liquid handling. It only
controls the stricter target-decision gate already applied after QC. For
example, a value of `15` nm rejects a condition with a QC-included replicate
SD above 15 nm as a target-EI incumbent and early-stop candidate, while its
approved observations may still contribute to GP training.

## Recommended roadmap from the current state

### 1. Next controlled validation: short three-variable output audit

Before adding further visualization or acquisition features, run a small,
human-supervised three-variable dry/debug or water-only workflow using the
current `Auto-RTG` commit. Confirm the output folders, plot manifest,
standalone slices, lifecycle messages, source preflight, recipe-design CSVs,
volume balances, performance log, and report in the actual lab environment.
This is an output/workflow check, not a claim of chemical optimization.

### 2. Before a medium supervised three-variable chemistry run

The next high-value work is operational and scientific hardening—not more
cosmetic plotting.

1. **Review the new λmax scan-quality diagnostics and specify a chemistry
   policy.** The performance log now records blank-corrected peak height and
   endpoint maxima as warning-only metadata. Low-amplitude or edge-dominated
   spectra can still yield numeric λmax values that are not scientifically
   informative. Before changing model eligibility, explicitly choose and
   validate any peak-height, prominence, or signal-to-noise threshold against
   chemistry-relevant controls.
2. **Verify source inventory against prepared vessels.** The source preflight
   is fail-closed only when cached source inventory is current. Confirm vessel
   identity, aspiratable volume, reserve, and full-batch demand. Do not assume
   that an aggregate controller-side check proves automatic backup-source
   switching at the robot-control layer. Keep
   `auto_source_volume_check` set to `required` for supervised chemistry runs;
   the reserve setting is intentionally inactive when the check is `off`.
3. **Resolve or audit water-tip reuse.** A transfer/tip audit CSV and explicit
   tip-policy decision are advisable before a larger or contamination-sensitive
   chemistry campaign, especially if water-used tips can transition to other
   reagents.
4. **Run the medium `core3` experiment under direct supervision.** With six
   seed conditions, two optimizer iterations, three modes, and three
   duplicates, the planned total is 36 wells:

   ```text
   seeds:      6 conditions × 3 duplicates = 18 wells
   iterations: 2 × 3 modes × 3 duplicates = 18 wells
   total:                                 36 wells
   ```

   Interpret it as controlled chemistry validation. Review every
   condition-level QC decision and selected acquisition mode before claiming
   optimization success.

### 3. After that run passes review

- repeat the chemistry system on another day to assess reproducibility;
- compare portfolio-mode outcomes using their common cumulative GP history;
- decide whether `target_ei` adds practical value for the chemistry;
- only then increase plate occupancy or iteration count.

### 4. Planned safety feature: batch-boundary source replenishment workflow

The implemented `auto_source_volume_check=required` preflight is intentionally
an operational gate rather than an acquisition constraint. After the optimizer
has selected the next scientifically appropriate batch, the controller builds
the exact duplicate-expanded protocol dataframe and checks its resolved
transfers against current robot-reported aggregate aspiratable inventory plus
the configured reserve. It must continue to reject a deficient batch before
any liquid handling begins.

The next planned enhancement is a human-supervised replenishment/resume
workflow at that safe batch boundary:

1. retain the already selected, fully audited batch without re-optimizing it;
2. present the insufficient source, available aspiratable volume, planned
   demand, reserve, and deficit in terminal output and a persistent audit file;
3. pause before protocol execution, allowing the operator to replenish or
   replace the affected stock, reweigh it, and verify its identity;
4. refresh and reconcile robot-reported inventory; then either execute the
   unchanged approved batch or fail closed if inventory remains insufficient.

The controller may keep a shadow withdrawal ledger for prediction and audit,
but a fresh robot inventory response at every batch boundary remains the
authority because actual consumption can differ through dead volume, priming,
retries, manual handling, or robot-side tube switching. This feature must not
silently alter GP acquisition scores or make a scientifically valuable recipe
unavailable merely because a stock is temporarily low.

The safe stop-before-batch behavior is computer-side and already partially
implemented. A true in-session reweigh-and-continue path requires verifying
that the deployed frozen Raspberry Pi runtime accepts a refreshed reagent or
inventory payload. Until that behavior is confirmed, the conservative response
to insufficient inventory remains an exported audit and clean stop; do not
claim automatic backup-tube switching from the aggregate controller preflight.

### Deferred, not immediate

- further plot styling unless a new controlled output audit finds a readability
  defect;
- batch-boundary replenishment/resume after a failed source-volume preflight;
- higher-dimensional GP-surface visualization beyond the existing projection,
  parallel-coordinate, PCA, and three-variable conditional-slice views;
- broader model changes such as heteroscedastic/noise-aware GP fitting;
- unattended or large-scale chemistry optimization;
- promotion of `Auto-RTG` into `Auto-RTG-v1`.

---


# Historical Stable-v1 Record — Not Current Auto-RTG Status

> [!IMPORTANT]
> Everything below this heading is retained as provenance for the stable-v1
> checkpoint and the development plan that preceded `Auto-RTG`. It is not a
> current implementation checklist. References to `stable/auto-rtg-v1`,
> `feature/acquisition-modes`, deferred acquisition modes, or a missing
> source-volume hard-stop describe the historical state, not the active
> `Auto-RTG` branch reconciled above.

# Stable V1 Branch Checkpoint

## Branch designation

The validated software baseline documented by this file is intended to live on:

```text
stable/auto-rtg-v1
```

New acquisition-function development should branch from this checkpoint into:

```text
feature/acquisition-modes
```

The stable branch should not receive experimental acquisition-function changes directly. It should move
only after a later feature branch passes static validation, synthetic optimizer tests, a controlled debug
run, and review of the generated scientific outputs.

## Validated code identity

The stable checkpoint was validated with the following file identities:

| File used for stable checkpoint | Equivalent uploaded copy | SHA-256 |
|---|---|---|
| `controller(199).py` | `controller(200).py` | `41b52de144646fc6b79731e4d6114aa733224cee9a40493fb828a95caa398081` |
| `optimizers(77).py` | `optimizers(78).py` | `85d08e67cb4788f5e337ed00478f031077d09a284638adb20274e8536e5fac23` |

The paired copies are byte-for-byte identical. The version numbers in parentheses are local upload labels,
not repository filenames; the repository files remain `controller.py` and `optimizers.py`.

## Stable-checkpoint validation evidence

The July 13 checkpoint includes:

- successful Python 3.9 parsing and compilation of the controller and optimizer,
- validation of the controller-to-optimizer public interface,
- corrected cumulative GP training history across successive batches,
- corrected 2D GP heatmap orientation,
- post-model-update prediction and uncertainty grid refresh,
- removal of stale pre-experiment grid refresh,
- condition-level replicate QC and model-training selection,
- target-hit and max-iteration stop-state preservation,
- standard lifecycle plot generation after measurement, after model update, and at finalization,
- successful generation of seed-only and full Auto design-space plots after the NumPy-array bounds fix,
- CSV, report, terminal-log, recipe-debug, and plot output generation,
- and clean finalization without a pre-success traceback or post-success cleanup warning.

## What “stable v1” means

`stable/auto-rtg-v1` is a reproducible software and protocol baseline for the currently implemented
GP-guided target-distance optimizer. It is suitable as a rollback point and as the parent branch for new
features.

It does **not** mean that every scientific or operational enhancement is complete. The following remain
known, nonblocking limitations:

- the current optimizer uses target-distance exploitation rather than selectable acquisition modes,
- UV overlay plots still use a fixed `0–1` absorbance axis and can look blank for near-zero signals,
- low-amplitude spectra can produce mathematically valid but scientifically weak λmax calls,
- source/deck-volume hard-stop validation is still deferred,
- water-used tip contamination policy is still deferred,
- and larger real-chemistry reproducibility testing is still required.

Where an older historical section below describes a feature as compile-only, pending, or using an earlier
filename, the July 13 stable-checkpoint sections are authoritative for the stable branch.

---

# High-Level Status

The stable v1 Auto mode branch has reached a substantially more mature validation milestone than the original June 8 notes. The branch now supports:

- volume-safe Auto recipe generation,
- true-zero variable reagent handling,
- mixed discrete/continuous mask optimization,
- condition-level model-performance logging,
- condition-level stopping logic,
- lambda-max progress plotting,
- final Auto progress summary plotting,
- fixed 300–1000 nm lambda-display-window plotting with error bars clipped only at the display boundary,
- 2D GP prediction heatmaps,
- 2D GP uncertainty heatmaps,
- scientifically correct 2D GP reagent-axis orientation,
- cumulative GP training history across all completed optimizer batches,
- lifecycle-coordinated per-batch and final Auto plotting,
- separate seed-only and full design-space exploration plots,
- organized debug output folders,
- terminal-output capture to the run-specific Debug folder,
- notebook-ready Auto run report generation,
- source-aligned padded Markdown condition tables,
- read-only dimension-aware initial-training/design-space plotting,
- conditional report embedding of generated design-space plots,
- and a cleaner division between primary results and debug/audit artifacts.

The latest validated checkpoint confirmed the complete two-seed/two-iteration Auto lifecycle, cumulative GP updates, post-batch heatmap timing, corrected heatmap orientation, replicate QC, target-based stopping, final reporting, and restored seed/exploration design-space plots.

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
| Correct 2D GP axis orientation | Implemented and validated |
| Post-batch GP grid refresh timing | Implemented and validated |
| Cumulative optimizer/GP training history | Implemented and validated |
| Auto plot profiles (`standard`, `final_only`, `off`) | Implemented and validated |
| Seed-only design-space plots | Implemented and validated |
| Seed-plus-iterations exploration plots | Implemented and validated |
| Stable v1 rollback checkpoint | Documented for `stable/auto-rtg-v1` |
| Terminal output saved to Debug folder | Implemented and validated |
| Notebook-ready Auto run report | Implemented and validated |
| Padded Markdown compact condition table | Implemented and validated |
| Dimension-aware seed and full-exploration design-space plots | Implemented and validated |
| Conditional design-plot report embedding | Implemented and validated |
| Auto recipe-design exports moved to Debug subfolder | Implemented and validated |
| Optimizer duplicate max-iteration print | Removed / controller owns user-facing stop print |
| Water-used tip handling | Deferred |
| Source/deck volume hard-stop | Deferred |
| Spreadsheet-selectable acquisition modes | Deferred to `feature/acquisition-modes` |

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
3. Exclude the third value only if it is more than the threshold away from **both** members of the closest pair. A value within the threshold of either agreeing member is retained.
4. If no tight pair exists, flag the condition as `flagged_not_excluded` and preserve all valid replicates.

Examples:

| Replicates | QC result | Training behavior |
|---|---|---|
| `650, 653, 980` | `excluded_replicate` | train on `650, 653` |
| `650, 720, 790` | `flagged_not_excluded` | train on all valid replicates, but tag condition |
| `650, 660, 670` | `passed` | train on all valid replicates |
| `600, 640, 690` | `passed` | `690` is within 50 nm of the nearer closest-pair member (`640`), so it is retained |
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

## June 16, 2026 — Notebook-Ready Report Tables and Read-Only Design-Space Visualization

> [!NOTE]  
> Stable v1 supersedes the early `initial_training_design_*` output names described historically in this section. The validated stable filenames use separate `initial_maximin_seed_design_*` and `auto_design_space_exploration_*` families.

### Purpose

This update improved the Auto mode reporting layer without changing optimizer behavior, recipe generation, QC handling, model training, robot execution, spreadsheet-triggered plot rows, or the existing 2D GPR prediction/uncertainty heatmaps.

The objective was to make each completed Auto run more self-documenting and notebook-ready by:

- cleaning up the compact condition table in `auto_run_report.md`,
- preserving raw Markdown readability for Git diffs and notebook review,
- adding read-only design-space visualizations of initial seed and optimizer-selected conditions,
- wiring design-space plot generation defensively into the final run export path,
- and conditionally embedding generated design-space plots in the Markdown report.

### Report table formatting update

The Auto report now uses reusable Markdown table-formatting helpers rather than manually concatenated pipe-table strings.

New report-helper behavior includes:

- escaping pipe characters and line breaks inside table cells,
- replacing missing compact-table values with `—`,
- formatting scalar numeric values more compactly,
- formatting replicate lists as clean comma-separated values rather than code-like list strings,
- and generating padded Markdown source tables so raw `.md` output aligns cleanly in text editors and Git diffs.

This fixes the issue where the rendered Markdown table was valid but the raw inline table grid appeared visually misaligned in source view.

### Compact condition table behavior

The compact condition table remains condition-level and includes:

- condition number,
- batch number,
- condition type,
- predicted λmax,
- GP SD,
- raw λmax replicate values,
- QC-used λmax replicate values,
- mean λmax,
- target error,
- and replicate QC status.

The update is formatting-only. It does not change the condition-level data, raw well-level data, QC-cleaned values, excluded replicate preservation, prediction values, or target-error calculations.

### Design-space plotting subsystem

A read-only, dimension-aware plotting subsystem was added for initial-training/design-space visualization.

The subsystem reads from a deep copy of:

```python
self.auto_model_performance_rows
```

and detects variable reagent concentration columns based on the existing convention:

```text
<reagent_name>_concentration
```

Visible axis labels follow the same convention already used by the existing 2D GPR heatmaps:

```text
<reagent_name> (mM)
```

This keeps the new 2D design-space plots visually consistent with the already implemented 2D GPR prediction and uncertainty plots.

### Automatic design-space plot behavior

The dispatcher `_plot_initial_training_designs_after_run()` generates plots based on the number of detected variable reagent concentration columns.

| Variable reagent count | Automatically generated design-space plots |
|---:|---|
| 0 | no plot; warning only |
| 1 | `initial_training_design_1d.png` |
| 2 | `initial_training_design_2d.png` |
| 3 | `initial_training_design_pairwise.png` and `initial_training_design_3d.png` |
| 4–6 | `initial_training_design_pairwise.png` and `initial_training_design_parallel_coordinates.png` |
| 7+ | `initial_training_design_pairwise_compact.png`, `initial_training_design_parallel_coordinates.png`, and `initial_training_design_pca.png` |

For exactly three variable reagents, the 3D scatter plot is intentionally generated every time, not treated as optional.

### Defensive plotting behavior

Design-space plotting is deliberately non-critical. The final `_run()` path now calls the dispatcher after final lambda plots and before report generation:

```text
_export_auto_model_performance_log()
_plot_lambda_progress_after_batch(... final ...)
_plot_lambda_replicate_progress_after_batch(... final ...)
_plot_initial_training_designs_after_run()
_write_auto_run_report()
```

The design-space plotting call is wrapped in `try/except`, and each individual plot type is also guarded inside the dispatcher.

Expected failure behavior:

- a plotting failure prints a controller warning,
- Auto mode continues,
- robot shutdown is not blocked,
- CSV export is not blocked,
- final lambda plots are not blocked,
- and report generation is not blocked.

### Report embedding behavior

The report now conditionally embeds generated design-space plots. The helper `_auto_report_plot_markdown_if_exists()` checks whether a plot exists before adding Markdown image references.

The report can now include:

- 1D reagent-space design plots,
- 2D reagent-space design plots,
- pairwise design-space projections,
- mandatory 3D design-space scatter plots for three-variable runs,
- parallel-coordinate design summaries,
- compact pairwise high-dimensional summaries,
- and PCA design-space summaries.

Only plots that actually exist are embedded in the report. The `Generated Files` section lists all possible design-space plot outputs as present or not generated.

### Spreadsheet input-template decision

No change was made to the spreadsheet-triggered plotting rows.

The current recommended plot rows remain:

```text
plot | auto_scan | auto_plot       | 2d_gpr
plot | auto_scan | auto_uv_overlay | OVERLAY
```

The new design-space plots are intentionally generated from the controller after the Auto run is complete, rather than from the spreadsheet input template. This prevents the new reporting plots from interfering with existing scan-driven plot behavior.

This preserves the existing roles:

| Plot family | Trigger/source | Purpose |
|---|---|---|
| `2d_gpr` | spreadsheet plot row | GP prediction and uncertainty heatmaps |
| `OVERLAY` | spreadsheet plot row | UV-vis scan overlays |
| `initial_training_design_*` | final controller export path | executable design-space visualization |

### Validation checkpoint

`controller(162).py` was syntax/compile validated after these changes.

Confirmed implementation state:

- report table helpers present once,
- old `_markdown_safe` and `_condition_table_value` references removed,
- read-only design-space plot methods present once,
- design-space dispatcher present once,
- design-space plotting wired into `_run()` with defensive `try/except`,
- design-space plots generated before report generation,
- optional report plot-embedding helper present,
- `auto_run_report.md` conditionally embeds generated design-space plots,
- `Generated Files` records design-space plot presence/absence,
- existing spreadsheet-driven `2d_gpr` and `OVERLAY` behavior preserved,
- no changes made to optimizer, QC, model training, recipe generation, robot execution, or loop data.

### Remaining validation task

The next validation task is to run or simulate against a completed Auto output folder such as `DEBUGRTG_007` and confirm:

- expected design-space PNG files are created for the detected dimensionality,
- generated plots use correct reagent labels and units,
- report image links render correctly,
- padded Markdown condition table aligns in raw source view,
- `Generated Files` correctly marks design plots as present or not generated,
- and design-space plotting remains warning-only if a plot fails.

---


## July 9–13, 2026 — Plot Lifecycle, Cumulative GP Repair, Heatmap Orientation, and Stable V1 Validation

### Purpose

This stage converted several Auto plotting and optimizer behaviors from loosely coupled features into a
validated lifecycle and corrected two scientifically important GP issues.

### General Auto plotting profiles

The optional Header setting:

```text
auto_plot_profile
```

supports:

| Value | Behavior |
|---|---|
| `standard` | generate applicable per-batch plots and final outputs |
| `final_only` | suppress per-batch Auto diagnostics and generate final plots/report |
| `off` | suppress automatic Auto diagnostics and summary plots |

Older spreadsheets default to `standard`.

The controller now coordinates automatic plotting by lifecycle stage:

```text
after_measurement
after_model_update
final
```

Scan-derived UV overlays remain spreadsheet-triggered, while GP and Auto diagnostic plots are generated
at the scientifically appropriate model lifecycle stage.

### Cumulative GP-history defect and correction

A major optimizer defect was identified in the model-update path. The GP model received cumulative arrays,
but the GPyOpt optimizer object's `X` and `Y` arrays were not synchronized after a successful update.

Before correction, a longer run could evolve as:

```text
seed
seed + batch 1
seed + batch 2
seed + batch 3
```

instead of preserving:

```text
seed
seed + batch 1
seed + batch 1 + batch 2
seed + batch 1 + batch 2 + batch 3
```

The corrected `update_experiment_data()` now updates the GP first and then synchronizes:

```python
self.optimizer.X
self.optimizer.Y
```

with complete cumulative copies. The update remains atomic: if GP updating fails, optimizer history,
iteration count, and stop state are left unchanged.

This was not merely a visualization fix. In runs with three or more optimizer iterations, the old behavior
could have caused later reaction selection to forget older optimizer-generated experiments.

### 2D GP heatmap-axis correction

The previous prediction-grid code used ordinary `meshgrid`/C-order flattening and then transposed the
reshaped prediction array. Under that coordinate ordering, the transpose swapped the relationship between
the reagent coordinates and the displayed matrix cells.

Stable v1 now explicitly uses:

```text
columns = first variable reagent = x-axis
rows    = second variable reagent = y-axis
```

with no final transpose.

The correction was validated with an intentionally asymmetric synthetic surface and confirmed against the
real debug plots. The GP prediction and uncertainty maps now share the same correct orientation.

### Post-update heatmap timing

The stale prediction-grid refresh inside `getNextReaction()` was removed. The controller now refreshes the
full GP prediction and uncertainty grids only after the completed batch has been incorporated into the
fitted model.

Therefore:

```text
gpr_predictions_batch_0.png
```

represents the seed-fitted model, and:

```text
gpr_predictions_batch_N.png
```

represents the cumulative model that actually includes Batch N.

### July 13 DEBUGRTG_009 validation configuration

| Setting | Value |
|---|---:|
| Initial seed conditions | `2` |
| Maximum optimizer iterations | `2` |
| Replicates per condition | `3` |
| True-zero mixed masks | enabled |
| Auto plot profile | `standard` |
| Target λmax | `625 nm` |

Observed lifecycle:

```text
2 seed conditions
+ 2 optimizer-selected conditions
= 4 unique conditions
= 12 physical replicate wells
```

The run stopped after Batch 2 because the controller accepted a validated condition-level target hit:

```text
QC-cleaned mean λmax = 630 nm
target error         = 5 nm
replicate SD         = 0 nm
```

One `683 nm` replicate was excluded from that condition under the configured `50 nm` outlier threshold;
the two included values were `630, 630 nm`.

### Cumulative model-update evidence

The optimizer terminal output recorded:

```text
after Batch 1: 7 cumulative observations, 3 new
after Batch 2: 9 cumulative observations, 2 new
```

The second update retained the earlier optimizer batch and added only the two QC-included observations from
the final condition.

### Restored seed and exploration plots

The first DEBUGRTG_009 archive exposed a final plotting defect:

```text
The truth value of an array with more than one element is ambiguous.
```

`self.variable_reagents` was a NumPy array, and the executable-bounds helper attempted to use it in a Python
truth-value expression.

The stable controller normalizes that collection explicitly. The post-fix validation generated:

```text
initial_maximin_seed_design_2d.png
auto_design_space_exploration_2d.png
```

Both use identical executable concentration bounds, identical 3% display padding, a physically square
plotting panel, and the centralized Auto font sizes. The seed-only figure therefore does not autoscale
tightly around the seed points and remains directly comparable with the full exploration figure.

### Stable v1 output set

For a two-variable run using `auto_plot_profile = standard`, the validated output family includes:

```text
Plots/
  auto_uv_overlay-0.png
  auto_uv_overlay-1.png
  auto_uv_overlay-2.png
  gpr_predictions_batch_0.png
  gpr_predictions_batch_1.png
  gpr_predictions_batch_2.png
  gpr_uncertainty_batch_0.png
  gpr_uncertainty_batch_1.png
  gpr_uncertainty_batch_2.png
  lambda_progress_after_batch_0.png
  lambda_progress_after_batch_1.png
  lambda_progress_after_batch_2.png
  lambda_progress_final.png
  lambda_replicates_after_batch_0.png
  lambda_replicates_after_batch_1.png
  lambda_replicates_after_batch_2.png
  lambda_replicates_final.png
  initial_maximin_seed_design_2d.png
  auto_design_space_exploration_2d.png
```

Primary and audit outputs also include:

```text
pr_data/experiment_data.csv
pr_data/auto_model_performance_log.csv
pr_data/auto_run_report.md
Debug/terminal_output.txt
Debug/auto_recipe_design/auto_recipe_design_batch_*.csv
Eve_Files/protocol_record.txt
Eve_Files/wellmap.tsv
Eve_Files/well_history.tsv
Eve_Files/translated_wellmap.tsv
```

### Scientific caveat from the debug spectra

The UV overlay traces in this debug run were close to baseline relative to the fixed `0–1` absorbance axis.
The run is therefore strong evidence for software, robot, plate-reader, data-flow, QC, GP-update, plotting,
and reporting behavior, but not strong evidence that the apparent λmax values represented robust chemical
absorbance peaks.

A later scientific-hardening feature should reject or flag λmax values when peak height, prominence, or
signal-to-noise is inadequate.

### Stable checkpoint outcome

The paired stable controller and optimizer are accepted as the rollback baseline for the current
target-distance Auto workflow. Acquisition-mode work should start from this checkpoint on
`feature/acquisition-modes`.

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

Stable v1 intentionally preserves the validated target-distance exploitation objective:

```text
choose the feasible candidate whose GP mean λmax is closest to the target
```

The optimizer stores:

- selected mask,
- predicted λmax mean in nm,
- predicted GP standard deviation in nm,
- suggested normalized recipe,
- and volume-balance information.

Although `optimizers.py` contains legacy names such as `EI`, `MPI`, and `LCB`, those names are not the
active mixed-mask recipe-selection policy in stable v1. The custom mask-constrained target-distance
objective is the operative selection behavior.

Spreadsheet-selectable acquisition modes are intentionally deferred to `feature/acquisition-modes`.
The planned canonical modes are:

| Planned mode | Intended behavior |
|---|---|
| `exploit` | preserve stable-v1 closest-to-target mean selection |
| `explore` | prioritize maximum GP predictive uncertainty |
| `balanced` | target-aware hybrid of target proximity and uncertainty |
| `target_ei` | expected improvement in best QC-approved target error |

Older spreadsheets must continue to default to `exploit` so the stable-v1 behavior remains reproducible.

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
| `initial_maximin_seed_design_2d.png` | seed-only 2D executable design-space view |
| `auto_design_space_exploration_2d.png` | seed plus optimizer-selected 2D design-space view |

## 6. Debug output organization

Current intended structure:

```text
DEBUGRTG_###/
  Plots/
    auto_uv_overlay-*.png
    lambda_progress_after_batch_*.png
    lambda_progress_final.png
    lambda_replicates_after_batch_*.png
    lambda_replicates_final.png
    gpr_predictions_batch_*.png
    gpr_uncertainty_batch_*.png
    initial_maximin_seed_design_*.png
    auto_design_space_exploration_*.png

  pr_data/
    experiment_data.csv
    auto_model_performance_log.csv
    auto_run_report.md
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


## 4. July 13, 2026 Stable V1 Checkpoint Run

This controlled debug run is the main validation checkpoint for `stable/auto-rtg-v1`.

### Configuration

| Setting | Value |
|---|---:|
| `initial_data` | `2` |
| `max_iterations` | `2` |
| `num_duplicates` | `3` |
| `allow_true_zero` | `TRUE` |
| `auto_plot_profile` | `standard` |
| target λmax | `625 nm` |

### Validated outcomes

- The seed batch and two optimizer-selected batches executed.
- All recipes remained volume-feasible.
- True-zero mixed masks were exercised.
- Batch 1 selected mask `[0, 1]`.
- Batch 2 selected mask `[1, 0]`.
- Batch 2 used the model containing seeds plus Batch 1.
- The final model retained seeds plus both optimizer batches.
- The final update used only the two QC-included `630 nm` replicates from condition 3.
- GP prediction and uncertainty grids refreshed after each model update.
- Heatmap x-axis data corresponded to silver nitrate.
- Heatmap y-axis data corresponded to potassium bromide.
- Condition-level target stopping accepted `630 ± 0 nm` against the `625 nm` target.
- The run exported well-level data, condition-level performance data, report, terminal log, and recipe audit CSVs.
- The post-fix plot rerun generated both the seed-only and full-exploration 2D design-space plots.
- No pre-success traceback or post-success cleanup warning was identified.

### Interpretation

This run validates the software and protocol behavior that defines stable v1. It does not establish robust
chemical optimization because the measured UV signals were close to baseline. The stable branch should
therefore be described as a validated automation and Bayesian-optimization software baseline, not as a
fully validated chemistry model.

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

## 3. Spreadsheet-selectable acquisition modes are deferred to the acquisition feature branch

Stable v1 uses target-distance exploitation based on GP mean λmax. GP uncertainty is logged and plotted,
but it does not yet alter recipe selection.

The next feature branch is:

```text
feature/acquisition-modes
```

The planned first release should support:

```text
exploit
explore
balanced
target_ei
```

All four modes must preserve the stable mixed-mask search, exact-zero behavior, executable transfer bounds,
water top-off rules, volume-feasibility penalties, cumulative GP history, replicate-QC model training, and
controller hard-stop.

The stable branch should not be modified during this work.

---

# Recommended Next Steps

## 1. Preserve the Stable V1 Checkpoint

Commit this document with the validated controller and optimizer on:

```text
stable/auto-rtg-v1
```

Treat that branch as the rollback baseline. Do not add acquisition-mode experiments directly to it.

Create the next development branch from the exact stable commit:

```text
feature/acquisition-modes
```

The first implementation step on that branch is to add and validate the optional Header
`acquisition_mode` parser without changing recipe selection yet.

## 2. Implement Acquisition Modes in Controlled Stages

Recommended canonical spreadsheet values:

| Mode | Stable-v1 relationship |
|---|---|
| `exploit` | exact backward-compatible stable-v1 behavior |
| `explore` | maximum predictive uncertainty |
| `balanced` | target-aware hybrid |
| `target_ei` | expected improvement in target error |

Required validation sequence:

```text
Header parsing and aliases
→ controller-to-optimizer propagation
→ target-aware acquisition score functions
→ mask and volume-feasibility integration
→ acquisition metadata logging
→ synthetic mode-separation tests
→ regression test of exploit against stable v1
→ controlled debug run
→ generated-output audit
```


## 3. Continued Small Real-Chemistry Validation / Scale-Up

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

## 4. Add Optimizer Mask-Result Export

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

## 5. Add Deck/Source-Volume Validation Before Scaling

The next major safety feature before larger runs should be a batch-level deck/source-volume validator.

Before execution, calculate total required source volume for the full batch, including duplicates, for every source reagent:

- Water,
- fixed reagents,
- variable reagents.

Then compare required volume against available deck/source volume with a safety margin.

### Purpose

Independent reagent bounds make high-volume reagent usage more likely than the old equal-split approach. Source depletion is therefore one of the next likely practical risks when scaling.

---

## 6. Consider Uncertainty-Plot Display Scaling

The uncertainty heatmap currently shows raw GP predictive SD in nm. In sparse early batches this can be very large, sometimes hundreds of nm.

This is technically correct, but may make the heatmap visually dominated by huge early uncertainty values.

A later visualization-only improvement could add robust color scaling or a display cap for the uncertainty heatmap, while preserving raw uncertainty values in logs.

---

## 7. Gradual Scale-Up Plan

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

## 8. Later Model/Chemistry Improvements

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

The Auto mode branch has reached the `stable/auto-rtg-v1` checkpoint. It has passed the main optimizer/protocol validation milestone for mixed-mask optimization and volume-safe recipe generation, as well as validation of cumulative GP history, corrected heatmap orientation, lifecycle-aware plotting, condition-level QC/logging, final reporting, and seed/full-exploration design-space plot generation.

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
- fixed-window lambda display with boundary-clipped error-bar annotation behavior,
- 2D GP prediction heatmap generation,
- 2D GP uncertainty heatmap generation,
- correct mapping of the first variable reagent to heatmap columns/x-axis,
- correct mapping of the second variable reagent to heatmap rows/y-axis,
- cumulative GP and optimizer history across successive batches,
- post-model-update heatmap refresh timing,
- `standard`, `final_only`, and `off` Auto plot-profile parsing and lifecycle behavior,
- seed-only and seed-plus-iterations design-space plots,
- stable controller/optimizer interface compatibility,
- terminal-output capture to `Debug/terminal_output.txt`,
- recipe-design debug export relocation to `Debug/auto_recipe_design/`,
- notebook-ready report table formatting,
- read-only design-space plotting methods,
- defensive design-space plot generation from `_run()`,
- conditional design-space plot embedding in `auto_run_report.md`,
- and optimizer/controller stop-message cleanup.

## Deferred

- water-used tip behavior in `ot2_robot.py`,
- transfer/tip audit CSV,
- deck/source-volume hard-stop,
- broader real-chemistry reproducibility / scale-up validation,
- spreadsheet-selectable acquisition modes on `feature/acquisition-modes`,
- spectral signal-quality gating,
- larger autonomous scale-up.

## Recommended immediate next step

Commit the validated code and this document to `stable/auto-rtg-v1`, create
`feature/acquisition-modes` from that exact checkpoint, and resume the acquisition-mode implementation with
the Header parser as the first isolated change.

Continue small, cautious real-chemistry validation in parallel and review:

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
