# Auto preparation worksheet and execution contract (Stages 9A–9B)

This document defines the opt-in Auto working-solution preparation interface.
The planner supplies a hardware-free chemistry manifest; Auto-main validates,
reserves, and executes it exactly once before seed or optimizer recipes are
generated. It is deliberately separate from legacy conversion-error dilution:
a malformed request fails before robot connection, and a failed preparation
prevents Auto batch execution.

## Worksheet

The future optional worksheet name is:

```text
auto_preparation
```

Its columns are:

| Column | Meaning |
| --- | --- |
| `enabled` | `yes`/`on`/`true`/`1` to request the row. `no`/`off`/`false`/`0`, or a blank cell, skips it. |
| `stock_source_group` | Reagent root name, such as `sodium_borohydride`. Spaces are normalized to underscores. |
| `stock_concentration_mM` | Measured concentration of the existing source container. |
| `working_concentration_mM` | Desired working concentration. It must be lower than the stock concentration. |
| `tube_count` | Number of identical working tubes to create. |
| `final_volume_per_tube_uL` | Final volume in each working tube. |
| `destination_labware` | Destination labware type, for example `temp_mod_24_tube`. |
| `destination_container` | Destination tube/container class, for example `Tube2000uL`. |
| `destination_locs` *(optional)* | Ordered empty-tube wells, separated by semicolons, such as `A1;A2`. It must list exactly `tube_count` unique locations when supplied. |
| `destination_deck_positions` *(optional)* | Ordered deck positions corresponding to `destination_locs`, separated by semicolons, such as `3;3`. It must list exactly `tube_count` positions when supplied. Both destination columns must be completed together or both left blank to retain Auto-main's compatible-empty-tube allocation. |
| `water_source_policy` *(optional)* | `auto` (default) follows stock temperature-module placement; `temperature_controlled` requires the water source on the temperature module; `ambient` requires the ordinary water source outside that module. |

Preparation destinations must be declared as `empty` tube locations in
`reagent_info`; `destination_locs` and `destination_deck_positions` narrow
that declared empty-tube pool to the exact positions for one group. The legacy Header ``dilution_cont``
and ``dilution_vol`` settings remain for manual dilution workflows; they do
not override an Auto preparation row.

Temporary worksheets using the earlier combined
`destination_tube_locations` column remain readable only when both new
destination columns are blank. Do not mix the old and new forms in one row;
new worksheets should use the separate columns exclusively.

Preparation is disabled by default for legacy worksheets. To require an
enabled worksheet request, add this Header row:

```text
auto_preparation_mode    required
```

`off` is the default and preserves existing Auto behavior. `required` is
fail-closed: an absent worksheet, no enabled rows, invalid chemistry, missing
stock source, or an existing destination name stops the run before an Auto
recipe is generated.

For example, a 130 mM stock prepared as 1000 uL of 6.25 mM working solution is:

| enabled | stock_source_group | stock_concentration_mM | working_concentration_mM | tube_count | final_volume_per_tube_uL | destination_labware | destination_container | destination_locs | destination_deck_positions | water_source_policy |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| yes | sodium_borohydride | 130 | 6.25 | 2 | 1000 | temp_mod_24_tube | Tube2000uL | A1;A2 | 3;3 | auto |

The planned calculation is the manual-controller dilution calculation:

```text
stock_transfer_per_tube_uL = final_volume_per_tube_uL × working_concentration_mM / stock_concentration_mM
water_transfer_per_tube_uL = final_volume_per_tube_uL − stock_transfer_per_tube_uL
```

In this example each tube requires 48.0769 uL stock and 951.9231 uL water.

## Safety checks already implemented

The pure manifest builder rejects a requested preparation when it has:

- a missing, non-finite, or non-positive required value;
- a target concentration at or above the stock concentration;
- final volume above `dilution_vol`;
- stock or water transfer in the non-executable `0 < volume < 5 uL` interval;
- more than one requested working concentration for the same Auto reagent;
- more than one requested output with the same resulting working chemical name;
- a stock chemical name absent from `reagent_info`;
- a working chemical name that already exists in `reagent_info`.

The one-working-source-per-reagent constraint is intentional. Auto's current
concentration-to-volume and GP-bound logic requires one active stock
concentration per reagent coordinate. Multiple same-concentration backup tubes
also require a separate grouped-source registration design. Neither situation
is silently inferred from duplicate worksheet rows.

## Implemented Stage 9B execution semantics

For a real Auto run, Stage 9B executes each validated preparation exactly once
after the Auto-main compatibility handshake and before Auto seed or optimizer
transfers:

1. reserve the specified empty destinations, or allocate compatible empty
   tubes only when both destination columns are blank;
2. select water according to `water_source_policy`: `auto` chooses the
   temperature-controlled water source only when the active stock source is in
   the temperature module (matching the manual controller);
   `temperature_controlled` requires that source and `ambient` requires the
   ordinary water source outside the module. The established internal key
   `ColdWaterC1.0` remains a compatibility identifier only; the Header
   temperature determines whether the module heats or cools the water.
3. transfer water, then stock reagent;
4. mix twice;
5. resolve and journal the destination returned by the robot;
6. resolve the destination returned by the Pi;
7. replace the controller and optimizer runtime source view for that reagent
   with the confirmed working source, refresh its physical bounds, and prevent
   later Auto conversion from falling back to the consumed stock name.

No Auto batch will start if an enabled preparation request cannot be validated
or completed.

During the local `simulate` preflight, Stage 9B records that the validated
preparation was deferred. It does not create a working tube or mutate model
bounds against the local simulator; the preparation runs once in the following
real Auto session.

## Current boundaries

- Stage 9B uses the existing Auto-main `init_containers`, `transfer`, `mix`,
  and location-query commands; it does not require a new Pi packet type.
- A checkpoint import cannot currently be combined with required preparation.
  Reconciliation between an imported model's coordinate history and a new
  working-source transformation needs a dedicated lineage stage.
- Prepared backup groups and unattended preparation-source replacement are not
  implemented. An operator must configure one valid preparation destination per
  prepared reagent.
