# Auto preparation worksheet contract (Stage 9A)

This document defines the preparation request format that will be executed in
Stage 9B.  Stage 9A does **not** read the worksheet during a run and does not
send preparation transfers to the robot.  It exists so the planned spreadsheet
interface and its chemical calculations can be reviewed before execution is
introduced.

## Worksheet

The future optional worksheet name is:

```text
auto_preparation
```

Its columns are:

| Column | Meaning |
| --- | --- |
| `enabled` | `yes`/`on`/`true`/`1` to request the row.  `no`/`off`/`false`/`0` skips it. |
| `stock_reagent` | Reagent root name, such as `sodium_borohydride`. Spaces are normalized to underscores. |
| `stock_concentration_mM` | Measured concentration of the existing source container. |
| `working_concentration_mM` | Desired working concentration. It must be lower than the stock concentration. |
| `final_volume_uL` | Total prepared volume. It must not exceed the existing Header `dilution_vol` capacity. |

The existing Header values remain the destination configuration:

```text
dilution_cont    <empty destination container type>
dilution_vol     <maximum prepared volume in uL>
```

For example, a 130 mM stock prepared as 1000 uL of 6.25 mM working solution is:

| enabled | stock_reagent | stock_concentration_mM | working_concentration_mM | final_volume_uL |
| --- | --- | ---: | ---: | ---: |
| yes | sodium_borohydride | 130 | 6.25 | 1000 |

The planned calculation is the manual-controller dilution calculation:

```text
stock_transfer_uL = final_volume_uL × working_concentration_mM / stock_concentration_mM
water_transfer_uL = final_volume_uL − stock_transfer_uL
```

In this example the manifest requires 48.0769 uL stock and 951.9231 uL water.

## Safety checks already implemented

The pure manifest builder rejects a requested preparation when it has:

- a missing, non-finite, or non-positive required value;
- a target concentration at or above the stock concentration;
- final volume above `dilution_vol`;
- stock or water transfer in the non-executable `0 < volume < 5 uL` interval;
- more than one requested output with the same resulting working chemical name.

The last constraint is intentional for now.  The current legacy robot
initialization maps one chemical name to one prepared destination.  Preparing
multiple same-concentration backup tubes will be added only after an explicit
grouped-source registration design is implemented and tested; it must not be
silently inferred from duplicate rows.

## Planned execution semantics

Stage 9B will execute each validated preparation exactly once before Auto seed
or optimizer transfers:

1. allocate an appropriate empty `dilution_cont` destination;
2. choose `WaterC1.0` or `ColdWaterC1.0` based on whether the stock source is
   stored in the temperature module, matching the manual controller;
3. transfer water, then stock reagent;
4. mix twice;
5. resolve and journal the destination returned by the robot;
6. make the prepared source available to Auto’s ordinary concentration-to-volume
   conversion only after completion is confirmed.

No Auto batch will start if an enabled preparation request cannot be validated
or completed.
