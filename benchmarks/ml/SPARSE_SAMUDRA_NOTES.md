# Sparse Samudra: modeling notes

Status: early results, from 2026-11-15. The code is on the `samudra-demo`
branch of xarray-sql (draft PR #300), in `benchmarks/ml/04_samudra_sparse.py`.

## What this is

[Samudra](https://github.com/m2lines/Samudra) is a neural emulator of the
global ocean: a U-Net (an encoder that halves the grid at each level and a
decoder that doubles it again) of ConvNeXt blocks, trained on the output of
OM4, GFDL's ocean model. Like most such emulators, it computes over every
grid cell, land included, and masks land out only at the end.

[xarray-sql](https://github.com/xqlsystems/xarray-sql) runs array programs as
SQL. In SQL, a grid is a table with a row per cell, and land can simply have
no rows. The sparse model is a Samudra-style U-Net in which land has no rows
at any layer: not in the input, any hidden layer or the output.

The question is whether a model that never sees land forecasts as well as,
or better than, one that does. Strict parity with Samudra's weights is not
the goal; accurate forecasts are.

## The model

Each operator at a coast does what SQL does with a missing row, plus a
correction where that alone would be wrong:

| Operator | At land | Why |
|---|---|---|
| 3 × 3 convolution | sums only the taps that exist, then rescales by the share present | partial convolution (Liu et al., 2018, https://arxiv.org/abs/1804.07723), so a coastal cell isn't pulled toward zero |
| average pooling | averages ocean cells only; a coarse cell exists if any of its fine cells does | `AVG` over the rows present |
| bilinear upsampling | reads only coarse cells that exist, with its weights rescaled | the same, for interpolation |
| batch norm | statistics from ocean cells only | land would bias them |
| output | each channel uses its own depth's mask | channels are variables at depths |

Hidden layers' channels aren't depths, so they use the column mask: a cell
with ocean at any depth. OM4's masks nest with depth (a cell that's ocean at
one depth is ocean at every depth above it), so the column mask is the
surface mask, `mask_0`. The script checks the nesting on the data it reads.

The SQL is checked against a PyTorch *mirror* of the same model, which does
the same operations on arrays and masks. The mirror trains the weights and
runs the long rollouts; the SQL forecast matches it to about 1e-16 on held-out
days, on DataFusion, einfold and DuckDB. A *dense* model, computed over land as
Samudra is, is trained the same way as the ablation.

Inputs follow Samudra's OM4 configuration (`om4.yaml`): the state is `thetao`
and `so` by depth plus `zos` (velocities are not in yet), and the forcing is
`tauuo`, `tauvo`, `hfds` and `hfds_anomalies`. The anomaly and the
climatology come from the training years only.

## Results so far

2° OM4, 4 depths (2.5, 105, 775 and 3,100 m), widths 8 and 16 at two levels,
trained on 1975 to 1984, single-step loss, 20 epochs, on CPU (i7-8700).
Tested on 2014-10-10 to 2022-12-24.

Five-day skill, 1 minus the RMSE over the better of persistence and the
day-of-year climatology (0 is no better than a baseline):

| | surface temp | surface salinity | sea surface height | 105 m and deeper |
|---|---|---|---|---|
| sparse, all ocean | 0.34 | 0.036 | 0.17 | ≈ 0 |
| dense, all ocean | 0.32 | 0.025 | 0.15 | ≈ 0 |
| sparse, coast | 0.31 | 0.053 | 0.29 | ≈ 0 |
| dense, coast | 0.28 | 0.039 | 0.22 | ≈ 0 |

Area-mean surface values at the end of an 8-year rollout (a retrain with the
same settings):

| | data | sparse | dense |
|---|---|---|---|
| temperature | 18.47 °C | 18.29 °C | 8.15 °C |
| salinity | 34.72 psu | 34.55 psu | 22.26 psu |

- The sparse model is slightly better at five days, mostly at the coast, and
  far more stable over 8 years.
- Changing the values under land doesn't change the sparse forecast at all.
  It moves the dense model's by up to 1.1 normalized units.
- Both pass the longitude-roll and batch-independence tests.
- Both are still physically wrong in places at the end of 8 years: 9%
  (sparse) and 21% (dense) of the surface ocean is below freezing.
- The deep ocean learned nothing. Weighting each depth's error by its
  five-day change's variance taught it something (skill 0.004 to 0.020 at
  105 m), but cost sea surface height and 8-year stability.

These are tiny models, far from Samudra 2's widths (280 to 520), trained on a
quarter of its training years. Scores with `samudra.metrics` against
observations are in progress.

## Open questions

1. **The baseline at 2°.** There is no published Samudra 2 run at 2°. A
   standard one is being trained.
2. **Does sparsity survive scale?** The 8-year stability gap may close, or
   grow, at Samudra 2's widths and training length.
3. **Samudra 2's own weights, masked.** Masking the public 1° checkpoint at
   every layer, without retraining, would show how much the trained model
   depends on land.
4. **Training through SQL.** The mirror trains in PyTorch. ddx, which
   differentiates SQL, could train the SQL model itself; that's a later demo.
5. **Speed.** One SQL forecast takes 2.4 to 4.5 s at these widths, so long
   rollouts run in the mirror. Making sparse arrays fast in SQL is tracked
   separately.
