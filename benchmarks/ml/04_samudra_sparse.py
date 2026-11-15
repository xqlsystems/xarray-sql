# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "xarray-sql[ml,duckdb]",
#   "ddxdb[duckdb]>=0.4.8",
#   "einfold[datafusion]>=0.1.0rc2",
#   "torch",
#   "zarr>=3",
#   "fsspec",
#   "aiohttp",
#   "cftime",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Demo 4, sparse: a Samudra-style U-Net in SQL with no rows for land.

Samudra (Dheeshjith et al., 2024, https://arxiv.org/abs/2412.03795) is an
emulator of the global ocean: from the ocean's state and the forcing at its
surface, it predicts the state five days later. Its network is a U-Net
(Ronneberger et al., 2015, https://arxiv.org/abs/1505.04597) of ConvNeXt
blocks (Liu et al., 2022, https://arxiv.org/abs/2201.03545). Demo 4,
``04_samudra.py``, writes that network in SQL and checks it against
Samudra's PyTorch, cell for cell, land included: Samudra computes every
layer over land and masks only its output.

This demo keeps the same network, view for view, but stores the ocean
only. A state is a table of ``(c, lat, lon, v)``, and a land cell has no
row, in the input, in every hidden layer and in the forecast. That is the
idea of XQL: an array is a table, and a table holds only the cells that
exist. Each operator is then the SQL a relational engine runs anyway, and
what it does at a coast is what SQL does with an absent row:

* ``conv``: a sum over the 3 × 3 taps that have a row. With
  ``--partial`` (the default) the sum is rescaled by the share of taps
  present, as partial convolution does for images with holes (Liu et al.,
  2018, https://arxiv.org/abs/1804.07723), so a coastal cell isn't pulled
  toward zero by the land around it. Without it, land counts as zero.
* ``avg_pool``: ``AVG`` over the ocean cells of each 2 × 2 block. A coarse
  cell exists where any of its four does.
* ``upsample``: bilinear interpolation from the coarse cells that exist,
  with the weights rescaled to sum to one.
* ``batch_norm``: trained with statistics over ocean cells only.

Each channel of the state is a variable at a depth, and has a row only
where the ocean reaches that depth (OM4's ``mask_k`` for level ``k``).
Hidden channels aren't depths, so a hidden layer has a row at every cell
with any ocean: the surface mask, ``mask_0``.

GELU needs the error function ``erf``. Neither DataFusion nor DuckDB
defines it, so this demo registers it on each host as a Python function,
as ddx (https://github.com/xqlsystems/ddx), which differentiates SQL,
asks since its 0.4.8. GELU is then exact, where demo 4 approximates
``erf`` in SQL by Abramowitz and Stegun's formula 7.1.26.

**The data** is OM4 (Adcroft et al., 2019,
https://doi.org/10.1029/2019MS001726), GFDL's ocean model, as Samudra's
authors publish it for training, at 2° (90 × 180 cells, five-day means,
1958 to 2022; https://m2lines.github.io/Samudra/docs/data/). The demo
streams the slices it uses over HTTP and keeps them in memory, or, with
``--cache``, in a directory. It models temperature (``thetao``) and
salinity (``so``) at four of OM4's 19 depths, 2.5 m, 105 m, 775 m and
3,100 m, and sea surface height (``zos``): 9 channels. The forcing is
Samudra 2's: wind stress (``tauuo``, ``tauvo``), the heat flux into the
ocean (``hfds``), and that flux's anomaly from its climatology
(``hfds_anomalies``), over the five days being predicted. The two
southernmost rows, all land, are dropped, so the grid is 88 × 180. Each
field is normalized by OM4's own mean and standard deviation over ocean
cells, so 0 is the mean.

Samudra's notes for contributors, ``AGENTS.md`` and
``docs/concepts/ocean-physics.md`` in its repository
(https://github.com/m2lines/Samudra), set out what a correct dataset and
model must satisfy, and this demo follows them:

* OM4's masks nest with depth, and its NaNs mark land exactly. The demo
  checks both as it streams.
* ``areacello``, each cell's area, is labelled km² but holds m², computed
  with Earth's polar radius. The demo uses it only as relative weights,
  times ``ocean_fraction``, each cell's share of ocean at a depth.
* ``hfds_anomalies``, and the climatology a forecast must beat, come from
  the training years only.
* A short forecast is judged against persistence (nothing changes) and
  climatology (the training years' mean for that step of the year).
  Raw error in the deep ocean is tiny even for a model that has learned
  nothing, so the demo reports each depth's *skill*: 1 minus its error
  over the better of the two baselines' errors. 0 is no better than a
  baseline.
* A long rollout, each forecast fed back as the next state, should keep
  the global means of temperature and salinity steady, area-mean ``zos``
  near 0, temperatures above freezing (−0.054 °C per psu, about −1.9 °C
  at 35 psu), and the water column stable, denser below.
* Invariants: values under land don't change a forecast, rolling the
  input in longitude by a multiple of the network's downsampling rolls
  the forecast, and a forecast doesn't depend on its batch-mates. The
  first holds by construction here, with no rows for land; the ``dense``
  model shows what it takes when land is computed.

**The weights** are trained here, in PyTorch, by a mirror of the SQL: the
same operators on arrays, with a mask for absent rows. Samudra's
published checkpoints are 1° models 280 to 520 channels wide, beyond what
a CPU can run as SQL; training the network through its SQL with ddx is
the next step this demo leaves open. The SQL forecast is checked against
the mirror on held-out days, in float64, and the mirror runs the long
evaluations. A ``dense`` mirror, the same network computed over land as
Samudra does, is trained the same way for comparison.

The network differs from Samudra 2 in ways the tests here chose: it
predicts the change in the state rather than the state (Samudra's
``pred_residuals`` option), starting from no change, so it begins as
persistence; it reads one state rather than two; and it is 8 and 16
channels wide at two levels rather than 280 to 520 at four.

Run ``uv run benchmarks/ml/04_samudra_sparse.py --help`` for the options.
``--data synthetic`` (the default) checks the SQL against the mirror on a
random grid with random weights; ``--data om4`` trains on OM4, checks the
SQL on held-out days and reports the forecast's accuracy.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as F
import xarray as xr
from _bench import Report, assert_close, cli, hosts
from _hosts import Host
from torch import nn

OM4 = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_twodeg/"
# OM4's depth levels this demo models: 2.5 m, 105 m, 775 m and 3,100 m.
LEVELS = [0, 5, 10, 15]
STATE = [f"{v}_{k}" for v in ("thetao", "so") for k in LEVELS] + ["zos"]
FORCING = ["tauuo", "tauvo", "hfds"]
STREAMED = STATE + FORCING
# The network's inputs: the state, its forcing, and the heat flux's
# anomaly from its climatology, computed here.
INPUTS = STREAMED + ["hfds_anomalies"]
# The two southernmost rows of the 2° grid are Antarctica at every depth.
FIRST_ROW = 2
WIDTHS = [8, 16]
DILATIONS = [1, 2]
# Five-day steps: 73 to a year.
YEAR = 73

# --------------------------------------------------------------------------
# The network, in SQL. A state is a table of (c, lat, lon, v), with a row
# only where there is ocean.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """The number of latitude and longitude cells at one level."""

    lat: int
    lon: int

    def halved(self) -> Grid:
        return Grid(self.lat // 2, self.lon // 2)


def table(module: str, leaf: str | None = None) -> str:
    """The table holding a parameter of one of the network's modules: its
    name in the mirror's state dict, with underscores for dots."""
    name = module if leaf is None else f"{module}.{leaf}"
    return name.replace(".", "_")


def conv(
    x: str,
    module: str,
    channels: int,
    dilation: int,
    grid: Grid,
    ocean: str,
    terms: tuple[str, int] | None,
) -> str:
    """A 3 × 3 convolution with bias, at the cells of ``ocean``: each
    input row, through each tap, reaches the cell the tap's offset points
    to. Longitude wraps around the globe; latitude stops at the poles.

    With ``terms``, the sum is scaled, as partial convolution does, by the
    number of terms a full window has (9 taps × ``channels``) over the
    number that have a row.
    ``terms`` names a table with a row per cell, or per channel and cell,
    where the input has rows, and how many input rows each of its rows
    stands for, so the count doesn't read the input itself."""
    if terms is None:
        scale = count = ""
    else:
        present, rows = terms
        scale = f" * {9 * channels} / n.terms"
        count = f"""
  JOIN (
    SELECT q.lat - (k.ky - 1) * {dilation} AS lat,
           (q.lon - (k.kx - 1) * {dilation} + {grid.lon}) % {grid.lon} AS lon,
           COUNT(*) * {rows} AS terms
    FROM {present} q CROSS JOIN kernel k
    GROUP BY 1, 2) n ON p.lat = n.lat AND p.lon = n.lon"""
    return f"""(
  SELECT p.c, p.lat, p.lon, p.v{scale} + b.v AS v
  FROM (
    SELECT w.co AS c,
           x.lat - (w.ky - 1) * {dilation} AS lat,
           (x.lon - (w.kx - 1) * {dilation} + {grid.lon}) % {grid.lon} AS lon,
           SUM(x.v * w.v) AS v
    FROM {x} x JOIN {table(module, "weight")} w ON x.c = w.ci
    GROUP BY 1, 2, 3) p
  JOIN {ocean} o ON p.lat = o.lat AND p.lon = o.lon{count}
  JOIN {table(module, "bias")} b ON p.c = b.co)"""


def pointwise(x: str, module: str) -> str:
    """A 1 × 1 convolution with bias: a matrix product over channels."""
    return f"""(
  SELECT p.c, p.lat, p.lon, p.v + b.v AS v
  FROM (
    SELECT w.co AS c, x.lat, x.lon, SUM(x.v * w.v) AS v
    FROM {x} x JOIN {table(module, "weight")} w ON x.c = w.ci
    GROUP BY w.co, x.lat, x.lon) p
  JOIN {table(module, "bias")} b ON p.c = b.co)"""


def batch_norm(x: str, module: str) -> str:
    """A batch norm in evaluation, with its running statistics."""
    return f"""(
  SELECT x.c, x.lat, x.lon,
         (x.v - n.running_mean) / sqrt(n.running_var + 1e-5) * n.weight
           + n.bias AS v
  FROM {x} x JOIN {table(module)} n ON x.c = n.c)"""


def capped_gelu(x: str) -> str:
    """Samudra's ``CappedGELU``: ``x · Φ(x)``, at most 10, where
    ``Φ(x) = (1 + erf(x / √2)) / 2``."""
    return f"""(
  SELECT c, lat, lon, least(0.5 * v * (1 + erf(v / sqrt(2))), 10.0) AS v
  FROM {x})"""


def convnext_block(
    x: str,
    module: str,
    a: int,
    b: int,
    dilation: int,
    grid: Grid,
    ocean: str,
    partial: bool,
    present: tuple[str, int] | None = None,
) -> str:
    """Samudra's ``ConvNeXtBlock`` from ``a`` to ``b`` channels, at the
    cells of ``ocean``: two dilated convolutions, each with a batch norm
    and a capped GELU, then a 1 × 1 convolution, added to the input
    (through a 1 × 1 convolution of its own when the widths differ).
    ``present`` says where the input has rows, for partial convolution,
    when that isn't every channel at every cell of ``ocean``."""
    first = present or (ocean, a)
    h = capped_gelu(
        batch_norm(
            conv(
                x,
                f"{module}.conv1",
                a,
                dilation,
                grid,
                ocean,
                first if partial else None,
            ),
            f"{module}.norm1",
        )
    )
    h = capped_gelu(
        batch_norm(
            conv(
                h,
                f"{module}.conv2",
                b,
                dilation,
                grid,
                ocean,
                (ocean, b) if partial else None,
            ),
            f"{module}.norm2",
        )
    )
    h = pointwise(h, f"{module}.mix")
    skip = x if a == b else pointwise(x, f"{module}.skip")
    return f"""
SELECT h.c, h.lat, h.lon, s.v + h.v AS v
FROM {h} h
JOIN {skip} s ON h.c = s.c AND h.lat = s.lat AND h.lon = s.lon"""


def avg_pool(x: str) -> str:
    """The mean of the ocean cells in each 2 × 2 block. A coarse cell has a
    row where any of its four cells does."""
    return f"""
SELECT c, floor(lat / 2.0) AS lat, floor(lon / 2.0) AS lon, AVG(v) AS v
FROM {x}
GROUP BY 1, 2, 3"""


def coarser(ocean: str) -> str:
    """The cells of the next level down: those with any ocean below."""
    return f"""
SELECT DISTINCT floor(lat / 2.0) AS lat, floor(lon / 2.0) AS lon
FROM {ocean}"""


def upsample(x: str, skip: str, coarse: Grid) -> str:
    """Bilinear interpolation to twice the resolution, periodic in
    longitude, at the cells the skip connection from the encoder has, then
    that skip connection added. The weights of the coarse cells that exist
    are rescaled to sum to one."""
    return f"""
WITH taps(r, d, w) AS (
  -- Fine cell 2i + r reads coarse cells i + d with weight w.
  VALUES (0, -1, 0.25), (0, 0, 0.75), (1, 0, 0.75), (1, 1, 0.25))
SELECT k.c, k.lat, k.lon,
       k.v + SUM(ty.w * tx.w * x.v) / SUM(ty.w * tx.w) AS v
FROM {skip} k
JOIN taps ty ON ty.r = k.lat % 2
JOIN taps tx ON tx.r = k.lon % 2
JOIN {x} x
  ON x.c = k.c
 AND x.lat = greatest(0, least({coarse.lat - 1}, floor(k.lat / 2.0) + ty.d))
 AND x.lon = (floor(k.lon / 2.0) + tx.d + {coarse.lon}) % {coarse.lon}
GROUP BY k.c, k.lat, k.lon, k.v"""


def program(grid: Grid, partial: bool) -> Iterator[tuple[str, str]]:
    """Each view of the network and its SQL, from the state's ocean rows
    to the forecast's. The tables it reads are ``state`` and ``mask``
    (the grid, with NaN and ``false`` on land), ``kernel`` (the taps of a
    3 × 3 kernel) and one per parameter."""
    yield "ocean_0", "SELECT lat, lon FROM mask WHERE wet"
    yield "ocean_state", "SELECT c, lat, lon, v FROM state WHERE NOT isnan(v)"
    channels = [len(INPUTS), *WIDTHS]
    x, grids = "ocean_state", []
    for i, (a, b) in enumerate(pairwise(channels)):
        yield (
            f"down_{i}",
            convnext_block(
                x,
                f"down.{i}",
                a,
                b,
                DILATIONS[i],
                grid,
                f"ocean_{i}",
                partial,
                ("ocean_state", 1) if i == 0 else None,
            ),
        )
        yield f"pool_{i}", avg_pool(f"down_{i}")
        yield f"ocean_{i + 1}", coarser(f"ocean_{i}")
        grids.append(grid)
        x, grid = f"pool_{i}", grid.halved()
    levels = len(WIDTHS)
    yield (
        "middle",
        convnext_block(
            x,
            "middle",
            WIDTHS[-1],
            WIDTHS[-1],
            DILATIONS[-1],
            grid,
            f"ocean_{levels}",
            partial,
        ),
    )
    x = "middle"
    for level in reversed(range(levels)):
        yield f"up_{level}", upsample(x, f"down_{level}", grid)
        grid = grids[level]
        b = WIDTHS[level - 1] if level else WIDTHS[0]
        yield (
            f"up_block_{level}",
            convnext_block(
                f"up_{level}",
                f"up.{level}",
                WIDTHS[level],
                b,
                DILATIONS[level],
                grid,
                f"ocean_{level}",
                partial,
            ),
        )
        x = f"up_block_{level}"
    # The forecast is the state plus the predicted change, at the cells
    # where each channel's depth has ocean: the state's own rows.
    change = conv(
        x,
        "decoder",
        WIDTHS[0],
        1,
        grid,
        "ocean_0",
        ("ocean_0", WIDTHS[0]) if partial else None,
    )
    yield (
        "forecast",
        f"""
SELECT s.c, s.lat, s.lon, s.v + f.v AS v
FROM {change} f
JOIN ocean_state s ON f.c = s.c AND f.lat = s.lat AND f.lon = s.lon""",
    )


def register_erf(host: Host) -> None:
    """Register ``erf`` on the host, as a Python function over a column."""

    def erf(v: np.ndarray) -> np.ndarray:
        return torch.special.erf(torch.from_numpy(v.copy())).numpy()

    host.function("erf", erf)


# --------------------------------------------------------------------------
# The mirror: the same network on arrays, in PyTorch, with a mask for the
# rows that exist. It trains the weights and runs the long evaluations.
# --------------------------------------------------------------------------


def _pad(x: torch.Tensor, d: int) -> torch.Tensor:
    """Pad ``d`` cells: around the globe in longitude, with absent cells
    (zeros) past the poles."""
    x = F.pad(x, (d, d, 0, 0), mode="circular")
    return F.pad(x, (0, 0, d, d))


class Conv(nn.Module):
    """``conv``: a 3 × 3 convolution over the rows present."""

    def __init__(self, a: int, b: int, dilation: int, partial: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(b, a, 3, 3) / (3 * a**0.5))
        self.bias = nn.Parameter(torch.zeros(b))
        self.dilation, self.partial = dilation, partial

    def forward(self, x, present, out):
        d = self.dilation
        y = F.conv2d(_pad(x * present, d), self.weight, dilation=d)
        if self.partial:
            ones = torch.ones(1, x.shape[1], 3, 3, dtype=x.dtype)
            terms = F.conv2d(_pad(present.expand_as(x), d), ones, dilation=d)
            y = y * (9 * x.shape[1]) / terms.clamp(min=1)
        return (y + self.bias[:, None, None]) * out


class Pointwise(nn.Module):
    """``pointwise``: a 1 × 1 convolution."""

    def __init__(self, a: int, b: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(b, a) / a**0.5)
        self.bias = nn.Parameter(torch.zeros(b))

    def forward(self, x, present, out):
        y = torch.einsum("oi,bihw->bohw", self.weight, x * present)
        return (y + self.bias[:, None, None]) * out


class Norm(nn.Module):
    """``batch_norm``, with its statistics over the rows present."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x, out):
        if self.training:
            n = out.sum() * x.shape[0] / out.shape[0]
            mean = (x * out).sum((0, 2, 3)) / n
            var = (((x - mean[:, None, None]) * out) ** 2).sum((0, 2, 3)) / n
            with torch.no_grad():
                self.running_mean.lerp_(mean, 0.1)
                self.running_var.lerp_(var * n / (n - 1), 0.1)
        else:
            mean, var = self.running_mean, self.running_var
        y = (x - mean[:, None, None]) / torch.sqrt(var[:, None, None] + 1e-5)
        return (y * self.weight[:, None, None] + self.bias[:, None, None]) * out


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(F.gelu(x), max=10.0)


class Block(nn.Module):
    """``convnext_block``."""

    def __init__(self, a: int, b: int, dilation: int, partial: bool):
        super().__init__()
        self.conv1 = Conv(a, b, dilation, partial)
        self.norm1 = Norm(b)
        self.conv2 = Conv(b, b, dilation, partial)
        self.norm2 = Norm(b)
        self.mix = Pointwise(b, b)
        self.skip = Pointwise(a, b) if a != b else None

    def forward(self, x, present, out):
        h = _gelu(self.norm1(self.conv1(x, present, out), out))
        h = _gelu(self.norm2(self.conv2(h, out, out), out))
        h = self.mix(h, out, out)
        s = x * out if self.skip is None else self.skip(x, present, out)
        return s + h


def _avg_pool(x, present):
    total = F.avg_pool2d(x * present, 2)
    count = F.avg_pool2d(present.expand_as(x), 2)
    return total / count.clamp(min=1e-12) * (count > 0)


def _upsample_taps(n: int, coarse: int, periodic: bool):
    """For each of ``n`` fine cells, its two coarse sources and weights."""
    fine = torch.arange(n)
    parent, r = fine // 2, fine % 2
    near = torch.full((n,), 0.75)
    other = torch.where(r == 0, parent - 1, parent + 1)
    other = other % coarse if periodic else other.clamp(0, coarse - 1)
    return [(parent, near), (other, 1 - near)]


def _upsample(x, coarse_present, fine_present):
    """``upsample``, without the skip connection."""
    _, _, h, w = fine_present.shape
    total = torch.zeros(x.shape[:2] + (h, w), dtype=x.dtype)
    weight = torch.zeros((1, 1, h, w), dtype=x.dtype)
    xm = x * coarse_present
    for sy, wy in _upsample_taps(h, x.shape[2], periodic=False):
        for sx, wx in _upsample_taps(w, x.shape[3], periodic=True):
            ww = (wy[:, None] * wx[None, :]).to(x.dtype)
            total += ww * xm[:, :, sy][:, :, :, sx]
            weight += ww * coarse_present[:, :, sy][:, :, :, sx]
    return total / weight.clamp(min=1e-12) * fine_present


class UNet(nn.Module):
    """The network: the SQL's ``program``, on arrays.

    With ``dense``, every cell is present, land included, as Samudra
    computes it; land then reads as 0, the mean. The forecast is masked
    to each channel's ocean either way.
    """

    def __init__(self, partial: bool = True, dense: bool = False):
        super().__init__()
        partial = partial and not dense
        self.dense = dense
        channels = [len(INPUTS), *WIDTHS]
        self.down = nn.ModuleList(
            Block(a, b, DILATIONS[i], partial)
            for i, (a, b) in enumerate(pairwise(channels))
        )
        self.middle = Block(WIDTHS[-1], WIDTHS[-1], DILATIONS[-1], partial)
        self.up = nn.ModuleList(
            Block(
                WIDTHS[i],
                WIDTHS[i - 1] if i else WIDTHS[0],
                DILATIONS[i],
                partial,
            )
            for i in range(len(WIDTHS))
        )
        self.decoder = Conv(WIDTHS[0], len(STATE), 1, partial)
        # The network predicts the change in the state, so it starts out
        # as persistence: no change.
        nn.init.zeros_(self.decoder.weight)

    def forward(self, x, wet):
        """``x``: ``(batch, channel, lat, lon)``, 0 where a row is absent;
        ``wet``: ``(channel, lat, lon)``, whether each row exists."""
        wet = wet[None].to(x.dtype)
        if self.dense:
            present = torch.ones_like(wet)
        else:
            present = wet
        ocean = [present.amax(1, keepdim=True)]
        h, skips = x, []
        for i, block in enumerate(self.down):
            h = block(h, present if i == 0 else ocean[i], ocean[i])
            skips.append(h)
            h = _avg_pool(h, ocean[i])
            ocean.append(F.max_pool2d(ocean[i], 2))
        h = self.middle(h, ocean[-1], ocean[-1])
        for level in reversed(range(len(self.down))):
            fine = ocean[level]
            h = skips[level] + _upsample(h, ocean[level + 1], fine)
            h = self.up[level](h, fine, fine)
        change = self.decoder(h, ocean[0], ocean[0])
        state_wet = wet[:, : len(STATE)]
        return (x[:, : len(STATE)] + change) * state_wet


def parameters(model: UNet) -> dict[str, xr.Dataset]:
    """The mirror's weights as Datasets, one table per parameter, named as
    the SQL reads them."""
    out = {}
    norms = {k.rsplit(".", 1)[0] for k in model.state_dict() if "running" in k}
    for module in norms:
        cols = {
            leaf: model.state_dict()[f"{module}.{leaf}"].double().numpy()
            for leaf in ("weight", "bias", "running_mean", "running_var")
        }
        out[table(module)] = xr.Dataset(
            {k: (("c",), v) for k, v in cols.items()}
        )
    for key, value in model.state_dict().items():
        if key.rsplit(".", 1)[0] in norms:
            continue
        dims = {1: ("co",), 2: ("co", "ci"), 4: ("co", "ci", "ky", "kx")}
        out[table(key)] = xr.Dataset(
            {"v": (dims[value.ndim], value.double().numpy())}
        )
    return out


KERNEL = xr.Dataset(
    {"tap": (("ky", "kx"), np.ones((3, 3)))},
    coords={"ky": np.arange(3), "kx": np.arange(3)},
)
QUERY = "SELECT c, lat, lon, v FROM forecast ORDER BY c, lat, lon"


def tables(state: np.ndarray, wet: np.ndarray) -> dict[str, xr.Dataset]:
    """The input tables for one forecast: the state, NaN where a row is
    absent, and the grid's mask."""
    v = np.where(wet, state, np.nan)
    return {
        "state": xr.Dataset({"v": (("c", "lat", "lon"), v)}),
        "mask": xr.Dataset({"wet": (("lat", "lon"), wet.any(0))}),
        "kernel": KERNEL,
    }


def matrix(rows: pa.Table, shape: tuple[int, ...]) -> np.ndarray:
    """The rows of a ``(c, lat, lon, v)`` result as an array, NaN where a
    cell has no row."""
    out = np.full(shape, np.nan)
    c, lat, lon = (rows[k].to_numpy().astype(int) for k in ("c", "lat", "lon"))
    out[c, lat, lon] = rows["v"].to_numpy()
    return out


def mirror(model: UNet, state: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """The mirror's forecast of one state, in float64, NaN where absent."""
    model = model.double().eval()
    with torch.no_grad():
        x = torch.from_numpy(np.where(wet, state, 0.0))[None]
        y = model(x, torch.from_numpy(wet))[0].numpy()
    model.float()
    return np.where(wet[: len(STATE)], y, np.nan)


def sql_forecast(
    host: Host, model: UNet, state: np.ndarray, wet: np.ndarray, grid: Grid
) -> pa.Table:
    """Load one state and the weights, and run the network a layer at a
    time: each view's rows stored as a table before the next reads it, as
    PyTorch keeps each layer's output. (As one query of views, a layer
    read twice, by a block and its skip connection, is computed twice, so
    the work doubles with each block.)"""
    for name, ds in {**tables(state, wet), **parameters(model)}.items():
        host.load(name, ds)
    for view, sql in program(grid, model.down[0].conv1.partial):
        host.execute(f"CREATE OR REPLACE TABLE {view} AS {sql}")
    return host.query(QUERY)


# --------------------------------------------------------------------------
# OM4: streaming, normalizing, training and scoring.
# --------------------------------------------------------------------------


def _step_of_year(times: list[str]) -> np.ndarray:
    """Each time's step within its year. OM4's steps restart on January 3
    each year, five days apart, 73 to a year."""
    doy = np.array(
        [np.datetime64(t) - np.datetime64(t[:4] + "-01-01") for t in times]
    )
    return np.clip((doy.astype(int) - 2) // 5, 0, YEAR - 1)


@dataclass
class Streamed:
    """OM4's state and forcing over a span of time, in their units, NaN on
    land: ``x`` is ``(time, channel, lat, lon)`` over ``STATE + FORCING``."""

    x: np.ndarray
    times: list[str]


@dataclass
class OM4Grid:
    """What OM4's 2° grid says about each input channel: ``wet``, where it
    has a row, and ``weight``, each state channel's ocean area per cell
    (``areacello`` times ``ocean_fraction`` at its depth), used only
    relative to itself."""

    wet: np.ndarray
    weight: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def om4_grid() -> OM4Grid:
    """The masks, areas and normalization statistics, from OM4's stores."""
    ds = xr.open_zarr(OM4 + "OM4.zarr").isel(y=slice(FIRST_ROW, None))
    means = xr.open_zarr(OM4 + "OM4_means.zarr")
    stds = xr.open_zarr(OM4 + "OM4_stds.zarr")
    level = [int(v.rsplit("_", 1)[1]) if v[-1].isdigit() else 0 for v in INPUTS]
    masks = np.stack([ds[f"mask_{k}"].values for k in range(ds.sizes["lev"])])
    # Samudra's data notes: a cell that is ocean at one depth is ocean at
    # every depth above it.
    assert (masks[1:] <= masks[:-1]).all(), "OM4's masks don't nest with depth"
    fraction = ds.ocean_fraction.values
    area = ds.areacello.values
    weight = np.stack([area * fraction[k] for k in level[: len(STATE)]])
    # The anomaly's statistics come from the training years (see main).
    mean = np.array([float(means[v]) for v in STREAMED] + [0.0])
    std = np.array([float(stds[v]) for v in STREAMED] + [1.0])
    return OM4Grid(masks[level], weight, mean, std)


def om4(
    start: str, end: str, grid: OM4Grid, cache: str | None = None
) -> Streamed:
    """Stream OM4's state and forcing from ``start`` to ``end``, as dates.
    With ``cache``, a directory, keep the streamed values there and read
    them from there next time."""
    path = cache and pathlib.Path(cache) / f"om4_2deg_{start}_{end}.npz"
    if path and path.exists():
        saved = np.load(path)
        return Streamed(saved["x"], list(saved["times"]))
    ds = xr.open_zarr(OM4 + "OM4.zarr").isel(y=slice(FIRST_ROW, None))
    keys = [(t.year, t.month, t.day) for t in ds.time.values]
    first = tuple(map(int, start.split("-")))
    last = tuple(map(int, end.split("-")))
    span = [i for i, k in enumerate(keys) if first <= k <= last]
    sub = ds[STREAMED].isel(time=slice(span[0], span[-1] + 1)).load()
    x = np.stack([sub[v].values for v in STREAMED], axis=1)
    # And the data notes say NaN marks land exactly.
    wet = grid.wet[: len(STREAMED)]
    assert (np.isnan(x) == ~wet[None]).all(), "OM4's NaNs don't match its masks"
    streamed = Streamed(x, [str(t)[:10] for t in sub.time.values])
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=streamed.x, times=np.array(streamed.times))
    return streamed


def climatology(data: Streamed) -> np.ndarray:
    """The mean of each channel at each of the year's 73 steps, over the
    years of ``data``: ``(73, channel, lat, lon)``."""
    step = _step_of_year(data.times)
    with warnings.catch_warnings():
        # Land is NaN in every year.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack(
            [np.nanmean(data.x[step == k], axis=0) for k in range(YEAR)]
        )


@dataclass
class Ocean:
    """OM4 normalized for the network: ``x`` is ``(time, channel, lat,
    lon)`` over ``INPUTS``, 0 on land, and ``clim`` the training years'
    climatology at each time, normalized the same way."""

    x: np.ndarray
    clim: np.ndarray
    grid: OM4Grid
    times: list[str]


def normalize(data: Streamed, grid: OM4Grid, clim: np.ndarray) -> Ocean:
    """Normalize ``data`` by OM4's statistics, and add ``hfds_anomalies``:
    ``hfds`` minus its climatology, as Samudra 2's forcing has it."""
    step = _step_of_year(data.times)
    hfds = STREAMED.index("hfds")
    anomaly = data.x[:, hfds] - clim[step, hfds]
    x = np.concatenate([data.x, anomaly[:, None]], axis=1)
    c = np.concatenate([clim[step], np.zeros_like(anomaly)[:, None]], axis=1)
    scale = (grid.mean[None, :, None, None], grid.std[None, :, None, None])
    x, c = ((v - scale[0]) / scale[1] for v in (x, c))
    wet = grid.wet[None]
    return Ocean(
        np.where(wet, np.nan_to_num(x), 0.0).astype(np.float32),
        np.where(wet, np.nan_to_num(c), 0.0).astype(np.float32),
        grid,
        data.times,
    )


def inputs(ocean: Ocean, t: np.ndarray, state=None) -> torch.Tensor:
    """The network's input for steps ``t``: the state at ``t`` (or
    ``state``) and the forcing over the five days to ``t + 1``."""
    s = len(STATE)
    state = ocean.x[t, :s] if state is None else state
    return torch.from_numpy(np.concatenate([state, ocean.x[t + 1, s:]], axis=1))


def train(
    model: UNet,
    ocean: Ocean,
    epochs: int,
    unroll: int = 1,
    by_change: bool = False,
    batch: int = 8,
    seed: int = 0,
) -> list[float]:
    """Train on every step of ``ocean``, for the mean squared error over
    ocean rows of the normalized forecast, rolled out ``unroll`` steps as
    Samudra trains. With ``by_change``, each channel's error is divided by
    the variance of its five-day change, so a channel that changes little,
    such as the deep ocean, counts as much as one that changes a lot, as
    Samudra 2's variance-weighted loss does. Returns each epoch's loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    wet = torch.from_numpy(ocean.grid.wet)
    target_wet = wet[: len(STATE)].float()
    if by_change:
        change = np.diff(ocean.x[:, : len(STATE)], axis=0)
        w = ocean.grid.wet[None, : len(STATE)]
        var = (change**2 * w).sum((0, 2, 3)) / (w.sum((0, 2, 3)) * len(change))
        target_wet = target_wet / torch.from_numpy(var).float()[:, None, None]
        target_wet = target_wet / target_wet.mean()
    steps = np.arange(len(ocean.x) - unroll)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    total = epochs * -(-len(steps) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=total)
    losses = []
    model.float().train()
    for _ in range(epochs):
        rng.shuffle(steps)
        run = []
        for i in range(0, len(steps), batch):
            t = steps[i : i + batch]
            loss, state = torch.zeros(()), None
            for k in range(unroll):
                x = inputs(ocean, t + k)
                if state is not None:
                    x = torch.cat([state, x[:, len(STATE) :]], dim=1)
                state = model(x, wet)
                y = torch.from_numpy(ocean.x[t + k + 1, : len(STATE)])
                err = (state - y) ** 2 * target_wet
                loss = loss + err.sum() / (
                    wet[: len(STATE)].sum() * len(t) * unroll
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run.append(loss.item())
        losses.append(float(np.mean(run)))
    return losses


def _coast(wet: np.ndarray) -> np.ndarray:
    """Ocean cells with land among their eight neighbours. Longitude wraps
    around; past a pole there is no neighbour."""
    near = np.zeros_like(wet)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            land = np.roll(~wet, (dy, dx), axis=(-2, -1))
            if dy:
                land[..., 0 if dy > 0 else -1, :] = False
            near |= land
    return wet & near


def rmse(ocean: Ocean, got: np.ndarray, want: np.ndarray, where: np.ndarray):
    """Area-weighted RMSE per state channel, in each channel's units, over
    the cells ``where`` holds. ``got`` and ``want`` are normalized."""
    s = len(STATE)
    err = (got - want) * ocean.grid.std[None, :s, None, None]
    w = ocean.grid.weight * where
    return np.sqrt(
        (err**2 * w[None]).sum((0, 2, 3)) / (w.sum((1, 2)) * len(got))
    )


def predict(model: UNet, ocean: Ocean, t: np.ndarray) -> np.ndarray:
    model.float().eval()
    with torch.no_grad():
        return model(inputs(ocean, t), torch.from_numpy(ocean.grid.wet)).numpy()


def rollout(model: UNet, ocean: Ocean, steps: int) -> np.ndarray:
    """Forecast ``steps`` steps from the first, each from the last
    forecast, with the forcing from the data."""
    state = ocean.x[:1, : len(STATE)]
    out = []
    model.float().eval()
    wet = torch.from_numpy(ocean.grid.wet)
    with torch.no_grad():
        for t in range(steps):
            state = model(inputs(ocean, np.array([t]), state), wet).numpy()
            out.append(state[0])
    return np.stack(out)


def physical(ocean: Ocean, x: np.ndarray, name: str) -> np.ndarray:
    """Channel ``name`` of normalized states ``x`` in its units, NaN on
    land."""
    c = INPUTS.index(name)
    g = ocean.grid
    return np.where(g.wet[c], x[:, c] * g.std[c] + g.mean[c], np.nan)


def area_mean(ocean: Ocean, x: np.ndarray, name: str) -> np.ndarray:
    """The ocean-area-weighted mean of channel ``name``, per state."""
    w = ocean.grid.weight[STATE.index(name)]
    return np.nansum(physical(ocean, x, name) * w, axis=(1, 2)) / w.sum()


# A linear equation of state, to compare stratification between the
# forecast and the data, not to compute density: thermal expansion and
# haline contraction near the surface.
ALPHA, BETA = 2e-4, 7.6e-4


def physics(ocean: Ocean, x: np.ndarray) -> dict[str, np.ndarray]:
    """Checks from Samudra's notes on ocean physics, per state: the means
    that shouldn't drift, the share of ocean below freezing, and the share
    of columns lighter at 105 m than at the surface."""
    t0, s0 = physical(ocean, x, "thetao_0"), physical(ocean, x, "so_0")
    t1, s1 = physical(ocean, x, "thetao_5"), physical(ocean, x, "so_5")
    w = ocean.grid.weight[STATE.index("thetao_0")]
    below = np.nansum((t0 < -0.054 * s0 - 0.05) * w, axis=(1, 2)) / w.sum()
    # Density, up to a constant: lighter below means an inversion.
    lighter = -ALPHA * (t1 - t0) + BETA * (s1 - s0) < 0
    deep = ocean.grid.wet[INPUTS.index("thetao_5")]
    w1 = ocean.grid.weight[STATE.index("thetao_5")]
    inverted = np.nansum((lighter & deep) * w1, axis=(1, 2)) / w1.sum()
    return {
        "mean thetao_0 (°C)": area_mean(ocean, x, "thetao_0"),
        "mean so_0 (psu)": area_mean(ocean, x, "so_0"),
        "mean so_10 (psu)": area_mean(ocean, x, "so_10"),
        "mean zos (m)": area_mean(ocean, x, "zos"),
        "below freezing": below,
        "inverted 0-105 m": inverted,
    }


def table_row(name: str, values) -> str:
    return f"| {name} | " + " | ".join(f"{v:.4g}" for v in values) + " |"


def score(models: dict[str, UNet], test: Ocean, steps: int) -> dict:
    """Print the accuracy tables, and return them: one-step RMSE over all
    ocean and over the coast, against persistence and climatology, with
    each model's skill against the better of the two, and a rollout's error
    and physics."""
    s = len(STATE)
    results: dict = {"one_step": {}, "skill": {}, "rollout": {}, "physics": {}}
    t = np.arange(len(test.x) - 1)
    want = test.x[t + 1, :s]
    wet = test.grid.wet[:s]
    header = "| | " + " | ".join(STATE) + " |"
    rule = "|---" * (s + 1) + "|"
    forecasts = {
        "persistence": test.x[t, :s],
        "climatology": test.clim[t + 1, :s],
    }
    for name, model in models.items():
        forecasts[name] = np.concatenate(
            [predict(model, test, t[i : i + 32]) for i in range(0, len(t), 32)]
        )
    for title, where in (("ocean", wet), ("coast", _coast(wet))):
        errors = {
            n: rmse(test, got, want, where) for n, got in forecasts.items()
        }
        best = np.minimum(errors["persistence"], errors["climatology"])
        print(
            f"\nOne-step RMSE over the {title}, {len(t)} steps from {test.times[0]}:\n"
        )
        print(header)
        print(rule)
        for name, e in errors.items():
            print(table_row(name, e))
        results["one_step"][title] = {
            n: dict(zip(STATE, e.tolist())) for n, e in errors.items()
        }
        results["skill"][title] = {}
        for name in models:
            skill = 1 - errors[name] / best
            print(table_row(f"{name}: skill", skill))
            results["skill"][title][name] = dict(zip(STATE, skill.tolist()))
    steps = min(steps, len(test.x) - 1)
    truth = test.x[1 : steps + 1, :s]
    marks = sorted({1, 4, YEAR, 2 * YEAR, steps} & set(range(1, steps + 1)))
    print(
        f"\nRollout from {test.times[0]}: RMSE against the data at each step\n"
    )
    print("| model | step | date | " + " | ".join(STATE) + " |")
    print("|---" * (s + 3) + "|")
    runs = {"persistence": np.repeat(test.x[:1, :s], steps, axis=0)}
    for name, model in models.items():
        runs[name] = rollout(model, test, steps)
    for name, run in runs.items():
        results["rollout"][name] = {}
        for k in marks:
            e = rmse(test, run[k - 1 : k], truth[k - 1 : k], wet)
            print(table_row(f"{name} | {k} | {test.times[k]}", e))
            results["rollout"][name][k] = dict(zip(STATE, e.tolist()))
    print(
        f"\nRollout physics at step {steps} ({test.times[steps]}), "
        "and the data's:\n"
    )
    checks = {"data": physics(test, truth)}
    checks.update(
        {n: physics(test, r) for n, r in runs.items() if n != "persistence"}
    )
    keys = list(checks["data"])
    print("| | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for name, c in checks.items():
        print(table_row(f"{name}, start", [c[k][0] for k in keys]))
        print(table_row(f"{name}, end", [c[k][-1] for k in keys]))
        results["physics"][name] = {k: [c[k][0], c[k][-1]] for k in keys}
    return results


def metamorphic(model: UNet, ocean: Ocean) -> dict[str, float]:
    """Invariants from Samudra's notes that a model must satisfy, as the
    largest change each makes to a forecast (0 when it holds):

    * values under land don't matter;
    * rolling the input (and masks) in longitude by a multiple of the
      network's downsampling rolls the forecast the same way;
    * in evaluation, a forecast doesn't depend on its batch-mates."""
    model.float().eval()
    wet = torch.from_numpy(ocean.grid.wet)
    x = inputs(ocean, np.array([0, 1]))
    with torch.no_grad():
        y = model(x, wet)
        noisy = torch.where(wet, x, torch.randn_like(x) * 10)
        land = (model(noisy, wet) - y).abs().max().item()
        n = 2 ** len(WIDTHS) * 3
        rolled = model(x.roll(n, -1), wet.roll(n, -1)).roll(-n, -1)
        roll = (rolled - y).abs().max().item()
        alone = (model(x[:1], wet) - y[:1]).abs().max().item()
    return {"land values": land, f"roll by {n}": roll, "batch": alone}


# --------------------------------------------------------------------------
# The command line.
# --------------------------------------------------------------------------


def synthetic(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A random state on an ``n × 2n`` grid with a random coastline, deeper
    channels wet on less of it."""
    rng = np.random.default_rng(seed)
    grid = (n, 2 * n)
    depth = rng.uniform(size=grid)
    wet = np.stack(
        [
            depth > (0.3 if "_" not in v or v.endswith("_0") else 0.5)
            for v in INPUTS
        ]
    )
    state = rng.normal(size=(len(INPUTS), *grid))
    return state, wet


def check(report: Report, hosts_: list[Host], model, state, wet, size, step):
    """Run the SQL forecast on each host and check it against the
    mirror's, in float64."""
    grid = Grid(*state.shape[1:])
    want = mirror(model, state, wet)
    for host in hosts_:
        register_erf(host)
        rows = report.time(
            step, host, size, sql_forecast, host, model, state, wet, grid
        )
        report.check(
            assert_close(
                step, matrix(rows, want.shape), want, rtol=1e-8, atol=1e-8
            )
        )


def main() -> None:
    parser = cli(__doc__.splitlines()[0], sizes=[8], size_name="lat")
    parser.add_argument(
        "--data",
        choices=["synthetic", "om4"],
        default="synthetic",
        help="a random grid and weights, or OM4 at 2° (streamed)",
    )
    parser.add_argument(
        "--partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rescale each convolution by the share of its taps present",
    )
    parser.add_argument(
        "--train",
        nargs=2,
        default=["1975-01-01", "1984-12-31"],
        metavar=("START", "END"),
        help="OM4 dates to train on",
    )
    parser.add_argument(
        "--test",
        nargs=2,
        default=["2014-10-10", "2022-12-24"],
        metavar=("START", "END"),
        help="held-out OM4 dates",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--unroll",
        type=int,
        default=1,
        help="steps each training sample rolls out",
    )
    parser.add_argument("--cache", help="a directory to keep streamed OM4 in")
    parser.add_argument("--json", help="also write the accuracy tables here")
    parser.add_argument(
        "--by-change",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="weight each channel's error by its five-day change's variance",
    )
    parser.add_argument(
        "--dense",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also train the model computed over land, to compare",
    )
    parser.add_argument(
        "--rollout",
        type=int,
        default=8 * YEAR,
        help="steps to forecast from the first test date",
    )
    parser.add_argument(
        "--sql-steps",
        type=int,
        default=2,
        help="held-out steps to forecast in SQL too",
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="PyTorch's threads for training"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report = Report("samudra_sparse", "lat", args.reps)
    if args.data == "synthetic":
        for n in args.sizes:
            state, wet = synthetic(n)
            torch.manual_seed(0)
            model = UNet(partial=args.partial)
            with torch.no_grad():
                for name, b in model.named_buffers():
                    if name.endswith("running_mean"):
                        b.copy_(0.1 * torch.randn(b.shape))
                    elif name.endswith("running_var"):
                        b.copy_(0.5 + torch.rand(b.shape))
                for name, p in model.named_parameters():
                    if name.endswith("bias"):
                        p.copy_(0.1 * torch.randn(p.shape))
                # A trained network's decoder isn't 0, as it starts.
                model.decoder.weight.normal_(0, 0.3)
            check(report, hosts(args.hosts), model, state, wet, n, "forecast")
        report.finish(args.csv)
        return
    started = time.time()
    grid = om4_grid()
    (start, end), (test_start, test_end) = args.train, args.test
    raw = om4(start, end, grid, cache=args.cache)
    held_out = om4(test_start, test_end, grid, cache=args.cache)
    print(
        f"Streamed {len(raw.x)} training and {len(held_out.x)} held-out "
        f"steps in {time.time() - started:.0f} s."
    )
    # The climatology, and the anomaly's spread, from the training years.
    clim = climatology(raw)
    hfds = STREAMED.index("hfds")
    anomaly = raw.x[:, hfds] - clim[_step_of_year(raw.times), hfds]
    grid.std[-1] = np.nanstd(anomaly)
    data, test = normalize(raw, grid, clim), normalize(held_out, grid, clim)
    models = {"sparse": UNet(partial=args.partial)}
    if args.dense:
        models["dense"] = UNet(dense=True)
    for name, model in models.items():
        started = time.time()
        losses = train(
            model,
            data,
            args.epochs,
            unroll=args.unroll,
            by_change=args.by_change,
        )
        print(
            f"Trained {name} for {args.epochs} epochs in "
            f"{time.time() - started:.0f} s; loss {losses[0]:.4f} → "
            f"{losses[-1]:.4f}."
        )
        print(f"Invariants, {name}: {metamorphic(model, test)}")
    for t in range(args.sql_steps):
        x = inputs(test, np.array([t]))[0].double().numpy()
        check(
            report,
            hosts(args.hosts),
            models["sparse"],
            x,
            grid.wet,
            test.x.shape[2],
            f"forecast of {test.times[t + 1]}",
        )
    report.finish(args.csv)
    results = score(models, test, args.rollout)
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
