# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "xarray-sql[ml,duckdb]",
#   "ddxdb[duckdb]",
#   "einfold[datafusion]>=0.1.0rc2",
#   "torch",
#   "samudra @ git+https://github.com/m2lines/Samudra@4f86be4",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Demo 4: Samudra 2's U-Net in SQL, checked against Samudra's PyTorch.

Samudra (Dheeshjith et al., 2024, https://arxiv.org/abs/2412.03795) is an
emulator of the global ocean: from the ocean's state at two times, five
days apart, and the forcing at its surface, it predicts the next two
states. Samudra 2 (Yuan et al., 2026, https://arxiv.org/abs/2606.02610)
scales it from 1° to 1/4°. Its network is a U-Net (Ronneberger et al.,
2015, https://arxiv.org/abs/1505.04597): an encoder that halves the grid
at each level and a decoder that doubles it again, joined at each level by
a skip connection. Each level is a ConvNeXt block (Liu et al., 2022,
https://arxiv.org/abs/2201.03545), two dilated 3 × 3 convolutions, each
followed by a batch norm and a GELU, and a 1 × 1 convolution to the
level's width, added to its input.

A state is a table of ``(c, lat, lon, v)``: a channel (a variable at a
depth), a grid cell, and its value. Every piece of the network is the SQL
of its PyTorch module, named after it:

* ``conv``: each input cell, through each tap of the kernel, reaches the
  output cell the tap's offset points to; the sum over taps and input
  channels is the convolution. Longitude wraps around the globe, so its
  offset is taken modulo the grid's width. Latitude doesn't: an offset
  past a pole reaches no cell, as Samudra's zero padding there gives;
* ``batch_norm``: the running mean and variance it was trained with;
* ``capped_gelu``: GELU, ``x · Φ(x)``, capped at 10, as Samudra uses it.
  GELU needs the error function ``erf``, which neither DataFusion nor
  DuckDB has, so the SQL computes it by Abramowitz and Stegun's formula
  7.1.26, whose error is below 1.5e-7;
* ``avg_pool``: the mean of each 2 × 2 block of cells;
* ``upsample``: bilinear interpolation to twice the resolution, periodic in
  longitude, plus the skip connection from the encoder.

Each block is a view of the views before it, and the forecast is one query
of the last. Each host (an engine that runs the SQL; see ``_hosts.py``)
plans it as a whole.

The check is Samudra's own model: its ``UNetBackbone``, ``ConvNeXtBlock``
and ``Samudra`` modules, built as Samudra 2's configuration builds them,
with random weights and random batch-norm statistics, and run in float32
as Samudra runs. This draft runs the network at reduced width (4 to 10
channels) on a small grid; ``--levels`` sets its depth, 1 to 4, where
Samudra 2 has 4.

Run ``uv run benchmarks/ml/04_samudra.py --help`` for the options.
``--sizes`` is the number of latitude cells; the grid is twice as wide.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import torch
import xarray as xr
from _bench import Report, assert_close, cli, hosts
from samudra.models import Samudra
from samudra.models.modules import (
    AvgPool,
    CappedGELU,
    ConvNeXtBlock,
    UNetBackbone,
    ZonallyPeriodicBilinearUpsample,
)

# Samudra 2's widths and dilations, by level (its model.yaml has widths
# 280, 380, 480 and 520); this draft's network is narrower.
WIDTHS = [4, 6, 8, 10]
DILATIONS = [1, 2, 4, 8]
IN_CHANNELS, OUT_CHANNELS = 3, 2

# --------------------------------------------------------------------------
# The network, in SQL. A state is a table of (c, lat, lon, v).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """The number of latitude and longitude cells at one level."""

    lat: int
    lon: int

    def halved(self) -> Grid:
        return Grid(self.lat // 2, self.lon // 2)


def table(module: str, leaf: str | None = None) -> str:
    """The table holding a parameter of one of Samudra's modules: its name
    in the model's state dict, with underscores for dots."""
    name = module if leaf is None else f"{module}.{leaf}"
    return name.replace(".", "_")


def conv(x: str, module: str, dilation: int, grid: Grid) -> str:
    """A 3 × 3 convolution with bias, as ``torch.nn.Conv2d`` after
    Samudra's padding: circular in longitude, zeros in latitude."""
    return f"""(
  SELECT p.c, p.lat, p.lon, p.v + b.v AS v
  FROM (
    SELECT w.co AS c,
           x.lat - (w.ky - 1) * {dilation} AS lat,
           (x.lon - (w.kx - 1) * {dilation} + {grid.lon}) % {grid.lon} AS lon,
           SUM(x.v * w.v) AS v
    FROM {x} x JOIN {table(module, "weight")} w ON x.c = w.ci
    GROUP BY 1, 2, 3) p
  JOIN {table(module, "bias")} b ON p.c = b.co
  WHERE p.lat BETWEEN 0 AND {grid.lat - 1})"""


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
    """``torch.nn.BatchNorm2d`` in evaluation, with its running statistics."""
    return f"""(
  SELECT x.c, x.lat, x.lon,
         (x.v - n.running_mean) / sqrt(n.running_var + 1e-5) * n.weight
           + n.bias AS v
  FROM {x} x JOIN {table(module)} n ON x.c = n.c)"""


def capped_gelu(x: str) -> str:
    """Samudra's ``CappedGELU``: ``x · Φ(x)``, at most 10, where
    ``Φ(x) = (1 + erf(x / √2)) / 2``."""
    return f"""(
  SELECT c, lat, lon,
         least(0.5 * v * (1 + CASE WHEN v < 0 THEN -erf ELSE erf END), 10.0)
           AS v
  FROM (
    -- erf(|v| / √2), by Abramowitz and Stegun's formula 7.1.26.
    SELECT c, lat, lon, v,
           1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
             + t * (-1.453152027 + t * 1.061405429)))) * exp(-v * v / 2)
             AS erf
    FROM (SELECT c, lat, lon, v, 1 / (1 + 0.3275911 * abs(v) / sqrt(2)) AS t
          FROM {x})))"""


def convnext_block(
    x: str, module: str, a: int, b: int, dilation: int, grid: Grid
) -> str:
    """Samudra's ``ConvNeXtBlock`` from ``a`` to ``b`` channels: two
    dilated convolutions, each with a batch norm and a capped GELU, then a
    1 × 1 convolution, added to the input (through a 1 × 1 convolution of
    its own when the widths differ)."""
    h = capped_gelu(
        batch_norm(
            conv(x, f"{module}.convblock.0", dilation, grid),
            f"{module}.convblock.1",
        )
    )
    h = capped_gelu(
        batch_norm(
            conv(h, f"{module}.convblock.3", dilation, grid),
            f"{module}.convblock.4",
        )
    )
    h = pointwise(h, f"{module}.convblock.6")
    skip = x if a == b else pointwise(x, f"{module}.skip_module")
    return f"""
SELECT h.c, h.lat, h.lon, s.v + h.v AS v
FROM {h} h
JOIN {skip} s ON h.c = s.c AND h.lat = s.lat AND h.lon = s.lon"""


def avg_pool(x: str, grid: Grid) -> str:
    """``torch.nn.AvgPool2d(2)``: the mean of each 2 × 2 block of cells. An
    odd last row or column is dropped."""
    return f"""
SELECT c, floor(lat / 2.0) AS lat, floor(lon / 2.0) AS lon, AVG(v) AS v
FROM {x}
WHERE lat < {2 * (grid.lat // 2)} AND lon < {2 * (grid.lon // 2)}
GROUP BY 1, 2, 3"""


def upsample(x: str, skip: str, coarse: Grid, fine: Grid) -> str:
    """Samudra's ``ZonallyPeriodicBilinearUpsample``, then the skip
    connection: the upsampled grid is centred on the encoder's, which may
    be a row or column larger, and added to it."""
    top = (fine.lat - 2 * coarse.lat) // 2
    left = (fine.lon - 2 * coarse.lon) // 2
    return f"""
WITH taps(r, d, w) AS (
  -- Fine cell 2i + r reads coarse cells i + d with weight w.
  VALUES (0, -1, 0.25), (0, 0, 0.75), (1, 0, 0.75), (1, 1, 0.25)),
reads AS (
  -- Each fine cell reads four coarse cells, with bilinear weights. Latitude
  -- stops at the poles; longitude wraps around.
  SELECT x.c, 2 * x.lat + ty.r + {top} AS lat, 2 * x.lon + tx.r + {left} AS lon,
         ty.w * tx.w AS w,
         greatest(0, least({coarse.lat - 1}, x.lat + ty.d)) AS source_lat,
         (x.lon + tx.d + {coarse.lon}) % {coarse.lon} AS source_lon
  FROM {x} x CROSS JOIN taps ty CROSS JOIN taps tx),
up AS (
  SELECT r.c, r.lat, r.lon, SUM(r.w * s.v) AS v
  FROM reads r
  JOIN {x} s ON s.c = r.c AND s.lat = r.source_lat AND s.lon = r.source_lon
  GROUP BY r.c, r.lat, r.lon)
SELECT k.c, k.lat, k.lon, k.v + coalesce(u.v, 0) AS v
FROM {skip} k
LEFT JOIN up u ON k.c = u.c AND k.lat = u.lat AND k.lon = u.lon"""


def program(levels: int, grid: Grid) -> Iterator[tuple[str, str]]:
    """Each view of the network and its SQL, in ``UNetBackbone``'s order:
    its modules are numbered as in the state dict, a block or a pooling
    or an upsampling each."""
    widths, dilations = WIDTHS[:levels], DILATIONS[:levels]
    channels = [IN_CHANNELS, *widths]
    layer, x, skips = 0, "state", []
    for i, (a, b) in enumerate(zip(channels, channels[1:])):
        yield (
            f"down_{i}",
            convnext_block(x, f"unet.layers.{layer}", a, b, dilations[i], grid),
        )
        yield f"pool_{i}", avg_pool(f"down_{i}", grid)
        skips.append(grid)
        layer, x, grid = layer + 2, f"pool_{i}", grid.halved()
    yield (
        "middle",
        convnext_block(
            x,
            f"unet.layers.{layer}",
            widths[-1],
            widths[-1],
            dilations[-1],
            grid,
        ),
    )
    x, layer = "middle", layer + 1
    ups = list(reversed(channels))[:-1]
    # UNetBackbone gives its last block the dilation of the block before
    # it (or, with one level, the first block's).
    up_dilations = list(reversed(dilations))[: levels - 1]
    up_dilations.append(up_dilations[-1] if up_dilations else dilations[0])
    for i, (a, b) in enumerate(zip(ups, ups[1:] + [widths[0]])):
        level = levels - 1 - i
        yield f"up_{level}", upsample(x, f"down_{level}", grid, skips[level])
        grid = skips[level]
        layer += 1
        name = f"up_block_{level}" if level else "unet"
        yield (
            name,
            convnext_block(
                f"up_{level}",
                f"unet.layers.{layer}",
                a,
                b,
                up_dilations[i],
                grid,
            ),
        )
        x, layer = name, layer + 1
    yield (
        "forecast",
        f"""
SELECT f.c, f.lat, f.lon, f.v
FROM {conv("unet", "decoder", 1, grid)} f
JOIN wet m ON f.c = m.c AND f.lat = m.lat AND f.lon = m.lon
WHERE m.wet""",
    )


# --------------------------------------------------------------------------
# The reference: Samudra's PyTorch modules.
# --------------------------------------------------------------------------


def samudra(levels: int, grid: Grid, seed: int) -> Samudra:
    """Samudra 2's network, as its configuration builds it, at this demo's
    widths, with random weights and batch-norm statistics."""

    def block(**kwargs) -> ConvNeXtBlock:
        return ConvNeXtBlock(
            **kwargs,
            kernel_size=3,
            activation=CappedGELU,
            upscale_factor=2,
            norm="batch",
        )

    unet = UNetBackbone(
        in_channels=IN_CHANNELS,
        ch_width=WIDTHS[:levels],
        dilation=DILATIONS[:levels],
        n_layers=[1] * levels,
        pad="circular",
        create_block=block,
        downsampling_block=AvgPool(),
        create_upsampling_block=lambda **_: ZonallyPeriodicBilinearUpsample(),
        checkpointing=None,
    )
    model = Samudra(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        unet=unet,
        pos_channels=0,
        add_3d_coordinates=None,
        input_steps=1,
        grid_size=(grid.lat, grid.lon),
        gradient_detach_interval=0,
        use_bfloat16=False,
    )
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(0.3 * torch.randn(p.shape, generator=rng))
        for name, b in model.named_buffers():
            if name.endswith("running_mean"):
                b.copy_(0.1 * torch.randn(b.shape, generator=rng))
            elif name.endswith("running_var"):
                b.copy_(0.5 + torch.rand(b.shape, generator=rng))
    return model.eval()


def forecast(model: Samudra, state: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Samudra's ``forward_once`` on one state, with the ocean mask."""

    class Batch:
        label_mask = torch.from_numpy(wet[None])
        input_resolution_cpu = None

    with torch.no_grad():
        x = torch.from_numpy(state[None])
        return model.forward_once(x, x[:, :0], Batch()).numpy()[0]


def parameters(model: Samudra) -> dict[str, xr.Dataset]:
    """Every parameter of the model as a Dataset: a convolution's weight
    over ``(co, ci, ky, kx)`` and bias over ``co``, and a batch norm's four
    vectors over ``c``, as one table."""
    state = {
        k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()
    }
    norms = {k.rpartition(".")[0] for k in state if k.endswith("running_mean")}
    out = {}
    for module in norms:
        out[table(module)] = xr.Dataset(
            {
                leaf: ("c", state[f"{module}.{leaf}"])
                for leaf in ("weight", "bias", "running_mean", "running_var")
            }
        )
    for key, value in state.items():
        module, _, leaf = key.rpartition(".")
        # The rest of the state dict holds a batch norm's step count and
        # each CappedGELU's cap, which the SQL writes as a constant.
        if module in norms or leaf not in ("weight", "bias"):
            continue
        dims = ("co", "ci", "ky", "kx") if leaf == "weight" else ("co",)
        out[table(key)] = xr.Dataset({"v": (dims, value)})
    return out


def matrix(rows: pa.Table, shape: tuple[int, ...]) -> np.ndarray:
    """The rows of a ``(c, lat, lon, v)`` result as an array, zero where a
    cell has no row (land)."""
    out = np.zeros(shape)
    c, lat, lon = (rows[k].to_numpy().astype(int) for k in ("c", "lat", "lon"))
    out[c, lat, lon] = rows["v"].to_numpy()
    return out


def main() -> None:
    parser = cli(__doc__.splitlines()[0], sizes=[10], size_name="lat")
    parser.add_argument(
        "--levels",
        type=int,
        default=2,
        choices=range(1, 5),
        help="the U-Net's depth (Samudra 2's is 4)",
    )
    args = parser.parse_args()
    report = Report("samudra", "lat", args.reps)
    rng = np.random.default_rng(0)
    for n in args.sizes:
        grid = Grid(n, 2 * n)
        model = samudra(args.levels, grid, seed=0)
        state = rng.normal(size=(IN_CHANNELS, grid.lat, grid.lon)).astype(
            np.float32
        )
        wet = rng.uniform(size=(OUT_CHANNELS, grid.lat, grid.lon)) > 0.3
        want = forecast(model, state, wet)
        data = {
            "state": xr.Dataset(
                {"v": (("c", "lat", "lon"), state.astype(np.float64))}
            ),
            "wet": xr.Dataset({"wet": (("c", "lat", "lon"), wet)}),
            **parameters(model),
        }
        for host in hosts(args.hosts):
            for name, ds in data.items():
                host.load(name, ds)
            for view, sql in program(args.levels, grid):
                host.execute(f"CREATE OR REPLACE VIEW {view} AS {sql}")
            query = "SELECT c, lat, lon, v FROM forecast ORDER BY c, lat, lon"
            rows = report.time(
                f"forecast ({args.levels} levels)", host, n, host.query, query
            )
            got = matrix(rows, want.shape)
            report.check(
                assert_close("forecast", got, want, rtol=1e-5, atol=1e-5)
            )
            report.note("operators", host.operators(query))
    report.finish(args.csv)


if __name__ == "__main__":
    main()
