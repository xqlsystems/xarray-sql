# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql[ml,duckdb]",
#   "jax",
#   "gcsfs",
#   "zarr>=3",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Demo 3: a GraphCast-shaped weather model in SQL, over ERA5 in Zarr.

GraphCast (Lam et al., 2023) forecasts the atmosphere 6 hours ahead from
the two previous states. It is a graph neural network in three stages:

* an **encoder** maps the latitude–longitude grid onto a "multi-mesh",
  a refined icosahedron that keeps the edges of every refinement level;
* a **processor** passes messages along the mesh's edges, round after
  round;
* a **decoder** maps the mesh back onto the grid, as the 6-hour change.

Here it is in miniature, as SQL over ERA5, ECMWF's reanalysis of the
atmosphere since 1940. xarray-sql reads it from the Zarr store of
WeatherBench 2, a benchmark for weather models (or a synthetic stand-in,
offline). Every
graph is a table of edges ``(edge, sender, receiver)``, every state a
table of ``(id, f, val)``, and every stage the same relational steps:

=================  ============================================
GNN step           In SQL
=================  ============================================
Gather             join the nodes' states to the edges, by sender
                   and by receiver
Concatenate        ``UNION ALL``, offsetting each part's features
MLP                einsums with ``params`` around swish, ``x / (1 + exp(-x))``
Aggregate          ``SUM`` of the edges' messages, ``GROUP BY`` receiver
=================  ============================================

The geometry (the mesh, and which grid cells and mesh nodes are joined)
is computed once with numpy; the model, its inputs and its forecast are
SQL. The script checks the forecast, and the gradient of the forecast
error, against the same model in JAX, Google's library for differentiable
array programs. ddx (https://github.com/xqlsystems/ddx), which
differentiates SQL queries, computes the gradient.

Then it measures what the third target demo of einfold
(https://github.com/xqlsystems/einfold, a query optimizer that rewrites
SQL plans into array algorithms; the demo is in its ``docs/demos.md``) is
about: how much less a regional forecast needs than a global one. It
reports the receptive field: how many mesh nodes and grid cells each
stage needs for the region.

Simplifications from GraphCast: random weights, a 64×32 grid and six
variables, a mesh refined 2 times rather than 6, 2 processor rounds
rather than 16, no biases or layer norms, edge features of 3-D
displacement and length, decoder edges from each cell's 3 nearest mesh
nodes, and edge states that are embedded once rather than updated.

Run ``uv run benchmarks/ml/03_graphcast.py --help`` for the options.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
from ddxdb import DdxError
from _bench import Report, assert_close, cli, hosts

jax.config.update("jax_enable_x64", True)

ERA5 = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"
)
ERA5_TABLES = {
    ("time", "longitude", "latitude"): "surface",
    ("time", "level", "longitude", "latitude"): "atmosphere",
}
# The forecast's variables, as (table, variable, pressure level).
VARIABLES = [
    ("surface", "2m_temperature", None),
    ("surface", "mean_sea_level_pressure", None),
    ("surface", "10m_u_component_of_wind", None),
    ("surface", "10m_v_component_of_wind", None),
    ("atmosphere", "geopotential", 500),
    ("atmosphere", "temperature", 850),
]
# The two input states and the state to forecast, 6 hours apart.
TIMES = pd.to_datetime(["2020-01-01T00", "2020-01-01T06", "2020-01-01T12"])

HIDDEN = 16  # channels of every node and edge state
INPUTS = 2 * len(VARIABLES) + 3  # two states, cos(lat), sin(lon), cos(lon)
EDGE_FEATURES = 4  # receiver minus sender in 3-D, and the edge's length
# A region to forecast: the contiguous United States, roughly.
BOUNDS = {"latitude": (25, 50), "longitude": (235, 295)}
REGION = " AND ".join(
    f"c.{coord} BETWEEN {low} AND {high}"
    for coord, (low, high) in BOUNDS.items()
)

# --------------------------------------------------------------------------
# The model, in SQL.
# --------------------------------------------------------------------------


def mlp(out: str, x: str, net: str) -> str:
    """A two-layer MLP over the features of each ``id`` of ``x``."""
    return f"""
{out}_hidden AS (
  SELECT x.id, p.g AS f, SUM(x.val * p.w) AS val
  FROM {x} x JOIN params p ON p.net = '{net}.0' AND x.f = p.f
  GROUP BY x.id, p.g),
{out} AS (
  SELECT h.id, p.g AS f, SUM(h.val / (1 + exp(-h.val)) * p.w) AS val
  FROM {out}_hidden h JOIN params p ON p.net = '{net}.1' AND h.f = p.f
  GROUP BY h.id, p.g),"""


def message_passing(out, net, edges, edge_h, senders, receivers) -> str:
    """One round of an interaction network on the graph ``edges``: each
    edge's message is an MLP of the edge, its sender and its receiver;
    each receiver adds an MLP of itself and its messages' sum to itself."""
    return f"""
{out}_edge_in AS (
  SELECT id, f, val FROM {edge_h}
  UNION ALL
  SELECT e.edge AS id, s.f + {HIDDEN} AS f, s.val
  FROM {edges} e JOIN {senders} s ON e.sender = s.id
  UNION ALL
  SELECT e.edge AS id, r.f + {2 * HIDDEN} AS f, r.val
  FROM {edges} e JOIN {receivers} r ON e.receiver = r.id),
{mlp(f"{out}_message", f"{out}_edge_in", f"{net}.edge")}
{out}_sum AS (
  SELECT e.receiver AS id, m.f, SUM(m.val) AS val
  FROM {out}_message m JOIN {edges} e ON m.id = e.edge
  GROUP BY e.receiver, m.f),
{out}_node_in AS (
  SELECT id, f, val FROM {receivers}
  UNION ALL
  SELECT id, f + {HIDDEN} AS f, val FROM {out}_sum),
{mlp(f"{out}_update", f"{out}_node_in", f"{net}.node")}
{out} AS (
  SELECT r.id, r.f, r.val + u.val AS val
  FROM {receivers} r JOIN {out}_update u ON r.id = u.id AND r.f = u.f),"""


def era5_state(time, first_feature: int) -> list[str]:
    """Each variable at ``time``, normalized, as (cell, feature, value)
    rows. Only these variables, levels and times are read from ERA5."""
    parts = []
    for var, (table, name, level) in enumerate(VARIABLES):
        on_level = f" AND x.level = {level}" if level else ""
        parts.append(
            f"""  SELECT c.cell AS id, {first_feature + var} AS f,
         (x."{name}" - n.mean) / n.std AS val
  FROM era5.{table} x
  JOIN cells c ON x.latitude = c.latitude AND x.longitude = c.longitude
  JOIN scale n ON n.var = {var}
  WHERE x.time = TIMESTAMP '{time}'{on_level}"""
        )
    return parts


def position(table: str, key: str, first_feature: int) -> list[str]:
    """Where each row of ``table`` is on the sphere, as 3 features."""
    features = [
        "cos(radians(latitude))",
        "sin(radians(longitude))",
        "cos(radians(longitude))",
    ]
    return [
        f"  SELECT {key} AS id, {first_feature + i} AS f, {feature} AS val "
        f"FROM {table}"
        for i, feature in enumerate(features)
    ]


def union(name: str, parts: list[str]) -> str:
    return f"\n{name} AS (\n" + "\n  UNION ALL\n".join(parts) + "),"


def edge_features(graph: str) -> str:
    return f"(SELECT edge AS id, f, val FROM {graph}.features)"


def model(rounds: int) -> str:
    """The CTEs of the whole model, ending in ``forecast``: the change of
    each normalized variable over the next 6 hours, at each cell."""
    n = len(VARIABLES)
    processor = "".join(
        message_passing(
            f"mesh{r + 1}",
            f"processor.{r}",
            "multimesh.edges",
            "multimesh_h",
            f"mesh{r}",
            f"mesh{r}",
        )
        for r in range(rounds)
    )
    return f"""{
        union(
            "grid_in",
            era5_state(TIMES[0], 0)
            + era5_state(TIMES[1], n)
            + position("cells", "cell", 2 * n),
        )
    }
{union("mesh_in", position("mesh", "node", 0))}
{mlp("grid_h", "grid_in", "embed.grid")}
{mlp("mesh_h", "mesh_in", "embed.mesh")}
{mlp("g2m_h", edge_features("g2m"), "embed.g2m")}
{mlp("multimesh_h", edge_features("multimesh"), "embed.multimesh")}
{mlp("m2g_h", edge_features("m2g"), "embed.m2g")}
{message_passing("mesh0", "encoder", "g2m.edges", "g2m_h", "grid_h", "mesh_h")}
{processor}
{
        message_passing(
            "grid_out",
            "decoder",
            "m2g.edges",
            "m2g_h",
            f"mesh{rounds}",
            "grid_h",
        )
    }
forecast AS (
  SELECT g.id, p.g AS f, SUM(g.val * p.w) AS val
  FROM grid_out g JOIN params p ON p.net = 'output' AND g.f = p.f
  GROUP BY g.id, p.g)"""


def forecast_statement(rounds: int, where: str = "TRUE") -> str:
    """The forecast change of each variable at each cell in ``where``."""
    return f"""WITH {model(rounds)}
SELECT c.latitude, c.longitude, f.f AS var, f.val AS change
FROM forecast f JOIN cells c ON f.id = c.cell
WHERE {where}
ORDER BY c.latitude, c.longitude, f.f"""


def gradient_statement(rounds: int) -> str:
    """Each weight's gradient of the forecast's mean squared error
    against what ERA5 recorded 6 hours later."""
    n = len(VARIABLES)
    return f"""WITH {model(rounds)},
{union("later", era5_state(TIMES[2], 0))}
error AS (
  SELECT AVG(power(f.val - (y.val - x.val), 2)) AS l
  FROM forecast f
  JOIN later y ON f.id = y.id AND f.f = y.f
  JOIN grid_in x ON f.id = x.id AND f.f + {n} = x.f)
SELECT weight, grad(error, w) AS w FROM params ORDER BY weight"""


def receptive_field(host, rounds: int) -> list[tuple[str, int]]:
    """How many cells and mesh nodes each stage needs for the region,
    found by walking the edges backward from it: semi-joins in SQL."""

    def count(sql):
        return host.query(f"SELECT COUNT(*) AS n FROM ({sql}) t")["n"][
            0
        ].as_py()

    needed = f"SELECT cell AS id FROM cells c WHERE {REGION}"
    stages = [("decoder: grid cells forecast", count(needed))]
    needed = (
        f"SELECT DISTINCT sender AS id FROM m2g.edges "
        f"WHERE receiver IN ({needed})"
    )
    stages.append((f"processor round {rounds}: mesh nodes", count(needed)))
    for r in reversed(range(rounds)):
        needed = (
            f"SELECT DISTINCT sender AS id FROM multimesh.edges "
            f"WHERE receiver IN ({needed}) UNION SELECT id FROM ({needed}) n"
        )
        label = f"processor round {r}" if r else "encoder"
        stages.append((f"{label}: mesh nodes", count(needed)))
    needed = (
        f"SELECT DISTINCT sender AS id FROM g2m.edges "
        f"WHERE receiver IN ({needed})"
    )
    stages.append(("encoder: grid cells read", count(needed)))
    return stages


# --------------------------------------------------------------------------
# The same model in JAX: the reference.
# --------------------------------------------------------------------------


def jax_mlp(p, net, x):
    h = x @ p[f"{net}.0"]
    return (h * jax.nn.sigmoid(h)) @ p[f"{net}.1"]


def jax_message_passing(p, net, edges, edge_h, senders, receivers):
    sender, receiver = edges
    edge_in = jnp.concatenate(
        [edge_h, senders[sender], receivers[receiver]], axis=1
    )
    message = jax_mlp(p, f"{net}.edge", edge_in)
    total = jax.ops.segment_sum(message, receiver, len(receivers))
    node_in = jnp.concatenate([receivers, total], axis=1)
    return receivers + jax_mlp(p, f"{net}.node", node_in)


def jax_forecast(p, graph, grid_in, rounds):
    grid_h = jax_mlp(p, "embed.grid", grid_in)
    mesh_h = jax_mlp(p, "embed.mesh", graph["mesh_in"])
    edge_h = {
        g: jax_mlp(p, f"embed.{g}", graph[g]["features"])
        for g in ("g2m", "multimesh", "m2g")
    }
    mesh = jax_message_passing(
        p, "encoder", graph["g2m"]["edges"], edge_h["g2m"], grid_h, mesh_h
    )
    for r in range(rounds):
        mesh = jax_message_passing(
            p,
            f"processor.{r}",
            graph["multimesh"]["edges"],
            edge_h["multimesh"],
            mesh,
            mesh,
        )
    grid_out = jax_message_passing(
        p, "decoder", graph["m2g"]["edges"], edge_h["m2g"], mesh, grid_h
    )
    return grid_out @ p["output"]


def jax_error(p, graph, grid_in, target, rounds):
    return jnp.mean((jax_forecast(p, graph, grid_in, rounds) - target) ** 2)


# --------------------------------------------------------------------------
# Geometry, with numpy: the multi-mesh and the edges between it and the grid.
# --------------------------------------------------------------------------


def icosahedron():
    t = (1 + 5**0.5) / 2
    vertices = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    vertices = [np.array(v) / np.linalg.norm(v) for v in vertices]
    return vertices, faces


def refine(vertices: list, faces: list) -> list:
    """Split each triangle into four, adding its edges' midpoints, on the
    sphere, to ``vertices``."""
    midpoints = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in midpoints:
            m = vertices[a] + vertices[b]
            vertices.append(m / np.linalg.norm(m))
            midpoints[key] = len(vertices) - 1
        return midpoints[key]

    finer = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        finer += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return finer


def multimesh(refinements: int):
    """The vertices of an icosahedron refined ``refinements`` times, the
    edges of every level of refinement in both directions, and the length
    of the finest level's longest edge."""
    vertices, faces = icosahedron()
    levels = [faces]
    for _ in range(refinements):
        levels.append(refine(vertices, levels[-1]))
    edges = {
        (u, v)
        for level in levels
        for face in level
        for u, v in zip(face, face[1:] + face[:1])
    }
    edges |= {(v, u) for u, v in edges}
    vertices = np.array(vertices)
    finest = max(
        np.linalg.norm(vertices[u] - vertices[v])
        for face in levels[-1]
        for u, v in zip(face, face[1:] + face[:1])
    )
    return vertices, np.array(sorted(edges)).T, finest


def xyz(latitude, longitude):
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1
    )


def graph_table(senders, receivers, points_from, points_to) -> xr.Dataset:
    """A graph as a Dataset: the table ``edges(edge, sender, receiver)``
    and the table ``features(edge, f, val)``."""
    offset = points_to[receivers] - points_from[senders]
    features = np.concatenate(
        [offset, np.linalg.norm(offset, axis=1, keepdims=True)], axis=1
    )
    return xr.Dataset(
        {
            "sender": ("edge", senders),
            "receiver": ("edge", receivers),
            "val": (("edge", "f"), features),
        },
        coords={"edge": np.arange(len(senders)), "f": np.arange(EDGE_FEATURES)},
    )


GRAPH_TABLES = {("edge",): "edges", ("edge", "f"): "features"}


def geometry(era5: xr.Dataset, refinements: int) -> dict[str, xr.Dataset]:
    """The tables of the model's fixed structure, as Datasets."""
    lat, lon = xr.broadcast(era5.latitude, era5.longitude)
    lat, lon = lat.values.ravel(), lon.values.ravel()
    cell_xyz = xyz(lat, lon)
    node_xyz, (senders, receivers), finest = multimesh(refinements)
    node_lat = np.degrees(np.arcsin(node_xyz[:, 2]))
    node_lon = np.degrees(np.arctan2(node_xyz[:, 1], node_xyz[:, 0])) % 360

    # Grid to mesh: each cell to the mesh nodes within 0.6 of the finest
    # mesh's longest edge, GraphCast's radius.
    distance = np.linalg.norm(cell_xyz[:, None] - node_xyz[None], axis=-1)
    g2m_cell, g2m_node = np.nonzero(distance < 0.6 * finest)
    # Mesh to grid: each cell from its 3 nearest mesh nodes.
    nearest = np.argsort(distance, axis=1)[:, :3]
    m2g_cell = np.repeat(np.arange(len(lat)), 3)
    m2g_node = nearest.ravel()

    return {
        "cells": xr.Dataset(
            {"latitude": ("cell", lat), "longitude": ("cell", lon)},
            coords={"cell": np.arange(len(lat))},
        ),
        "mesh": xr.Dataset(
            {"latitude": ("node", node_lat), "longitude": ("node", node_lon)},
            coords={"node": np.arange(len(node_xyz))},
        ),
        "g2m": graph_table(g2m_cell, g2m_node, cell_xyz, node_xyz),
        "multimesh": graph_table(senders, receivers, node_xyz, node_xyz),
        "m2g": graph_table(m2g_node, m2g_cell, node_xyz, cell_xyz),
    }


# --------------------------------------------------------------------------
# Data and parameters.
# --------------------------------------------------------------------------


def open_era5() -> xr.Dataset:
    """WeatherBench 2's ERA5, lazily; a synthetic stand-in offline."""
    try:
        return xr.open_zarr(
            ERA5, chunks=None, storage_options={"token": "anon"}
        )
    except Exception as e:  # noqa: BLE001 - any failure to reach the store
        print(f"  ERA5 unavailable ({type(e).__name__}); using synthetic data")
    lon = np.arange(64) * 5.625
    lat = -87.1875 + np.arange(32) * 5.625
    rng = np.random.default_rng(3)
    surface = rng.normal(size=(len(TIMES), 64, 32))
    atmosphere = rng.normal(size=(len(TIMES), 2, 64, 32))
    data: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        name: (
            ("time", "longitude", "latitude"),
            surface + i,
        )
        for i, (table, name, _) in enumerate(VARIABLES)
        if table == "surface"
    }
    data["geopotential"] = (
        ("time", "level", "longitude", "latitude"),
        atmosphere,
    )
    data["temperature"] = (
        ("time", "level", "longitude", "latitude"),
        -atmosphere,
    )
    return xr.Dataset(
        data,
        coords={
            "time": TIMES,
            "level": [500, 850],
            "longitude": lon,
            "latitude": lat,
        },
    )


def era5_arrays(era5: xr.Dataset, time) -> np.ndarray:
    """Each variable at ``time`` as a column, one row per cell, read with
    xarray: the reference's input, independent of the SQL's."""
    columns = []
    for _, name, level in VARIABLES:
        field = era5[name].sel(time=time)
        if level:
            field = field.sel(level=level)
        columns.append(field.transpose("latitude", "longitude").values.ravel())
    return np.stack(columns, axis=1)


def init_params(rounds: int, rng) -> dict[str, np.ndarray]:
    """Each network's weight matrices, by name, with fan-in scaling."""
    shapes = {"output": (HIDDEN, len(VARIABLES))}

    def two_layer(net, inputs):
        shapes[f"{net}.0"] = (inputs, HIDDEN)
        shapes[f"{net}.1"] = (HIDDEN, HIDDEN)

    two_layer("embed.grid", INPUTS)
    two_layer("embed.mesh", 3)
    for graph in ("g2m", "multimesh", "m2g"):
        two_layer(f"embed.{graph}", EDGE_FEATURES)
    for net in ["encoder", "decoder"] + [
        f"processor.{r}" for r in range(rounds)
    ]:
        two_layer(f"{net}.edge", 3 * HIDDEN)
        two_layer(f"{net}.node", 2 * HIDDEN)
    return {
        net: rng.normal(0, 1 / np.sqrt(shape[0]), shape)
        for net, shape in shapes.items()
    }


def params_table(params: dict[str, np.ndarray]) -> xr.Dataset:
    """Every weight as a row of ``params(weight, net, f, g, w)``."""
    rows = [
        (net, f, g, w)
        for net, matrix in params.items()
        for (f, g), w in np.ndenumerate(matrix)
    ]
    net, f, g, w = map(np.array, zip(*rows))
    return xr.Dataset(
        {
            "net": ("weight", net),
            "f": ("weight", f),
            "g": ("weight", g),
            "w": ("weight", w),
        },
        coords={"weight": np.arange(len(rows))},
    )


def main():
    parser = cli(__doc__.splitlines()[0], sizes=[2], size_name="refinements")
    parser.add_argument(
        "--rounds", type=int, default=2, help="processor rounds"
    )
    args = parser.parse_args()
    report = Report("graphcast", "refinements", args.reps)
    rounds = args.rounds
    era5 = open_era5()

    # The normalization: each variable's mean and spread at the inputs.
    now, before, later = (era5_arrays(era5, t) for t in TIMES[[1, 0, 2]])
    mean = np.concatenate([before, now]).mean(axis=0)
    std = np.concatenate([before, now]).std(axis=0)
    scale = xr.Dataset(
        {"mean": ("var", mean), "std": ("var", std)},
        coords={"var": np.arange(len(VARIABLES))},
    )
    params = init_params(rounds, np.random.default_rng(0))

    for refinements in args.sizes:
        tables = geometry(era5, refinements)
        cells = tables["cells"]
        print(
            f"  mesh refined {refinements}×: {tables['mesh'].sizes['node']} "
            f"nodes, {tables['multimesh'].sizes['edge']} mesh edges, "
            f"{tables['g2m'].sizes['edge']} grid-to-mesh, "
            f"{tables['m2g'].sizes['edge']} mesh-to-grid; "
            f"{len(params_table(params).weight)} weights"
        )

        # The reference, in JAX, from arrays read with xarray.
        graph = {
            name: {
                "edges": (
                    tables[name].sender.values,
                    tables[name].receiver.values,
                ),
                "features": tables[name].val.values,
            }
            for name in ("g2m", "multimesh", "m2g")
        }
        graph["mesh_in"] = np.stack(
            [
                np.cos(np.radians(tables["mesh"].latitude.values)),
                np.sin(np.radians(tables["mesh"].longitude.values)),
                np.cos(np.radians(tables["mesh"].longitude.values)),
            ],
            axis=1,
        )
        position = np.stack(
            [
                np.cos(np.radians(cells.latitude.values)),
                np.sin(np.radians(cells.longitude.values)),
                np.cos(np.radians(cells.longitude.values)),
            ],
            axis=1,
        )
        grid_in = np.concatenate(
            [(before - mean) / std, (now - mean) / std, position], axis=1
        )
        target = (later - now) / std
        want = jax_forecast(params, graph, grid_in, rounds)

        want_grad = jax.grad(jax_error)(params, graph, grid_in, target, rounds)
        want_grad = np.concatenate([want_grad[net].ravel() for net in params])

        for host in hosts(args.hosts):
            host.load("era5", era5, table_names=ERA5_TABLES)
            for name, table in tables.items():
                options = (
                    {"table_names": GRAPH_TABLES}
                    if "edge" in table.dims
                    else {}
                )
                host.load(name, table, **options)
            host.load("scale", scale)
            host.load("params", params_table(params))

            statement = forecast_statement(rounds)
            rows = report.time(
                "forecast, global", host, refinements, host.query, statement
            )
            got = rows["change"].to_numpy().reshape(len(cells.cell), -1)
            # The rows are ordered by latitude then longitude, as cells are.
            report.check(assert_close("forecast", got, want, rtol=1e-9))
            report.note("operators", host.operators(statement))

            statement = forecast_statement(rounds, REGION)
            rows = report.time(
                "forecast, regional", host, refinements, host.query, statement
            )
            in_region = np.logical_and.reduce(
                [
                    (cells[c] >= lo) & (cells[c] <= hi)
                    for c, (lo, hi) in BOUNDS.items()
                ]
            )
            regional = rows["change"].to_numpy().reshape(-1, len(VARIABLES))
            report.check(assert_close("regional", regional, want[in_region]))
            report.note("operators", host.operators(statement))

            step = "gradient of the error"
            if not host.differentiates:
                report.skip(step, host, refinements, "no ddx backend")
                continue
            try:
                rows = report.time(
                    step,
                    host,
                    refinements,
                    host.query,
                    gradient_statement(rounds),
                )
            except DdxError as e:
                # Recorded rather than raised: a gap in ddx, which this
                # row tracks until ddx closes it.
                reason = str(e).removeprefix("not supported by ddx-ad yet: ")
                report.skip(step, host, refinements, f"ddx: {reason}")
                continue
            report.check(
                assert_close("gradient", rows["w"].to_numpy(), want_grad)
            )

        print("\n  receptive field of the regional forecast:")
        for stage, n in receptive_field(host, rounds):
            print(f"    {stage:<34} {n:>6}")

    report.finish(args.csv)


if __name__ == "__main__":
    main()
