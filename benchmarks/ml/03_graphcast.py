# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql[ml,duckdb]",
#   "weathernext @ git+https://github.com/google-deepmind/graphcast",
#   "gcsfs",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Demo 3: GraphCast in SQL, with DeepMind's published weights.

GraphCast (Lam et al., 2023, https://arxiv.org/abs/2212.12794) is
DeepMind's graph neural network for weather forecasting. It forecasts the
atmosphere 6 hours ahead from the two previous states. This runs
GraphCast_small, as DeepMind publishes it (1° grid, 13 pressure levels, a
mesh refined 5 times, 16 rounds of message passing, 36 million weights),
as a program of SQL statements over tables xarray-sql registers:

* ``era5``: DeepMind's example inputs, two states 6 hours apart from ERA5
  (ECMWF's reanalysis of the atmosphere since 1940), and the forcings at
  the forecast's time: inputs known in advance, such as the sunlight
  reaching the top of the atmosphere;
* ``weights``, ``biases``, ``norms`` and ``nets``: every parameter in
  DeepMind's checkpoint, one row each;
* ``grid2mesh``, ``mesh`` and ``mesh2grid``: the three graphs, as tables of
  edges and of edge features, built by DeepMind's code.

The model has three stages:

* an **encoder** embeds each grid cell's 183 input channels, and passes
  one round of messages from the grid to the mesh;
* a **processor** passes 16 rounds of messages along the multi-mesh, the
  icosahedron's edges at every level of refinement;
* a **decoder** passes one round back to the grid, and turns each cell's
  state into 83 output channels: the 6-hour change of each variable.

Every network is the same SQL: an MLP (linear, swish, linear, layer norm)
over a table of ``(id, f, val)``, one row per node or edge and feature.
Swish is the activation ``x · sigmoid(x)``. Each stage is one statement,
and its result is stored as a table, so no host (an engine that runs the
SQL; see ``_hosts.py``) computes a stage twice.

The check is DeepMind's own model. Its code, in JAX (Google's library for
differentiable array programs), run on the same inputs,
gives the stages' outputs; the SQL must match them. A plain JAX
transcription of the model (``_graphcast.mirror``), checked against
DeepMind's, also gives each processor round, so a run can stop after a
few rounds and still be checked. A full forecast is about two trillion
multiply-adds, which hash joins take hours over; DeepMind's JAX code
takes about 15 seconds on a 6-core CPU, including its compilation.

Run ``uv run benchmarks/ml/03_graphcast.py --help`` for the options.
``--sizes`` is the number of processor rounds to run; with 16, the
forecast is complete.
"""

from __future__ import annotations

import jax
import numpy as np

import _graphcast as gc
from _bench import Report, assert_close, cli, hosts

jax.config.update("jax_enable_x64", True)

INPUTS = 183  # input channels: the variables at each time and level
ROUNDS = 16  # processor rounds in GraphCast_small

# --------------------------------------------------------------------------
# The networks, in SQL. A state is a table of (id, f, val).
# --------------------------------------------------------------------------


def weights(name: str) -> str:
    """The weight matrix of one linear layer, by DeepMind's name."""
    return f"""(
  SELECT w.f, w.g, w.w FROM nets n JOIN weights w ON w.net = n.net
  WHERE n.name = '{name}')"""


def biases(name: str) -> str:
    """The bias vector of one linear layer, by DeepMind's name."""
    return f"""(
  SELECT b.g, b.b FROM nets n JOIN biases b ON b.net = n.net
  WHERE n.name = '{name}')"""


def mlp(out: str, x: str, net: str, layer_norm: bool = True) -> str:
    """DeepMind's MLP over the features of each ``id`` of ``x``: linear,
    swish, linear, and (unless it is the decoder's last) a layer norm with
    a learned scale and offset. A linear layer is a matrix product, a
    join and a sum, and then its bias."""
    first, second = f"{net}_mlp/linear_0", f"{net}_mlp/linear_1"
    sql = f"""
{out}_product_0 AS (
  SELECT x.id, w.g AS f, SUM(x.val * w.w) AS val
  FROM {x} x JOIN {weights(first)} w ON x.f = w.f
  GROUP BY x.id, w.g),
{out}_hidden AS (
  SELECT h.id, h.f, (h.val + b.b) / (1 + exp(-(h.val + b.b))) AS val
  FROM {out}_product_0 h JOIN {biases(first)} b ON h.f = b.g),
{out}_product_1 AS (
  SELECT h.id, w.g AS f, SUM(h.val * w.w) AS val
  FROM {out}_hidden h JOIN {weights(second)} w ON h.f = w.f
  GROUP BY h.id, w.g),
{out}_linear AS (
  SELECT y.id, y.f, y.val + b.b AS val
  FROM {out}_product_1 y JOIN {biases(second)} b ON y.f = b.g),"""
    if not layer_norm:
        return sql + f"\n{out} AS (SELECT * FROM {out}_linear),"
    return (
        sql
        + f"""
{out} AS (
  SELECT y.id, y.f,
         (y.val - AVG(y.val) OVER (PARTITION BY y.id))
         / sqrt(VAR_POP(y.val) OVER (PARTITION BY y.id) + 1e-5)
         * n.scale + n.offset AS val
  FROM {out}_linear y
  JOIN norms n ON n.g = y.f
  JOIN nets m ON m.net = n.net
  WHERE m.name = '{net}_layer_norm'),"""
    )


def select(ctes: str, out: str) -> str:
    return f"WITH {ctes.rstrip(',')}\nSELECT id, f, val FROM {out}"


def embed(x: str, net: str) -> str:
    """A statement: an MLP of every row of ``x``."""
    return select(mlp("embedded", x, net), "embedded")


def messages(net: str, graph: str, edges: str, senders: str, receivers: str):
    """A statement: each edge's message, an MLP of the edge's state, its
    sender's and its receiver's, concatenated."""
    return select(
        f"""
edge_in AS (
  SELECT x.id, x.f, x.val FROM {edges} x JOIN {graph} e ON x.id = e.edge
  UNION ALL
  SELECT e.edge AS id, s.f + 512 AS f, s.val
  FROM {graph} e JOIN {senders} s ON e.sender = s.id
  UNION ALL
  SELECT e.edge AS id, r.f + 1024 AS f, r.val
  FROM {graph} e JOIN {receivers} r ON e.receiver = r.id),"""
        + mlp("message", "edge_in", net),
        "message",
    )


def update(net: str, graph: str, nodes: str, messages: str | None) -> str:
    """A statement: each node plus an MLP of itself and the sum of the
    messages it receives (none, for the encoder's grid cells)."""
    node_in = f"SELECT id, f, val FROM {nodes}"
    if messages:
        node_in += f"""
  UNION ALL
  SELECT e.receiver AS id, m.f + 512 AS f, SUM(m.val) AS val
  FROM {messages} m JOIN {graph} e ON m.id = e.edge
  GROUP BY e.receiver, m.f"""
    return select(
        f"\nnode_in AS (\n  {node_in}),"
        + mlp("updated", "node_in", net)
        + f"""
new AS (
  SELECT x.id, x.f, x.val + u.val AS val
  FROM {nodes} x JOIN updated u ON x.id = u.id AND x.f = u.f),""",
        "new",
    )


def residual(state: str, change: str) -> str:
    """A statement: ``state + change``, row by row."""
    return f"""SELECT x.id, x.f, x.val + d.val AS val
FROM {state} x JOIN {change} d ON x.id = d.id AND x.f = d.f"""


ERA5_TABLES = {
    ("time", "lat", "lon"): "surface",
    ("time", "level", "lat", "lon"): "atmosphere",
    ("lat", "lon"): "static",
    ("time", "lon"): "day",
    ("time",): "year",
}
# How each era5 table's rows reach the grid's cells.
CELLS = {
    "surface": "JOIN cells c ON x.lat = c.lat AND x.lon = c.lon",
    "atmosphere": "JOIN cells c ON x.lat = c.lat AND x.lon = c.lon",
    "static": "JOIN cells c ON x.lat = c.lat AND x.lon = c.lon",
    "day": "JOIN cells c ON x.lon = c.lon",
    "year": "CROSS JOIN cells c",
}


def grid_inputs(era5) -> str:
    """A statement: each cell's input channels, normalized, then its
    position. Every variable is read from its ``era5`` table at the times
    and levels the channels ask for."""
    parts = []
    for name in sorted(era5.data_vars):
        dims = era5[name].dims
        table = ERA5_TABLES[dims]
        time = "ch.time = x.time" if "time" in dims else "ch.time = -1"
        level = "ch.level = x.level" if "level" in dims else "ch.level = 0"
        parts.append(
            f"""SELECT c.cell AS id, ch.channel AS f,
       (x."{name}" - ch.mean) / ch.std AS val
FROM era5.{table} x
{CELLS[table]}
JOIN channels ch ON ch.variable = '{name}' AND {time} AND {level}"""
        )
    parts.append(
        f"SELECT cell AS id, f + {INPUTS} AS f, val FROM cell_features"
    )
    return "\nUNION ALL\n".join(parts)


def forecast(era5) -> str:
    """A statement: the forecast of each variable at each cell and level,
    the last input plus the output's normalized 6-hour change."""
    targets = sorted(set(gc.outputs_table().variable.values))
    last = "\n  UNION ALL\n".join(
        f"""  SELECT c.cell AS id, '{name}' AS variable,
         {"x.level" if "level" in era5[name].dims else "0"} AS level,
         x."{name}" AS value
  FROM era5.{ERA5_TABLES[era5[name].dims]} x
  {CELLS[ERA5_TABLES[era5[name].dims]]}
  WHERE x.time = 0"""
        for name in targets
    )
    return f"""WITH last_input AS (
{last})
SELECT c.lat, c.lon, o.variable, o.level,
       x.value + y.val * o.diff_std AS value
FROM output y
JOIN outputs o ON o.channel = y.f
JOIN cells c ON c.cell = y.id
JOIN last_input x
  ON x.id = y.id AND x.variable = o.variable AND x.level = o.level
ORDER BY o.variable, o.level, c.lat, c.lon"""


def flatten(ds) -> np.ndarray:
    """A forecast Dataset in ``forecast``'s row order: by variable, level,
    latitude and longitude."""
    return np.concatenate(
        [
            ds[name].transpose(..., "lat", "lon").values.ravel()
            for name in sorted(ds.data_vars)
        ]
    )


# --------------------------------------------------------------------------
# The program: one stored table per stage.
# --------------------------------------------------------------------------


# A region to forecast: the contiguous United States, roughly.
BOUNDS = {"lat": (25, 50), "lon": (235, 295)}
REGION = " AND ".join(
    f"c.{coord} BETWEEN {low} AND {high}"
    for coord, (low, high) in BOUNDS.items()
)


def receptive_field(rounds: int):
    """The cells and mesh nodes a regional forecast needs, as stored
    ``(table, statement)``s, found by walking the graphs backward from the
    region: semi-joins of each graph's edges with what the next stage
    needs."""
    yield "needed_cells", f"SELECT cell AS id FROM cells c WHERE {REGION}"
    yield (
        f"needed_mesh_{rounds}",
        """SELECT DISTINCT sender AS id
FROM mesh2grid.edges WHERE receiver IN (SELECT id FROM needed_cells)""",
    )
    for r in reversed(range(rounds)):
        yield (
            f"needed_mesh_{r}",
            f"""SELECT id FROM needed_mesh_{r + 1}
UNION
SELECT sender AS id FROM mesh.edges
WHERE receiver IN (SELECT id FROM needed_mesh_{r + 1})""",
        )
    yield (
        "needed_grid",
        """SELECT id FROM needed_cells
UNION
SELECT sender AS id FROM grid2mesh.edges
WHERE receiver IN (SELECT id FROM needed_mesh_0)""",
    )


def program(era5, rounds: int, regional: bool = False):
    """The stages in order, as ``(table, statement)``; ``(table, None)``
    marks a table the reference checks. A regional program runs the same
    stages on the rows its receptive field needs."""

    def within(table: str, column: str, needed: str) -> str:
        if not regional:
            return table
        return f"(SELECT * FROM {table} WHERE {column} IN (SELECT id FROM {needed}))"

    def features(graph: str, edges: str) -> str:
        return f"""(SELECT x.edge AS id, x.f, x.val FROM {graph}.features x
   WHERE x.edge IN (SELECT edge FROM {edges}))"""

    checked = not regional
    yield (
        "grid_in",
        (
            f"SELECT * FROM ({grid_inputs(era5)}) g\n"
            "WHERE id IN (SELECT id FROM needed_grid)"
            if regional
            else grid_inputs(era5)
        ),
    )
    if checked:
        yield "grid_in", None
    # Encoder: embed the grid, the mesh and the grid-to-mesh edges, then
    # one round of messages from the grid to the mesh.
    g2m = "grid2mesh_gnn"
    g2m_graph = within("grid2mesh.edges", "receiver", "needed_mesh_0")
    yield "grid", embed("grid_in", f"{g2m}/encoder_nodes_grid_nodes")
    # DeepMind gives the mesh nodes 183 zero inputs; zeros are absent rows.
    yield (
        "mesh",
        embed(
            f"""(SELECT node AS id, f + {INPUTS} AS f, val
   FROM {within("node_features", "node", "needed_mesh_0")} n)""",
            f"{g2m}/encoder_nodes_mesh_nodes",
        ),
    )
    yield (
        "g2m_edges",
        embed(
            features("grid2mesh", g2m_graph), f"{g2m}/encoder_edges_grid2mesh"
        ),
    )
    yield (
        "g2m_messages",
        messages(
            f"{g2m}/processor_edges_0_grid2mesh",
            g2m_graph,
            "g2m_edges",
            "grid",
            "mesh",
        ),
    )
    yield (
        "mesh_0",
        update(
            f"{g2m}/processor_nodes_0_mesh_nodes",
            g2m_graph,
            "mesh",
            "g2m_messages",
        ),
    )
    yield (
        "grid_0",
        update(
            f"{g2m}/processor_nodes_0_grid_nodes",
            g2m_graph,
            within("grid", "id", "needed_cells"),
            None,
        ),
    )
    if checked:
        yield "mesh_0", None
        yield "grid_0", None
    # Processor: rounds of messages along the multi-mesh, edges and nodes
    # each updated with a residual.
    graphs = [
        within("mesh.edges", "receiver", f"needed_mesh_{r + 1}")
        for r in range(rounds)
    ]
    yield (
        "mesh_edges_0",
        embed(
            features("mesh", graphs[0] if rounds else "mesh.edges"),
            "mesh_gnn/encoder_edges_mesh",
        ),
    )
    for r in range(rounds):
        yield (
            f"mesh_messages_{r}",
            messages(
                f"mesh_gnn/processor_edges_{r}_mesh",
                graphs[r],
                f"mesh_edges_{r}",
                f"mesh_{r}",
                f"mesh_{r}",
            ),
        )
        yield (
            f"mesh_{r + 1}",
            update(
                f"mesh_gnn/processor_nodes_{r}_mesh_nodes",
                graphs[r],
                within(f"mesh_{r}", "id", f"needed_mesh_{r + 1}"),
                f"mesh_messages_{r}",
            ),
        )
        yield (
            f"mesh_edges_{r + 1}",
            residual(f"mesh_edges_{r}", f"mesh_messages_{r}"),
        )
        if checked:
            yield f"mesh_{r + 1}", None
    if rounds < ROUNDS:
        return
    # Decoder: one round of messages from the mesh to the grid, then each
    # cell's 83 output channels.
    m2g = "mesh2grid_gnn"
    m2g_graph = within("mesh2grid.edges", "receiver", "needed_cells")
    yield (
        "m2g_edges",
        embed(
            features("mesh2grid", m2g_graph), f"{m2g}/encoder_edges_mesh2grid"
        ),
    )
    yield (
        "m2g_messages",
        messages(
            f"{m2g}/processor_edges_0_mesh2grid",
            m2g_graph,
            "m2g_edges",
            f"mesh_{rounds}",
            "grid_0",
        ),
    )
    yield (
        "grid_1",
        update(
            f"{m2g}/processor_nodes_0_grid_nodes",
            m2g_graph,
            "grid_0",
            "m2g_messages",
        ),
    )
    yield (
        "output",
        select(
            mlp(
                "out",
                "grid_1",
                f"{m2g}/decoder_nodes_grid_nodes",
                layer_norm=False,
            ),
            "out",
        ),
    )
    if checked:
        yield "output", None


def matrix(host, table: str) -> np.ndarray:
    """A stored ``(id, f, val)`` table as an (id, f) matrix."""
    rows = host.query(f"SELECT id, f, val FROM {table} ORDER BY id, f")
    n = rows["id"][-1].as_py() + 1
    return rows["val"].to_numpy().reshape(n, -1)


def main():
    parser = cli(__doc__.splitlines()[0], sizes=[1], size_name="rounds")
    args = parser.parse_args()
    report = Report("graphcast", "rounds", args.reps)

    print("  loading DeepMind's checkpoint, example and graphs")
    reference = gc.deepmind_stages()
    era5 = gc.era5()
    tables = {
        "era5": era5,
        "channels": gc.channels_table(),
        "outputs": gc.outputs_table(),
        **gc.graph_tables(),
        **gc.param_tables(),
    }

    for rounds in args.sizes:
        mirror = gc.mirror(reference["features"], rounds=rounds)
        expected = {
            "grid_in": np.concatenate(
                [reference["features"], gc.graphs()["grid_nodes"]], axis=1
            ),
            "grid_0": reference["grid"],
            "mesh_0": reference["mesh"],
            **{f"mesh_{r + 1}": mirror[f"round_{r}"] for r in range(rounds)},
            "output": reference["output"],
        }
        for host in hosts(args.hosts, args.memory):
            for name, table in tables.items():
                options = {}
                if name == "era5":
                    options = {"table_names": ERA5_TABLES}
                elif "edge" in table.dims:
                    options = {"table_names": gc.GRAPH_TABLES}
                host.load(name, table, **options)

            for table, statement in program(era5, rounds):
                if statement is None:
                    got = matrix(host, table)
                    # The reference runs in float32, the SQL in float64.
                    error = assert_close(
                        table, got, expected[table], rtol=1e-4, atol=1e-3
                    )
                    report.check(error, step=table)
                    continue
                report.time(table, host, rounds, host.store, table, statement)

            if rounds == ROUNDS:
                rows = report.time(
                    "forecast", host, rounds, host.query, forecast(era5)
                )
                want = flatten(gc.deepmind_forecast().isel(time=0))
                error = assert_close(
                    "forecast", rows["value"].to_numpy(), want, rtol=1e-4
                )
                report.check(error)

    report.finish(args.csv)


if __name__ == "__main__":
    main()
