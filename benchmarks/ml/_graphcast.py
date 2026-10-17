"""GraphCast_small, as DeepMind publishes it, for the SQL demo to match.

Everything here is DeepMind's code or data, or a direct transcription of
it into tables:

* :func:`checkpoint`, :func:`example` and :func:`stats` fetch the
  published weights, a 1° ERA5 example and the normalization statistics
  from DeepMind's public bucket, cached in ``~/.cache/xarray-sql``;
* :func:`graphs` builds the three graphs with DeepMind's own code: which
  cells and mesh nodes are joined, and their geometric features;
* :func:`deepmind_stages` runs DeepMind's model, in float32, and returns
  the output of each stage: the reference the SQL must match;
* :func:`mirror` is the same model written plainly in JAX, from the same
  weights. It is checked against :func:`deepmind_stages`, and it can stop
  after any processor round, which DeepMind's code cannot, so each SQL
  stage can be checked on its own;
* :func:`params_table`, :func:`channels_table` and :func:`graph_tables`
  turn the weights, the input channels and the graphs into Datasets that
  xarray-sql registers as tables.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib

import gcsfs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from weathernext.utils import checkpoint as checkpoint_lib
from weathernext.utils import data_utils, normalization
from weathernext.weathernext1_graph import graphcast

CACHE = pathlib.Path.home() / ".cache" / "xarray-sql" / "graphcast"
BUCKET = "dm_graphcast/graphcast"
PARAMS = (
    "params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)
EXAMPLE = "dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"
STATS = {
    "diffs_stddev": "stats/diffs_stddev_by_level.nc",
    "mean": "stats/mean_by_level.nc",
    "stddev": "stats/stddev_by_level.nc",
}


def fetch(path: str) -> pathlib.Path:
    """A file from DeepMind's bucket, downloaded once."""
    local = CACHE / pathlib.Path(path).name
    if not local.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        gcsfs.GCSFileSystem(token="anon").get(f"{BUCKET}/{path}", str(local))
    return local


@functools.cache
def checkpoint() -> graphcast.CheckPoint:
    with open(fetch(PARAMS), "rb") as f:
        return checkpoint_lib.load(f, graphcast.CheckPoint)


@functools.cache
def stats() -> dict[str, xr.Dataset]:
    return {
        name: xr.load_dataset(fetch(path)).compute()
        for name, path in STATS.items()
    }


@functools.cache
def example() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """The inputs, the 6-hour target and the forcings, as DeepMind's
    notebook extracts them from the example, without the batch dim."""
    data = xr.load_dataset(fetch(EXAMPLE), decode_timedelta=True).compute()
    task = dataclasses.asdict(checkpoint().task_config)
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        data, target_lead_times=slice("6h", "6h"), **task
    )
    return tuple(
        ds.isel(batch=0).drop_vars("datetime", errors="ignore")
        for ds in (inputs, targets, forcings)
    )


# --------------------------------------------------------------------------
# DeepMind's model: the graphs it builds and the output of each stage.
# --------------------------------------------------------------------------


@functools.cache
def _model() -> graphcast.GraphCast:
    """A GraphCast with its graphs built for the example's grid."""
    ckpt = checkpoint()
    built = {}

    def build():
        model = graphcast.GraphCast(ckpt.model_config, ckpt.task_config)
        model._maybe_init(example()[0])
        built["model"] = model

    hk.transform(build).init(None)
    return built["model"]


@functools.cache
def graphs() -> dict:
    """The grid-to-mesh, mesh and mesh-to-grid graphs: their edges'
    senders and receivers, and the nodes' and edges' features."""
    model = _model()
    out = {}
    for name, graph in (
        ("grid2mesh", model._grid2mesh_graph_structure),
        ("mesh", model._mesh_graph_structure),
        ("mesh2grid", model._mesh2grid_graph_structure),
    ):
        (edges,) = graph.edges.values()
        out[name] = {
            "senders": np.asarray(edges.indices.senders),
            "receivers": np.asarray(edges.indices.receivers),
            "features": np.asarray(edges.features),
        }
    nodes = model._grid2mesh_graph_structure.nodes
    out["grid_nodes"] = np.asarray(nodes["grid_nodes"].features)
    out["mesh_nodes"] = np.asarray(nodes["mesh_nodes"].features)
    out["grid_lat"] = model._grid_nodes_lat
    out["grid_lon"] = model._grid_nodes_lon
    out["mesh_lat"] = model._mesh_nodes_lat
    out["mesh_lon"] = model._mesh_nodes_lon
    return out


def _normalized(inputs, forcings):
    s = stats()
    return (
        normalization.normalize(inputs, s["stddev"], s["mean"]),
        normalization.normalize(forcings, s["stddev"], s["mean"]),
    )


@functools.cache
def deepmind_stages() -> dict[str, np.ndarray]:
    """DeepMind's model, in float32, stage by stage: the grid features it
    reads, the latent grid and mesh after the encoder, the mesh after the
    processor, and the decoder's normalized output."""
    ckpt = checkpoint()
    inputs, _, forcings = example()

    @hk.transform_with_state
    def stages(inputs, forcings):
        model = graphcast.GraphCast(ckpt.model_config, ckpt.task_config)
        inputs, forcings = _normalized(inputs, forcings)
        model._maybe_init(inputs)
        features = model._inputs_to_grid_node_features(
            inputs.expand_dims("batch"), forcings.expand_dims("batch")
        )
        mesh, grid = model._run_grid2mesh_gnn(features)
        processed = model._run_mesh_gnn(mesh)
        output = model._run_mesh2grid_gnn(processed, grid)
        return dict(
            features=features, grid=grid, mesh=mesh,
            processed=processed, output=output,
        )

    out, _ = jax.jit(stages.apply)(
        ckpt.params, {}, None,
        inputs.astype(np.float32), forcings.astype(np.float32),
    )
    return {k: np.asarray(v)[:, 0] for k, v in out.items()}


@functools.cache
def deepmind_forecast() -> xr.Dataset:
    """DeepMind's 6-hour forecast, in float32, with its normalization and
    residual: what the notebook's ``run_forward`` returns, without the
    bfloat16 cast."""
    ckpt = checkpoint()
    inputs, targets, forcings = example()
    s = stats()

    @hk.transform_with_state
    def forward(inputs, targets, forcings):
        model = normalization.InputsAndResiduals(
            graphcast.GraphCast(ckpt.model_config, ckpt.task_config),
            diffs_stddev_by_level=s["diffs_stddev"],
            mean_by_level=s["mean"],
            stddev_by_level=s["stddev"],
        )
        return model(inputs, targets_template=targets, forcings=forcings)

    batch = [ds.expand_dims("batch").astype(np.float32)
             for ds in (inputs, targets * np.nan, forcings)]
    out, _ = forward.apply(ckpt.params, {}, None, *batch)
    return out.isel(batch=0).compute()


# --------------------------------------------------------------------------
# The same model, written plainly: the oracle for each stage.
# --------------------------------------------------------------------------


def weights(net: str) -> dict[str, np.ndarray]:
    """One network's parameters, by DeepMind's name for it, e.g.
    ``mesh_gnn/processor_edges_3_mesh``."""
    params = checkpoint().params
    gnn, block = net.split("/")
    prefix = f"{gnn}/~_networks_builder/{block}"
    out = {}
    for layer in ("linear_0", "linear_1"):
        for p, v in params[f"{prefix}_mlp/~/{layer}"].items():
            out[f"{layer}:{p}"] = np.asarray(v, np.float64)
    if f"{prefix}_layer_norm" in params:
        for p, v in params[f"{prefix}_layer_norm"].items():
            out[f"layer_norm:{p}"] = np.asarray(v, np.float64)
    return out


def mlp(net: str, x):
    """Linear, swish, linear, then a layer norm, as DeepMind's MLPs are."""
    p = weights(net)
    h = jax.nn.swish(x @ p["linear_0:w"] + p["linear_0:b"])
    y = h @ p["linear_1:w"] + p["linear_1:b"]
    if "layer_norm:scale" not in p:
        return y
    mean = y.mean(-1, keepdims=True)
    var = ((y - mean) ** 2).mean(-1, keepdims=True)
    return (y - mean) / jnp.sqrt(var + 1e-5) * p["layer_norm:scale"] + p[
        "layer_norm:offset"
    ]


def interaction(gnn, step, edge_set, graph, edges, senders, receivers, node_set):
    """One message-passing step: new edges, then the receivers' update,
    each with its residual. Returns the new edges and receivers."""
    s, r = graph["senders"], graph["receivers"]
    message = mlp(
        f"{gnn}/processor_edges_{step}_{edge_set}",
        jnp.concatenate([edges, senders[s], receivers[r]], axis=1),
    )
    total = jax.ops.segment_sum(message, r, len(receivers))
    update = mlp(
        f"{gnn}/processor_nodes_{step}_{node_set}",
        jnp.concatenate([receivers, total], axis=1),
    )
    return edges + message, receivers + update


def mirror(features: np.ndarray, rounds: int | None = None) -> dict:
    """The model from the grid features DeepMind's model reads, in float64.
    With ``rounds``, it stops after that many processor rounds."""
    g = graphs()
    x = jnp.asarray(features, jnp.float64)
    grid_in = jnp.concatenate([x, g["grid_nodes"]], axis=1)
    mesh_in = jnp.concatenate(
        [jnp.zeros((len(g["mesh_nodes"]), x.shape[1])), g["mesh_nodes"]], axis=1
    )
    grid = mlp("grid2mesh_gnn/encoder_nodes_grid_nodes", grid_in)
    mesh = mlp("grid2mesh_gnn/encoder_nodes_mesh_nodes", mesh_in)
    edges = mlp("grid2mesh_gnn/encoder_edges_grid2mesh", g["grid2mesh"]["features"])
    _, mesh = interaction(
        "grid2mesh_gnn", 0, "grid2mesh", g["grid2mesh"], edges, grid, mesh,
        "mesh_nodes",
    )
    grid = grid + mlp("grid2mesh_gnn/processor_nodes_0_grid_nodes", grid)
    out = {"grid": grid, "mesh": mesh}

    edges = mlp("mesh_gnn/encoder_edges_mesh", g["mesh"]["features"])
    total = checkpoint().model_config.gnn_msg_steps
    for step in range(total if rounds is None else rounds):
        edges, mesh = interaction(
            "mesh_gnn", step, "mesh", g["mesh"], edges, mesh, mesh, "mesh_nodes"
        )
        out[f"round_{step}"] = mesh
    if rounds is not None and rounds < total:
        return out
    out["processed"] = mesh

    edges = mlp("mesh2grid_gnn/encoder_edges_mesh2grid", g["mesh2grid"]["features"])
    _, grid = interaction(
        "mesh2grid_gnn", 0, "mesh2grid", g["mesh2grid"], edges, mesh, grid,
        "grid_nodes",
    )
    out["output"] = mlp("mesh2grid_gnn/decoder_nodes_grid_nodes", grid)
    return out


# --------------------------------------------------------------------------
# Tables: the example, the input channels, the graphs and the weights.
# --------------------------------------------------------------------------


def _hours(ds: xr.Dataset) -> xr.Dataset:
    """``time`` as whole hours from the forecast's start."""
    return ds.assign_coords(time=(ds.time / np.timedelta64(1, "h")).astype(int))


def era5() -> xr.Dataset:
    """The model's inputs at -6 h and 0 h, and the forcings at +6 h."""
    inputs, _, forcings = example()
    return xr.merge(
        [_hours(inputs), _hours(forcings)], compat="no_conflicts", join="outer"
    )


def _stacking(ds: xr.Dataset, time_offset: int = 0) -> list[tuple]:
    """DeepMind's channel order for ``ds``: its variables sorted by name,
    each over its own dimensions other than lat and lon, as
    ``(variable, time, level)``; ``level`` is 0 for a surface variable."""
    channels = []
    for name in sorted(ds.data_vars):
        var = ds[name]
        times = var.time.values if "time" in var.dims else [None]
        levels = var.level.values if "level" in var.dims else [0]
        for t in times:
            for level in levels:
                time = None if t is None else int(t / np.timedelta64(1, "h"))
                channels.append((name, time, int(level)))
    return channels


def channels_table() -> xr.Dataset:
    """Each input channel: its variable, time and level, and the mean and
    standard deviation that normalize it."""
    inputs, _, forcings = example()
    rows = _stacking(inputs) + _stacking(forcings)
    s = stats()

    def stat(kind, name, level):
        value = s[kind][name]
        return float(value.sel(level=level) if "level" in value.dims else value)

    return xr.Dataset(
        {
            "variable": ("channel", [r[0] for r in rows]),
            # Static variables have no time; -1 never matches a row's time.
            "time": ("channel", [-1 if r[1] is None else r[1] for r in rows]),
            "level": ("channel", [r[2] for r in rows]),
            "mean": ("channel", [stat("mean", r[0], r[2]) for r in rows]),
            "std": ("channel", [stat("stddev", r[0], r[2]) for r in rows]),
        },
        coords={"channel": np.arange(len(rows))},
    )


def outputs_table() -> xr.Dataset:
    """Each output channel: its variable and level, and the standard
    deviation of its 6-hour change, which turns the output into one."""
    _, targets, _ = example()
    rows = _stacking(targets.isel(time=0, drop=True))
    diffs = stats()["diffs_stddev"]

    def std(name, level):
        value = diffs[name]
        return float(value.sel(level=level) if "level" in value.dims else value)

    return xr.Dataset(
        {
            "variable": ("channel", [r[0] for r in rows]),
            "level": ("channel", [r[2] for r in rows]),
            "diff_std": ("channel", [std(r[0], r[2]) for r in rows]),
        },
        coords={"channel": np.arange(len(rows))},
    )


def _long(name: str, values: np.ndarray, key: str) -> xr.Dataset:
    """A (key, feature) matrix as the table ``(key, f, val)``."""
    return xr.Dataset(
        {"val": ((key, "f"), values.astype(np.float64))},
        coords={key: np.arange(len(values)), "f": np.arange(values.shape[1])},
    )


GRAPH_TABLES = {("edge",): "edges", ("edge", "f"): "features"}


def graph_tables() -> dict[str, xr.Dataset]:
    """The grid, the mesh, and the three graphs, as Datasets. Grid cells
    are numbered as DeepMind numbers them: by latitude, then longitude."""
    g = graphs()
    tables = {
        "cells": xr.Dataset(
            {"lat": ("cell", g["grid_lat"]), "lon": ("cell", g["grid_lon"])},
            coords={"cell": np.arange(len(g["grid_lat"]))},
        ),
        "cell_features": _long("cell_features", g["grid_nodes"], "cell"),
        "node_features": _long("node_features", g["mesh_nodes"], "node"),
    }
    for name in ("grid2mesh", "mesh", "mesh2grid"):
        graph = g[name]
        tables[name] = xr.Dataset(
            {
                "sender": ("edge", graph["senders"]),
                "receiver": ("edge", graph["receivers"]),
                "val": (("edge", "f"), graph["features"].astype(np.float64)),
            },
            coords={
                "edge": np.arange(len(graph["senders"])),
                "f": np.arange(graph["features"].shape[1]),
            },
        )
    return tables


def _short(module: str) -> str:
    """DeepMind's module path without Haiku's scaffolding:
    ``mesh_gnn/processor_edges_3_mesh_mlp/linear_0``."""
    return module.replace("~_networks_builder/", "").replace("~/", "")


def param_tables() -> dict[str, xr.Dataset]:
    """Every parameter, long form: ``weights(net, f, g, w)``,
    ``biases(net, g, b)``, ``norms(net, g, scale, offset)``, and
    ``nets(net, name)`` naming each network by DeepMind's path."""
    params = checkpoint().params
    names = sorted(_short(m) for m in params)
    ids = {name: i for i, name in enumerate(names)}
    weights, biases, norms = [], [], []
    for module, p in params.items():
        net = ids[_short(module)]
        if "w" in p:
            f, g = np.indices(p["w"].shape).reshape(2, -1)
            weights.append((np.full(f.size, net), f, g, p["w"].ravel()))
            biases.append((np.full(p["b"].size, net), np.arange(p["b"].size), p["b"]))
        else:
            norms.append(
                (np.full(p["scale"].size, net), np.arange(p["scale"].size),
                 p["scale"], p["offset"])
            )

    def table(dim, columns, parts):
        arrays = [np.concatenate(c) for c in zip(*parts)]
        return xr.Dataset(
            {
                name: (dim, a.astype(np.float64 if a.dtype.kind == "f" else np.int32))
                for name, a in zip(columns, arrays)
            },
            coords={dim: np.arange(len(arrays[0]))},
        )

    return {
        "nets": xr.Dataset(
            {"name": ("net", np.array(names, dtype=object))},
            coords={"net": np.arange(len(names), dtype=np.int32)},
        ),
        "weights": table("weight", ("net", "f", "g", "w"), weights),
        "biases": table("bias", ("net", "g", "b"), biases),
        "norms": table("norm", ("net", "g", "scale", "offset"), norms),
    }
