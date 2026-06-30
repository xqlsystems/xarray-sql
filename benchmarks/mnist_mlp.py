# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "xarray_sql",
#   "xarray",
#   "numpy",
# ]
#
# [tool.uv.sources]
# xarray_sql = { path = "..", editable = true }
# ///
"""Train an MNIST MLP as relational tensor algebra — with the architecture as data.

A neural network is a chain of **tensor contractions** (einsums), and an einsum
over coordinate-indexed arrays *is* relational algebra:

    C[i,k] = sum_j A[i,j] * B[j,k]   <=>   JOIN A, B ON A.j = B.j
                                           GROUP BY i, k -> SUM(A.val * B.val)

Contracting a shared index is a join on it followed by a grouped SUM over the
indices that survive. In xarray-sql an array indexed by named dims is a table
keyed by those dims, so **the dimension names are the join keys**.

The whole model is **one ``xr.Dataset``**. Each layer's weight is a data variable
whose two dims are the widths it connects — ``w0(u0, u1)``, ``w1(u1, u2)``, … —
sharing the boundary dims (``u1`` is the output of layer 0 and the input of layer
1, so it is the join key between them). **The architecture is therefore data: the
Dataset's dim sizes are the layer widths, and the number of layers is how many
weights it holds.** Differing neuron counts per layer are just differing dim
sizes — no padding, because the relational (long) form is naturally ragged.
``from_dataset`` splits that one Dataset into a table per weight automatically.

A single ``contract()`` turns an einsum spec into one query, and a single generic
loop trains a net of any depth/width:

* **forward** — contract the activation with each layer's weight, add bias, tanh
  (softmax on the last layer).
* **backward is the same operator transposed** — the VJP of a contraction is a
  contraction — with ``grad(tanh(z), z)`` for the one local-derivative step.
  Linear algebra is joins; the derivatives of the nonlinearities are ``grad``.

Every stage is an inspectable relation; the trained model, predictions, and loss
curve come back out as ``xarray`` via ``to_dataset``. Change ``WIDTHS`` and the
same code trains a different network.

This is not a numpy replacement — relational matmul carries join overhead a BLAS
inner product doesn't. What it buys is a declarative, inspectable pipeline whose
data side is chunked xarray (parallel over the batch, larger-than-memory).

Run standalone (builds the local extension on first use):

    uv run benchmarks/mnist_mlp.py
"""

from __future__ import annotations

import gzip
import struct
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

import xarray_sql as xql

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist"
CACHE = Path(tempfile.gettempdir()) / "mnist-xql"

# The architecture, as data: layer widths. 196 pooled pixels -> 32 tanh -> 10.
# Add an entry (e.g. 196, 64, 32, 10) and the same code trains the deeper net.
WIDTHS = [196, 32, 10]
DEPTH = len(WIDTHS) - 1  # number of weight layers
N_TRAIN, N_TEST = 1000, 500
LR, STEPS, CHUNK = 0.5, 60, 250


# --- the one idea: a tensor contraction is a relational query -----------------


def contract(spec: str, left: str, right: str) -> str:
    """An einsum over two coordinate-indexed tables, as one SQL query.

    ``contract("sample,u0 * u0,u1 -> sample,u1", "x", "w0")`` joins ``x`` and
    ``w0`` on their shared dim ``u0``, groups by the output dims, and sums the
    product of values — a matmul. Every table has its dims as columns plus a
    ``val`` column. Indices in the inputs but not the output are contracted; the
    same helper expresses the transposed contractions of backprop.
    """
    spec = spec.replace(" ", "")
    lhs, out = spec.split("->")
    da, db = (operand.split(",") for operand in lhs.split("*"))
    out_dims = out.split(",")
    shared = [d for d in da if d in db]
    join = (
        f"JOIN {right} r ON " + " AND ".join(f"l.{d} = r.{d}" for d in shared)
        if shared
        else f"CROSS JOIN {right} r"
    )
    pick = ", ".join(f"{'l' if d in da else 'r'}.{d} AS {d}" for d in out_dims)
    return (
        f"SELECT {pick}, SUM(l.val * r.val) AS val "
        f"FROM {left} l {join} GROUP BY {', '.join(out_dims)}"
    )


def register_tensor(
    ctx: xql.XarrayContext,
    name: str,
    arr: np.ndarray,
    dims: tuple[str, ...],
    var: str = "val",
    chunk: int | None = None,
) -> None:
    """Register a numpy array as a relation, the array-relational way: wrap it as
    an ``xr.Dataset`` whose named dims become the table's key columns, then hand
    it to ``from_dataset``. A tensor is an array at the edge and a relation
    inside; ``from_dataset`` is the bridge, and the dims become the join keys."""
    arr = np.asarray(arr, dtype=np.float64)
    ds = xr.Dataset(
        {var: (dims, arr)},
        coords={d: np.arange(n) for d, n in zip(dims, arr.shape)},
    )
    ctx.from_dataset(name, ds, chunks={dims[0]: chunk or arr.shape[0]})


class Tensors:
    """A step rewrites a handful of relations; ``put`` materialises a query as a
    named table (the stages of the forward/backward pass)."""

    def __init__(self, ctx: xql.XarrayContext):
        self.ctx = ctx

    def put(self, name: str, sql: str) -> None:
        batches = self.ctx.sql(sql).collect()
        if self.ctx.table_exist(name):
            self.ctx.deregister_table(name)
        self.ctx.register_record_batches(name, [batches])


# --- the model as one xarray Dataset ------------------------------------------


def build_model(rng: np.random.Generator) -> xr.Dataset:
    """The whole model as one Dataset: weight ``w{L}`` over dims ``(u{L}, u{L+1})``
    and bias ``b{L}`` over ``(u{L+1},)``. The shared boundary dims tie the layers
    together; the dim sizes *are* the architecture."""
    data_vars: dict = {}
    for layer in range(DEPTH):
        n_in, n_out = WIDTHS[layer], WIDTHS[layer + 1]
        data_vars[f"w{layer}"] = (
            (f"u{layer}", f"u{layer + 1}"),
            rng.standard_normal((n_in, n_out)) * 0.1,
        )
        data_vars[f"b{layer}"] = ((f"u{layer + 1}",), np.zeros(n_out))
    coords = {f"u{i}": np.arange(w) for i, w in enumerate(WIDTHS)}
    return xr.Dataset(data_vars, coords=coords)


def seed_weights(t: Tensors) -> None:
    """Unpack the one model Dataset (registered as the ``model`` schema) into
    working weight/bias relations with a uniform ``val`` column."""
    for layer in range(DEPTH):
        i, o = f"u{layer}", f"u{layer + 1}"
        t.put(
            f"w{layer}", f"SELECT {i}, {o}, w{layer} AS val FROM model.w{layer}"
        )
        t.put(f"b{layer}", f"SELECT {o}, b{layer} AS val FROM model.b{layer}")


# --- the network, as contractions (generic over depth) ------------------------


def forward(t: Tensors, inp: str = "x") -> None:
    """Forward pass from ``inp``: a contraction + bias + tanh per layer, leaving
    the pre-activations ``a{L}.z`` for backprop and the output ``logits``."""
    prev = inp
    for layer in range(DEPTH):
        i, o = f"u{layer}", f"u{layer + 1}"
        zc = contract(f"sample,{i} * {i},{o} -> sample,{o}", prev, f"w{layer}")
        if layer < DEPTH - 1:
            t.put(
                f"a{layer + 1}",
                f"""WITH zc AS ({zc})
                SELECT zc.sample, zc.{o}, zc.val + b{layer}.val AS z,
                       tanh(zc.val + b{layer}.val) AS val
                FROM zc JOIN b{layer} ON zc.{o} = b{layer}.{o}""",
            )
            prev = f"a{layer + 1}"
        else:
            t.put(
                "logits",
                f"""WITH zc AS ({zc})
                SELECT zc.sample, zc.{o}, zc.val + b{layer}.val AS z
                FROM zc JOIN b{layer} ON zc.{o} = b{layer}.{o}""",
            )


def softmax_delta_sql() -> str:
    """Output error delta = softmax(logits) - onehot(label). The one hand-derived
    rule: softmax couples classes through a per-sample normaliser an aggregate
    grad() does not cross."""
    o = f"u{DEPTH}"
    return f"""
    WITH m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
         e AS (SELECT logits.sample, logits.{o}, exp(logits.z - m.m) AS e
               FROM logits JOIN m ON logits.sample = m.sample),
         s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
    SELECT e.sample, e.{o},
           e.e / s.s - CASE WHEN e.{o} = y.label THEN 1.0 ELSE 0.0 END AS val
    FROM e JOIN s ON e.sample = s.sample JOIN y ON y.sample = e.sample"""


def train_step(t: Tensors) -> None:
    """Forward, backward (the same contraction transposed), SGD update."""
    forward(t)
    t.put(f"delta{DEPTH}", softmax_delta_sql())
    # Backward: walk the layers in reverse, the gradients are contractions.
    for layer in reversed(range(DEPTH)):
        i, o = f"u{layer}", f"u{layer + 1}"
        a_in = "x" if layer == 0 else f"a{layer}"
        gw = contract(
            f"sample,{i} * sample,{o} -> {i},{o}", a_in, f"delta{layer + 1}"
        )
        t.put(
            f"gw{layer}", f"SELECT {i}, {o}, val / {N_TRAIN} AS val FROM ({gw})"
        )
        t.put(
            f"gb{layer}",
            f"SELECT {o}, AVG(val) AS val FROM delta{layer + 1} GROUP BY {o}",
        )
        if layer > 0:  # propagate the cotangent, scaled by the local derivative
            dc = contract(
                f"sample,{o} * {i},{o} -> sample,{i}",
                f"delta{layer + 1}",
                f"w{layer}",
            )
            t.put(
                f"delta{layer}",
                f"""WITH dh AS ({dc})
                SELECT dh.sample, dh.{i}, dh.val * grad(tanh(a{layer}.z), a{layer}.z) AS val
                FROM dh JOIN a{layer} ON dh.sample = a{layer}.sample AND dh.{i} = a{layer}.{i}""",
            )
    # SGD: each weight relation becomes w - lr * grad.
    for layer in range(DEPTH):
        i, o = f"u{layer}", f"u{layer + 1}"
        t.put(
            f"w{layer}",
            f"SELECT w{layer}.{i}, w{layer}.{o}, w{layer}.val - {LR} * gw{layer}.val AS val "
            f"FROM w{layer} JOIN gw{layer} ON w{layer}.{i} = gw{layer}.{i} "
            f"AND w{layer}.{o} = gw{layer}.{o}",
        )
        t.put(
            f"b{layer}",
            f"SELECT b{layer}.{o}, b{layer}.val - {LR} * gb{layer}.val AS val "
            f"FROM b{layer} JOIN gb{layer} ON b{layer}.{o} = gb{layer}.{o}",
        )


def accuracy(t: Tensors, inp: str, lab: str) -> float:
    """A forward pass over ``inp`` + argmax, compared to ``lab`` — all in SQL."""
    forward(t, inp)
    o = f"u{DEPTH}"
    return float(
        t.ctx.sql(
            f"""WITH pred AS (
                SELECT sample, {o},
                       ROW_NUMBER() OVER (PARTITION BY sample ORDER BY z DESC) AS rk
                FROM logits)
            SELECT AVG(CASE WHEN p.{o} = l.label THEN 1.0 ELSE 0.0 END) AS acc
            FROM pred p JOIN {lab} l ON p.sample = l.sample WHERE p.rk = 1"""
        ).to_pandas()["acc"][0]
    )


def record_metrics(t: Tensors, step: int) -> None:
    """Append a (step, loss, train_acc, test_acc) row to the ``metrics`` table.

    NN training emits a lot of data — loss curves, per-step accuracies — and like
    everything else here it lives as rows in a relation, grown each time, not a
    Python list. Read it back at the end as a tidy ``(step,)`` xarray.
    """
    o = f"u{DEPTH}"
    train = accuracy(t, "x", "y")  # leaves the training forward in `logits`
    loss = float(
        t.ctx.sql(
            f"""WITH m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
                e AS (SELECT logits.sample, logits.{o}, exp(logits.z - m.m) AS e
                      FROM logits JOIN m ON logits.sample = m.sample),
                s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
            SELECT -AVG(ln(e.e / s.s)) AS loss
            FROM e JOIN s ON e.sample = s.sample JOIN y ON y.sample = e.sample
            WHERE e.{o} = y.label"""
        ).to_pandas()["loss"][0]
    )
    test = accuracy(t, "x_te", "y_te")
    row = (
        f"SELECT CAST({step} AS BIGINT) AS step, CAST({loss} AS DOUBLE) AS loss, "
        f"CAST({train} AS DOUBLE) AS train_acc, CAST({test} AS DOUBLE) AS test_acc"
    )
    t.put(
        "metrics",
        f"SELECT * FROM metrics UNION ALL {row}"
        if t.ctx.table_exist("metrics")
        else row,
    )
    print(
        f"step {step:2d}: loss {loss:.3f}  train {train:.3f}  test {test:.3f}"
    )


# --- MNIST loading ------------------------------------------------------------


def _download(url: str, dest: Path, tries: int = 5) -> None:
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
            if len(data) < 1024:
                raise OSError(f"suspiciously small download: {len(data)} bytes")
            dest.write_bytes(data)
            return
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last = exc
    raise OSError(f"failed to download {url}: {last}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        (magic,) = struct.unpack(">I", f.read(4))
        if magic == 2051:  # images
            n, r, c = struct.unpack(">III", f.read(12))
            return np.frombuffer(f.read(), np.uint8).reshape(n, r, c)
        struct.unpack(">I", f.read(4))  # labels: skip the count
        return np.frombuffer(f.read(), np.uint8)


def load_mnist():
    CACHE.mkdir(exist_ok=True)
    files = {
        "images": "train-images-idx3-ubyte.gz",
        "labels": "train-labels-idx1-ubyte.gz",
    }
    paths = {}
    for key, name in files.items():
        dest = CACHE / name
        if not dest.exists():
            _download(f"{MIRROR}/{name}", dest)
        paths[key] = dest
    imgs = _read_idx(paths["images"]).astype(np.float32) / 255.0
    labs = _read_idx(paths["labels"]).astype(np.int64)
    side = WIDTHS[0]  # pooled pixels per image
    pool = int(round((28 * 28 / side) ** 0.5))  # 2 for 196 pixels
    k = 28 // pool
    pooled = (
        imgs.reshape(-1, k, pool, k, pool).mean(axis=(2, 4)).reshape(-1, side)
    )
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pooled))
    tr, te = idx[:N_TRAIN], idx[N_TRAIN : N_TRAIN + N_TEST]
    return pooled[tr], labs[tr], pooled[te], labs[te]


# --- driver -------------------------------------------------------------------


def main() -> None:
    Xtr, ytr, Xte, yte = load_mnist()
    print(f"MNIST: train {Xtr.shape}, test {Xte.shape}  architecture {WIDTHS}")

    ctx = xql.XarrayContext()
    # The whole model is one Dataset; from_dataset splits it into a table per
    # weight (the shared boundary dims become the join keys).
    rng = np.random.default_rng(1)
    model = build_model(rng)
    ctx.from_dataset(
        "model",
        model,
        table_names={
            (f"u{layer}", f"u{layer + 1}"): f"w{layer}"
            for layer in range(DEPTH)
        }
        | {(f"u{layer + 1}",): f"b{layer}" for layer in range(DEPTH)},
        chunks={f"u{i}": w for i, w in enumerate(WIDTHS)},
    )
    t = Tensors(ctx)
    seed_weights(t)

    # Inputs and labels, registered once; the queries read x / x_te by name.
    register_tensor(ctx, "x", Xtr, ("sample", "u0"), chunk=CHUNK)
    register_tensor(ctx, "y", ytr, ("sample",), var="label")
    register_tensor(ctx, "x_te", Xte, ("sample", "u0"))
    register_tensor(ctx, "y_te", yte, ("sample",), var="label")

    print(f"init: test acc {accuracy(t, 'x_te', 'y_te'):.3f}")

    t0 = time.time()
    for step in range(STEPS):
        train_step(t)
        if step % 10 == 0 or step == STEPS - 1:
            record_metrics(t, step)
    dt = time.time() - t0

    # The trained model comes back out as one xarray Dataset.
    parts = []
    for layer in range(DEPTH):
        i, o = f"u{layer}", f"u{layer + 1}"
        parts.append(
            ctx.sql(f"SELECT {i}, {o}, val FROM w{layer}")
            .to_dataset(dims=[i, o])
            .rename({"val": f"w{layer}"})
        )
        parts.append(
            ctx.sql(f"SELECT {o}, val FROM b{layer}")
            .to_dataset(dims=[o])
            .rename({"val": f"b{layer}"})
        )
    trained = xr.merge(parts)
    # The loss curve and accuracies were recorded as rows; read them back as a
    # tidy (step,) xarray of training metrics.
    metrics = ctx.sql("SELECT * FROM metrics ORDER BY step").to_dataset(
        dims=["step"]
    )

    print(
        f"\ntrained a {WIDTHS} MLP as relational tensor algebra in {dt:.0f}s: "
        f"test accuracy {accuracy(t, 'x_te', 'y_te'):.3f}."
    )
    print(
        f"the model is one xarray Dataset again "
        f"(vars {list(trained.data_vars)}, dims {dict(trained.sizes)}); "
        f"metrics are a table -> xarray {list(metrics.data_vars)} over "
        f"{dict(metrics.sizes)}."
    )


if __name__ == "__main__":
    main()
