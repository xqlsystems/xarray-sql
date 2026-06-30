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
"""Train an MNIST MLP as relational tensor algebra — the whole net is one table.

A neural network is a chain of **tensor contractions** (einsums), and an einsum
over coordinate-indexed arrays *is* relational algebra:

    C[i,k] = sum_j A[i,j] * B[j,k]   <=>   JOIN A, B ON A.j = B.j
                                           GROUP BY i, k -> SUM(A.val * B.val)

Contracting a shared index is a join on it followed by a grouped SUM. In
xarray-sql an array indexed by named dims is a table keyed by those dims, so the
dim names are the join keys.

Two simplifications make the whole model **one relation**:

* **Bias folded into the weights (an ``nn.Linear``).** Each layer's bias is the
  weight of a constant-``1`` input, stored as the extra row ``inp = width`` in the
  same weight array — so a layer is a single matrix. The forward reads the matmul
  rows and that bias row from the one relation (no separate bias table).
* **A ``layer`` dimension.** Every layer's weight lives in one
  ``weight(layer, inp, out)`` array, so the forward/backward filter on the
  ``layer`` *column* instead of referencing a table per layer. The whole network
  is one ``xr.Dataset`` registered with ``from_dataset``; differing layer widths
  are NaN-padded in the dense array and dropped on the way in (the relational
  form is naturally ragged). The architecture is data — change ``WIDTHS`` and the
  same code trains a different net.

A single ``contract()`` and one generic loop train a net of any depth: forward
contracts the activation with ``weight WHERE layer = L``; backward is the same
contraction transposed (the VJP of a contraction is a contraction), with
``grad(tanh(z), z)`` for the one local-derivative step. Even the weight update is
one query over the whole ``weight`` relation. Linear algebra is joins; the
derivatives of the nonlinearities are ``grad``.

Everything stays relational and inspectable: activations, errors, gradients, and
the per-step training metrics are all tables; the trained model, predictions, and
metrics come back out as ``xarray`` via ``to_dataset``.

This is not a numpy replacement — the long form puts one matrix entry per row, so
the matmul-as-join carries overhead a BLAS inner product doesn't. What it buys is
a declarative, inspectable pipeline whose data side is chunked xarray (parallel
over the batch, larger-than-memory). Recovering BLAS speed would mean storing
dense *tiles* per cell and contracting them with a tile-matmul — a future
direction, not done here.

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
import pyarrow as pa
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
    """An einsum over two coordinate-indexed relations, as one SQL query.

    ``contract("sample,inp * inp,out -> sample,out", "x", w)`` joins ``x`` and
    ``w`` on their shared dim ``inp``, groups by the output dims, and sums the
    product of values — a matmul. ``left`` / ``right`` are table names or
    parenthesised subqueries; each exposes its dims plus a ``val`` column.
    Indices in the inputs but not the output are contracted (summed over).
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
        # UNION branches can yield batches that differ only in field nullability;
        # cast them all to one (nullable) schema so registration accepts them.
        if batches:
            target = pa.schema(
                [pa.field(f.name, f.type) for f in batches[0].schema]
            )
            batches = [b.cast(target) for b in batches]
        if self.ctx.table_exist(name):
            self.ctx.deregister_table(name)
        self.ctx.register_record_batches(name, [batches])


# --- the model: one weight relation, bias folded in ---------------------------


def build_model(rng: np.random.Generator) -> xr.Dataset:
    """The whole network as one ``weight(layer, inp, out)`` Dataset.

    Layer ``L`` connects ``WIDTHS[L]`` inputs (plus a constant-1 bias input, index
    ``WIDTHS[L]``) to ``WIDTHS[L+1]`` outputs. The dense array is NaN-padded to the
    widest layer; the padding is dropped when the relation is seeded, so the live
    table is the ragged set of real weights.
    """
    max_in = max(WIDTHS[layer] + 1 for layer in range(DEPTH))
    max_out = max(WIDTHS[layer + 1] for layer in range(DEPTH))
    arr = np.full((DEPTH, max_in, max_out), np.nan)
    for layer in range(DEPTH):
        n_in, n_out = WIDTHS[layer], WIDTHS[layer + 1]
        arr[layer, :n_in, :n_out] = rng.standard_normal((n_in, n_out)) * 0.1
        arr[layer, n_in, :n_out] = (
            0.0  # bias row (weight of the constant input)
        )
    return xr.Dataset(
        {"weight": (("layer", "inp", "out"), arr)},
        coords={
            "layer": np.arange(DEPTH),
            "inp": np.arange(max_in),
            "out": np.arange(max_out),
        },
    )


def matmul_rows(layer: int) -> str:
    """The matmul (non-bias) rows of one layer's weight, as a subquery."""
    return f"(SELECT inp, out, val FROM weight WHERE layer = {layer} AND inp < {WIDTHS[layer]})"


def bias_row(layer: int) -> str:
    """The bias row (inp = width) of one layer's weight, as a subquery over out."""
    return f"(SELECT out, val FROM weight WHERE layer = {layer} AND inp = {WIDTHS[layer]})"


# --- the network, as contractions (generic over depth) ------------------------


def forward(t: Tensors, inp: str = "x") -> None:
    """Forward pass from ``inp``: per layer, contract with the matmul rows and add
    the bias row (both from the one weight relation), then tanh on the hidden
    layers. Leaves ``a{L}.z`` for backprop and the output ``logits``."""
    prev = inp
    for layer in range(DEPTH):
        zc = contract(
            "sample,inp * inp,out -> sample,out", prev, matmul_rows(layer)
        )
        if layer < DEPTH - 1:
            t.put(
                f"a{layer + 1}",
                f"""WITH zc AS ({zc})
                SELECT zc.sample, zc.out AS inp, zc.val + b.val AS z,
                       tanh(zc.val + b.val) AS val
                FROM zc JOIN {bias_row(layer)} b ON zc.out = b.out""",
            )
            prev = f"a{layer + 1}"
        else:
            t.put(
                "logits",
                f"""WITH zc AS ({zc})
                SELECT zc.sample, zc.out, zc.val + b.val AS z
                FROM zc JOIN {bias_row(layer)} b ON zc.out = b.out""",
            )


def softmax_delta_sql() -> str:
    """Output error delta = softmax(logits) - onehot(label). The one hand-derived
    rule: softmax couples classes through a per-sample normaliser an aggregate
    grad() does not cross."""
    return """
    WITH m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
         e AS (SELECT logits.sample, logits.out, exp(logits.z - m.m) AS e
               FROM logits JOIN m ON logits.sample = m.sample),
         s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
    SELECT e.sample, e.out,
           e.e / s.s - CASE WHEN e.out = y.label THEN 1.0 ELSE 0.0 END AS val
    FROM e JOIN s ON e.sample = s.sample JOIN y ON y.sample = e.sample"""


def train_step(t: Tensors) -> None:
    """Forward, backward (the same contraction transposed), one SGD update."""
    forward(t)
    t.put(f"delta{DEPTH}", softmax_delta_sql())
    # Backward: gradients are contractions over the batch, accumulated into one
    # gweight relation tagged by layer. delta{L} is the error at layer L's units.
    for layer in reversed(range(DEPTH)):
        a_in = "x" if layer == 0 else f"a{layer}"
        # matmul gradient (mean over batch) + bias gradient (mean of delta),
        # both tagged with this layer, as rows of one gweight relation.
        gw = contract(
            "sample,inp * sample,out -> inp,out", a_in, f"delta{layer + 1}"
        )
        rows = (
            f"SELECT CAST({layer} AS BIGINT) AS layer, inp, out, "
            f"val / {N_TRAIN} AS val FROM ({gw}) "
            f"UNION ALL "
            f"SELECT CAST({layer} AS BIGINT) AS layer, "
            f"CAST({WIDTHS[layer]} AS BIGINT) AS inp, out, AVG(val) AS val "
            f"FROM delta{layer + 1} GROUP BY out"
        )
        t.put(
            "gweight",
            f"SELECT * FROM gweight UNION ALL {rows}"
            if t.ctx.table_exist("gweight")
            else rows,
        )
        if layer > 0:  # propagate the cotangent, scaled by the local derivative
            dc = contract(
                "sample,out * inp,out -> sample,inp",
                f"delta{layer + 1}",
                matmul_rows(layer),
            )
            t.put(
                f"delta{layer}",
                f"""WITH dc AS ({dc})
                SELECT dc.sample, dc.inp AS out,
                       dc.val * grad(tanh(a{layer}.z), a{layer}.z) AS val
                FROM dc JOIN a{layer}
                  ON dc.sample = a{layer}.sample AND dc.inp = a{layer}.inp""",
            )
    # One SGD update for the whole network: weight <- weight - lr * gweight.
    t.put(
        "weight",
        f"""SELECT w.layer, w.inp, w.out, w.val - {LR} * g.val AS val
        FROM weight w JOIN gweight g
          ON w.layer = g.layer AND w.inp = g.inp AND w.out = g.out""",
    )
    t.ctx.deregister_table("gweight")


def accuracy(t: Tensors, inp: str, lab: str) -> float:
    """A forward pass over ``inp`` + argmax, compared to ``lab`` — all in SQL."""
    forward(t, inp)
    return float(
        t.ctx.sql(
            f"""WITH pred AS (
                SELECT sample, out,
                       ROW_NUMBER() OVER (PARTITION BY sample ORDER BY z DESC) AS rk
                FROM logits)
            SELECT AVG(CASE WHEN p.out = l.label THEN 1.0 ELSE 0.0 END) AS acc
            FROM pred p JOIN {lab} l ON p.sample = l.sample WHERE p.rk = 1"""
        ).to_pandas()["acc"][0]
    )


def record_metrics(t: Tensors, step: int) -> None:
    """Append a (step, loss, train_acc, test_acc) row to the ``metrics`` table.

    NN training emits a lot of data — loss curves, per-step accuracies — and like
    everything else here it lives as rows in a relation, grown each time, not a
    Python list. Read it back at the end as a tidy ``(step,)`` xarray.
    """
    train = accuracy(t, "x", "y")  # leaves the training forward in `logits`
    loss = float(
        t.ctx.sql(
            """WITH m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
                e AS (SELECT logits.sample, logits.out, exp(logits.z - m.m) AS e
                      FROM logits JOIN m ON logits.sample = m.sample),
                s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
            SELECT -AVG(ln(e.e / s.s)) AS loss
            FROM e JOIN s ON e.sample = s.sample JOIN y ON y.sample = e.sample
            WHERE e.out = y.label"""
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
    pool = 28 // int(round(side**0.5))  # 2 for 196 pixels (14x14)
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
    # The whole model is one Dataset with a layer dim; from_dataset gives one
    # `net` table, and seeding drops the NaN padding to the live `weight` relation.
    rng = np.random.default_rng(1)
    model = build_model(rng)
    ctx.from_dataset(
        "net",
        model,
        chunks={
            "layer": DEPTH,
            "inp": model.sizes["inp"],
            "out": model.sizes["out"],
        },
    )
    t = Tensors(ctx)
    t.put(
        "weight",
        "SELECT layer, inp, out, weight AS val FROM net WHERE weight IS NOT NULL",
    )

    # Inputs and labels (the bias is in the weight relation, so no augmentation).
    register_tensor(ctx, "x", Xtr, ("sample", "inp"), chunk=CHUNK)
    register_tensor(ctx, "y", ytr, ("sample",), var="label")
    register_tensor(ctx, "x_te", Xte, ("sample", "inp"))
    register_tensor(ctx, "y_te", yte, ("sample",), var="label")

    print(f"init: test acc {accuracy(t, 'x_te', 'y_te'):.3f}")
    t0 = time.time()
    for step in range(STEPS):
        train_step(t)
        if step % 10 == 0 or step == STEPS - 1:
            record_metrics(t, step)
    dt = time.time() - t0

    # The trained model, predictions, and metrics all come back out as xarray.
    weights = (
        ctx.sql("SELECT layer, inp, out, val FROM weight")
        .to_dataset(dims=["layer", "inp", "out"])
        .rename({"val": "weight"})
    )
    metrics = ctx.sql("SELECT * FROM metrics ORDER BY step").to_dataset(
        dims=["step"]
    )

    print(
        f"\ntrained a {WIDTHS} MLP as relational tensor algebra in {dt:.0f}s: "
        f"test accuracy {accuracy(t, 'x_te', 'y_te'):.3f}."
    )
    print(
        f"the whole model is one weight relation -> xarray "
        f"{dict(weights.sizes)}; metrics are a table -> xarray "
        f"{list(metrics.data_vars)} over {dict(metrics.sizes)}."
    )


if __name__ == "__main__":
    main()
