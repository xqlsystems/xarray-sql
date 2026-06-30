# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "xarray_sql",
#   "xarray",
#   "numpy",
#   "pyarrow",
# ]
#
# [tool.uv.sources]
# xarray_sql = { path = "..", editable = true }
# ///
"""Train an MNIST MLP classifier in SQL.

A one-hidden-layer network (196->32 tanh->10 softmax, on 2x2-pooled 14x14 =
196-pixel images) trained by gradient descent where **every gradient is computed
in SQL** — and the whole model, with its entire training history, lives in a
single table.

The model is one append-only table ``model(step, layer, i, j, val)``: every
parameter ``W1[i, j]`` / ``b1[i]`` / ``W2`` / ``b2`` is a row, tagged by which
generation (``step``) it belongs to. **A training step never mutates anything —
it appends the next generation's rows.** The full optimisation trajectory is the
table; ``WHERE step = N`` is the model at iteration N.

Each step is a *single* SQL statement (``STEP`` below) that reads the current
generation and writes the next. It is reverse-mode autodiff as relational
algebra:

* **matmul = join + GROUP BY SUM.** A layer's pre-activation is
  ``SUM(input * weight)`` grouped by (sample, unit), joining the data to the
  current weight rows.
* **local derivatives = grad().** The hidden Jacobian is ``grad(tanh(z), z)`` —
  the autograd feature differentiates the nonlinearity, per (sample, unit).
* **cotangent propagation = join.** The output error is pushed back through W2 by
  another join + SUM, then scaled by the local ``grad`` factor.
* **parameter gradients = join + GROUP BY AVG**, and the update is ``w - lr*g``,
  emitted as the next generation's rows.

The only hand-written gradient is softmax + cross-entropy's ``delta = softmax -
onehot`` (softmax couples classes through a per-sample normaliser, which an
aggregate ``grad`` does not cross — staying faithful to SQL). Everything else is
grad and joins. Evaluation is SQL too: a forward pass with ``ROW_NUMBER()`` for
the argmax.

Why is the *outer* loop still Python rather than one recursive query (like
``grad_descent.py``)? A recursive CTE may reference the recursive relation only
once, but a 2-layer net uses the current weights several times per step (W1 and
W2 in the forward pass, W2 again in backprop), so it cannot be a single recursive
statement. Training is also inherently sequential and reuses each step's result,
so the steps must be *materialised* between iterations — which is exactly what the
thin Python loop does (append a generation, then query it). All the maths stays
in SQL; Python only sequences the steps.

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

# Network dimensions: 14x14 pooled pixels -> 32 hidden (tanh) -> 10 classes.
N_TRAIN, N_TEST, N_PIX, N_HID, N_CLS = 1000, 500, 196, 32, 10
LR, STEPS = 0.5, 60


def _download(url: str, dest: Path, tries: int = 5) -> None:
    """Fetch a URL to dest, reading the whole body (retries on truncation)."""
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
        (n,) = struct.unpack(">I", f.read(4))  # labels
        return np.frombuffer(f.read(), np.uint8)


def load_mnist():
    """Download (and cache) MNIST, 2x2 mean-pool to 14x14, subsample."""
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
    pooled = imgs.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, N_PIX)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pooled))
    tr, te = idx[:N_TRAIN], idx[N_TRAIN : N_TRAIN + N_TEST]
    return pooled[tr], labs[tr], pooled[te], labs[te]


# --- the model as rows --------------------------------------------------------

_MODEL_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("layer", pa.utf8()),
        ("i", pa.int64()),
        ("j", pa.int64()),
        ("val", pa.float64()),
    ]
)


def _param_rows(step: int, layer: str, arr: np.ndarray) -> dict:
    """One layer's parameters as ``(step, layer, i, j, val)`` columns.

    A matrix ``W[i, j]`` becomes rows ``(i, j, w)``; a bias vector ``b[i]``
    becomes ``(i, 0, b)``.
    """
    if arr.ndim == 2:
        ii, jj = np.meshgrid(
            np.arange(arr.shape[0]), np.arange(arr.shape[1]), indexing="ij"
        )
        ii, jj = ii.ravel(), jj.ravel()
    else:
        ii, jj = np.arange(arr.size), np.zeros(arr.size, np.int64)
    n = arr.size
    return {
        "step": np.full(n, step, np.int64),
        "layer": [layer] * n,
        "i": ii.astype(np.int64),
        "j": jj.astype(np.int64),
        "val": arr.ravel().astype(np.float64),
    }


def _generation_batch(step, w1, b1, w2, b2) -> pa.RecordBatch:
    """All four layers of one generation as a single RecordBatch."""
    cols: dict[str, list] = {k: [] for k in ("step", "layer", "i", "j", "val")}
    for layer, arr in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2)):
        for k, v in _param_rows(step, layer, arr).items():
            cols[k].extend(list(v))
    return pa.RecordBatch.from_arrays(
        [
            pa.array(cols["step"], pa.int64()),
            pa.array(cols["layer"], pa.utf8()),
            pa.array(cols["i"], pa.int64()),
            pa.array(cols["j"], pa.int64()),
            pa.array(cols["val"], pa.float64()),
        ],
        schema=_MODEL_SCHEMA,
    )


# One training step, as one SQL statement: read the current generation of the
# model table, run the forward + backward pass over the data, and SELECT the next
# generation's parameter rows (which the loop appends to the model table).
STEP = f"""
WITH cur AS (SELECT max(step) AS s FROM model),
 w1 AS (SELECT i AS pix, j AS hid, val AS w FROM model, cur
        WHERE step = cur.s AND layer = 'w1'),
 b1 AS (SELECT i AS hid, val AS b FROM model, cur
        WHERE step = cur.s AND layer = 'b1'),
 w2 AS (SELECT i AS hid, j AS cls, val AS w FROM model, cur
        WHERE step = cur.s AND layer = 'w2'),
 b2 AS (SELECT i AS cls, val AS b FROM model, cur
        WHERE step = cur.s AND layer = 'b2'),
 -- forward: hidden pre-activation z and activation a = tanh(z)
 zt AS (SELECT i.sample, w.hid, SUM(i.val * w.w) + MAX(bb.b) AS z
        FROM imgs i JOIN w1 w ON i.pix = w.pix JOIN b1 bb ON w.hid = bb.hid
        GROUP BY i.sample, w.hid),
 h AS (SELECT sample, hid, z, tanh(z) AS a FROM zt),
 -- output logits, then a stable softmax
 lg AS (SELECT h.sample, w.cls, SUM(h.a * w.w) + MAX(bb.b) AS z
        FROM h JOIN w2 w ON h.hid = w.hid JOIN b2 bb ON w.cls = bb.cls
        GROUP BY h.sample, w.cls),
 mx AS (SELECT sample, MAX(z) AS m FROM lg GROUP BY sample),
 ex AS (SELECT l.sample, l.cls, exp(l.z - mx.m) AS e
        FROM lg l JOIN mx ON l.sample = mx.sample),
 zs AS (SELECT sample, SUM(e) AS z FROM ex GROUP BY sample),
 -- output error delta2 = softmax - onehot(label)
 d2 AS (SELECT ex.sample, ex.cls,
               ex.e / zs.z
                 - CASE WHEN ex.cls = lb.label THEN 1.0 ELSE 0.0 END AS d
        FROM ex JOIN zs ON ex.sample = zs.sample
                JOIN labels lb ON lb.sample = ex.sample),
 -- backprop to hidden: push delta2 through W2, scale by grad(tanh(z), z)
 da AS (SELECT d.sample, w.hid, SUM(d.d * w.w) AS da
        FROM d2 d JOIN w2 w ON d.cls = w.cls GROUP BY d.sample, w.hid),
 d1 AS (SELECT da.sample, da.hid, da.da * grad(tanh(h.z), h.z) AS d
        FROM da JOIN h ON da.sample = h.sample AND da.hid = h.hid),
 -- parameter gradients: dW = AVG(input * delta) over the batch
 gw1 AS (SELECT i.pix, d.hid, AVG(i.val * d.d) AS g
         FROM imgs i JOIN d1 d ON i.sample = d.sample GROUP BY i.pix, d.hid),
 gb1 AS (SELECT hid, AVG(d) AS g FROM d1 GROUP BY hid),
 gw2 AS (SELECT h.hid, d.cls, AVG(h.a * d.d) AS g
         FROM h JOIN d2 d ON h.sample = d.sample GROUP BY h.hid, d.cls),
 gb2 AS (SELECT cls, AVG(d) AS g FROM d2 GROUP BY cls)
-- the next generation: w - lr*grad, tagged step+1, as model rows
SELECT (SELECT s FROM cur) + 1 AS step, 'w1' AS layer,
       w.pix AS i, w.hid AS j, w.w - {LR} * g.g AS val
FROM w1 w JOIN gw1 g ON w.pix = g.pix AND w.hid = g.hid
UNION ALL
SELECT (SELECT s FROM cur) + 1, 'b1', b.hid, CAST(0 AS BIGINT), b.b - {LR} * g.g
FROM b1 b JOIN gb1 g ON b.hid = g.hid
UNION ALL
SELECT (SELECT s FROM cur) + 1, 'w2', w.hid, w.cls, w.w - {LR} * g.g
FROM w2 w JOIN gw2 g ON w.hid = g.hid AND w.cls = g.cls
UNION ALL
SELECT (SELECT s FROM cur) + 1, 'b2', b.cls, CAST(0 AS BIGINT), b.b - {LR} * g.g
FROM b2 b JOIN gb2 g ON b.cls = g.cls
"""


def eval_sql(imgs_table: str, labels_table: str) -> str:
    """Accuracy of the latest model on a dataset — a forward pass in SQL.

    ``ROW_NUMBER()`` picks each sample's argmax class; it is compared to the
    label. No softmax needed at inference: the argmax of the logits is the
    prediction.
    """
    return f"""
    WITH cur AS (SELECT max(step) AS s FROM model),
     w1 AS (SELECT i AS pix, j AS hid, val AS w FROM model, cur
            WHERE step = cur.s AND layer = 'w1'),
     b1 AS (SELECT i AS hid, val AS b FROM model, cur
            WHERE step = cur.s AND layer = 'b1'),
     w2 AS (SELECT i AS hid, j AS cls, val AS w FROM model, cur
            WHERE step = cur.s AND layer = 'w2'),
     b2 AS (SELECT i AS cls, val AS b FROM model, cur
            WHERE step = cur.s AND layer = 'b2'),
     h AS (SELECT i.sample, w.hid,
                  tanh(SUM(i.val * w.w) + MAX(bb.b)) AS a
           FROM {imgs_table} i JOIN w1 w ON i.pix = w.pix
                               JOIN b1 bb ON w.hid = bb.hid
           GROUP BY i.sample, w.hid),
     lg AS (SELECT h.sample, w.cls, SUM(h.a * w.w) + MAX(bb.b) AS z
            FROM h JOIN w2 w ON h.hid = w.hid JOIN b2 bb ON w.cls = bb.cls
            GROUP BY h.sample, w.cls),
     pred AS (SELECT sample, cls,
                     ROW_NUMBER() OVER (PARTITION BY sample ORDER BY z DESC) AS rk
              FROM lg)
    SELECT AVG(CASE WHEN p.cls = l.label THEN 1.0 ELSE 0.0 END) AS acc
    FROM pred p JOIN {labels_table} l ON p.sample = l.sample
    WHERE p.rk = 1
    """


def _register_images(ctx, name, X):
    ctx.from_dataset(
        name,
        xr.Dataset(
            {"val": (("sample", "pix"), X)},
            coords={
                "sample": np.arange(X.shape[0]),
                "pix": np.arange(N_PIX),
            },
        ),
        chunks={"sample": X.shape[0]},
    )


def _register_labels(ctx, name, y):
    ctx.from_dataset(
        name,
        xr.Dataset(
            {"label": (("sample",), y.astype(np.float64))},
            coords={"sample": np.arange(len(y))},
        ),
        chunks={"sample": len(y)},
    )


def main() -> None:
    Xtr, ytr, Xte, yte = load_mnist()
    print(
        f"MNIST: train {Xtr.shape}, test {Xte.shape}  "
        f"({N_PIX} pix, {N_HID} hidden, {N_CLS} classes)"
    )

    ctx = xql.XarrayContext()
    # The data is registered as xarray (the library's core); the model below is
    # the one append-only table that holds every layer and every generation.
    _register_images(ctx, "imgs", Xtr)
    _register_labels(ctx, "labels", ytr)
    _register_images(ctx, "imgs_te", Xte)
    _register_labels(ctx, "labels_te", yte)

    # Generation 0: small random weights, zero biases.
    rng = np.random.default_rng(1)
    gen0 = _generation_batch(
        0,
        rng.standard_normal((N_PIX, N_HID)) * 0.1,
        np.zeros(N_HID),
        rng.standard_normal((N_HID, N_CLS)) * 0.1,
        np.zeros(N_CLS),
    )
    generations = [gen0]
    ctx.register_record_batches("model", [generations])

    def test_acc() -> float:
        return float(
            ctx.sql(eval_sql("imgs_te", "labels_te")).to_pandas()["acc"][0]
        )

    print(f"init: test acc {test_acc():.3f}")
    t0 = time.time()
    for s in range(STEPS):
        # One SQL statement computes the next generation; appending its rows to
        # the model table *is* the parameter update.
        generations.extend(ctx.sql(STEP).collect())
        ctx.deregister_table("model")
        ctx.register_record_batches("model", [generations])
        if s % 10 == 0 or s == STEPS - 1:
            tr = float(
                ctx.sql(eval_sql("imgs", "labels")).to_pandas()["acc"][0]
            )
            print(f"step {s:2d}: train {tr:.3f}  test {test_acc():.3f}")

    n_rows = ctx.sql("SELECT count(*) AS n FROM model").to_pandas()["n"][0]
    print(
        f"\ntrained an MNIST MLP in SQL: test accuracy {test_acc():.3f} "
        f"in {time.time() - t0:.0f}s.\nThe model and its entire training "
        f"history are one table of {n_rows} rows ({STEPS + 1} generations)."
    )


if __name__ == "__main__":
    main()
