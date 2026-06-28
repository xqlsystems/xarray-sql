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

A one-hidden-layer neural network (196->32 tanh->10 softmax, on 2x2-pooled
14x14 = 196-pixel images) trained by gradient descent where **every gradient is
computed in SQL**. The MNIST images are registered as xarray (the library's
core); the model weights and per-step intermediates live in DataFusion
in-memory tables. The optimisation loop is plain Python; all the math is
relational.

The design is reverse-mode autodiff expressed in relational algebra:

* **matmul = join + GROUP BY SUM.** A layer's pre-activation is
  ``SUM(input * weight)`` grouped by (sample, unit), joining the data table to a
  weight table on the shared index.
* **local derivatives = grad().** The hidden activation's Jacobian is
  ``grad(tanh(z), z)`` — the engine differentiates the nonlinearity for us,
  evaluated per (sample, unit). This is where the autograd feature does its
  work; the rest is ordinary SQL.
* **cotangent propagation = join.** The output error is pushed back through the
  second weight matrix by another join + SUM, then multiplied by the local
  ``grad`` factor to get the hidden-layer error.
* **parameter gradients = join + GROUP BY AVG.** ``dW = AVG(input * delta)``
  grouped by the weight's indices.

The only hand-written gradient is softmax + cross-entropy's ``delta = softmax -
onehot`` (softmax couples classes through a per-sample normaliser, an aggregate
``grad`` does not cross — staying faithful to SQL). Everything else is grad and
joins.

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


def _download(url: str, dest: Path, tries: int = 5) -> None:
    """Fetch a URL to dest, reading the whole body (retries on truncation)."""
    last = None
    for attempt in range(tries):
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


class SqlTables:
    """Model parameters and intermediates as DataFusion in-memory tables.

    The MNIST data stays registered as xarray (the library's core); the model
    weights and the per-step intermediate results (hidden activations, errors)
    are plain in-memory tables, rebuilt from Arrow each step. Matrices are stored
    in long form — a weight ``W[i, j]`` is a row ``(i, j, w)`` — so a matmul is a
    join + ``GROUP BY``.
    """

    def __init__(self, ctx: xql.XarrayContext):
        self.ctx = ctx

    def _replace(self, name: str, batches: list[pa.RecordBatch]) -> None:
        if self.ctx.table_exist(name):
            self.ctx.deregister_table(name)
        self.ctx.register_record_batches(name, [batches])

    def matrix(
        self, name: str, var: str, arr: np.ndarray, di: str, dj: str
    ) -> None:
        """Register a 2-D array as a long ``(di, dj, var)`` in-memory table."""
        ni, nj = arr.shape
        ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
        batch = pa.RecordBatch.from_pydict(
            {di: ii.ravel(), dj: jj.ravel(), var: arr.ravel()}
        )
        self._replace(name, [batch])

    def vector(self, name: str, var: str, arr: np.ndarray, d0: str) -> None:
        """Register a 1-D array as a ``(d0, var)`` in-memory table."""
        batch = pa.RecordBatch.from_pydict(
            {d0: np.arange(len(arr)), var: np.asarray(arr, dtype=np.float64)}
        )
        self._replace(name, [batch])

    def materialize(self, name: str, sql: str) -> None:
        """Run a query and register its Arrow result as the next stage's table."""
        self._replace(name, self.ctx.sql(sql).collect())


def main() -> None:
    Xtr, ytr, Xte, yte = load_mnist()
    print(
        f"MNIST: train {Xtr.shape}, test {Xte.shape}  ({N_PIX} pix, {N_HID} hidden)"
    )

    ctx = xql.XarrayContext()
    # The data is registered as xarray (the library's core); model state below
    # lives in DataFusion in-memory tables.
    ctx.from_dataset(
        "imgs",
        xr.Dataset(
            {"val": (("sample", "pix"), Xtr)},
            coords={"sample": np.arange(N_TRAIN), "pix": np.arange(N_PIX)},
        ),
        chunks={"sample": N_TRAIN},
    )
    ctx.from_dataset(
        "labels",
        xr.Dataset(
            {"label": (("sample",), ytr.astype(np.float64))},
            coords={"sample": np.arange(N_TRAIN)},
        ),
        chunks={"sample": N_TRAIN},
    )
    t = SqlTables(ctx)

    rng = np.random.default_rng(1)
    W1 = rng.standard_normal((N_PIX, N_HID)) * 0.1
    b1 = np.zeros(N_HID)
    W2 = rng.standard_normal((N_HID, N_CLS)) * 0.1
    b2 = np.zeros(N_CLS)

    def dense_to(df, ni, nj, ci, cj):
        g = np.zeros((ni, nj))
        g[df[ci].to_numpy(), df[cj].to_numpy()] = df["g"].to_numpy()
        return g

    def step(lr: float) -> None:
        nonlocal W1, b1, W2, b2
        t.matrix("w1", "w", W1, "pix", "hid")
        t.vector("b1", "b", b1, "hid")
        t.matrix("w2", "w", W2, "hid", "cls")
        t.vector("b2", "b", b2, "cls")

        # Forward: hidden pre-activation z and activation a = tanh(z).
        t.materialize(
            "h",
            """
            WITH z AS (
              SELECT i.sample, w.hid, SUM(i.val * w.w) + MAX(bb.b) AS z
              FROM imgs i JOIN w1 w ON i.pix = w.pix
                          JOIN b1 bb ON w.hid = bb.hid
              GROUP BY i.sample, w.hid)
            SELECT sample, hid, z, tanh(z) AS a FROM z
            """,
        )
        # Output softmax, then output error delta2 = softmax - onehot(label).
        t.materialize(
            "delta2",
            """
            WITH logit AS (
              SELECT h.sample, w.cls, SUM(h.a * w.w) + MAX(bb.b) AS z
              FROM h JOIN w2 w ON h.hid = w.hid
                     JOIN b2 bb ON w.cls = bb.cls
              GROUP BY h.sample, w.cls),
            mx AS (SELECT sample, MAX(z) AS m FROM logit GROUP BY sample),
            ex AS (SELECT l.sample, l.cls, exp(l.z - mx.m) AS e
                   FROM logit l JOIN mx ON l.sample = mx.sample),
            zsum AS (SELECT sample, SUM(e) AS z FROM ex GROUP BY sample)
            SELECT ex.sample, ex.cls,
                   ex.e / zsum.z
                     - CASE WHEN ex.cls = lb.label THEN 1.0 ELSE 0.0 END AS d
            FROM ex JOIN zsum ON ex.sample = zsum.sample
                    JOIN labels lb ON ex.sample = lb.sample
            """,
        )
        # Backprop to the hidden layer: push delta2 back through W2 (join + SUM),
        # then multiply by the LOCAL activation derivative grad(tanh(z), z).
        t.materialize(
            "delta1",
            """
            WITH da AS (
              SELECT d.sample, w.hid, SUM(d.d * w.w) AS da
              FROM delta2 d JOIN w2 w ON d.cls = w.cls
              GROUP BY d.sample, w.hid)
            SELECT da.sample, da.hid, da.da * grad(tanh(h.z), h.z) AS d
            FROM da JOIN h ON da.sample = h.sample AND da.hid = h.hid
            """,
        )

        # Parameter gradients: dW = AVG(input * delta) over the batch.
        gW2 = dense_to(
            ctx.sql(
                f"SELECT h.hid, d.cls, SUM(h.a * d.d) / {N_TRAIN}.0 AS g "
                "FROM h JOIN delta2 d ON h.sample = d.sample "
                "GROUP BY h.hid, d.cls"
            ).to_pandas(),
            N_HID,
            N_CLS,
            "hid",
            "cls",
        )
        gW1 = dense_to(
            ctx.sql(
                f"SELECT i.pix, d.hid, SUM(i.val * d.d) / {N_TRAIN}.0 AS g "
                "FROM imgs i JOIN delta1 d ON i.sample = d.sample "
                "GROUP BY i.pix, d.hid"
            ).to_pandas(),
            N_PIX,
            N_HID,
            "pix",
            "hid",
        )
        gb2 = ctx.sql(
            f"SELECT cls, SUM(d) / {N_TRAIN}.0 AS g FROM delta2 GROUP BY cls"
        ).to_pandas()
        gb1 = ctx.sql(
            f"SELECT hid, SUM(d) / {N_TRAIN}.0 AS g FROM delta1 GROUP BY hid"
        ).to_pandas()
        vb2 = np.zeros(N_CLS)
        vb2[gb2["cls"].to_numpy()] = gb2["g"].to_numpy()
        vb1 = np.zeros(N_HID)
        vb1[gb1["hid"].to_numpy()] = gb1["g"].to_numpy()

        W2 -= lr * gW2
        b2 -= lr * vb2
        W1 -= lr * gW1
        b1 -= lr * vb1

    def accuracy(X, y) -> float:
        a = np.tanh(X @ W1 + b1)
        return float(((a @ W2 + b2).argmax(1) == y).mean())

    print(f"init: test acc {accuracy(Xte, yte):.3f}")
    t0 = time.time()
    steps = 60
    for s in range(steps):
        step(lr=0.5)
        if s % 10 == 0 or s == steps - 1:
            print(
                f"step {s:2d}: train {accuracy(Xtr, ytr):.3f}  "
                f"test {accuracy(Xte, yte):.3f}"
            )
    print(
        f"\ntrained an MNIST MLP in SQL: test accuracy "
        f"{accuracy(Xte, yte):.3f} in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
