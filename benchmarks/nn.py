# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "xarray_sql",
#   "xarray",
#   "numpy",
#   "s3fs",
#   "zarr<3",
# ]
#
# [tool.uv.sources]
# xarray_sql = { path = "..", editable = true }
# ///


from __future__ import annotations

from typing import Callable

import numpy as np
import xarray as xr
import datetime

import xarray_sql as xql

SIDE = 28  # images are 28x28; flatten index is height * SIDE + width
WIDTHS = (
    SIDE * SIDE,
    196,
    32,
    10,
)  # 784 pixels -> 196 -> 32 tanh -> 10 softmax
N_TRAIN, N_TEST = 500, 200
LR, STEPS, CHUNK = 0.5, 60, 250

# Drop zero-valued pixels from the (dominant) layer-0 contraction. A background
# pixel contributes 0 * weight = 0, so skipping those rows shrinks the join
# *exactly* — the result is identical, and the speedup scales with the fraction
# of zeros (a dark background). On dense inputs it is a no-op. Toggle to compare.
SKIP_ZERO_PIXELS = True


def fashion_mnist():
    try:
        ds = (
            xr.open_dataset(
                "s3://carbonplan-share/xbatcher/fashion-mnist-train.zarr",
                engine="zarr",
                chunks=None,
                backend_kwargs={"storage_options": {"anon": True}},
            )
            .isel(sample=slice(N_TRAIN + N_TEST))
            .load()
        )
        if "channel" in ds.dims:
            ds = ds.isel(channel=0, drop=True)
        images = ds["images"].astype("float64").values
        labels = ds["labels"].values.astype("int64")
    except Exception:
        # Offline fallback: a separable synthetic set (per-class template +
        # noise), so the same pipeline still learns without the network.
        rng = np.random.default_rng(0)
        n = N_TRAIN + N_TEST
        templates = rng.standard_normal((10, SIDE, SIDE))
        labels = rng.integers(0, 10, n).astype("int64")
        images = templates[labels] + 0.6 * rng.standard_normal((n, SIDE, SIDE))
    if images.max() > 1.0:
        images = images / 255.0
    return xr.Dataset(
        {
            "images": (("sample", "height", "width"), images),
            "labels": (("sample",), labels),
        },
        coords={
            "sample": np.arange(images.shape[0]),
            "height": np.arange(SIDE),
            "width": np.arange(SIDE),
        },
    )


def build_model_with_table_names(
    init_weight: Callable[[int, int], np.array],
    widths=WIDTHS,
) -> tuple[xr.Dataset, dict[tuple[str, str], str]]:
    """The network as one Dataset that splits into one table per layer.

    Layer ``i`` is ``layer_i (inp_i, out_i)`` with the folded bias as an extra
    ``inp_i = widths[i]`` row, so ``inp_i`` has ``widths[i] + 1`` entries.
    """
    weights = {
        f"layer_{i}": ((f"inp_{i}", f"out_{i}"), init_weight(inp, out))
        for i, (inp, out) in enumerate(zip(widths[:-1], widths[1:]))
    }
    coords = {}
    coords.update(
        {f"inp_{i}": np.arange(inp + 1) for i, inp in enumerate(widths[:-1])}
    )
    coords.update(
        {f"out_{i}": np.arange(out) for i, out in enumerate(widths[1:])}
    )
    ds = xr.Dataset(weights, coords=coords)
    names = {(f"inp_{i}", f"out_{i}"): f"layer{i}" for i in range(len(weights))}
    return ds, names


def main():
    rng = np.random.default_rng(1)
    mnist = fashion_mnist()

    ctx = xql.XarrayContext()
    # One Dataset splits into two tables: pixels (sample, height, width) and
    # labels (sample). The dim names are the join keys.
    ctx.from_dataset(
        "mnist",
        mnist,
        chunks=dict(sample=CHUNK),
        table_names={
            ("sample", "height", "width"): "pixels",
            ("sample",): "labels",
        },
    )

    frac = N_TRAIN / (N_TRAIN + N_TEST)  # default ratio: ~0.7
    # Train-test split
    data = ctx.sql(f"""
    SELECT sample,
    CASE WHEN random() < {frac} THEN 'train' ELSE 'test' END AS split
    FROM mnist.labels
    """).cache()
    ctx.register_table("data", data)
    # The gradient averages over the actual train count (random, ~frac * N),
    # read once from the materialized split.
    n_train = ctx.sql(
        "SELECT COUNT(*) AS n FROM data WHERE split = 'train'"
    ).to_pandas()["n"][0]

    def init_weight(inp: int, out: int):
        """Small random weights with a zero bias row appended."""
        weight = rng.standard_normal((inp, out)) * 0.1
        bias = np.zeros((1, out))
        return np.concatenate((weight, bias), axis=0)  # (inp + 1, out)

    model, table_names = build_model_with_table_names(init_weight)
    ctx.from_dataset(
        "model",
        model,
        table_names=table_names,
        chunks={
            f"inp_{i}": model.sizes[f"inp_{i}"] for i in range(len(WIDTHS) - 1)
        },
    )

    # Unify the per-layer tables into one working weight(layer, inp, out, val)
    # relation the loop rewrites in place, tagging each layer with its index.
    seed = " UNION ALL ".join(
        f"SELECT {i} AS layer, inp_{i} AS inp, out_{i} AS out, layer_{i} AS val, "
        f"{width} AS width FROM model.layer{i}"
        for i, width in enumerate(WIDTHS[:-1])
    )
    ctx.register_table("weight", ctx.sql(seed).cache())

    # The zero-pixel skip. fwd0 has no WHERE (it forwards all samples), so it
    # needs a fresh `WHERE`; g0 already filters to the train split, so it
    # appends an `AND`. Empty strings when the flag is off.
    zero_where = "WHERE images <> 0" if SKIP_ZERO_PIXELS else ""
    zero_and = "AND images <> 0" if SKIP_ZERO_PIXELS else ""

    for step in range(STEPS):
        #
        # --- forward pass -----------------------------------------------------
        #
        # Each layer augments its activation with a constant-1 bias unit (
        # index = width), contracts with the weight table (JOIN on the shared
        # index + grouped SUM), and keeps the pre-activation z (tanh(z) for
        # hidden, linear output). .cache() materialises each stage so the
        # per-step plan stays flat.
        #
        # The forward runs over ALL samples: train rows drive learning, test
        # rows ride along so we can score them from the same logits. Only delta2
        # is restricted to train, so the gradients (and the trained weights) are
        # identical to a train-only forward — test is never backpropagated.
        fwd0 = ctx.sql(f"""
        WITH a AS (
          SELECT sample, height * {SIDE} + width AS inp, images AS val
          FROM mnist.pixels
          {zero_where}
          UNION ALL
          -- the constant-1 bias unit
          SELECT sample,
            (SELECT DISTINCT width FROM weight WHERE layer = 0) AS inp,
            1.0 AS val
          FROM mnist.labels
        )
        SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z,
               tanh(SUM(a.val * w.val)) AS val
        FROM a JOIN weight w ON a.inp = w.inp AND w.layer = 0
        GROUP BY a.sample, w.out
        """).cache()
        ctx.deregister_table("fwd0")
        ctx.register_table("fwd0", fwd0)

        fwd1 = ctx.sql(f"""
        WITH a AS (
          SELECT sample, out AS inp, val FROM fwd0
          UNION ALL
          SELECT DISTINCT sample,
                 (SELECT DISTINCT width FROM weight WHERE layer = 1) AS inp,
                 1.0 AS val FROM fwd0
        )
        SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z,
               tanh(SUM(a.val * w.val)) AS val
        FROM a JOIN weight w ON a.inp = w.inp AND w.layer = 1
        GROUP BY a.sample, w.out
        """).cache()
        ctx.deregister_table("fwd1")
        ctx.register_table("fwd1", fwd1)

        # Output layer is linear (softmax lives in the loss / output error).
        logits = ctx.sql(f"""
        WITH a AS (
          SELECT sample, out AS inp, val FROM fwd1
          UNION ALL
          SELECT DISTINCT sample,
                 (SELECT DISTINCT width FROM weight WHERE layer = 2) AS inp,
                 1.0 AS val FROM fwd1
        )
        SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
        FROM a JOIN weight w ON a.inp = w.inp AND w.layer = 2
        GROUP BY a.sample, w.out
        """).cache()
        ctx.deregister_table("logits")
        ctx.register_table("logits", logits)
        #
        # --- backward pass ----------------------------------------------------
        #
        # Output error delta2 = softmax(logits) - onehot(label). The one
        # hand-derived rule: softmax couples classes through a per-sample
        # normaliser.
        delta2 = ctx.sql(f"""
        WITH m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
             e AS (SELECT logits.sample, logits.out, exp(logits.z - m.m) AS e
                   FROM logits JOIN m ON logits.sample = m.sample),
             s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
        SELECT e.sample, e.out,
               e.e / s.s - CASE WHEN e.out = y.labels THEN 1.0 ELSE 0.0 END AS val
        FROM e JOIN s ON e.sample = s.sample
               JOIN mnist.labels y ON y.sample = e.sample
        -- restrict the error to train, so every downstream gradient is train-only
        WHERE e.sample IN (SELECT sample FROM data WHERE split = 'train')
        """).cache()
        ctx.deregister_table("delta2")
        ctx.register_table("delta2", delta2)

        # Weight gradient of layer 2: (bias-augmented fwd1).T @ delta2 / N.
        # The bias row (inp = width) falls out for free — its gradient is the
        # mean error.
        g2 = ctx.sql(f"""
        WITH a AS (
          SELECT sample, out AS inp, val FROM fwd1
          UNION ALL
          SELECT DISTINCT sample,
                 (SELECT DISTINCT width FROM weight WHERE layer = 2) AS inp,
                 1.0 AS val FROM fwd1
        )
        SELECT a.inp AS inp, d.out AS out, SUM(a.val * d.val) / {n_train} AS val
        FROM a JOIN delta2 d ON a.sample = d.sample
        GROUP BY a.inp, d.out
        """).cache()
        ctx.deregister_table("g2")
        ctx.register_table("g2", g2)

        # Propagate to layer 1: delta1 = (delta2 @ W2[non-bias].T) * tanh'(
        # z1). The local derivative is grad(tanh(z), z) at fwd1's
        # pre-activation.
        delta1 = ctx.sql(f"""
        WITH dc AS (
          SELECT d.sample, w.inp AS out, SUM(d.val * w.val) AS val
          FROM delta2 d JOIN weight w ON d.out = w.out AND w.layer = 2
          WHERE w.inp < w.width
          GROUP BY d.sample, w.inp
        )
        SELECT dc.sample, dc.out,
               dc.val * grad(tanh(fwd1.z), fwd1.z) AS val
        FROM dc JOIN fwd1 ON dc.sample = fwd1.sample AND dc.out = fwd1.out
        """).cache()
        ctx.deregister_table("delta1")
        ctx.register_table("delta1", delta1)

        g1 = ctx.sql(f"""
        WITH a AS (
          SELECT sample, out AS inp, val FROM fwd0
          UNION ALL
          SELECT DISTINCT sample,
                 (SELECT DISTINCT width FROM weight WHERE layer = 1) AS inp,
                 1.0 AS val FROM fwd0
        )
        SELECT a.inp AS inp, d.out AS out, SUM(a.val * d.val) / {n_train} AS val
        FROM a JOIN delta1 d ON a.sample = d.sample
        GROUP BY a.inp, d.out
        """).cache()
        ctx.deregister_table("g1")
        ctx.register_table("g1", g1)

        # Propagate to layer 0: delta0 = (delta1 @ W1[non-bias].T) * tanh'(z0).
        delta0 = ctx.sql(f"""
        WITH dc AS (
          SELECT d.sample, w.inp AS out, SUM(d.val * w.val) AS val
          FROM delta1 d JOIN weight w ON d.out = w.out AND w.layer = 1
          WHERE w.inp < w.width
          GROUP BY d.sample, w.inp
        )
        SELECT dc.sample, dc.out,
               dc.val * grad(tanh(fwd0.z), fwd0.z) AS val
        FROM dc JOIN fwd0 ON dc.sample = fwd0.sample AND dc.out = fwd0.out
        """).cache()
        ctx.deregister_table("delta0")
        ctx.register_table("delta0", delta0)

        g0 = ctx.sql(f"""
        WITH a AS (
          SELECT sample, height * {SIDE} + width AS inp, images AS val
          FROM mnist.pixels
          WHERE sample IN (SELECT sample FROM data WHERE split = 'train')
          {zero_and}
          UNION ALL
          SELECT sample,
                 (SELECT DISTINCT width FROM weight WHERE layer = 0) AS inp,
                 1.0 AS val
          FROM mnist.labels
          WHERE sample IN (SELECT sample FROM data WHERE split = 'train')
        )
        SELECT a.inp AS inp, d.out AS out, SUM(a.val * d.val) / {n_train} AS val
        FROM a JOIN delta0 d ON a.sample = d.sample
        GROUP BY a.inp, d.out
        """).cache()
        ctx.deregister_table("g0")
        ctx.register_table("g0", g0)

        #
        # --- SGD update: one query over the whole relation --------------------
        #
        # weight <- weight - lr * gradient, joining every layer at once
        # against the per-layer gradients tagged with their layer index.
        w = ctx.sql(f"""
        WITH grad AS (
          SELECT 0 AS layer, inp, out, val FROM g0
          UNION ALL SELECT 1 AS layer, inp, out, val FROM g1
          UNION ALL SELECT 2 AS layer, inp, out, val FROM g2
        )
        SELECT w.layer, w.inp, w.out,
               w.val - {LR} * COALESCE(g.val, 0) AS val, w.width
        FROM weight w LEFT JOIN grad g
          ON w.layer = g.layer AND w.inp = g.inp AND w.out = g.out
        """).cache()
        ctx.deregister_table("weight")
        ctx.register_table("weight", w)

        if step % 5 == 0 or step == STEPS - 1:
            # Train cross-entropy (logits span all samples, so filter to train).
            loss = ctx.sql(f"""
              WITH m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
                   e AS (SELECT logits.sample, logits.out, exp(logits.z - m.m) AS e
                         FROM logits JOIN m ON logits.sample = m.sample),
                   s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
              SELECT -AVG(ln(e.e / s.s)) AS loss
              FROM e JOIN s ON e.sample = s.sample
                     JOIN mnist.labels y ON y.sample = e.sample
              WHERE e.out = y.labels
                AND e.sample IN (SELECT sample FROM data WHERE split = 'train')
              """).to_pandas()["loss"][0]
            # Accuracy per split: argmax the shared logits, join the split label.
            # Both come from the one all-samples forward — no second pass.
            acc = (
                ctx.sql(f"""
              WITH pred AS (
                SELECT sample, out,
                       ROW_NUMBER() OVER (PARTITION BY sample ORDER BY z DESC) AS rk
                FROM logits)
              SELECT d.split,
                     AVG(CASE WHEN p.out = y.labels THEN 1.0 ELSE 0.0 END) AS acc
              FROM pred p JOIN mnist.labels y ON p.sample = y.sample
                          JOIN data d ON d.sample = p.sample
              WHERE p.rk = 1
              GROUP BY d.split
              """)
                .to_pandas()
                .set_index("split")["acc"]
            )
            print(
                f"step {step:2d}: loss {loss:.3f}  "
                f"train_acc {acc['train']:.3f}  test_acc {acc['test']:.3f}"
            )

    # The trained weights come back out as xarray as one relation: a ragged
    # weight(layer, inp, out) array (absent cells are NaN where layers are narrower).
    trained = (
        ctx.sql("SELECT layer, inp, out, val FROM weight")
        .to_dataset(dims=["layer", "inp", "out"])
        .rename({"val": "weight"})
    )
    print(f"trained {WIDTHS} MLP; weights -> xarray {dict(trained.sizes)}.")
    print(trained)
    trained.to_zarr(
        f"fashion_mnist_mlp_"
        f"{datetime.datetime.now().isoformat(timespec='seconds')}.zarr"
    )


if __name__ == "__main__":
    main()
