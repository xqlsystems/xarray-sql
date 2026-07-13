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
N_SAMPLES, TRAIN_FRAC = 700, 0.7  # total samples; fraction used for training
LR, STEPS, CHUNK = 0.5, 60, 250

# Drop zero-valued pixels from the (dominant) layer-0 contraction. A background
# pixel contributes 0 * weight = 0, so skipping those rows shrinks the join
# *exactly* — the result is identical, and the speedup scales with the fraction
# of zeros (a dark background). On dense inputs it is a no-op.
#
# Measured ~1.8x on real Fashion-MNIST (~50% zero pixels): 2.56 -> 1.45 s/step.
SKIP_ZERO_PIXELS = True


def fashion_mnist():
    """The whole training set, left lazy so SQL streams and samples it.

    The real path returns a dask-backed (chunked) Dataset — nothing is pulled
    into memory here; ``from_dataset`` reads it chunk by chunk on demand, and
    the random subsample happens later in SQL. The offline fallback is a small
    synthetic set built in memory.
    """
    try:
        ds = xr.open_dataset(
            "s3://carbonplan-share/xbatcher/fashion-mnist-train.zarr",
            engine="zarr",
            chunks=None,
            backend_kwargs={"storage_options": {"anon": True}},
        )
        if "channel" in ds.dims:
            ds = ds.isel(channel=0, drop=True)
        # To float64, lazily (no full read). This zarr already stores images
        # as float in [0, 1]; only integer-encoded sources ([0, 255]) rescale.
        images = ds["images"].astype("float64")
        if not np.issubdtype(ds["images"].dtype, np.floating):
            images = images / 255.0
        ds = ds.assign(images=images, labels=ds["labels"].astype("int64"))
    except Exception:
        # Offline fallback: a separable synthetic set (per-class template +
        # noise), so the same pipeline still learns without the network. A pool
        # larger than N_SAMPLES so the SQL subsample still has something to pick.
        rng = np.random.default_rng(0)
        n = 3 * N_SAMPLES
        templates = rng.standard_normal((10, SIDE, SIDE))
        labels = rng.integers(0, 10, n).astype("int64")
        images = templates[labels] + 0.6 * rng.standard_normal((n, SIDE, SIDE))
        ds = xr.Dataset(
            {
                "images": (("sample", "height", "width"), images),
                "labels": (("sample",), labels),
            }
        )
    # Integer index coords are the SQL join keys (sample, height, width).
    return ds[["images", "labels"]].assign_coords(
        sample=np.arange(ds.sizes["sample"]),
        height=np.arange(ds.sizes["height"]),
        width=np.arange(ds.sizes["width"]),
    )


def build_model_with_table_names(
    init_weight: Callable[[int, int], np.ndarray],
    init_bias: Callable[[int], np.ndarray],
    widths=WIDTHS,
) -> tuple[xr.Dataset, dict[tuple[str, ...], str]]:
    """The network as one Dataset that splits into tables per layer.

    Layer ``i`` is a weight matrix ``layer_i (inp_i, out_i)`` and a separate
    bias vector ``bias_i (out_i,)``.
    """
    weights = {
        f"layer_{i}": ((f"inp_{i}", f"out_{i}"), init_weight(inp, out))
        for i, (inp, out) in enumerate(zip(widths[:-1], widths[1:]))
    }
    biases = {
        f"bias_{i}": ((f"out_{i}",), init_bias(out))
        for i, out in enumerate(widths[1:])
    }
    coords = {}
    coords.update(
        {f"inp_{i}": np.arange(inp) for i, inp in enumerate(widths[:-1])}
    )
    coords.update(
        {f"out_{i}": np.arange(out) for i, out in enumerate(widths[1:])}
    )
    ds = xr.Dataset({**weights, **biases}, coords=coords)
    names: dict[tuple[str, ...], str] = {}
    for i in range(len(weights)):
        names[(f"inp_{i}", f"out_{i}")] = f"layer{i}"
        names[(f"out_{i}",)] = f"bias{i}"
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

    # Draw a random N_SAMPLES subset in SQL (ORDER BY random() LIMIT), carrying
    # each sample's label and a train/test tag. `data` is the working label
    # table: cache() pins the chosen subset so every downstream query sees the
    # same split without rescanning the source. `ORDER BY random()` shuffles the
    # whole label column, so the subset is order-independent even if the on-disk
    # samples are class-sorted.
    data = ctx.sql(f"""
    SELECT sample, labels,
    CASE WHEN random() < {TRAIN_FRAC} THEN 'train' ELSE 'test' END AS split
    FROM mnist.labels
    ORDER BY random()
    LIMIT {N_SAMPLES}
    """).cache()
    ctx.register_table("data", data)

    # Materialise just the sampled images once: a single lazy scan of the full
    # dataset extracts the ~N_SAMPLES subset into `pixels`, which the per-step
    # forward joins instead of rescanning the source 60x. Only the subset lives
    # in memory; the full set stays lazy.
    pixels = ctx.sql("""
    SELECT p.sample, p.height, p.width, p.images
    FROM mnist.pixels p JOIN data d ON p.sample = d.sample
    """).cache()
    ctx.register_table("pixels", pixels)

    # The gradient averages over the actual train count (random, ~frac * N),
    # read once from the materialized split.
    n_train = ctx.sql(
        "SELECT COUNT(*) AS n FROM data WHERE split = 'train'"
    ).to_pandas()["n"][0]

    def init_weight(inp: int, out: int):
        """Small random weights."""
        return rng.standard_normal((inp, out)) * 0.1

    def init_bias(out: int):
        """Biases start at zero."""
        return np.zeros(out)

    model, table_names = build_model_with_table_names(init_weight, init_bias)
    ctx.from_dataset(
        "model",
        model,
        table_names=table_names,
        # Each layer table is one chunk: weights on (inp_i, out_i) and the bias
        # vector on (out_i,), so every dim needs a size here.
        chunks={
            **{
                f"inp_{i}": model.sizes[f"inp_{i}"]
                for i in range(len(WIDTHS) - 1)
            },
            **{
                f"out_{i}": model.sizes[f"out_{i}"]
                for i in range(len(WIDTHS) - 1)
            },
        },
    )

    # Unify the per-layer weight tables into one working weight(layer, inp, out,
    # val) relation the loop rewrites in place, tagging each layer with its
    # index.
    seed = " UNION ALL ".join(
        f"SELECT {i} AS layer, inp_{i} AS inp, out_{i} AS out, layer_{i} AS val "
        f"FROM model.layer{i}"
        for i in range(len(WIDTHS) - 1)
    )
    ctx.register_table("weight", ctx.sql(seed).cache())

    # The biases live in their own bias(layer, out, val) relation, summed into
    # each layer's pre-activation as a separate term (z = W @ a + b).
    bias_seed = " UNION ALL ".join(
        f"SELECT {i} AS layer, out_{i} AS out, bias_{i} AS val FROM model.bias{i}"
        for i in range(len(WIDTHS) - 1)
    )
    ctx.register_table("bias", ctx.sql(bias_seed).cache())

    # The zero-pixel skip. fwd0 has no WHERE (it forwards all samples), so it
    # needs a fresh `WHERE`; g0 already filters to the train split, so it
    # appends an `AND`. Empty strings when the flag is off.
    zero_where = "WHERE images <> 0" if SKIP_ZERO_PIXELS else ""
    zero_and = "AND images <> 0" if SKIP_ZERO_PIXELS else ""

    for step in range(STEPS):
        #
        # --- forward pass -----------------------------------------------------
        #
        # Each layer contracts its activation with the weight table (JOIN on the
        # shared index + grouped SUM), then adds the layer's bias as a separate
        # term (JOIN the bias table on `out`), and keeps the pre-activation z
        # (tanh(z) for hidden, linear output). .cache() materialises each stage
        # so the per-step plan stays flat.
        #
        # The forward runs over ALL samples: train rows drive learning, test
        # rows ride along so we can score them from the same logits. Only delta2
        # is restricted to train, so the gradients (and the trained weights) are
        # identical to a train-only forward — test is never backpropagated.
        fwd0 = ctx.sql(f"""
        WITH c AS (
          -- z = x @ W: matmul of the input and first weight matrix
          SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
          FROM (
            SELECT sample, height * {SIDE} + width AS inp, images AS val
            FROM pixels
            {zero_where}
          ) a
          JOIN weight w ON a.inp = w.inp AND w.layer = 0
          GROUP BY a.sample, w.out
        )
        -- activation(z + b): Add in the bias term, then perform activation
        SELECT c.sample, c.out AS out, c.z + b.val AS z,
               tanh(c.z + b.val) AS val
        FROM c JOIN bias b ON c.out = b.out AND b.layer = 0
        """).cache()
        ctx.deregister_table("fwd0")
        ctx.register_table("fwd0", fwd0)

        fwd1 = ctx.sql(f"""
        WITH c AS (
          SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
          FROM (SELECT sample, out AS inp, val FROM fwd0) a
          JOIN weight w ON a.inp = w.inp AND w.layer = 1
          GROUP BY a.sample, w.out
        )
        SELECT c.sample, c.out AS out, c.z + b.val AS z,
               tanh(c.z + b.val) AS val
        FROM c JOIN bias b ON c.out = b.out AND b.layer = 1
        """).cache()
        ctx.deregister_table("fwd1")
        ctx.register_table("fwd1", fwd1)

        # Output layer is linear (softmax lives in the loss / output error),
        # but still gets its bias summed in.
        logits = ctx.sql(f"""
        WITH c AS (
          SELECT a.sample, w.out AS out, SUM(a.val * w.val) AS z
          FROM (SELECT sample, out AS inp, val FROM fwd1) a
          JOIN weight w ON a.inp = w.inp AND w.layer = 2
          GROUP BY a.sample, w.out
        )
        SELECT c.sample, c.out AS out, c.z + b.val AS z
        FROM c JOIN bias b ON c.out = b.out AND b.layer = 2
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
               JOIN data y ON y.sample = e.sample
        -- restrict the error to train, so every downstream gradient is train-only
        WHERE e.sample IN (SELECT sample FROM data WHERE split = 'train')
        """).cache()
        ctx.deregister_table("delta2")
        ctx.register_table("delta2", delta2)

        # Weight gradient of layer 2: fwd1.T @ delta2 / N.
        g2 = ctx.sql(f"""
        SELECT a.inp AS inp, d.out AS out, SUM(a.val * d.val) / {n_train} AS val
        FROM (SELECT sample, out AS inp, val FROM fwd1) a
        JOIN delta2 d ON a.sample = d.sample
        GROUP BY a.inp, d.out
        """).cache()
        ctx.deregister_table("g2")
        ctx.register_table("g2", g2)

        # Bias gradient of layer 2: the mean output error per unit.
        gb2 = ctx.sql(f"""
        SELECT out, SUM(val) / {n_train} AS val FROM delta2 GROUP BY out
        """).cache()
        ctx.deregister_table("gb2")
        ctx.register_table("gb2", gb2)

        # Propagate to layer 1: delta1 = (delta2 @ W2.T) * tanh'(z1). The local
        # derivative is grad(tanh(z), z) at fwd1's pre-activation.
        delta1 = ctx.sql(f"""
        WITH dc AS (
          SELECT d.sample, w.inp AS out, SUM(d.val * w.val) AS val
          FROM delta2 d JOIN weight w ON d.out = w.out AND w.layer = 2
          GROUP BY d.sample, w.inp
        )
        SELECT dc.sample, dc.out,
               dc.val * grad(tanh(fwd1.z), fwd1.z) AS val
        FROM dc JOIN fwd1 ON dc.sample = fwd1.sample AND dc.out = fwd1.out
        """).cache()
        ctx.deregister_table("delta1")
        ctx.register_table("delta1", delta1)

        g1 = ctx.sql(f"""
        SELECT a.inp AS inp, d.out AS out, SUM(a.val * d.val) / {n_train} AS val
        FROM (SELECT sample, out AS inp, val FROM fwd0) a
        JOIN delta1 d ON a.sample = d.sample
        GROUP BY a.inp, d.out
        """).cache()
        ctx.deregister_table("g1")
        ctx.register_table("g1", g1)

        gb1 = ctx.sql(f"""
        SELECT out, SUM(val) / {n_train} AS val FROM delta1 GROUP BY out
        """).cache()
        ctx.deregister_table("gb1")
        ctx.register_table("gb1", gb1)

        # Propagate to layer 0: delta0 = (delta1 @ W1.T) * tanh'(z0).
        delta0 = ctx.sql(f"""
        WITH dc AS (
          SELECT d.sample, w.inp AS out, SUM(d.val * w.val) AS val
          FROM delta1 d JOIN weight w ON d.out = w.out AND w.layer = 1
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
          FROM pixels
          WHERE sample IN (SELECT sample FROM data WHERE split = 'train')
          {zero_and}
        )
        SELECT a.inp AS inp, d.out AS out, SUM(a.val * d.val) / {n_train} AS val
        FROM a JOIN delta0 d ON a.sample = d.sample
        GROUP BY a.inp, d.out
        """).cache()
        ctx.deregister_table("g0")
        ctx.register_table("g0", g0)

        gb0 = ctx.sql(f"""
        SELECT out, SUM(val) / {n_train} AS val FROM delta0 GROUP BY out
        """).cache()
        ctx.deregister_table("gb0")
        ctx.register_table("gb0", gb0)

        #
        # --- SGD update: one query per relation -------------------------------
        #
        # weight <- weight - lr * gradient and bias <- bias - lr * gradient,
        # joining every layer at once against the per-layer gradients tagged
        # with their layer index.
        w = ctx.sql(f"""
        WITH grad AS (
          SELECT 0 AS layer, inp, out, val FROM g0
          UNION ALL SELECT 1 AS layer, inp, out, val FROM g1
          UNION ALL SELECT 2 AS layer, inp, out, val FROM g2
        )
        SELECT w.layer, w.inp, w.out,
               w.val - {LR} * COALESCE(g.val, 0) AS val
        FROM weight w LEFT JOIN grad g
          ON w.layer = g.layer AND w.inp = g.inp AND w.out = g.out
        """).cache()
        ctx.deregister_table("weight")
        ctx.register_table("weight", w)

        b = ctx.sql(f"""
        WITH gb AS (
          SELECT 0 AS layer, out, val FROM gb0
          UNION ALL SELECT 1 AS layer, out, val FROM gb1
          UNION ALL SELECT 2 AS layer, out, val FROM gb2
        )
        SELECT b.layer, b.out,
               b.val - {LR} * COALESCE(g.val, 0) AS val
        FROM bias b LEFT JOIN gb g
          ON b.layer = g.layer AND b.out = g.out
        """).cache()
        ctx.deregister_table("bias")
        ctx.register_table("bias", b)

        if step % 5 == 0 or step == STEPS - 1:
            # Train cross-entropy (logits span all samples, so filter to train).
            loss = ctx.sql(f"""
              WITH m AS (SELECT sample, MAX(z) AS m FROM logits GROUP BY sample),
                   e AS (SELECT logits.sample, logits.out, exp(logits.z - m.m) AS e
                         FROM logits JOIN m ON logits.sample = m.sample),
                   s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
              SELECT -AVG(ln(e.e / s.s)) AS loss
              FROM e JOIN s ON e.sample = s.sample
                     JOIN data y ON y.sample = e.sample
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
                     AVG(CASE WHEN p.out = d.labels THEN 1.0 ELSE 0.0 END) AS acc
              FROM pred p JOIN data d ON d.sample = p.sample
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

    # The trained parameters come back out as xarray in the *same shape as the
    # input model*: one weight variable per layer with its own (inp_i, out_i)
    # dims, plus one bias variable per layer on (out_i,). Each is read from its
    # relation by the `layer` column, so the result is a ragged set of per-layer
    # matrices and vectors — no dense array padded with NaN.
    trained = xr.Dataset(
        {
            **{
                f"layer_{i}": ctx.sql(
                    f"SELECT inp AS inp_{i}, out AS out_{i}, val AS layer_{i} "
                    f"FROM weight WHERE layer = {i}"
                ).to_dataset(dims=[f"inp_{i}", f"out_{i}"])[f"layer_{i}"]
                for i in range(len(WIDTHS) - 1)
            },
            **{
                f"bias_{i}": ctx.sql(
                    f"SELECT out AS out_{i}, val AS bias_{i} "
                    f"FROM bias WHERE layer = {i}"
                ).to_dataset(dims=[f"out_{i}"])[f"bias_{i}"]
                for i in range(len(WIDTHS) - 1)
            },
        }
    )
    print(f"trained {WIDTHS} MLP; weights -> xarray {dict(trained.sizes)}.")
    print(trained)
    trained.to_zarr(
        f"fashion_mnist_mlp_"
        f"{datetime.datetime.now().isoformat(timespec='seconds')}.zarr"
    )


if __name__ == "__main__":
    main()
