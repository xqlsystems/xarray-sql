# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "xarray_sql",
#   "xarray",
#   "numpy",
# ]
#
# [tool.uv.sources]
# xarray_sql = { path = "..", editable = true }
# ///


from __future__ import annotations

from typing import Callable

import numpy as np
import xarray as xr

import xarray_sql as xql

# N_TRAIN, N_TEST = 1000, 500
LR, STEPS, CHUNK = 0.5, 60, 250


def fashion_mnist():
  try:
    return xr.open_dataset(
      's3://carbonplan-share/xbatcher/fashion-mnist-train.zarr',
      engine='zarr',
      chunks=None,
      backend_kwargs={'storage_options': {'anon': True}},
    )
  except:
    N = 12
    return xr.Dataset(
      {
        'images': (('sample', 'channel', 'height', 'width'), np.random.rand(N, 1, 28, 28)),
        'labels': (('sample',), np.array([i % 10 for i in range(N)])),
      },
    )


def build_model_with_table_names(
    init_weight: Callable[[int, int], np.array],
    widths = (28 * 28, 196, 32, 10)
) -> tuple[xr.Dataset, dict[tuple[str, ...], str]]:
  """The whole network as one ``weight(layer, inp, out)`` Dataset."""
  weights = {
    f'layer_{i}': ((f'inp_{i}', f'out_{i}'), init_weight(inp, out))
    for i, (inp, out) in enumerate(zip(widths[:-1], widths[1:]))
  }
  # model metadata
  coords = {}
  coords.update({f"inp_{i}": np.arange(inp) for i, inp in enumerate(widths[:-1])})
  coords.update({f"out_{i}": np.arange(out) for i, out in enumerate(widths[1:])})

  ds = (
    xr.Dataset(weights, coords=coords)
    .expand_dims({"layer": np.arange(len(weights))})
  )

  names = {
    ("layer", f"inp_{i}", f"out_{i}"): f"layer{i}" for i in range(len(ds))
  }

  return ds, names


def main():
  rng = np.random.default_rng(1)
  mnist = fashion_mnist()

  ctx = xql.XarrayContext()
  ctx.from_dataset(
    "mnist",
    mnist,
    chunks=dict(sample=1),
    table_names={
      ("sample", "channel", "height", "width"): "X",
      ("sample",): "y"
    }
  )

  def init_weight(inp: int, out: int):
    """inp contains is inclusive of a bias term."""
    weight = rng.standard_normal((inp - 1, out)) * 0.1
    bias = np.zeros((1, out))
    return np.concatenate((weight, bias), axis=0)

  model, table_names = build_model_with_table_names(init_weight)
  ctx.from_dataset("model", model, table_names=table_names, chunks=dict(layer=1))

  # # TOOD(alxmrs): Add (train,val,test)-split column
  # data = ctx.sql(
  #   """
  #   SELECT *
  #   FROM mnist.'X' x
  #   JOIN mnist.y y
  #   ON x.sample = y.sample
  #   """
  # )
  # ctx.register_table("data", data)

  for _ in range(STEPS):
    #
    # --- forward pass ---------------------------------------------------------
    #
    fwd0 = ctx.sql(
      """
      SELECT x.sample, (x.height + x.width) as inp, h.out_0 as out,
       tanh(SUM(x.images * h.layer_0)) as val
      FROM mnist.'X' x JOIN model.layer0 h ON (x.height + x.width) = h.inp_0
      GROUP BY x.sample, x.height, x.width, out
      """
    )
    ctx.deregister_table("fwd0")
    ctx.register_table("fwd0", fwd0)

    fwd1 = ctx.sql(
      """
        SELECT x.sample, x.inp as inp, h.out_1 as out,
         tanh(SUM(x.val * h.layer_1)) AS val
        FROM fwd0 x JOIN model.layer1 h ON x.out = h.inp_1
        GROUP BY x.sample, inp, h.out_1
      """
    )
    ctx.deregister_table("fwd1")
    ctx.register_table('fwd1', fwd1)

    fwd2 = ctx.sql(
      """
      SELECT x.sample, x.inp as inp, h.out_2 as out,
       tanh(SUM(x.val * h.layer_2)) AS val
      FROM fwd1 x JOIN model.layer2 h ON x.out = h.inp_2
      GROUP BY x.sample, inp, h.out_2
      """
    )
    ctx.deregister_table("logits")
    ctx.register_table('logits', fwd2)

    #
    # --- backward pass --------------------------------------------------------
    #

    # TODO(alxmrs): Taken from an agent -- this is suspect.
    # Output error = softmax(logits) - onehot(label).
    err = ctx.sql(
      """
      WITH m AS (SELECT sample, MAX(val) AS m FROM logits GROUP BY sample),
           e AS (SELECT logits.sample, logits.out, exp(logits.val - m.m) AS e
                 FROM logits JOIN m ON logits.sample = m.sample),
           s AS (SELECT sample, SUM(e) AS s FROM e GROUP BY sample)
      SELECT e.sample, e.out,
             e.e / s.s - (
              CASE WHEN e.out = mnist.y.labels THEN 1.0 ELSE 0.0 END
             ) AS val
      FROM e 
      JOIN s ON e.sample = s.sample JOIN mnist.y ON mnist.y.sample = e.sample
      """
    )
    ctx.deregister_table("err")
    ctx.register_table('err', err)

    bwd2 = ctx.sql(
      """
      with dh as (
        SELECT h.inp, e.out, SUM(h.val * e.val) as val,
        FROM logits h JOIN err e ON h.sample = e.sample
        GROUP BY h.inp, e.out
      )
      SELECT dh.inp, dh.out, dh.val * grad(tanh(logits.val), logits.val) as val
      FROM dh JOIN logits ON dh.inp = logits.inp AND dh.out = logits.out
      """
    )
    ctx.deregister_table("bwd2")
    ctx.register_table('bwd2', bwd2)

    bwd1 = ctx.sql(
      """
      WITH dh AS (
        SELECT h.inp, e.out, SUM(h.val * e.val) as val,
        FROM fwd1 h JOIN bwd2 e ON h.inp = e.inp
        GROUP BY h.inp, e.out
      )
      SELECT dh.inp, dh.out, dh.val * grad(tanh(fwd1.val), fwd1.val) as val
      FROM dh JOIN fwd1 ON dh.inp = fwd1.inp AND dh.out = fwd1.out
      """
    )
    ctx.deregister_table("bwd1")
    ctx.register_table('bwd1', bwd1)

    bwd0 = ctx.sql(
      """
      WITH dh AS (
        SELECT h.inp, e.out, SUM(h.val * e.val) as val,
        FROM fwd0 h JOIN bwd1 e ON h.inp = e.inp
        GROUP BY h.inp, e.out
      )
      SELECT dh.inp, dh.out, dh.val * grad(tanh(fwd0.val), fwd0.val) as val
      FROM dh JOIN fwd0 ON dh.inp = fwd0.inp AND dh.out = fwd0.out 
      """
    )
    ctx.deregister_table("bwd0")
    ctx.register_table('bwd0', bwd0)


