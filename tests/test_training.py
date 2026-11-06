"""Owned SQL training state preserves updates, failures and snapshots."""

import os

import numpy as np
import pytest
import xarray as xr

from xarray_sql import Context

pytest.importorskip("ddxdb")
pytest.importorskip("einfold.dense_program")
from xarray_sql.training import TrainingSession

_PRECISIONS = [
    "float64",
    pytest.param(
        "float32",
        marks=pytest.mark.skipif(
            os.environ.get("MLX_ENABLE_TF32") != "0",
            reason="GPU requires MLX_ENABLE_TF32=0",
        ),
    ),
]


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_training_state_reuses_primal_and_owns_inputs(precision):
    ctx = Context()
    initial = xr.Dataset({"v": ("i", [1.0, 2.0, 3.0])}, coords={"i": [2, 4, 7]})
    ctx.from_dataset("w", initial)
    session = TrainingSession(ctx, precision=precision)
    session.bind("w")
    initial.v.values[:] = 99
    step = session.prepare_step(
        gradients={
            "g": "WITH loss AS (SELECT SUM(v*v)/2 AS l FROM w) SELECT i,grad(loss,v) AS g FROM w ORDER BY i"
        },
        stages={},
        updates={
            "u": "SELECT p.i,p.v-0.1*g.g AS v FROM w p JOIN g ON p.i=g.i ORDER BY p.i"
        },
        assignments={"w": ("u", ["i", "v"])},
    )
    before = session.export("w")["w"]
    for k in range(3):
        result = step()
        np.testing.assert_allclose(
            float(result.loss), 7 * 0.9 ** (2 * k), rtol=1e-6
        )
        np.testing.assert_allclose(
            session.export("w")["w"]["v"],
            np.array([1, 2, 3]) * 0.9 ** (k + 1),
            rtol=1e-6,
        )
    assert before.to_pydict() == {"i": [2, 4, 7], "v": [1.0, 2.0, 3.0]}
    session.bind(
        "w", xr.Dataset({"v": ("i", [4.0, 5.0, 6.0])}, coords={"i": [2, 4, 7]})
    )
    np.testing.assert_allclose(float(step().loss), 38.5, rtol=1e-6)
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        step()


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_training_failure_keeps_all_state(precision):
    ctx = Context()
    for name, values in (("a", [1.0, 2.0]), ("b", [3.0, 4.0])):
        ctx.from_dataset(name, xr.Dataset({"v": ("i", values)}))
    session = TrainingSession(ctx, precision=precision)
    session.bind("a")
    session.bind("b")
    before = session.export("a", "b")
    step = session.prepare_step(
        gradients={
            "g": "WITH loss AS (SELECT SUM(v*v) AS l FROM a) SELECT i,grad(loss,v) AS g FROM a ORDER BY i"
        },
        stages={},
        updates={
            "good": "SELECT i,v+1 AS v FROM a ORDER BY i",
            "bad": "SELECT i,ln(-v) AS v FROM b ORDER BY i",
        },
        assignments={"a": ("good", ["i", "v"]), "b": ("bad", ["i", "v"])},
    )
    with pytest.raises(ValueError, match="finite|exceptional"):
        step()
    after = session.export("a", "b")
    assert all(after[name].equals(before[name]) for name in before)
    assert session.version == 0


@pytest.mark.skipif(
    os.environ.get("MLX_ENABLE_TF32") != "0", reason="requires GPU"
)
def test_compiled_training_uses_new_tokens_and_learning_rates():
    ctx = Context()
    weights = np.arange(1, 7, dtype=float).reshape(3, 2)
    ctx.from_dataset("w", xr.Dataset({"v": (("tok", "e"), weights)}))
    session = TrainingSession(ctx)
    session.bind("w")
    session.bind("batch", xr.Dataset({"tok": ("pos", [2, 2, 0])}))
    session.index_domains("batch", {"tok": ("w", "tok")})
    session.bind("batch")
    session.bind("rate", xr.Dataset({"lr": xr.DataArray(0.1)}))
    step = session.prepare_step(
        gradients={
            "g": """WITH model AS (
          SELECT b.pos,w.e,w.v FROM batch b JOIN w ON b.tok=w.tok),
          loss AS (SELECT SUM(v*v)/2 AS l FROM model)
          SELECT tok,e,grad(loss,v) AS g FROM w ORDER BY tok,e"""
        },
        stages={},
        updates={
            "u": "SELECT w.tok,w.e,w.v-r.lr*g.g AS v FROM w JOIN g ON w.tok=g.tok AND w.e=g.e CROSS JOIN rate r ORDER BY w.tok,w.e"
        },
        assignments={"w": ("u", ["tok", "e", "v"])},
        compile=True,
    )
    for tokens, lr in (([2, 2, 0], 0.1), ([1, 0, 1], 0.03), ([0, 0, 2], 0.07)):
        session.bind("batch", xr.Dataset({"tok": ("pos", tokens)}))
        session.bind("rate", xr.Dataset({"lr": xr.DataArray(lr)}))
        expected_loss = np.sum(weights[tokens] ** 2) / 2
        weights = (
            weights - lr * np.bincount(tokens, minlength=3)[:, None] * weights
        )
        result = step()
        assert result.measurements["gpu_readback_bytes"] == 0
        np.testing.assert_allclose(float(result.loss), expected_loss, rtol=3e-6)
        rows = session.export("w")["w"]
        np.testing.assert_allclose(
            np.asarray(rows["v"]).reshape(3, 2), weights, rtol=3e-6
        )
    before = session.export("w")["w"]
    with pytest.raises(ValueError, match="domain"):
        session.bind("batch", xr.Dataset({"tok": ("pos", [9, 0, 1])}))
    assert session.export("w")["w"].equals(before)


def test_training_rejects_schema_and_coordinate_changes():
    ctx = Context()
    ctx.from_dataset("w", xr.Dataset({"v": ("i", [1.0, 2.0])}))
    session = TrainingSession(ctx, precision="float64")
    session.bind("w")
    before = session.export("w")["w"]
    for ds in (
        xr.Dataset({"other": ("i", [1.0, 2.0])}),
        xr.Dataset({"v": ("i", [1.0, 2.0])}, coords={"i": [4, 5]}),
    ):
        with pytest.raises(ValueError, match="schema|coordinate"):
            session.bind("w", ds)
    assert session.export("w")["w"].equals(before)


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_loss_snapshot_cannot_modify_scalar_state(precision):
    ctx = Context()
    ctx.from_dataset("w", xr.Dataset({"v": ("i", [2.0])}))
    session = TrainingSession(ctx, precision=precision)
    session.bind("w")
    step = session.prepare_step(
        gradients={
            "g": "WITH loss AS (SELECT SUM(v) AS l FROM w) SELECT grad(loss,v) AS g FROM w"
        },
        stages={},
        updates={"u": "SELECT i,v FROM w"},
        assignments={"w": ("u", ["i", "v"])},
    )
    result = step()
    result.loss[...] = 99
    assert session.export("w")["w"].to_pydict() == {"i": [0], "v": [2.0]}
