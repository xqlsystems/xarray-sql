"""Real SQL training can resume without changing sampled data or updates."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ddx = pytest.importorskip("ddxdb")
pytest.importorskip("einfold.dense_program")
pytestmark = pytest.mark.skipif(
    "erf" not in ddx.supported_functions(),
    reason="requires the optional ddx erf patch",
)


@pytest.mark.parametrize(
    "execution",
    [
        "arrow",
        "resident",
        pytest.param(
            "compiled",
            marks=pytest.mark.skipif(
                os.environ.get("MLX_ENABLE_TF32") != "0", reason="requires GPU"
            ),
        ),
    ],
)
def test_shakespeare_sql_checkpoint_resume(tmp_path, monkeypatch, execution):
    scripts = Path(__file__).resolve().parents[1] / "benchmarks/ml"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "_test_shakespeare", scripts / "03_shakespeare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = "To be, or not to be.\n" * 30
    vocab = sorted(set(text))
    ids = np.array([vocab.index(c) for c in text], dtype=np.uint16)
    (tmp_path / "input.txt").write_text(text)
    split = int(len(ids) * 0.9)
    ids[:split].tofile(tmp_path / "train.bin")
    ids[split:].tofile(tmp_path / "val.bin")

    def run(out, steps, resume=None):
        args = [
            "03_shakespeare.py",
            "--backend",
            "dense_gpu" if execution == "compiled" else "dense_cpu",
            "--data-dir",
            str(tmp_path),
            "--out-dir",
            str(out),
            "--steps",
            str(steps),
            "--batch_size",
            "2",
            "--block_size",
            "4",
            "--n_layer",
            "1",
            "--n_head",
            "2",
            "--n_embd",
            "8",
            "--eval_iters",
            "1",
            "--eval_interval",
            "2",
        ]
        if execution == "resident":
            args += ["--resident"]
        elif execution == "compiled":
            args += ["--compile-training", "--gpu-dropout"]
        if resume:
            args += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", args)
        module.main()

    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(full, 3)
    run(resumed, 2)
    run(resumed, 1, resumed / "last.npz")
    with (
        np.load(full / "last.npz", allow_pickle=False) as expected,
        np.load(resumed / "last.npz", allow_pickle=False) as actual,
    ):
        for key in expected.files:
            if key != "metadata":
                if execution == "compiled":
                    np.testing.assert_allclose(
                        actual[key],
                        expected[key],
                        atol=3e-6,
                        rtol=3e-6,
                        err_msg=key,
                    )
                else:
                    np.testing.assert_array_equal(
                        actual[key], expected[key], err_msg=key
                    )
        a, b = (
            json.loads(str(actual["metadata"])),
            json.loads(str(expected["metadata"])),
        )
        assert a["iteration"] == b["iteration"] == 3
        assert a["data_rng"] == b["data_rng"]
        assert a["mask_rng"] == b["mask_rng"]
        assert a["best_val_loss"] == b["best_val_loss"]
    events = [
        json.loads(line)
        for line in (full / "metrics.jsonl").read_text().splitlines()
    ]
    assert [e["iteration"] for e in events if e["kind"] == "eval"] == [0, 2]
    assert [e["lr"] for e in events if e["kind"] == "train"] == [
        1e-3 * (i + 1) / 101 for i in range(3)
    ]
