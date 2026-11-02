"""Time the original nanoGPT training loop on Apple's PyTorch MPS backend.

nanoGPT (https://github.com/karpathy/nanoGPT) supplies train.py and its
Shakespeare character recipe. An AST instrumentation adds synchronized
timers around each native forward/backward/AdamW update, leaving the
model, dropout, optimizer and data pipeline unchanged. Source files in
the nanoGPT checkout are only read.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


def _call(statement, owner, attribute):
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and isinstance(statement.value.func.value, ast.Name)
        and statement.value.func.value.id == owner
        and statement.value.func.attr == attribute
    )


def _instrument(source):
    tree = ast.parse(source)
    candidates = [node for node in tree.body if isinstance(node, ast.While)]
    if len(candidates) != 1:
        raise ValueError("expected one top-level nanoGPT training loop")
    loop = candidates[0]
    starts = [
        i
        for i, node in enumerate(loop.body)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "micro_step"
    ]
    ends = [
        i
        for i, node in enumerate(loop.body)
        if _call(node, "optimizer", "zero_grad")
    ]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError(
            "cannot identify nanoGPT's forward/backward/update interval"
        )
    begin = ast.parse(
        "torch.mps.synchronize(); _xql_started = _xql_clock()"
    ).body
    end = ast.parse("""torch.mps.synchronize()
_xql_elapsed = _xql_clock() - _xql_started
_xql_record(iter_num, lr, loss, _xql_elapsed)
""").body
    # Insert from the end so the original statement indexes remain valid.
    loop.body[ends[0] + 1 : ends[0] + 1] = end
    loop.body[starts[0] : starts[0]] = begin
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanogpt-source", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sql-summary",
        type=Path,
        default=Path(__file__).with_name("shakespeare-summary.json"),
    )
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("at least two updates are required for a warm mean")
    if not torch.backends.mps.is_available():
        parser.error("MPS GPU is unavailable")
    source_dir, output = args.nanogpt_source.resolve(), args.output.resolve()
    source = (source_dir / "train.py").read_text()
    tree = _instrument(source)
    arguments = [
        "train.py",
        "config/train_shakespeare_char.py",
        "--device=mps",
        "--dtype=float32",
        "--compile=False",
        "--batch_size=12",
        "--block_size=256",
        "--n_layer=6",
        "--n_head=6",
        "--n_embd=384",
        "--gradient_accumulation_steps=1",
        f"--max_iters={args.steps - 1}",
        "--lr_decay_iters=2000",
        "--warmup_iters=100",
        "--eval_interval=200",
        "--eval_iters=20",
        "--log_interval=1",
        "--wandb_log=False",
        f"--out_dir={output.parent / (output.stem + '-out')}",
    ]
    records = []

    def record(iteration, lr, loss, seconds):
        value = float(loss.detach().cpu())
        if not math.isfinite(value) or seconds <= 0:
            raise ValueError("invalid native training loss or duration")
        records.append(
            {
                "iteration": iteration,
                "lr": lr,
                "loss": value,
                "seconds": seconds,
                "tokens_per_second": 3072 / seconds,
            }
        )

    namespace = {
        "__name__": "__main__",
        "__file__": str(source_dir / "train.py"),
        "_xql_clock": time.perf_counter,
        "_xql_record": record,
    }
    previous_dir, previous_argv = Path.cwd(), sys.argv
    sys.path.insert(0, str(source_dir))
    try:
        os.chdir(source_dir)
        sys.argv = arguments
        # Run the explicitly selected local trainer, with only timing hooks.
        exec(compile(tree, str(source_dir / "train.py"), "exec"), namespace)  # noqa: S102
    finally:
        sys.argv = previous_argv
        os.chdir(previous_dir)
        sys.path.remove(str(source_dir))
    if [r["iteration"] for r in records] != list(range(args.steps)):
        raise ValueError(
            "native train.py did not execute the requested update count"
        )
    for row in records:
        expected = 1e-3 * (row["iteration"] + 1) / 101
        if not math.isclose(row["lr"], expected, rel_tol=1e-12):
            raise ValueError(
                "native learning-rate warmup differs from the SQL recipe"
            )
    config = namespace["config"]
    assert config["dropout"] == 0.2 and config["bias"] is False
    assert config["learning_rate"] == 1e-3 and config["min_lr"] == 1e-4
    assert config["beta1"] == 0.9 and config["beta2"] == 0.99
    assert config["weight_decay"] == 0.1 and config["grad_clip"] == 1.0
    assert namespace["meta_vocab_size"] == 65
    parameters = sum(p.numel() for p in namespace["model"].parameters())
    assert parameters == 10745088
    warm = statistics.mean(row["seconds"] for row in records[1:])
    result = {
        "backend": "torch_mps",
        "date": "2026-11-01",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "steps": args.steps,
        "parameters": parameters,
        "batch_tokens": 3072,
        "config": config,
        "nanogpt_revision": subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in (
                "train.py",
                "model.py",
                "config/train_shakespeare_char.py",
            )
        },
        "instrumentation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "native_argv": arguments,
        "rows": records,
        "warm_mean_seconds": warm,
        "warm_tokens_per_second": 3072 / warm,
        "attention": "original model.py scaled_dot_product_attention selection",
        "scope": "Synchronized native forward, backward, gradient clipping, AdamW, zero_grad and next-batch prefetch. Evaluation, initialization, logging and checkpoint I/O excluded. Weights and optimizer state remain on MPS.",
        "matching": "Same model, data split, optimizer, dropout and LR schedule as the SQL recipe; stock PyTorch RNGs, rather than SQL NumPy RNGs. max_iters=steps-1 stops native's inclusive loop after exactly steps updates. compile=False.",
    }
    sql = json.loads(args.sql_summary.read_text())
    result["sql_comparison"] = [
        {
            "backend": run["backend"],
            "seconds": run["warm_mean_seconds"],
            "sql_over_torch_ratio": run["warm_mean_seconds"] / warm,
        }
        for run in sql["runs"]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "backend",
                    "warm_mean_seconds",
                    "warm_tokens_per_second",
                    "sql_comparison",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
