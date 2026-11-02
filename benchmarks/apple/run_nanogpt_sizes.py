"""Run checked nanoGPT size comparisons in bounded, separate processes.

Each host starts from identical seeds. Logs, checked CSV checkpoints, exact
commands, exit status and sampled peak resident memory are retained. The
limits apply only to these benchmark processes, never to other applications.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["tiny", "demo", "shakespeare", "gpt2"],
        required=True,
    )
    parser.add_argument(
        "--hosts", nargs="+", default=["shared_cpu", "shared_gpu"]
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="run later hosts after a failure",
    )
    parser.add_argument("--max-minutes", type=float, default=20)
    parser.add_argument("--rss-limit-gib", type=float, default=96)
    parser.add_argument(
        "--dense-cache-gib",
        type=float,
        default=8,
        help="computed array cache for compact hosts",
    )
    parser.add_argument(
        "--contended",
        action="store_true",
        help="record concurrent game/device load",
    )
    parser.add_argument(
        "--prefix", help="output basename (defaults to nanogpt-shared-PRESET)"
    )
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.batch,
            args.max_minutes,
            args.rss_limit_gib,
            args.dense_cache_gib,
        )
        <= 0
    ):
        parser.error("steps, batch, timeout and memory limit must be positive")
    output = Path(__file__).resolve().parent
    prefix = args.prefix or f"nanogpt-shared-{args.preset}"
    results = []
    for host in args.hosts:
        stem = output / f"{prefix}-{host}"
        csv_path, log_path = stem.with_suffix(".csv"), stem.with_suffix(".log")
        command = [
            sys.executable,
            "-u",
            "benchmarks/ml/02_nanogpt.py",
            "--preset",
            args.preset,
            "--steps",
            str(args.steps),
            "--sizes",
            str(args.batch),
            "--optimizer",
            args.optimizer,
            "--learning-rate",
            str(args.learning_rate),
            "--hosts",
            host,
            "--csv",
            str(csv_path),
        ]
        if args.context is not None:
            command.extend(["--context", str(args.context)])
        if args.forward_only:
            command.append("--forward-only")
        vocab = args.vocab_size or (50257 if args.preset == "gpt2" else None)
        if vocab is not None:
            command.extend(["--vocab-size", str(vocab)])
        # A checkpoint from an older process must never survive a failed run.
        csv_path.unlink(missing_ok=True)
        started = time.monotonic()
        record = {
            "host": host,
            "command": command,
            "contended": args.contended,
            "dense_cache_gib": args.dense_cache_gib,
            "log": log_path.name,
            "csv": csv_path.name,
            "peak_rss_gib": 0.0,
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        print(
            f"starting {host}: {args.preset}, steps={args.steps}, batch={args.batch}",
            flush=True,
        )
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "MLX_ENABLE_TF32": "0",
                    "XQL_DENSE_CACHE_BYTES": str(
                        int(args.dense_cache_gib * (1 << 30))
                    ),
                    "PYTHONPATH": str(_ROOT)
                    + os.pathsep
                    + os.environ.get("PYTHONPATH", ""),
                },
            )
            while process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    sample = subprocess.run(
                        ["ps", "-o", "rss=", "-p", str(process.pid)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    rss = sample.stdout.strip()
                    if rss:
                        record["peak_rss_gib"] = max(
                            record["peak_rss_gib"], int(rss) / (1 << 20)
                        )
                    reason = None
                    if record["peak_rss_gib"] > args.rss_limit_gib:
                        reason = "rss_limit"
                    elif time.monotonic() - started > args.max_minutes * 60:
                        reason = "timeout"
                    if reason:
                        record["status"] = reason
                        _stop(process)
            record["exit_code"] = process.returncode
        record["wall_seconds"] = time.monotonic() - started
        checked = (
            list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
        )
        record["checked_rows"] = sum(
            r["check"].startswith("ok (") for r in checked
        )
        if record["status"] == "running":
            record["status"] = (
                "passed"
                if process.returncode == 0
                and record["checked_rows"]
                == args.steps * (1 if args.forward_only else 2)
                else "failed"
            )
        results.append(record)
        (output / f"{prefix}-runs.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(
            f"{host}: {record['status']}, {record['checked_rows']} checked rows, "
            f"{record['wall_seconds']:.1f}s wall, sampled peak RSS {record['peak_rss_gib']:.1f} GiB",
            flush=True,
        )
        if record["status"] != "passed" and not args.keep_going:
            break
    if any(r["status"] != "passed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
