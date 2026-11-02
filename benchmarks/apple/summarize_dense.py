"""Summarize checked serial SQL benchmarks without including failed runs.

Warm time is the mean paired loss/update API time at steps 1–4. Setup and
first-use costs are reported separately. Per-parameter validation and
conversion of returned Arrow rows to Datasets are outside the API timers.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _summarize(manifest):
    results = []
    for record in json.loads(manifest.read_text()):
        result = dict(record) | {"manifest": manifest.name}
        if record["status"] != "passed":
            results.append(result)
            continue
        rows = list(csv.DictReader((manifest.parent / record["csv"]).open()))
        command = record["command"]
        steps = int(command[command.index("--steps") + 1])
        forward = "--forward-only" in command
        if len(rows) != steps * (1 if forward else 2) or any(
            not r["check"].startswith("ok (") for r in rows
        ):
            raise ValueError(f"incomplete correctness records: {record['csv']}")
        loss = {
            int(r["step"].rsplit(" ", 1)[1]): float(r["seconds"])
            for r in rows
            if r["step"].startswith("loss")
        }
        train = {
            int(r["step"].rsplit(" ", 1)[1]): float(r["seconds"])
            for r in rows
            if r["step"].startswith("train")
        }
        pairs = [loss[i] + train.get(i, 0) for i in range(steps)]
        first, last = rows[0], rows[-1]
        setup = sum(
            float(last.get(k) or 0) for k in ("load_seconds", "model_seconds")
        )
        result |= {
            "steps": steps,
            "batch": int(first["batch"]),
            "parameters": int(first["parameters"]),
            "context": int(first["context"]),
            "optimizer": first["optimizer"],
            "forward_only": forward,
            "rtol": float(first["rtol"]),
            "atol": float(first["atol"]),
            "paired_api_seconds": pairs,
            "warm_loss_seconds": statistics.mean(
                loss[i] for i in range(1, steps)
            ),
            "warm_training_seconds": statistics.mean(
                train[i] for i in range(1, steps)
            )
            if train
            else 0,
            "warm_combined_seconds": statistics.mean(pairs[1:]),
            "five_api_seconds": sum(pairs),
            "loading_and_model_seconds": setup,
            "five_api_with_loading_model_seconds": sum(pairs) + setup,
            "final_statistics": {
                k: v
                for k, v in last.items()
                if k.startswith(
                    (
                        "dense_",
                        "gpu_",
                        "physical_",
                        "planning_",
                        "stage_",
                        "key_",
                    )
                )
            },
        }
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = [r for path in args.manifests for r in _summarize(path)]
    args.output.write_text(
        json.dumps({"method": __doc__, "runs": runs}, indent=2) + "\n"
    )
    for run in runs:
        if run["status"] == "passed":
            print(
                run["manifest"],
                run["host"],
                f"{run['parameters']:,}",
                f"ctx={run['context']}",
                f"warm={run['warm_combined_seconds']:.6f}s",
                f"rss={run['peak_rss_gib']:.2f}GiB",
            )
        else:
            print(run["manifest"], run["host"], run["status"])


if __name__ == "__main__":
    main()
