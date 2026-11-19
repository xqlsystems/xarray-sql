"""Rebuild the resident-training summary from recorded benchmark artifacts."""

import hashlib
import json
import statistics
from pathlib import Path


def summarize(root):
    baseline = json.loads((root / "shakespeare-summary.json").read_text())
    before = next(r for r in baseline["runs"] if r["backend"] == "dense_gpu")
    native = json.loads((root / "shakespeare-resident-torch-mps-100.json").read_text())
    runs = []
    for name in ("shakespeare-resident-verified-gpu", "shakespeare-resident-steady-gpu"):
        path = root / name / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        train = [r for r in rows if r["kind"] == "train"]
        seconds = statistics.mean(r["seconds"] for r in train[1:])
        device = [r["device_step"] for r in train]
        run = {
            "metrics": str(path.relative_to(root.parent.parent)),
            "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "updates": len(train),
            "verified_all_updates": all(r["verified"] for r in train),
            "first_step_seconds": train[0]["seconds"],
            "warm_mean_seconds": seconds,
            "warm_tokens_per_second": 3072 / seconds,
            "speedup_over_arrow_gpu": before["warm_mean_seconds"] / seconds,
            "ratio_to_native_mps_100": seconds / native["warm_mean_seconds"],
            "total_training_compilations": sum(r["gpu_training_compilations"] for r in device),
            "timed_array_readback_bytes": sum(r["gpu_readback_bytes"] for r in device),
            "timed_training_exports": sum(r["training_exports"] for r in device),
            "status_reads_per_step": sorted({r["gpu_status_readbacks"] for r in device}),
            "loss_reads_per_step": sorted({r["gpu_loss_readbacks"] for r in device}),
        }
        if "rss_bytes" in train[0]:
            run.update({
                "first_rss_bytes": train[0]["rss_bytes"],
                "last_rss_bytes": train[-1]["rss_bytes"],
                "last_mlx_memory": train[-1]["mlx_memory"],
            })
        runs.append(run)
    return {
        "date": "2026-11-01",
        "device": baseline["device"],
        "platform": baseline["platform"],
        "parameters": 10745088,
        "batch_tokens": 3072,
        "arithmetic": "float32; MLX_ENABLE_TF32=0; compile_training and gpu_dropout enabled",
        "packages": {**baseline["packages"], "ddxdb": "0.4.6+xqlmath.3", "einfold": "0.2.0+xqlmlx.6"},
        "ddx_base": baseline["ddx_revision"],
        "einfold_base": "39b25210d3b39027ef91584383fb114dfa8a4a48",
        "nanogpt_revision": native["nanogpt_revision"],
        "baseline_gpu_seconds": before["warm_mean_seconds"],
        "native_mps_100_warm_seconds": native["warm_mean_seconds"],
        "native_evidence": "benchmarks/apple/shakespeare-resident-torch-mps-100.json",
        "scope": "Synchronized SQL/ddx forward, backward, clipping and AdamW, resident state publication and loss scalar. Evaluation, batch sampling/upload, oracle and checkpoint I/O excluded; GPU dropout is inside. First-step excludes prior evaluation and process startup. Caches may already be warm. Warm mean excludes first update. Native includes next-batch prefetch and uses its own RNG trajectory.",
        "runs": runs,
    }


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    (directory / "resident-summary.json").write_text(json.dumps(summarize(directory), indent=2) + "\n")
