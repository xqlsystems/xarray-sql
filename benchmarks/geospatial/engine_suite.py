"""The geospatial suite across engines and VM sizes, via Coiled Functions.

Runs the nine geospatial cases (``01_ndvi`` … ``09_warp``) under every
SQL engine the suite supports — DataFusion (the original path), DuckDB,
and Polars, selected per process through ``GEOBENCH_ENGINE`` and the
``_engines`` facade — on one reused Coiled VM per machine size, driven
in parallel across sizes.

The measurement protocol is exactly ``run_perf.sh``'s: every repetition
is a **fresh process** with no warm-up (``GEOBENCH_PROFILE=1
GEOBENCH_WARMUP=0 GEOBENCH_REPS=1``), so the SQL side and the xarray
reference each pay a cold read on every rep, and each case's own
correctness assertion (SQL answer == array reference) must pass for the
timing to count. The xarray-reference timings are engine-independent;
the tables report the reference column from the DataFusion runs.

Coverage notes, recorded rather than hidden: cases 07 and 09 build
DataFusion scalar UDFs, so DuckDB/Polars are marked n/a; case 08 reads
through Earth Engine and is left on the original context (EE-gated);
cases 07–09 skip cleanly wherever Earth Engine auth is unavailable
(e.g. on the benchmark VMs) with the reason recorded.

Each (vm, case, engine) cell returns a plain dict; every completed cell
is appended to a local ``--jsonl`` file immediately, and the driver
prints one timestamped line per event.

Usage::

    python benchmarks/geospatial/engine_suite.py --local --reps 1 \
        --cases 02_climatology --vms local            # in-process check
    python benchmarks/geospatial/engine_suite.py     # 3 VMs, full suite
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

REGION = "us-central1"

CASES = [
    "01_ndvi",
    "02_climatology",
    "03_zonal_mean",
    "04_anomaly",
    "05_forecast_skill",
    "06_zonal_vector",
    "07_reproject_udf",
    "08_regrid_weights",
    "09_warp",
]
ENGINES = ["datafusion", "duckdb", "polars"]
# Cases whose SQL builds DataFusion scalar UDFs (07, and the UDF half of
# 09): not expressible on the other engines. Case 08 is portable SQL but
# Earth-Engine-gated, so it stays on the original context.
NOT_PORTABLE = {
    "07_reproject_udf": "n/a (DataFusion scalar UDF)",
    "09_warp": "n/a (DataFusion scalar UDF)",
    "08_regrid_weights": "not ported (Earth-Engine-gated case)",
}
VM_SIZES = ["e2-standard-8", "e2-standard-16", "e2-standard-32"]


def cluster_name(vm: str) -> str:
    return "xql-geo-" + vm.replace("standard-", "")


# --------------------------------------------------------------------------
# Remote side (runs inside the coiled function, or locally with --local)
# --------------------------------------------------------------------------


def _install_src(src_targz: bytes | None) -> tuple[str, str]:
    """Unpack the shipped source tree; returns (sys.path root, geo dir).

    The root is keyed by the tarball's hash so a reused warm VM never
    serves a stale tree from an earlier driver run.
    """
    import hashlib

    digest = hashlib.md5(src_targz or b"local").hexdigest()[:10]
    root = f"/tmp/xql_geo_src_{digest}"
    marker = os.path.join(root, "benchmarks", "geospatial", "_engines.py")
    if src_targz is not None and not os.path.exists(marker):
        os.makedirs(root, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(src_targz), mode="r:gz") as tf:
            tf.extractall(root)  # noqa: S202 — our own tarball
    return root, os.path.join(root, "benchmarks", "geospatial")


def run_case_cell(
    case: str,
    engine: str,
    reps: int,
    src_targz: bytes | None = None,
    rep_timeout: float = 600.0,
) -> dict:
    """One (case, engine) cell: ``reps`` fresh-process cold runs."""
    result = {"case": case, "engine": engine, "status": "ok", "reps": []}
    try:
        src_root, geo_dir = _install_src(src_targz)
        env = dict(
            os.environ,
            GEOBENCH_ENGINE=engine,
            GEOBENCH_PROFILE="1",
            GEOBENCH_WARMUP="0",
            GEOBENCH_REPS="1",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=src_root,
        )
        rows: list[dict] = []
        for rep in range(1, reps + 1):
            with tempfile.NamedTemporaryFile(suffix=".csv") as csv_file:
                env["GEOBENCH_CSV"] = csv_file.name
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(
                        [sys.executable, f"{case}.py"],
                        cwd=geo_dir,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=rep_timeout,
                    )
                except subprocess.TimeoutExpired:
                    result["reps"].append({"rep": rep, "status": "timeout"})
                    result["status"] = "timeout"
                    break
                wall = round(time.perf_counter() - t0, 3)
                out = proc.stdout
                if "SKIPPED" in out:
                    reason = next(
                        (
                            line.split("SKIPPED:", 1)[1].strip()
                            for line in out.splitlines()
                            if "SKIPPED:" in line
                        ),
                        "skipped",
                    )
                    result.update(status="skip", reason=reason[:300])
                    break
                if proc.returncode != 0:
                    result.update(
                        status="error",
                        error=(proc.stderr.strip() or out.strip())[-600:],
                    )
                    break
                flavor = next(
                    (
                        line.split("engine:", 1)[1].strip()
                        for line in out.splitlines()
                        if "engine:" in line
                    ),
                    engine,
                )
                result["flavor"] = flavor
                with open(csv_file.name) as fh:
                    for row in csv.DictReader(fh):
                        row["rep"] = rep
                        rows.append(row)
                result["reps"].append(
                    {"rep": rep, "status": "ok", "wall_s": wall}
                )
                print(f"[vm] {case} x {engine}: rep {rep} {wall}s", flush=True)
        steps: dict[str, dict] = {}
        for row in rows:
            step = steps.setdefault(
                row["step"], {"times_s": [], "peak_mb": 0.0}
            )
            step["times_s"].append(float(row["t_median_s"]))
            step["peak_mb"] = max(step["peak_mb"], float(row["peak_mb"]))
        for step in steps.values():
            times = step["times_s"]
            step["median_s"] = round(statistics.median(times), 3)
            step["min_s"] = round(min(times), 3)
            step["max_s"] = round(max(times), 3)
            step["n"] = len(times)
        result["steps"] = steps
        if result["status"] == "ok" and not steps:
            result.update(status="error", error="no perf rows produced")
    except Exception as exc:  # noqa: BLE001 — cell errors are data
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


def probe_environment(src_targz: bytes | None = None) -> dict:
    """Machine spec + package versions, gathered where the cells run."""
    import platform

    _install_src(src_targz)
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "node": platform.node(),
    }
    try:
        import psutil

        info["mem_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:  # noqa: BLE001
        pass
    versions = {}
    for pkg in ["duckdb", "polars", "datafusion", "pyarrow", "xarray"]:
        try:
            from importlib import metadata

            versions[pkg] = metadata.version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = "missing"
    return {"machine": info, "versions": versions}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

_PRINT_LOCK = threading.Lock()


def log(vm: str, msg: str) -> None:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    with _PRINT_LOCK:
        print(f"[{now}][{vm}] {msg}", flush=True)


def _pack_src() -> bytes:
    """gzip tar of xarray_sql + benchmarks/geospatial (pure Python)."""
    repo = Path(__file__).resolve().parents[2]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in ["xarray_sql", "benchmarks/geospatial"]:
            for path in sorted((repo / rel).rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                tf.add(path, arcname=str(path.relative_to(repo)))
    return buf.getvalue()


def _drive_vm(vm, cells, args, src, results, jsonl_lock):
    """Run every (case, engine) cell for one VM size, sequentially."""
    if vm == "local":
        remote_cell, remote_probe = run_case_cell, probe_environment
        submit = None
    else:
        import coiled

        deco = coiled.function(
            name=cluster_name(vm),
            vm_type=vm,
            region=REGION,
            keepalive="10m",
            idle_timeout="20 minutes",
            spot_policy="on-demand",
            package_sync_ignore=["xarray_sql", "xarray-sql"],
            environ={"PYTHONUNBUFFERED": "1"},
        )
        remote_cell, remote_probe = deco(run_case_cell), deco(probe_environment)
        submit = remote_cell.submit

    log(vm, "probing environment (provisions the VM on first call)...")
    meta = None
    for attempt in range(1, 4):
        try:
            meta = remote_probe(src)
            break
        except Exception as exc:  # noqa: BLE001 — transient control plane
            log(vm, f"probe attempt {attempt} failed: {exc}"[:200])
            time.sleep(30 * attempt)
    if meta is None:
        log(vm, "giving up: VM never came up")
        return
    log(vm, f"machine: {json.dumps(meta['machine'])}")
    total = len(cells)
    for k, (case, engine) in enumerate(cells, 1):
        tag = f"cell {k}/{total} {case} x {engine}"
        if case in NOT_PORTABLE and engine != "datafusion":
            rec = {
                "case": case,
                "engine": engine,
                "status": "n/a",
                "reason": NOT_PORTABLE[case],
            }
        else:
            log(vm, f"{tag}: submitted")
            t0 = time.monotonic()
            try:
                if submit is None:
                    rec = run_case_cell(case, engine, args.reps, src)
                else:
                    fut = submit(case, engine, args.reps, src)
                    rec = fut.result(timeout=args.cell_timeout)
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "case": case,
                    "engine": engine,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            rec["cell_wall_s"] = round(time.monotonic() - t0, 1)
        rec["vm"] = vm
        results.append(rec)
        with jsonl_lock, open(args.jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if rec["status"] == "ok":
            sql_step = next(
                (
                    s
                    for name, s in rec.get("steps", {}).items()
                    if name.startswith("SQL")
                ),
                None,
            )
            brief = (
                f"SQL median {sql_step['median_s']}s (n={sql_step['n']})"
                if sql_step
                else "ok"
            )
            log(vm, f"{tag}: ok {brief} [{rec.get('flavor', engine)}]")
        else:
            detail = rec.get("reason") or rec.get("error", "")
            log(vm, f"{tag}: {rec['status']} {detail[:200]}")
    results.append({"vm": vm, "case": "_meta", "engine": "", **meta})
    with jsonl_lock, open(args.jsonl, "a") as fh:
        fh.write(json.dumps(results[-1]) + "\n")
    if submit is not None:
        # Shut the VM down the moment its last cell finishes — don't
        # leave the teardown to keepalive expiry.
        try:
            remote_cell.cluster.shutdown()
            log(vm, "cluster shut down")
        except Exception as exc:  # noqa: BLE001 — teardown best-effort
            log(vm, f"cluster shutdown failed: {exc}"[:200])


def _markdown(results: list[dict]) -> str:
    """One case x engine table per VM (SQL median s; reference column)."""
    out = []
    vms = list(dict.fromkeys(r["vm"] for r in results))
    for vm in vms:
        rows = [r for r in results if r["vm"] == vm and r["case"] != "_meta"]
        if not rows:
            continue
        cases = list(dict.fromkeys(r["case"] for r in rows))
        out.append(f"\n### {vm}\n")
        out.append("| Case | " + " | ".join(ENGINES) + " | xarray reference |")
        out.append("|---|" + "---|" * (len(ENGINES) + 1))
        by = {(r["case"], r["engine"]): r for r in rows}
        for case in cases:
            cells = []
            for engine in ENGINES:
                r = by.get((case, engine))
                if r is None:
                    cells.append("-")
                elif r["status"] != "ok":
                    detail = r.get("reason") or r.get("error", "")
                    cells.append(f"{r['status']}: {detail[:40]}")
                else:
                    s = next(
                        (
                            v
                            for k, v in r["steps"].items()
                            if k.startswith("SQL")
                        ),
                        None,
                    )
                    cells.append(
                        f"{s['median_s']:.3f}s (n={s['n']}, "
                        f"{s['peak_mb']:.0f} MB)"
                        if s
                        else "?"
                    )
            df_run = by.get((case, "datafusion"), {})
            ref = (df_run.get("steps") or {}).get("xarray reference")
            ref_text = (
                f"{ref['median_s']:.3f}s ({ref['peak_mb']:.0f} MB)"
                if ref
                else "-"
            )
            out.append(f"| {case} | " + " | ".join(cells) + f" | {ref_text} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--vms", default=",".join(VM_SIZES))
    ap.add_argument("--cell-timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="engine_suite_results.json")
    ap.add_argument("--jsonl", default="engine_suite_results.jsonl")
    args = ap.parse_args()

    cases = [c for c in args.cases.split(",") if c]
    engines = [e for e in args.engines.split(",") if e]
    vms = ["local"] if args.local else [v for v in args.vms.split(",") if v]
    cells = [(c, e) for c in cases for e in engines]
    log("plan", f"{len(vms)} VMs x {len(cells)} cells, reps={args.reps}")
    for c, e in cells:
        note = (
            f"  [{NOT_PORTABLE[c]}]"
            if c in NOT_PORTABLE and e != "datafusion"
            else ""
        )
        log("plan", f"  {c} x {e}{note}")
    src = _pack_src()
    log("plan", f"packed source: {len(src) / 1024:.0f} KiB")

    open(args.jsonl, "w").close()
    results: list[dict] = []
    jsonl_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_drive_vm,
            args=(vm, cells, args, src, results, jsonl_lock),
            name=vm,
        )
        for vm in vms
    ]
    for i, t in enumerate(threads):
        if i:  # stagger: concurrent package-sync scans trip the server
            time.sleep(20)
        t.start()
    for t in threads:
        t.join()

    payload = {
        "meta": {
            "region": REGION,
            "reps": args.reps,
            "protocol": "fresh process per rep, no warmup, cold reads",
        },
        "results": results,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    md = _markdown([r for r in results if r.get("case")])
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w") as fh:
        fh.write(md + "\n")
    print(md)
    log("done", f"wrote {args.out}, {md_path}, {args.jsonl}")


if __name__ == "__main__":
    main()
