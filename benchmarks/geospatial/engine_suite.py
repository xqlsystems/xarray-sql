"""The geospatial suite across engines and VM sizes, via Coiled Functions.

Runs the nine geospatial cases (``01_ndvi`` … ``09_warp``) under every
SQL engine the suite supports — DataFusion over the native table
provider (``datafusion``, the original path; requires the compiled
native module), DataFusion over the pure-Python pyarrow dataset
(``datafusion-arrow``), DuckDB, and Polars, selected per process
through ``GEOBENCH_ENGINE`` and the ``_engines`` facade — on one reused
Coiled VM per machine size, driven in parallel across sizes.

``datafusion`` cells need the compiled native module; the first such
cell on a VM provisions it (see :func:`_ensure_native`) and records the
outcome under ``native`` in its result, so a failed build surfaces as
that cell's error rather than a VM startup failure. The driver's own
build is copied in when it imports on that platform; otherwise rustup
(minimal profile) is installed and the shipped crate is built via the
project's maturin build backend, once per source digest.

The measurement protocol is exactly ``run_perf.sh``'s: every repetition
is a **fresh process** with no warm-up (``GEOBENCH_PROFILE=1
GEOBENCH_WARMUP=0 GEOBENCH_REPS=1``), so the SQL side and the xarray
reference each pay a cold read on every rep, and each case's own
correctness assertion (SQL answer == array reference) must pass for the
timing to count. The xarray-reference timings are engine-independent;
the tables report the reference column from the DataFusion runs.

Coverage notes, recorded rather than hidden: cases 07 and 09 build
DataFusion scalar UDFs on ``xql.XarrayContext`` directly, so every
engine except ``datafusion`` is marked n/a; case 08 reads
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
ENGINES = ["datafusion", "datafusion-arrow", "duckdb", "polars"]
# Cases whose SQL builds DataFusion scalar UDFs on XarrayContext directly
# (07, and the UDF half of 09): they run only under ``datafusion``. Case
# 08 is portable SQL but Earth-Engine-gated, so it stays on the original
# context.
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


def _run_logged(cmd, **kwargs) -> None:
    """subprocess.run(check=True) that surfaces stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{cmd if isinstance(cmd, str) else ' '.join(cmd)} failed:\n"
            f"{proc.stderr[-800:]}"
        )


def _ensure_native(src_root: str) -> str:
    """Make ``xarray_sql._native`` importable from ``src_root``.

    Tries, in order: the module already present in the tree; the one
    installed in this interpreter's environment (copied in, when built
    for this platform); a from-source build of the shipped crate —
    rustup (minimal profile) plus ``pip wheel``, which drives the
    project's maturin build backend — installed over the pure-Python
    copy. The built module lands in ``src_root``, which is keyed by
    source digest, so a warm VM builds at most once per source state.

    Returns a status string for the run log.
    """
    import glob
    import importlib.util
    import shutil

    # cwd well inside the tree, so `-c` resolves xarray_sql only through
    # PYTHONPATH=src_root — the same view the case subprocesses get.
    geo_dir = os.path.join(src_root, "benchmarks", "geospatial")
    env = dict(os.environ, PYTHONPATH=src_root)

    def _importable() -> bool:
        return (
            subprocess.run(
                [sys.executable, "-c", "import xarray_sql._native"],
                env=env,
                cwd=geo_dir,
                capture_output=True,
            ).returncode
            == 0
        )

    if _importable():
        return "importable"

    try:
        spec = importlib.util.find_spec("xarray_sql._native")
    except ImportError:
        spec = None
    if spec is not None and spec.origin:
        shutil.copy2(
            spec.origin,
            os.path.join(src_root, "xarray_sql", os.path.basename(spec.origin)),
        )
        if _importable():
            return "copied from driver environment"

    t0 = time.monotonic()
    build_env = dict(env)
    cargo_bin = os.path.expanduser("~/.cargo/bin")
    build_env["PATH"] = cargo_bin + os.pathsep + build_env.get("PATH", "")
    if shutil.which("cargo", path=build_env["PATH"]) is None:
        _run_logged(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y --profile minimal --default-toolchain stable",
            shell=True,
            env=build_env,
        )
    wheel_dir = os.path.join(src_root, "wheelhouse")
    _run_logged(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "-w", wheel_dir, src_root],
        env=build_env,
    )
    wheel = sorted(glob.glob(os.path.join(wheel_dir, "xarray_sql-*.whl")))[-1]
    _run_logged(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade",
         "--target", src_root, wheel],
    )
    if not _importable():
        raise RuntimeError(f"built {wheel} but xarray_sql._native still fails")
    return f"built from source in {time.monotonic() - t0:.0f}s"


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
        if engine == "datafusion":
            try:
                result["native"] = _ensure_native(src_root)
            except Exception:
                result["native"] = "provisioning failed"
                raise
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
    """gzip tar of xarray_sql, benchmarks/geospatial, and the Rust crate.

    Byte-identical for identical file contents (gzip and tar metadata
    normalized): _install_src keys its extraction root — and therefore
    _ensure_native's build cache on a warm VM — on the digest of these
    bytes.
    """
    import gzip

    repo = Path(__file__).resolve().parents[2]

    def _normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mtime = 0
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    paths: list[Path] = []
    for rel in ["xarray_sql", "benchmarks/geospatial"]:
        paths += [
            p
            for p in sorted((repo / rel).rglob("*.py"))
            if "__pycache__" not in p.parts
        ]
    # The crate sources, so `datafusion` cells can build the native
    # module where it is not already importable (see _ensure_native).
    for rel in ["src", "Cargo.toml", "Cargo.lock", "pyproject.toml",
                "README.md"]:
        target = repo / rel
        paths += (
            [p for p in sorted(target.rglob("*")) if p.is_file()]
            if target.is_dir()
            else [target]
        )
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            for path in paths:
                tf.add(
                    path,
                    arcname=str(path.relative_to(repo)),
                    filter=_normalize,
                )
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
            if attempt < 3:
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
        if rec.get("native", "importable") != "importable":
            log(vm, f"{tag}: native module {rec['native']}")
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
    meta_rec = {"vm": vm, "case": "_meta", "engine": "", **meta}
    results.append(meta_rec)
    with jsonl_lock, open(args.jsonl, "a") as fh:
        fh.write(json.dumps(meta_rec) + "\n")
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
    # dict.fromkeys: duplicate VM names would share one cluster and defeat
    # the incomplete-run detection, which matches records by VM name.
    vms = (
        ["local"]
        if args.local
        else list(dict.fromkeys(v for v in args.vms.split(",") if v))
    )
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
    # A VM that never produced its _meta record never ran its cells;
    # exit nonzero so partial runs cannot pass for complete ones.
    incomplete = [
        vm
        for vm in vms
        if not any(
            r.get("vm") == vm and r.get("case") == "_meta" for r in results
        )
    ]
    if incomplete:
        log("done", f"incomplete run: no results from {', '.join(incomplete)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
