"""Per-engine performance matrix on public ARCO-ERA5, via Coiled Functions.

Runs the same bounded workloads through every SQL engine xarray-sql
serves via the pyarrow dataset protocol -- DuckDB, Polars
(``scan_pyarrow_dataset``), and DataFusion
(``ctx.register_dataset(xql.arrow_dataset(ds))``) -- plus a native
xarray+dask baseline, all against the anonymous-access ARCO-ERA5 bucket:

    gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3

Workloads (2m_temperature, hourly, 721x1440 global grid):

* ``day_bbox``    -- 1 day, Iberia bbox, AVG (31,680 rows).
* ``week_globe``  -- 7 days, full globe, AVG (~174M rows).
* ``month_globe`` -- 31 days, full globe, AVG (~772M rows).
* ``clim_region`` -- 2 months over Iberia, monthly climatology per grid
  cell (``GROUP BY latitude, longitude, month`` vs xarray
  ``groupby("time.month")``).
* ``count_jan``   -- ``count(*)`` over January (xarray side is pure
  metadata arithmetic, reported as such).

Every SQL engine reads through the same pushdown dataset,
``xql.arrow_dataset(ds, {"time": 1}, prefetch=16,
coalesce_rows=8_000_000)`` -- a pure-Python path that needs no compiled
extension (the driver ships the ``xarray_sql`` source tree to the VM as
a tarball, so no Rust build happens anywhere). The xarray baseline
opens the store with dask ``chunks={"time": 24}``.

Execution model: one ``@coiled.function`` invocation per (workload,
engine) cell, all sharing one named VM (``name=`` keeps the VM warm
across invocations; ``keepalive`` tears it down after the run). Each
cell returns a plain dict -- results come back as Python objects, and
every completed cell is appended to a local ``results.jsonl``
immediately, so a crash loses nothing. Rep policy: 1 warm-up + a first
timed rep; if that rep is faster than ``--slow-rep`` (15s) two more
timed reps follow (median of 3), otherwise the single rep stands.
Errors are recorded per cell and the matrix continues -- an engine can
fail, never silently vanish.

Usage::

    python benchmarks/engine_matrix.py --local --smoke  # in-process check
    python benchmarks/engine_matrix.py                  # matrix on Coiled

Writes JSON + a markdown table next to ``--out`` and streams
timestamped progress lines throughout.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import statistics
import sys
import tarfile
import threading
import time
from pathlib import Path

URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VAR = "2m_temperature"
# lat_s, lat_n, lon_w, lon_e (ERA5 longitude is 0-360E; Iberia box).
IBERIA = (36.0, 44.0, 350.0, 360.0)

ENGINES = ["duckdb", "polars", "datafusion", "xarray"]

PACKAGES = [
    "duckdb",
    "polars",
    "pyarrow",
    "datafusion",
    "xarray",
    "zarr",
    "gcsfs",
    "dask",
    "numpy",
    "pandas",
]

VM_TYPE = "n2-standard-8"
REGION = "us-central1"
CLUSTER_NAME = "xql-engine-matrix"
SRC_ROOT = "/tmp/xql_src"


def workloads(smoke: bool) -> dict[str, tuple[str, str, tuple | None, str]]:
    """(t0, t1_exclusive, bbox, kind) per workload; tiny windows in smoke."""
    if smoke:
        return {
            "day_bbox": ("2020-01-01", "2020-01-01 03:00", IBERIA, "mean"),
            "week_globe": ("2020-01-03", "2020-01-03 03:00", None, "mean"),
            "month_globe": ("2020-01-01", "2020-01-01 06:00", None, "mean"),
            "clim_region": ("2020-06-01", "2020-06-03", IBERIA, "clim"),
            "count_jan": ("2020-01-01", "2020-01-02", None, "count"),
        }
    return {
        "day_bbox": ("2020-01-01", "2020-01-02", IBERIA, "mean"),
        "week_globe": ("2020-01-03", "2020-01-10", None, "mean"),
        "month_globe": ("2020-01-01", "2020-02-01", None, "mean"),
        "clim_region": ("2020-06-01", "2020-08-01", IBERIA, "clim"),
        "count_jan": ("2020-01-01", "2020-02-01", None, "count"),
    }


# --------------------------------------------------------------------------
# Cell machinery — executes inside the coiled function (or locally)
# --------------------------------------------------------------------------


def _install_src(src_targz: bytes | None) -> None:
    """Unpack the shipped xarray_sql source once and put it on sys.path."""
    marker = os.path.join(SRC_ROOT, "xarray_sql", "__init__.py")
    if src_targz is not None and not os.path.exists(marker):
        os.makedirs(SRC_ROOT, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(src_targz), mode="r:gz") as tf:
            tf.extractall(SRC_ROOT)  # noqa: S202 — our own tarball
    if os.path.exists(marker) and SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)


def _open_ds():
    import xarray as xr

    return xr.open_zarr(
        URL, chunks=None, storage_options={"token": "anon"}, consolidated=True
    )[[VAR]]


def _pushdown(ds):
    import xarray_sql as xql

    return xql.arrow_dataset(
        ds, {"time": 1}, prefetch=16, coalesce_rows=8_000_000
    )


def _expected_rows(ds, t0, t1, bbox) -> int:
    import pandas as pd

    hours = int((pd.Timestamp(t1) - pd.Timestamp(t0)) / pd.Timedelta("1h"))
    lat, lon = ds.latitude.values, ds.longitude.values
    if bbox:
        s, n, w, e = bbox
        npts = int(((lat >= s) & (lat <= n)).sum()) * int(
            ((lon >= w) & (lon <= e)).sum()
        )
    else:
        npts = lat.size * lon.size
    return hours * npts


def _ts(t: str) -> str:
    """Full 'YYYY-MM-DD HH:MM:SS' literal (DataFusion rejects bare HH:MM)."""
    import pandas as pd

    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _sql(kind: str, t0: str, t1: str, bbox) -> str:
    where = f"time >= TIMESTAMP '{_ts(t0)}' AND time < TIMESTAMP '{_ts(t1)}'"
    if bbox:
        s, n, w, e = bbox
        where += (
            f" AND latitude BETWEEN {s} AND {n}"
            f" AND longitude BETWEEN {w} AND {e}"
        )
    if kind == "mean":
        return f'SELECT avg("{VAR}") - 273.15 AS mean_c FROM era5 WHERE {where}'
    if kind == "count":
        return f"SELECT count(*) AS n FROM era5 WHERE {where}"
    if kind == "clim":
        return (
            "SELECT latitude, longitude,"
            " date_part('month', time) AS month,"
            f' avg("{VAR}") - 273.15 AS clim_c'
            f" FROM era5 WHERE {where}"
            " GROUP BY latitude, longitude, date_part('month', time)"
        )
    raise ValueError(kind)


def _build_duckdb(kind, t0, t1, bbox):
    import duckdb

    con = duckdb.connect()
    con.register("era5", _pushdown(_open_ds()))
    sql = _sql(kind, t0, t1, bbox)

    def run():
        return con.execute(sql).fetchall()

    def summarize(rows):
        if kind == "clim":
            vals = [r[3] for r in rows]
            return {"cells": len(rows), "mean_c": _fmean(vals)}
        return {"value": rows[0][0]}

    return run, summarize


def _build_polars(kind, t0, t1, bbox):
    import pandas as pd
    import polars as pl

    lf = pl.scan_pyarrow_dataset(_pushdown(_open_ds()))
    pred = (pl.col("time") >= pd.Timestamp(t0)) & (
        pl.col("time") < pd.Timestamp(t1)
    )
    if bbox:
        s, n, w, e = bbox
        pred = (
            pred
            & pl.col("latitude").is_between(s, n)
            & pl.col("longitude").is_between(w, e)
        )
    lf = lf.filter(pred)
    if kind == "mean":
        q = lf.select((pl.col(VAR).mean() - 273.15).alias("mean_c"))
    elif kind == "count":
        q = lf.select(pl.len().alias("n"))
    else:
        q = lf.group_by(
            "latitude",
            "longitude",
            pl.col("time").dt.month().alias("month"),
        ).agg((pl.col(VAR).mean() - 273.15).alias("clim_c"))

    def run():
        return q.collect()

    def summarize(df):
        if kind == "clim":
            return {
                "cells": df.height,
                "mean_c": float(df["clim_c"].mean()),
            }
        return {"value": _jsonable(df.row(0)[0])}

    return run, summarize


def _build_datafusion(kind, t0, t1, bbox):
    from datafusion import SessionContext

    ctx = SessionContext()
    ctx.register_dataset("era5", _pushdown(_open_ds()))
    sql = _sql(kind, t0, t1, bbox)

    def run():
        import pyarrow as pa

        return pa.Table.from_batches(ctx.sql(sql).collect())

    def summarize(table):
        if kind == "clim":
            vals = table.column("clim_c").to_pylist()
            return {"cells": table.num_rows, "mean_c": _fmean(vals)}
        return {"value": _jsonable(table.column(0)[0].as_py())}

    return run, summarize


def _build_xarray(kind, t0, t1, bbox):
    import pandas as pd
    import xarray as xr

    da = xr.open_zarr(
        URL,
        chunks={"time": 24},
        storage_options={"token": "anon"},
        consolidated=True,
    )[VAR]
    # xarray slices are label-inclusive; the window is [t0, t1), so end
    # one hour (one sample) before t1. ERA5 latitude descends.
    end = pd.Timestamp(t1) - pd.Timedelta("1h")
    sel = da.sel(time=slice(pd.Timestamp(t0), end))
    if bbox:
        s, n, w, e = bbox
        sel = sel.sel(latitude=slice(n, s), longitude=slice(w, e))

    if kind == "mean":

        def run():
            return float(sel.mean().compute()) - 273.15

        def summarize(v):
            return {"value": v}
    elif kind == "count":
        # No data read: the row count of a selection is coordinate
        # arithmetic in the array paradigm.
        def run():
            return int(sel.size)

        def summarize(v):
            return {"value": v, "note": "metadata arithmetic, no data read"}
    else:

        def run():
            return (sel.groupby("time.month").mean("time") - 273.15).compute()

        def summarize(out):
            return {"cells": int(out.size), "mean_c": float(out.mean())}

    return run, summarize


_BUILDERS = {
    "duckdb": _build_duckdb,
    "polars": _build_polars,
    "datafusion": _build_datafusion,
    "xarray": _build_xarray,
}


def _fmean(vals):
    vals = [v for v in vals if v is not None]
    return float(statistics.fmean(vals)) if vals else None


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


class _RssSampler:
    """Peak process RSS, sampled by a daemon thread."""

    def __init__(self):
        import psutil

        self._proc = psutil.Process()
        self._stop = threading.Event()
        self.base = self._proc.memory_info().rss
        self.peak = self.base
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(0.05):
            self.peak = max(self.peak, self._proc.memory_info().rss)

    def stop(self) -> tuple[float, float]:
        self._stop.set()
        self._thread.join(timeout=1)
        self.peak = max(self.peak, self._proc.memory_info().rss)
        return self.base / 2**20, (self.peak - self.base) / 2**20


def run_cell(
    workload: str,
    engine: str,
    smoke: bool,
    src_targz: bytes | None = None,
    slow_rep_s: float = 15.0,
) -> dict:
    """One (workload, engine) cell; returns a plain dict, never raises.

    Rep policy: 1 warm-up + first timed rep; if the first rep beats
    ``slow_rep_s`` two more timed reps follow (median of 3), otherwise
    the single rep stands. ``--smoke``: no warm-up, single rep.
    """
    import platform

    def vlog(msg: str) -> None:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[vm {now}] {workload} x {engine}: {msg}", flush=True)

    t0s, t1s, bbox, kind = workloads(smoke)[workload]
    result = {
        "workload": workload,
        "engine": engine,
        "kind": kind,
        "window": [t0s, t1s],
        "bbox": bbox,
        "smoke": smoke,
        "host": platform.node(),
    }
    try:
        _install_src(src_targz)
        vlog("setup")
        t_setup = time.perf_counter()
        run, summarize = _BUILDERS[engine](kind, t0s, t1s, bbox)
        result["rows"] = _expected_rows(_open_ds(), t0s, t1s, bbox)
        result["setup_s"] = round(time.perf_counter() - t_setup, 3)

        sampler = _RssSampler()
        times: list[float] = []
        summary = None
        if not smoke:
            t = time.perf_counter()
            summary = summarize(run())
            result["warmup_s"] = round(time.perf_counter() - t, 3)
            vlog(f"warm-up {result['warmup_s']:.2f}s")
        n_reps = 1
        for i in range(3):
            if i >= n_reps:
                break
            t = time.perf_counter()
            summary = summarize(run())
            times.append(round(time.perf_counter() - t, 3))
            vlog(f"rep {i + 1} {times[-1]:.2f}s")
            if i == 0 and not smoke and times[0] < slow_rep_s:
                n_reps = 3
        rss_base_mb, rss_delta_mb = sampler.stop()
        result.update(
            status="ok",
            median_s=round(statistics.median(times), 3),
            times_s=times,
            n=len(times),
            rss_base_mb=round(rss_base_mb, 1),
            peak_rss_delta_mb=round(rss_delta_mb, 1),
            result=summary,
        )
        vlog(f"done median={result['median_s']}s n={len(times)}")
    except Exception as exc:  # noqa: BLE001 — cell errors are data
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        vlog(f"ERROR {result['error']}")
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
    for pkg in PACKAGES:
        try:
            from importlib import metadata

            versions[pkg] = metadata.version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = "missing"
    try:
        import xarray_sql

        versions["xarray_sql"] = getattr(
            xarray_sql, "__version__", "source tree"
        )
    except Exception as exc:  # noqa: BLE001
        versions["xarray_sql"] = f"import failed: {exc}"
    return {"machine": info, "versions": versions}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _pack_src() -> bytes:
    """gzip tar of the pure-Python xarray_sql package next to this file."""
    root = Path(__file__).resolve().parents[1] / "xarray_sql"
    if not (root / "__init__.py").exists():
        raise SystemExit(f"xarray_sql source not found at {root}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in sorted(root.rglob("*.py")):
            arcname = Path("xarray_sql") / path.relative_to(root)
            tf.add(path, arcname=str(arcname))
    return buf.getvalue()


def _cell_text(r: dict) -> str:
    if r["status"] == "timeout":
        return "timeout"
    if r["status"] == "error":
        return "error"
    note = r.get("result") or {}
    star = "*" if note.get("note") else ""
    return (
        f"{r['median_s']:.2f}s{star} (n={r['n']}, "
        f"ΔRSS {r['peak_rss_delta_mb']:.0f} MB)"
    )


def _markdown(results: list[dict], meta: dict) -> str:
    names = list(dict.fromkeys(r["workload"] for r in results))
    engines = list(dict.fromkeys(r["engine"] for r in results))
    by = {(r["workload"], r["engine"]): r for r in results}
    lines = [
        "| workload | rows | " + " | ".join(engines) + " |",
        "|---|--:|" + "---|" * len(engines),
    ]
    for w in names:
        rows = next(
            (
                by[(w, e)].get("rows")
                for e in engines
                if (w, e) in by and by[(w, e)].get("rows")
            ),
            None,
        )
        cells = [
            _cell_text(by[(w, e)]) if (w, e) in by else "-" for e in engines
        ]
        rows_text = f"{rows:,}" if rows else "?"
        lines.append(f"| {w} | {rows_text} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Median of 3 reps after 1 warm-up (single rep when the first rep "
        "exceeds the slow-rep threshold); ΔRSS is peak process RSS above "
        "the pre-cell baseline. `*` = metadata-only (no data read)."
    )
    lines.append(f"\nMachine: `{json.dumps(meta.get('machine', {}))}`")
    lines.append(f"\nVersions: `{json.dumps(meta.get('versions', {}))}`")
    return "\n".join(lines)


def _report_cell(rec: dict, k: int, total: int, wall: float) -> None:
    tag = f"cell {k}/{total} {rec['workload']} x {rec['engine']}"
    if rec["status"] == "ok":
        for i, t in enumerate(rec.get("times_s", []), 1):
            log(f"{tag}: rep {i} {t:.2f}s")
        log(
            f"{tag}: ok median={rec['median_s']}s n={rec['n']} "
            f"warmup={rec.get('warmup_s', '-')}s "
            f"dRSS={rec.get('peak_rss_delta_mb')}MB ({wall:.1f}s wall)"
        )
    else:
        log(f"{tag}: {rec['status']} {rec.get('error', '')[:300]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny windows, 1 rep")
    ap.add_argument(
        "--local", action="store_true", help="run in-process, no Coiled"
    )
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--slow-rep", type=float, default=15.0)
    ap.add_argument("--out", default="engine_matrix_results.json")
    ap.add_argument("--jsonl", default="engine_matrix_results.jsonl")
    ap.add_argument("--engines", default=",".join(ENGINES))
    ap.add_argument("--workloads", default="")
    ap.add_argument("--keepalive", default="10m")
    args = ap.parse_args()

    engines = [e for e in args.engines.split(",") if e]
    names = [w for w in args.workloads.split(",") if w] or list(
        workloads(args.smoke)
    )
    cells = [(w, e) for w in names for e in engines]
    total = len(cells)
    log(f"plan: {total} cells (smoke={args.smoke})")
    for k, (w, e) in enumerate(cells, 1):
        log(f"  {k}/{total}: {w} x {e}")

    src = _pack_src()
    log(f"packed xarray_sql source: {len(src) / 1024:.0f} KiB")

    if args.local:
        remote_cell, remote_probe, cluster = run_cell, probe_environment, None
    else:
        import coiled

        deco = coiled.function(
            name=CLUSTER_NAME,
            vm_type=VM_TYPE,
            region=REGION,
            keepalive=args.keepalive,
            idle_timeout="20 minutes",
            spot_policy="on-demand",
            package_sync_ignore=["xarray_sql", "xarray-sql"],
            environ={"PYTHONUNBUFFERED": "1"},
        )
        remote_cell, remote_probe = deco(run_cell), deco(probe_environment)
        cluster = remote_cell

    log("probing benchmark environment (provisions the VM on first call)...")
    meta_env = remote_probe(src)
    log(f"machine: {json.dumps(meta_env['machine'])}")
    log(f"versions: {json.dumps(meta_env['versions'])}")

    open(args.jsonl, "w").close()  # fresh partials file
    results: list[dict] = []
    for k, (w, e) in enumerate(cells, 1):
        log(f"cell {k}/{total} {w} x {e}: submitted")
        t_cell = time.monotonic()
        try:
            if args.local:
                rec = run_cell(w, e, args.smoke, src, args.slow_rep)
            else:
                fut = remote_cell.submit(w, e, args.smoke, src, args.slow_rep)
                rec = fut.result(timeout=args.budget)
        except Exception as exc:  # noqa: BLE001 — timeout or transport
            rec = {
                "workload": w,
                "engine": e,
                "status": "timeout"
                if "imeout" in type(exc).__name__ + str(exc)
                else "error",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
            try:
                fut.cancel()
            except Exception:  # noqa: BLE001
                pass
        wall = time.monotonic() - t_cell
        rec["cell_wall_s"] = round(wall, 1)
        results.append(rec)
        with open(args.jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        _report_cell(rec, k, total, wall)

    missing = [
        (w, e)
        for w, e in cells
        if not any(r["workload"] == w and r["engine"] == e for r in results)
    ]
    assert not missing, f"cells dropped from results: {missing}"

    meta = {
        **meta_env,
        "smoke": args.smoke,
        "url": URL,
        "vm": {"vm_type": VM_TYPE, "region": REGION, "local": args.local},
        "pushdown": (
            "xql.arrow_dataset(ds, {'time': 1}, prefetch=16, "
            "coalesce_rows=8_000_000)"
        ),
        "xarray_baseline": "open_zarr(chunks={'time': 24}) + dask",
    }
    payload = {"meta": meta, "results": results}
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    md = _markdown(results, meta)
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w") as fh:
        fh.write(md + "\n")
    print("\n" + md)
    log(f"wrote {args.out}, {md_path}, {args.jsonl}")
    if cluster is not None:
        log(f"VM stays warm for --keepalive={args.keepalive}, then stops")


if __name__ == "__main__":
    main()
