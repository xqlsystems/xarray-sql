"""Benchmark: DuckDB re-scannable stream vs pushdown dataset vs ceilings.

Usage: python benchmarks/duckdb_pushdown.py  (needs duckdb installed)
"""

import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import xarray as xr

import xarray_sql as xql
from xarray_sql.backends.duckdb import XarrayArrowStream

np.random.seed(0)
N_TIME, N_LAT, N_LON = 1000, 100, 100  # 10M rows
ds = xr.Dataset(
    {
        "temperature": (
            ["time", "lat", "lon"],
            np.random.rand(N_TIME, N_LAT, N_LON),
        ),
        "humidity": (
            ["time", "lat", "lon"],
            np.random.rand(N_TIME, N_LAT, N_LON),
        ),
    },
    coords={
        "time": pd.date_range("2020-01-01", periods=N_TIME, freq="h"),
        "lat": np.linspace(-90, 90, N_LAT),
        "lon": np.linspace(-180, 180, N_LON),
    },
).chunk({"time": 50})  # 20 partitions

con = duckdb.connect()

QUERIES = {
    "full AVG scan": "SELECT AVG(temperature) FROM {t}",
    "1pct time filter": (
        "SELECT AVG(temperature) FROM {t} WHERE time < '2020-01-01 10:00:00'"
    ),
    "bbox filter": (
        "SELECT AVG(temperature) FROM {t} "
        "WHERE lat BETWEEN 0 AND 10 AND lon BETWEEN 0 AND 20"
    ),
    "projection (1 of 2 vars)": "SELECT AVG(humidity) FROM {t}",
    "count only": "SELECT COUNT(*) FROM {t}",
}


def bench(table, label, n=3):
    print(f"\n== {label} ==")
    for qname, q in QUERIES.items():
        sql = q.format(t=table)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = con.sql(sql).fetchall()
            times.append(time.perf_counter() - t0)
        print(f"  {qname:28s} {min(times):8.3f}s   -> {r[0][0]:.6g}")


# re-scannable stream, registered via the stream wrapper explicitly:
# DuckDB scans every row, no filter/projection pushdown
con.register("t_stream", XarrayArrowStream(ds))
bench("t_stream", "stream (no pushdown)")

# default register(): the pushdown pyarrow-dataset path
xql.register(con, "t_pushdown", ds)
bench("t_pushdown", "register() [pushdown]")

# ceiling: materialized pa.Table via pyarrow.dataset
table = xql.read_xarray(ds).read_all()
con.register("t_ceiling", pads.dataset(table))
bench("t_ceiling", "ceiling: in-memory pyarrow.dataset")

# reference: DataFusion engine on the same dataset
ctx = xql.XarrayContext()
xql.register(ctx, "t_df", ds)
print("\n== DataFusion reference ==")
for qname, q in QUERIES.items():
    sql = q.format(t="t_df")
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        r = ctx.sql(sql).to_pandas()
        times.append(time.perf_counter() - t0)
    print(f"  {qname:28s} {min(times):8.3f}s   -> {float(r.iloc[0, 0]):.6g}")
