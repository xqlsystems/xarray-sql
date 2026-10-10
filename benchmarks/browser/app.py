"""The Python half of the browser demo: open a remote Zarr, query it with SQL.

``worker.js`` loads this module into Pyodide and calls :func:`setup` once
and :func:`run` per query. Both return JSON strings so the worker can hand
results to the page without converting Python objects.
"""

from __future__ import annotations

import json
import math
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from chunk_cache import ChunkCache
from fetch_store import FetchStore

import xarray_sql as xql

GFS_URL = "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr"
GFS_VARIABLES = [
    "temperature_2m",
    "precipitation_surface",
    "wind_u_10m",
    "wind_v_10m",
]
# The store's inner chunks; one cached chunk is then one range request.
GFS_CHUNKS = {
    "init_time": 1,
    "lead_time": 105,
    "latitude": 121,
    "longitude": 121,
}

_state: dict[str, Any] = {}


def _ts(value: Any) -> str:
    return str(pd.Timestamp(value))[:19]


def _examples(ds: xr.Dataset) -> list[dict[str, str]]:
    latest = _ts(ds.init_time.values[-1])
    earliest = _ts(ds.init_time.values[0])
    return [
        {
            "title": "San Francisco: 2 m temperature for the next 3 days",
            "sql": f"""SELECT init_time + lead_time AS valid_time,
       round(avg(temperature_2m), 1) AS temperature_c
FROM gfs
WHERE init_time = TIMESTAMP '{latest}'
  AND lead_time <= INTERVAL 72 HOUR
  AND latitude BETWEEN 37.5 AND 38 AND longitude BETWEEN -122.75 AND -122.25
GROUP BY ALL ORDER BY valid_time""",
        },
        {
            "title": "Map: Europe 2 m temperature, 24 h ahead",
            "sql": f"""SELECT latitude, longitude, temperature_2m
FROM gfs
WHERE init_time = TIMESTAMP '{latest}'
  AND lead_time = INTERVAL 24 HOUR
  AND latitude BETWEEN 35 AND 70 AND longitude BETWEEN -10 AND 40""",
        },
        {
            "title": "Map: contiguous US wind speed, 12 h ahead",
            "sql": f"""SELECT latitude, longitude,
       round(sqrt(wind_u_10m ^ 2 + wind_v_10m ^ 2), 1) AS wind_speed
FROM gfs
WHERE init_time = TIMESTAMP '{latest}'
  AND lead_time = INTERVAL 12 HOUR
  AND latitude BETWEEN 24 AND 50 AND longitude BETWEEN -125 AND -66""",
        },
        {
            "title": "Wettest places in the contiguous US, next 24 h",
            "sql": f"""SELECT latitude, longitude,
       round(sum(precipitation_surface) * 3600, 1) AS approx_precip_mm
FROM gfs
WHERE init_time = TIMESTAMP '{latest}'
  AND lead_time BETWEEN INTERVAL 1 HOUR AND INTERVAL 24 HOUR
  AND latitude BETWEEN 24 AND 50 AND longitude BETWEEN -125 AND -66
GROUP BY ALL ORDER BY approx_precip_mm DESC LIMIT 10""",
        },
        {
            "title": "How the forecast for San Francisco changed across runs",
            "sql": f"""SELECT init_time,
       round(max(temperature_2m), 1) AS max_temperature_c
FROM gfs
WHERE init_time BETWEEN TIMESTAMP '{earliest}' AND TIMESTAMP '{latest}'
  AND init_time + lead_time BETWEEN TIMESTAMP '{latest}' + INTERVAL 24 HOUR
                                AND TIMESTAMP '{latest}' + INTERVAL 48 HOUR
  AND lead_time <= INTERVAL 72 HOUR
  AND latitude BETWEEN 37.5 AND 38 AND longitude BETWEEN -122.75 AND -122.25
GROUP BY ALL ORDER BY init_time""",
        },
    ]


async def setup(
    url: str = GFS_URL,
    variables: list[str] | None = None,
    runs: int = 4,
    max_cache_mib: int = 768,
) -> str:
    """Open the dataset and register it on a DuckDB connection as ``gfs``."""
    store = FetchStore(url)
    ds = xr.open_zarr(store, chunks=None)
    # DuckDB does not push INTERVAL predicates down to Arrow scans, so a
    # lead_time filter cannot prune chunks. Keeping only the first lead-time
    # chunk (forecast hours 0-104) halves what every query downloads.
    ds = ds[variables or GFS_VARIABLES].isel(
        init_time=slice(-runs, None),
        lead_time=slice(0, GFS_CHUNKS["lead_time"]),
    )
    cache = ChunkCache(ds, GFS_CHUNKS, max_bytes=max_cache_mib * 2**20)
    con = duckdb.connect()
    con.sql("SET enable_progress_bar = false")
    xql.register(con, "gfs", cache.dataset())
    _state.update(store=store, cache=cache, con=con)
    return json.dumps(
        {
            "url": url,
            "variables": {
                name: {
                    "dims": list(var.dims),
                    "units": var.attrs.get("units", ""),
                    "long_name": var.attrs.get("long_name", name),
                }
                for name, var in ds.data_vars.items()
            },
            "sizes": dict(ds.sizes),
            "init_times": [_ts(t) for t in ds.init_time.values],
            "examples": _examples(ds),
        }
    )


def _jsonable(column: pa.ChunkedArray) -> list[Any]:
    if pa.types.is_temporal(column.type):
        return [None if v is None else str(v) for v in column.to_pylist()]
    values = column.to_pylist()
    return [
        None if isinstance(v, float) and not math.isfinite(v) else v
        for v in values
    ]


def _grid(table: pa.Table) -> dict[str, Any] | None:
    """Columns for the page's map: latitude, longitude, and one number."""
    names = table.column_names
    if "latitude" not in names or "longitude" not in names:
        return None
    values = [
        n
        for n in names
        if n not in ("latitude", "longitude")
        and (
            pa.types.is_floating(table[n].type)
            or pa.types.is_integer(table[n].type)
        )
    ]
    if len(values) != 1 or table.num_rows < 4:
        return None
    value = table[values[0]].to_numpy().astype("float64")
    finite = np.isfinite(value)
    return {
        "name": values[0],
        "latitude": table["latitude"].to_numpy()[finite].tolist(),
        "longitude": table["longitude"].to_numpy()[finite].tolist(),
        "value": value[finite].tolist(),
    }


async def run(sql: str, max_rows: int = 500) -> str:
    """Run ``sql``, fetching whatever chunks it needs, and return the result."""
    store, cache, con = _state["store"], _state["cache"], _state["con"]
    requests, nbytes = store.requests, store.bytes_read
    table, stats = await cache.query(lambda: con.sql(sql).to_arrow_table())
    shown = table.slice(0, max_rows)
    return json.dumps(
        {
            "columns": shown.column_names,
            "rows": list(
                zip(*(_jsonable(shown[c]) for c in shown.column_names))
            ),
            "num_rows": table.num_rows,
            "grid": _grid(table),
            "stats": {
                "runs": stats.rounds,
                "chunks_fetched": stats.chunks_fetched,
                "requests": store.requests - requests,
                "mb_downloaded": round((store.bytes_read - nbytes) / 1e6, 1),
                "seconds_fetching": round(stats.seconds_fetching, 2),
                "seconds_querying": round(stats.seconds_querying, 2),
                "cached_chunks": len(cache),
            },
        }
    )
