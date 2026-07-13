"""Tests for the engine-neutral pyarrow dataset view.

``xql.arrow_dataset`` returns a real ``pyarrow.dataset.Dataset``, so it
serves any consumer of that protocol — pyarrow itself and Polars are
exercised here; DuckDB has its own suite in ``test_duckdb_backend.py``.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import xarray as xr

import xarray_sql as xql


@pytest.fixture
def ds() -> xr.Dataset:
    np.random.seed(3)
    return xr.Dataset(
        {
            "temperature": (
                ["time", "lat"],
                20 + 5 * np.random.randn(20, 6),
            ),
            "humidity": (["time", "lat"], np.random.rand(20, 6)),
        },
        coords={
            "time": pd.date_range("2022-01-01", periods=20, freq="D"),
            "lat": np.linspace(-25.0, 25.0, 6),
        },
    ).chunk({"time": 5})


def test_to_table_projects_and_filters(ds):
    table = xql.arrow_dataset(ds).to_table(
        columns=["time", "temperature"],
        filter=pc.field("lat") > 0,
    )
    assert table.column_names == ["time", "temperature"]
    assert table.num_rows == 20 * 3  # lat > 0 keeps 3 of 6 latitudes


def test_count_rows_and_head(ds):
    dataset = xql.arrow_dataset(ds)
    assert dataset.count_rows() == 20 * 6
    assert dataset.head(7).num_rows == 7


def test_get_fragments_prunes_and_scans(ds):
    dataset = xql.arrow_dataset(ds)
    assert len(dataset.get_fragments()) == 4  # time chunked by 5

    # A time predicate covering the first chunk keeps one fragment.
    early = pc.field("time") < pa.scalar(
        pd.Timestamp("2022-01-06"), type=pa.timestamp("ns")
    )
    kept = dataset.get_fragments(filter=early)
    assert len(kept) == 1
    assert kept[0].to_table().num_rows == 5 * 6

    # An unsatisfiable predicate prunes everything.
    assert dataset.get_fragments(filter=pc.field("lat") > 100) == []


def test_datafusion_register_dataset_round_trips(ds):
    from datafusion import SessionContext

    ctx = SessionContext()
    ctx.register_dataset("t", xql.arrow_dataset(ds))
    out = ctx.sql(
        "SELECT time, AVG(temperature) AS temperature FROM t "
        "WHERE lat > 0 GROUP BY time ORDER BY time"
    ).to_pandas()
    expected = ds.temperature.sel(lat=ds.lat[ds.lat > 0]).mean("lat").compute()
    np.testing.assert_allclose(out["temperature"].values, expected.values)


def test_dask_from_map_over_fragments(ds):
    dd = pytest.importorskip("dask.dataframe")

    frags = xql.arrow_dataset(ds).get_fragments()
    ddf = dd.from_map(lambda f: f.to_table().to_pandas(), frags)
    assert len(ddf.compute()) == 20 * 6


def test_polars_scan_pushdown_round_trip(ds):
    pl = pytest.importorskip("polars")

    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))
    out = (
        lf.filter(pl.col("lat") > 0)
        .group_by("time")
        .agg(pl.col("temperature").mean())
        .sort("time")
        .collect()
    )
    expected = (
        ds.temperature.sel(lat=ds.lat[ds.lat > 0]).mean("lat").compute().values
    )
    np.testing.assert_allclose(out["temperature"].to_numpy(), expected)


def test_polars_result_round_trips_to_xarray(ds):
    pl = pytest.importorskip("polars")

    lf = pl.scan_pyarrow_dataset(xql.arrow_dataset(ds))
    frame = (
        lf.group_by("time")
        .agg(pl.col("temperature").mean().alias("temperature"))
        .sort("time")
        .collect()
    )
    # Polars DataFrames export Arrow via the PyCapsule protocol, so the
    # engine-agnostic round-trip works unchanged.
    out = xql.to_dataset(frame, template=ds)
    assert list(out.dims) == ["time"]
    assert out.sizes["time"] == 20
