"""SQL functionality tests for xarray-sql using pytest."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xarray_sql import XarrayContext
from xarray_sql.sql import _group_vars_by_dims


def test_sanity(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    result = ctx.sql(
        'SELECT "lat", "lon", "time", "air" FROM "air" LIMIT 100'
    ).to_pandas()
    assert len(result) > 0
    assert len(result) <= 1320
    assert all(col in result.columns for col in ["lat", "lon", "time", "air"])


def test_aggregation_small(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    query = """
        SELECT lat, lon, SUM(air) AS air_total
        FROM air
        GROUP BY lat, lon
    """
    result = ctx.sql(query).to_pandas()
    expected_rows = (
        air_dataset_small.sizes["lat"] * air_dataset_small.sizes["lon"]
    )
    assert len(result) == expected_rows


def test_aggregation_large(air_dataset_large):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_large)
    query = """
        SELECT lat, lon, AVG(air) AS air_avg
        FROM air
        GROUP BY lat, lon
    """
    result = ctx.sql(query).to_pandas()
    expected_rows = (
        air_dataset_large.sizes["lat"] * air_dataset_large.sizes["lon"]
    )
    assert len(result) == expected_rows


def test_basic_select_all(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    result = ctx.sql("SELECT * FROM air LIMIT 10").to_pandas()
    assert len(result) <= 10
    for col in ["lat", "lon", "time", "air"]:
        assert col in result.columns


def test_weather_queries(weather_dataset):
    ctx = XarrayContext()
    ctx.from_dataset("weather", weather_dataset)
    # Selecting specific columns
    result = ctx.sql(
        "SELECT lat, lon, temperature, precipitation FROM weather LIMIT 20"
    ).to_pandas()
    assert "temperature" in result.columns
    assert "precipitation" in result.columns
    # Filtering
    result = ctx.sql(
        "SELECT * FROM weather WHERE temperature > 10 LIMIT 50"
    ).to_pandas()
    assert len(result) > 0
    assert (result["temperature"] > 10).all()


def test_synthetic_aggregations(synthetic_dataset):
    ctx = XarrayContext()
    ctx.from_dataset("synthetic", synthetic_dataset)
    # COUNT aggregation
    result = ctx.sql(
        "SELECT COUNT(*) AS total_count FROM synthetic"
    ).to_pandas()
    assert result["total_count"].iloc[0] > 0
    # MIN, MAX, AVG
    query = """
        SELECT MIN(temperature) AS min_temp,
               MAX(temperature) AS max_temp,
               AVG(temperature) AS avg_temp
        FROM synthetic
    """
    result = ctx.sql(query).to_pandas()
    assert result["min_temp"].iloc[0] < result["max_temp"].iloc[0]
    assert (
        result["min_temp"].iloc[0]
        <= result["avg_temp"].iloc[0]
        <= result["max_temp"].iloc[0]
    )


def test_invalid_table_name(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(Exception):
        ctx.sql("SELECT * FROM nonexistent_table")


def test_invalid_column_name(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(Exception):
        ctx.sql("SELECT nonexistent_column FROM air")


def test_sql_syntax_error(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(Exception):
        ctx.sql("SELECT * FORM air")  # Typo: FORM instead of FROM
    with pytest.raises(Exception):
        ctx.sql("SELECT * FROM air WHERE")  # Incomplete WHERE


def test_cross_join(air_and_stations):
    air, stations = air_and_stations
    ctx = XarrayContext()
    ctx.from_dataset("air_data", air)
    ctx.from_dataset("stations", stations)
    result = ctx.sql(
        "SELECT COUNT(*) AS total FROM air_data CROSS JOIN stations"
    ).to_pandas()
    assert result["total"].iloc[0] > 0


def test_string_coordinates():
    """String-typed coordinates should not crash during registration."""
    ds = xr.Dataset(
        {"score": (["student", "subject"], np.random.rand(3, 2))},
        coords={
            "student": ["alice", "bob", "charlie"],
            "subject": ["math", "science"],
        },
    )
    ctx = XarrayContext()
    ctx.from_dataset("scores", ds.chunk({"student": 3, "subject": 2}))
    result = ctx.sql("SELECT * FROM scores").to_pandas()
    assert len(result) == 6
    assert "student" in result.columns
    assert "subject" in result.columns
    assert "score" in result.columns


class TestNanAsNull:
    """NaN in float columns should become Arrow nulls so SQL aggregates work."""

    @pytest.fixture
    def nan_ds(self):
        data = np.array(
            [[[1.0, 2.0], [np.nan, 4.0]], [[5.0, np.nan], [7.0, 8.0]]]
        )
        return xr.Dataset(
            {"temp": (["time", "x", "y"], data)},
            coords={
                "time": pd.date_range("2020-01-01", periods=2),
                "x": [0, 1],
                "y": [0, 1],
            },
        ).chunk({"time": 1})

    def test_nan_aggregates(self, nan_ds):
        ctx = XarrayContext()
        ctx.from_dataset("data", nan_ds)

        # Test multiple aggregates at once:
        # MAX/MIN/AVG should ignore NaN, COUNT(col) should exclude NaN,
        # and WHERE col IS NULL should match NaN.
        query = """
        SELECT
            MAX(temp) AS mx,
            MIN(temp) AS mn,
            AVG(temp) AS avg,
            COUNT(temp) AS cnt,
            COUNT(*) FILTER (WHERE temp IS NULL) AS null_cnt
        FROM data
    """
        result = ctx.sql(query).to_pandas().iloc[0]

        assert result["mx"] == 8.0
        assert result["mn"] == 1.0
        expected_avg = np.nanmean([1.0, 2.0, 4.0, 5.0, 7.0, 8.0])
        assert abs(result["avg"] - expected_avg) < 1e-6
        assert result["cnt"] == 6
        assert result["null_cnt"] == 2


class TestCftimeGregorianLike:
    """Tests for Gregorian-like cftime calendars (noleap, standard, etc.).

    These use pa.timestamp('us') and support string-based SQL filters.
    """

    def test_noleap_dataset_registers(self, rasm_ds):
        """A noleap dataset should register without errors."""
        ctx = XarrayContext()
        ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
        result = ctx.sql("SELECT COUNT(*) AS cnt FROM rasm").to_pandas()
        assert result["cnt"].iloc[0] > 0

    def test_select_time_column(self, rasm_ds):
        """Querying the time column should return valid timestamps."""
        ctx = XarrayContext()
        ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
        result = ctx.sql(
            "SELECT DISTINCT time FROM rasm ORDER BY time LIMIT 5"
        ).to_pandas()
        assert len(result) == 5
        times = result["time"].tolist()
        assert times == sorted(times)

    def test_string_filter_works(self, rasm_ds):
        """String-based time filters should work for Gregorian-like calendars."""
        ctx = XarrayContext()
        ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
        result = ctx.sql(
            "SELECT COUNT(*) AS cnt FROM rasm WHERE time >= '1980-10-01'"
        ).to_pandas()
        full = ctx.sql("SELECT COUNT(*) AS cnt FROM rasm").to_pandas()
        assert 0 < result["cnt"].iloc[0] < full["cnt"].iloc[0]

    def test_aggregation(self, rasm_ds):
        """MIN/MAX on timestamp columns should work."""
        ctx = XarrayContext()
        ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
        result = ctx.sql(
            "SELECT MIN(time) AS t_min, MAX(time) AS t_max FROM rasm"
        ).to_pandas()
        assert result["t_min"].iloc[0] < result["t_max"].iloc[0]

    def test_row_count_matches_xarray(self, rasm_ds):
        """Total row count should equal the product of dimension sizes."""
        ctx = XarrayContext()
        ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
        result = ctx.sql("SELECT COUNT(*) AS cnt FROM rasm").to_pandas()
        expected = int(
            np.prod([rasm_ds.sizes[d] for d in rasm_ds["Tair"].dims])
        )
        assert result["cnt"].iloc[0] == expected


class TestCftimeNonGregorian:
    """Tests for non-Gregorian cftime calendars (360_day, julian).

    These use pa.int64() with CF-convention metadata and the cftime() UDF.
    """

    @pytest.fixture
    def ds_360day(self):
        """Synthetic 360-day calendar dataset."""
        import cftime

        times = [cftime.Datetime360Day(2000, m, 1) for m in range(1, 13)]
        return xr.Dataset(
            {"temp": ("time", np.arange(12, dtype=np.float32))},
            coords={"time": times},
        )

    def test_360day_registers(self, ds_360day):
        """A 360-day dataset should register without errors."""
        ctx = XarrayContext()
        ctx.from_dataset("ds360", ds_360day, chunks={"time": 6})
        result = ctx.sql("SELECT COUNT(*) AS cnt FROM ds360").to_pandas()
        assert result["cnt"].iloc[0] == 12

    def test_360day_select_ordered(self, ds_360day):
        """Integer offsets should be orderable."""
        ctx = XarrayContext()
        ctx.from_dataset("ds360", ds_360day, chunks={"time": 6})
        result = ctx.sql(
            "SELECT DISTINCT time FROM ds360 ORDER BY time"
        ).to_pandas()
        times = result["time"].tolist()
        assert times == sorted(times)
        assert len(times) == 12

    def test_360day_integer_filter(self, ds_360day):
        """Direct integer comparisons should work on int64 time columns."""
        ctx = XarrayContext()
        ctx.from_dataset("ds360", ds_360day, chunks={"time": 6})
        # Get all distinct time values to find a midpoint
        all_times = (
            ctx.sql("SELECT DISTINCT time FROM ds360 ORDER BY time")
            .to_pandas()["time"]
            .tolist()
        )
        mid = all_times[len(all_times) // 2]
        result = ctx.sql(
            f"SELECT COUNT(*) AS cnt FROM ds360 WHERE time >= {mid}"
        ).to_pandas()
        assert 0 < result["cnt"].iloc[0] < 12

    def test_360day_cftime_udf_registered(self, ds_360day):
        """from_dataset should auto-register a cftime() UDF for 360-day calendars."""
        ctx = XarrayContext()
        ctx.from_dataset("ds360", ds_360day, chunks={"time": 6})
        # The cftime() UDF should convert a date string to the int64 offset,
        # enabling ergonomic filtering.
        result = ctx.sql(
            "SELECT COUNT(*) AS cnt FROM ds360 "
            "WHERE time >= cftime('2000-07-01')"
        ).to_pandas()
        # July through December = 6 months
        assert result["cnt"].iloc[0] == 6

    def test_gregorian_like_no_cftime_udf(self):
        """Gregorian-like calendars should NOT register a cftime() UDF."""
        ds = xr.tutorial.open_dataset("rasm")
        ctx = XarrayContext()
        ctx.from_dataset("rasm", ds, chunks={"time": 12})
        # Using cftime() should fail since it's not registered for noleap.
        with pytest.raises(Exception):
            ctx.sql(
                "SELECT COUNT(*) FROM rasm WHERE time >= cftime('1980-01-01')"
            ).collect()


class TestGroupVarsByDims:
    """Unit tests for the private _group_vars_by_dims helper."""

    def test_single_dim_group(self):
        ds = xr.Dataset(
            {
                "a": (["x", "y"], np.zeros((2, 3))),
                "b": (["x", "y"], np.ones((2, 3))),
            }
        )
        groups = _group_vars_by_dims(ds)
        assert groups == {("x", "y"): ["a", "b"]}

    def test_multiple_dim_groups(self):
        ds = xr.Dataset(
            {
                "surface": (["time", "lat", "lon"], np.zeros((2, 3, 4))),
                "upper": (
                    ["time", "lat", "lon", "level"],
                    np.zeros((2, 3, 4, 5)),
                ),
            }
        )
        groups = _group_vars_by_dims(ds)
        assert set(groups.keys()) == {
            ("time", "lat", "lon"),
            ("time", "lat", "lon", "level"),
        }
        assert groups[("time", "lat", "lon")] == ["surface"]
        assert groups[("time", "lat", "lon", "level")] == ["upper"]

    def test_empty_dataset(self):
        assert _group_vars_by_dims(xr.Dataset()) == {}

    def test_includes_scalar_group(self):
        """Scalar (0-dim) variables group under the empty dims tuple."""
        ds = xr.Dataset(
            {"band": (["y", "x"], np.zeros((2, 3))), "projection": ((), 0)}
        )
        groups = _group_vars_by_dims(ds)
        assert groups == {("y", "x"): ["band"], (): ["projection"]}

    def test_ignores_coords(self):
        """Coordinate variables shouldn't be returned as groups."""
        ds = xr.Dataset(
            {"v": (["x"], np.arange(3))},
            coords={"x": np.arange(3), "label": ("x", ["a", "b", "c"])},
        )
        groups = _group_vars_by_dims(ds)
        assert groups == {("x",): ["v"]}


class TestFromDatasetMultiDims:
    """from_dataset should split datasets with mixed dims into multiple tables."""

    @pytest.fixture
    def mixed_ds(self):
        np.random.seed(0)
        return xr.Dataset(
            {
                "temperature_2m": (
                    ["time", "lat", "lon"],
                    np.random.rand(2, 3, 4),
                ),
                "pressure": (
                    ["time", "lat", "lon", "level"],
                    np.random.rand(2, 3, 4, 2),
                ),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=2),
                "lat": np.linspace(-90, 90, 3),
                "lon": np.linspace(-180, 180, 4),
                "level": [500, 1000],
            },
        ).chunk({"time": 1})

    def test_registers_multiple_tables(self, mixed_ds):
        ctx = XarrayContext()
        ctx.from_dataset("era5", mixed_ds)
        surface = ctx.sql("SELECT * FROM era5.time_lat_lon").to_pandas()
        upper = ctx.sql("SELECT * FROM era5.time_lat_lon_level").to_pandas()
        assert "temperature_2m" in surface.columns
        assert "pressure" in upper.columns
        assert len(surface) == 2 * 3 * 4
        assert len(upper) == 2 * 3 * 4 * 2

    @pytest.fixture
    def scalar_and_array_ds(self):
        """A gridded variable plus a scalar metadata variable (GOES-like)."""
        np.random.seed(0)
        return xr.Dataset(
            {
                "temperature_2m": (
                    ["time", "lat", "lon"],
                    np.random.rand(2, 3, 4),
                ),
                "projection": ((), 0),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=2),
                "lat": np.linspace(-90, 90, 3),
                "lon": np.linspace(-180, 180, 4),
            },
        ).chunk({"time": 1})

    def test_registers_scalar_var_as_single_row_table(
        self, scalar_and_array_ds
    ):
        ctx = XarrayContext()
        ctx.from_dataset("goes", scalar_and_array_ds)
        surface = ctx.sql("SELECT * FROM goes.time_lat_lon").to_pandas()
        scalar = ctx.sql("SELECT * FROM goes.scalar").to_pandas()
        assert "temperature_2m" in surface.columns
        assert "projection" in scalar.columns
        assert len(scalar) == 1

    def test_scalar_group_in_catalog(self, scalar_and_array_ds):
        ctx = XarrayContext()
        ctx.from_dataset("goes", scalar_and_array_ds)
        tables = set(ctx.catalog().schema("goes").table_names())
        assert tables == {"time_lat_lon", "scalar"}

    def test_scalar_table_name_override(self, scalar_and_array_ds):
        ctx = XarrayContext()
        ctx.from_dataset("goes", scalar_and_array_ds, table_names={(): "meta"})
        result = ctx.sql("SELECT * FROM goes.meta").to_pandas()
        assert "projection" in result.columns
        assert len(result) == 1

    def test_table_names_override(self, mixed_ds):
        ctx = XarrayContext()
        ctx.from_dataset(
            "era5",
            mixed_ds,
            table_names={("time", "lat", "lon"): "surface"},
        )
        result = ctx.sql("SELECT * FROM era5.surface").to_pandas()
        assert "temperature_2m" in result.columns
        # Non-aliased group falls back to the default joined-dim table name.
        upper = ctx.sql("SELECT * FROM era5.time_lat_lon_level").to_pandas()
        assert "pressure" in upper.columns

    def test_schema_registered_in_catalog(self, mixed_ds):
        """Mixed-dim datasets should create a SQL schema under the catalog."""
        ctx = XarrayContext()
        ctx.from_dataset("era5", mixed_ds)
        assert "era5" in ctx.catalog().schema_names()
        tables = set(ctx.catalog().schema("era5").table_names())
        assert tables == {"time_lat_lon", "time_lat_lon_level"}

    def test_uniform_dims_uses_name_directly(self, mixed_ds):
        """A single dim group should register under the bare name."""
        ds = mixed_ds[["temperature_2m"]]
        ctx = XarrayContext()
        ctx.from_dataset("surface", ds)
        result = ctx.sql("SELECT * FROM surface").to_pandas()
        assert "temperature_2m" in result.columns

    def test_table_names_is_keyword_only(self, mixed_ds):
        ctx = XarrayContext()
        with pytest.raises(TypeError):
            ctx.from_dataset("era5", mixed_ds, {("time",): "x"})

    @pytest.fixture
    def coordless_dims_ds(self):
        """Mirror the fashion-mnist layout from issue #203: a dimension
        coordinate (``sample``) alongside dimensions without coordinates
        (``channel``/``height``/``width``)."""
        n_sample, n_channel, n_height, n_width = 4, 1, 3, 3
        return xr.Dataset(
            {
                "images": (
                    ["sample", "channel", "height", "width"],
                    np.arange(
                        n_sample * n_channel * n_height * n_width,
                        dtype="float32",
                    ).reshape(n_sample, n_channel, n_height, n_width),
                ),
                "labels": (["sample"], np.arange(n_sample, dtype="int64")),
            },
            coords={"sample": ("sample", np.arange(n_sample, dtype="int64"))},
        ).chunk({"sample": 1})

    def test_coordless_dims_appear_as_columns(self, coordless_dims_ds):
        """Regression for #203: dimensions without coordinates must still be
        emitted as columns, not silently dropped from the schema."""
        ctx = XarrayContext()
        ctx.from_dataset(
            "mnist",
            coordless_dims_ds,
            table_names={
                ("sample", "channel", "height", "width"): "X",
                ("sample",): "y",
            },
        )
        result = ctx.sql('SELECT * FROM mnist."X"').to_pandas()
        assert set(result.columns) == {
            "sample",
            "channel",
            "height",
            "width",
            "images",
        }

    def test_coordless_dims_values_match_xarray(self, coordless_dims_ds):
        """The X table's rows must match xarray's own pivot exactly, including
        the synthesized index values for the coordinate-less dimensions."""
        ctx = XarrayContext()
        ctx.from_dataset(
            "mnist",
            coordless_dims_ds,
            table_names={
                ("sample", "channel", "height", "width"): "X",
                ("sample",): "y",
            },
        )
        dim_cols = ["sample", "channel", "height", "width"]
        result = (
            ctx.sql('SELECT * FROM mnist."X"')
            .to_pandas()
            .sort_values(dim_cols)
            .reset_index(drop=True)
        )
        expected = (
            coordless_dims_ds[["images"]]
            .to_dataframe()
            .reset_index()
            .sort_values(dim_cols)
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            result, expected, check_dtype=False, check_like=True
        )

    def test_coordless_dims_y_table_unaffected(self, coordless_dims_ds):
        """The 1-D ``y`` group (sample coordinate + labels) is unchanged."""
        ctx = XarrayContext()
        ctx.from_dataset(
            "mnist",
            coordless_dims_ds,
            table_names={
                ("sample", "channel", "height", "width"): "X",
                ("sample",): "y",
            },
        )
        result = ctx.sql('SELECT * FROM mnist."y"').to_pandas()
        assert set(result.columns) == {"sample", "labels"}
        assert len(result) == coordless_dims_ds.sizes["sample"]

    def test_single_table_all_coordless_dims(self):
        """A uniform-dim dataset whose dims lack coordinates registers as one
        table with every dimension present as a column, and the coordinate-less
        dimensions carry their ABSOLUTE index even when chunked (issue #203)."""
        ds = xr.Dataset(
            {"a": (("x", "y"), np.arange(6, dtype="float32").reshape(3, 2))}
        ).chunk({"x": 1})  # chunked along the coordinate-less 'x' dim
        ctx = XarrayContext()
        ctx.from_dataset("grid", ds)
        result = (
            ctx.sql("SELECT * FROM grid")
            .to_pandas()
            .sort_values(["x", "y"])
            .reset_index(drop=True)
        )
        assert set(result.columns) == {"x", "y", "a"}
        # x must span 0..2 across the three chunks, not restart at 0 each block.
        expected = (
            ds.to_dataframe()
            .reset_index()
            .sort_values(["x", "y"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            result, expected, check_dtype=False, check_like=True
        )
