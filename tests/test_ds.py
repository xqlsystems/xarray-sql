"""Tests for the SQL -> xarray reverse path.

Covers the user-facing contract of ``ctx.sql(...).to_dataset(...)``:

* Wrapper behavior on the object returned by ``ctx.sql`` and DataFusion
  method passthrough.
* Round-trip identity across varied source Datasets (one parametrized
  ``assert_identical`` test, not eight per-aspect checks).
* Aggregation, ``dimension_columns`` inference, and the template /
  ``template`` resolution rules (name or Dataset) with their error paths.
* Sparsity handling and ``fill_value`` dtype behavior.
* The vectorized-indexer fallback through xarray's adapter.

The tests favor checking the user-visible contract (values, dims,
attrs) over the implementation path (call counts, internal class
identity), so the suite stays useful as the lazy backend evolves.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xarray_sql import XarrayContext
from xarray_sql.ds import XarrayDataFrame


# ---------------------------------------------------------------------------
# Wrapper: ctx.sql(...) returns XarrayDataFrame
# ---------------------------------------------------------------------------


def test_ctx_sql_returns_xarray_dataframe(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    result = ctx.sql("SELECT * FROM air LIMIT 5")
    assert isinstance(result, XarrayDataFrame)


def test_to_pandas_unchanged_behavior(air_dataset_small):
    """Wrapped ``.to_pandas()`` is bit-for-bit equal to the un-wrapped path."""
    from datafusion import SessionContext

    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    wrapped = ctx.sql("SELECT * FROM air LIMIT 7").to_pandas()
    raw = SessionContext.sql(ctx, "SELECT * FROM air LIMIT 7").to_pandas()
    pd.testing.assert_frame_equal(wrapped, raw)


def test_passthrough_methods(air_dataset_small):
    """DataFusion methods we did not override forward via ``__getattr__``."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    result = ctx.sql("SELECT * FROM air LIMIT 5")
    names = [f.name for f in result.schema()]
    assert {"lat", "lon", "time", "air"}.issubset(set(names))


# ---------------------------------------------------------------------------
# Round-trip identity (parametrized over local + tutorial datasets)
# ---------------------------------------------------------------------------


def _clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Strip ``encoding`` from a Dataset and all its variables.

    Round-trip identity tests should not be coupled to encoding choices,
    since template-recovery deliberately drops dtype-bound keys.
    """
    ds = ds.copy()
    for v in ds.variables.values():
        v.encoding.clear()
    ds.encoding.clear()
    return ds


def _load_tutorial(name: str) -> xr.Dataset | None:
    """Return a small xarray tutorial Dataset, or None when unavailable.

    Used to widen round-trip coverage beyond the conftest fixtures without
    requiring network in CI. Pooch caches downloads locally on first run.
    """
    try:
        return xr.tutorial.open_dataset(name)
    except (OSError, ValueError, ImportError):
        return None


@pytest.mark.parametrize(
    "fixture_name",
    ["air_dataset_small", "weather_dataset", "synthetic_dataset", "eraint_uvz"],
)
def test_round_trip_identity(request, fixture_name):
    """``SELECT *`` round-trips to a Dataset that is ``assert_identical``
    to the source: values, dims, coord values, dtypes, non-dim coords,
    and attrs all match (modulo coord ordering, normalized on both
    sides). One test covers what was previously a fan of narrow checks,
    parametrized over local fixtures and one xarray tutorial dataset.
    """
    if fixture_name == "eraint_uvz":
        source = _load_tutorial("eraint_uvz")
        if source is None:
            pytest.skip("eraint_uvz tutorial dataset unavailable")
        source = source.chunk()
    else:
        source = request.getfixturevalue(fixture_name).copy()
        source.attrs["round_trip_marker"] = "yes"
        first_var = next(iter(source.data_vars))
        source[first_var].attrs["units"] = "test_units"

    ctx = XarrayContext()
    ctx.from_dataset("t", source)
    out = ctx.sql("SELECT * FROM t").to_dataset().compute()

    sort_keys = list(out.dims)
    actual = _clear_encoding(out.sortby(sort_keys))
    expected = _clear_encoding(source.compute().sortby(sort_keys))
    xr.testing.assert_identical(actual, expected)


def test_aggregation_drops_dim(air_dataset_small):
    """``GROUP BY lat, lon`` over time -> 2D Dataset with the alias."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dims=["lat", "lon"])
    assert set(out.dims) == {"lat", "lon"}
    assert "air_avg" in out.data_vars
    assert "air" not in out.data_vars
    expected = (
        air_dataset_small.compute()
        .sortby(["lat", "lon"])
        .mean(dim="time")["air"]
        .values
    )
    actual = out.sortby(["lat", "lon"])["air_avg"].values
    np.testing.assert_allclose(actual, expected)


def test_aggregation_infers_dims(air_dataset_small):
    """#189: to_dataset() infers the surviving GROUP BY dims without dims=."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)

    # GROUP BY lat, lon: time is aggregated away, lat/lon survive.
    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset()
    assert set(out.dims) == {"lat", "lon"}
    assert "air_avg" in out.data_vars
    expected = (
        air_dataset_small.compute()
        .sortby(["lat", "lon"])
        .mean(dim="time")["air"]
        .values
    )
    np.testing.assert_allclose(
        out.sortby(["lat", "lon"])["air_avg"].values, expected
    )

    # The reporter's exact case: GROUP BY the time coordinate.
    single = ctx.sql(
        'SELECT "time", AVG("air") AS air FROM "air" GROUP BY "time"'
    ).to_dataset()
    assert set(single.dims) == {"time"}
    assert "air" in single.data_vars
    expected_t = (
        air_dataset_small.compute()
        .sortby("time")
        .mean(dim=["lat", "lon"])["air"]
        .values
    )
    np.testing.assert_allclose(single.sortby("time")["air"].values, expected_t)


def test_barrier_query_scans_source_once(air_dataset_small):
    """A barrier plan (aggregation) executes the source exactly once.

    The lazy scan path re-runs the whole upstream plan for every coordinate
    discovery and every variable access; for an aggregation -- which cannot push
    an indexer filter below the GROUP BY -- that is pure re-computation of an
    expensive scan. ``to_dataset()`` on a barrier plan must instead make a
    single streamed pass over the source, and ``.compute()`` must trigger no
    further reads.
    """
    from xarray_sql.df import block_slices
    from xarray_sql.reader import read_xarray_table

    reads: list = []
    table = read_xarray_table(
        air_dataset_small,
        chunks={"time": 6},
        _iteration_callback=lambda block, proj: reads.append(block),
    )
    n_partitions = len(list(block_slices(air_dataset_small, {"time": 6})))

    ctx = XarrayContext()
    ctx.register_table("air", table)
    ctx._registered_datasets["air"] = air_dataset_small

    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dims=["lat", "lon"])
    reads_after_construct = len(reads)
    out.compute()
    reads_after_compute = len(reads)

    # Exactly one pass over the source (each partition read once) ...
    assert reads_after_construct == n_partitions
    # ... and computing the materialized result re-reads nothing.
    assert reads_after_compute == reads_after_construct


def test_order_by_direction_sets_dim_order(air_dataset_small):
    """A barrier query's ORDER BY direction carries through to the Dataset
    dimension order, rather than being force-sorted ascending.

    ``ORDER BY lat DESC`` must yield a strictly descending ``lat`` dimension,
    with data still correctly aligned to those (descending) coordinates.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql(
        "SELECT lat, AVG(air) AS air_avg FROM air GROUP BY lat ORDER BY lat DESC"
    ).to_dataset(dims=["lat"])

    lat = out["lat"].values
    assert (np.diff(lat) < 0).all(), f"expected descending lat, got {lat}"

    # Values stay aligned to the descending coordinate (scatter handles order).
    expected = (
        air_dataset_small.compute()
        .mean(dim=["time", "lon"])["air"]
        .sortby("lat", ascending=False)
    )
    np.testing.assert_allclose(out["air_avg"].values, expected.values)


def test_unfiltered_scan_skips_dim_discovery(air_dataset_small):
    """``SELECT * FROM <registered>`` does no per-dim discovery scans.

    The pre-fix :func:`_build_lazy_scan` ran one full source scan per dim
    via ``inner_df.select(col(d)).distinct().sort(...)`` -- a single-column
    projection per dim, multiplied across the dim count. With a registered
    template and an unfiltered plan, :func:`_maybe_template_coords` returns
    the template's coord arrays directly and the discovery loop is skipped
    entirely. Verified by the iteration callback's projection list: no read
    where the projection is a single dim column.
    """
    from xarray_sql.reader import read_xarray_table

    reads: list = []
    table = read_xarray_table(
        air_dataset_small,
        chunks={"time": 6},
        _iteration_callback=lambda block, proj: reads.append(list(proj)),
    )
    ctx = XarrayContext()
    ctx.register_table("air", table)
    ctx._registered_datasets["air"] = air_dataset_small

    reads.clear()
    out = ctx.sql('SELECT * FROM "air"').to_dataset()
    single_dim_reads = [
        proj for proj in reads if set(proj) <= {"time", "lat", "lon"}
    ]
    assert not single_dim_reads, (
        f"fast path should skip per-dim discovery scans, got "
        f"{single_dim_reads!r}"
    )
    # The fast path must not break correctness on first access.
    out.compute()  # safety: the Dataset still resolves


def test_filtered_scan_still_uses_dim_discovery(air_dataset_small):
    """A ``WHERE`` clause forbids the template-coord fast path.

    With a filter the result's coord extent is a subset of the source, so
    reusing template coords would silently widen the output. The dispatch
    in :func:`_maybe_template_coords` falls back to per-dim discovery in
    that case, which is observable as a positive read count on the source.
    """
    from xarray_sql.reader import read_xarray_table

    reads: list = []
    table = read_xarray_table(
        air_dataset_small,
        chunks={"time": 6},
        _iteration_callback=lambda block, proj: reads.append(block),
    )
    ctx = XarrayContext()
    ctx.register_table("air", table)
    ctx._registered_datasets["air"] = air_dataset_small

    reads.clear()
    ctx.sql('SELECT * FROM "air" WHERE lat > 30').to_dataset()
    assert reads, "filtered scan must hit the source to discover coord extent"


def test_fast_path_uses_scanned_tables_coords_not_user_template(
    air_dataset_small,
):
    """Fast path sources coord values from the scanned table, not ``template=``.

    With multiple registered Datasets, a user may pass ``template=other``
    for metadata recovery while the query scans a different registered
    table. The coord values must come from the **scanned** table's
    registered Dataset; using ``other``'s coords would silently widen the
    output if their coord ranges differ.
    """
    other = air_dataset_small.isel(lat=slice(0, 5))
    assert other.sizes["lat"] != air_dataset_small.sizes["lat"]
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small, chunks={"time": 24})
    ctx.from_dataset("other", other, chunks={"time": 24})
    out = ctx.sql('SELECT * FROM "air"').to_dataset(
        dims=["time", "lat", "lon"], template=other
    )
    # Scanned table is "air", so lat must match air's full lat axis.
    np.testing.assert_array_equal(
        out["lat"].values, air_dataset_small["lat"].values
    )


def test_round_trip_preserves_descending_lat_on_lazy_path(air_dataset_small):
    """Lazy round-trip preserves source dim order (xarray-sql#171).

    NCEP ``air_temperature`` ships descending lat (75.0 -> 15.0). The
    discovery path's ``.distinct().sort()`` previously flipped lat to
    ascending on the lazy result. The template-coord fast path returns the
    source's coord arrays as-is, so the descending order survives.
    """
    ds = air_dataset_small
    assert (np.diff(ds["lat"].values) < 0).all(), (
        "test relies on a descending-lat fixture"
    )
    ctx = XarrayContext()
    ctx.from_dataset("air", ds, chunks={"time": 24})
    lazy = ctx.sql('SELECT * FROM "air"').to_dataset()
    np.testing.assert_array_equal(lazy["lat"].values, ds["lat"].values)


def test_chunks_argument_controls_partitioning(synthetic_dataset):
    """``chunks`` controls eager-vs-chunked and inherits the source grid.

    The default ``"inherit"`` reuses the source's genuinely multi-chunk
    dimensions, so the output chunk grid maps onto the source partitions;
    ``chunks=None`` forces an eager, in-memory result. Both reproduce the source.
    """
    import dask.array as da

    ctx = XarrayContext()
    ctx.from_dataset("t", synthetic_dataset)
    var = next(iter(synthetic_dataset.data_vars))

    inherited = ctx.sql("SELECT * FROM t").to_dataset()
    assert isinstance(inherited[var].data, da.Array)
    # Output time chunks align to the source's time partitions.
    assert inherited.chunksizes["time"] == synthetic_dataset.chunksizes["time"]

    eager = ctx.sql("SELECT * FROM t").to_dataset(chunks=None)
    assert not isinstance(eager[var].data, da.Array)

    xr.testing.assert_allclose(
        inherited.compute().sortby(["time", "lat", "lon"]),
        synthetic_dataset.compute().sortby(["time", "lat", "lon"]),
    )


def test_chunks_auto_snaps_to_source_partitions():
    """``chunks="auto"`` coarsens to the byte budget but snaps chunk boundaries
    to whole source partitions (so no chunk splits a source partition)."""
    import dask

    # 12 source partitions of size 2 along time.
    ds = xr.Dataset(
        {
            "v": (
                ("time", "x"),
                np.arange(24 * 4, dtype="float64").reshape(24, 4),
            )
        },
        coords={"time": np.arange(24), "x": np.arange(4)},
    ).chunk({"time": 2})
    ctx = XarrayContext()
    ctx.from_dataset("t", ds)

    # block bytes = 8 * 2(time) * 4(x) = 64; target 192 -> merge 3 partitions.
    with dask.config.set({"array.chunk-size": "192B"}):
        out = ctx.sql("SELECT * FROM t").to_dataset(chunks="auto")

    time_chunks = out.chunksizes["time"]
    assert all(c % 2 == 0 for c in time_chunks)  # aligned to source size 2
    assert time_chunks[0] > 2  # genuinely coarsened
    assert len(time_chunks) < 12  # fewer chunks than source partitions

    xr.testing.assert_allclose(
        out.compute().sortby(["time", "x"]),
        ds.compute().sortby(["time", "x"]),
    )


# ---------------------------------------------------------------------------
# dimension_columns / template resolution rules
# ---------------------------------------------------------------------------


def test_to_dataset_multi_registered_requires_explicit_template(
    air_dataset_small,
):
    """With more than one registered Dataset, the caller disambiguates by
    passing a registered table name as ``template=``."""
    ctx = XarrayContext()
    ctx.from_dataset("air1", air_dataset_small)
    ctx.from_dataset("air2", air_dataset_small)
    out = ctx.sql("SELECT * FROM air1").to_dataset(template="air1")
    assert set(out.dims) == {"time", "lat", "lon"}


def test_to_dataset_infer_fails_when_no_dim_survives(air_dataset_small):
    """A global aggregation leaves no registered dim in the result -> clear error."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="dims cannot be inferred"):
        ctx.sql("SELECT AVG(air) AS air_avg FROM air").to_dataset()


def test_template_accepts_name_or_dataset(air_dataset_small):
    """``template=`` accepts either a registered table name or a Dataset
    object, with equivalent metadata recovery."""
    other = air_dataset_small.copy()
    other.attrs = {"flag": "other"}
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    ctx.from_dataset("other", other)

    by_name = ctx.sql("SELECT * FROM air").to_dataset(
        dims=["time", "lat", "lon"], template="other"
    )
    by_object = ctx.sql("SELECT * FROM air").to_dataset(
        dims=["time", "lat", "lon"], template=other
    )
    assert by_name.attrs == {"flag": "other"}
    assert by_object.attrs == {"flag": "other"}


def test_template_unknown_name_raises(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="not a registered table"):
        ctx.sql("SELECT * FROM air").to_dataset(
            dims=["time", "lat", "lon"], template="missing"
        )


def test_template_recovers_var_encoding_strips_dtype(air_dataset_small):
    """``zlib`` survives; dtype-bound keys are stripped (SQL may have cast)."""
    ds = air_dataset_small.copy()
    ds["air"].encoding = {
        "zlib": True,
        "dtype": "int16",
        "_FillValue": -999,
        "missing_value": -999,
    }
    ctx = XarrayContext()
    ctx.from_dataset("air", ds)
    out = ctx.sql("SELECT * FROM air").to_dataset(dims=["time", "lat", "lon"])
    assert out["air"].encoding.get("zlib") is True
    assert "dtype" not in out["air"].encoding
    assert "_FillValue" not in out["air"].encoding
    assert "missing_value" not in out["air"].encoding


def test_template_aggregation_alias_no_attrs(air_dataset_small):
    """``air_avg`` from ``AVG(air)`` does NOT inherit attrs from ``air``."""
    ds = air_dataset_small.copy()
    ds["air"].attrs = {"units": "K"}
    ctx = XarrayContext()
    ctx.from_dataset("air", ds)
    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dims=["lat", "lon"])
    assert "air_avg" in out.data_vars
    assert out["air_avg"].attrs == {}


def test_to_dataset_explicit_template_overrides_auto_resolve(
    air_dataset_small,
):
    """Explicit template= wins over the auto-resolved FROM-clause table."""
    other = air_dataset_small.copy()
    other.attrs = {"flag": "explicit"}
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dims=["time", "lat", "lon"], template=other
    )
    assert out.attrs == {"flag": "explicit"}


# ---------------------------------------------------------------------------
# Lazy backend: value-level contract (not call counts)
# ---------------------------------------------------------------------------


def test_lazy_isel_int_round_trip(air_dataset_small):
    """``isel(time=0)`` on the lazy result matches the eager equivalent."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    lazy = ctx.sql("SELECT * FROM air").to_dataset()
    eager = lazy.compute()
    actual = lazy["air"].isel(time=0).sortby(["lat", "lon"]).values
    expected = eager["air"].isel(time=0).sortby(["lat", "lon"]).values
    np.testing.assert_array_equal(actual, expected)


def test_lazy_isel_slice_round_trip(air_dataset_small):
    """isel(time=slice(0, 3)) round-trip matches the source."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    actual = out["air"].isel(time=slice(0, 3)).sortby(["lat", "lon"]).values
    expected = (
        air_dataset_small["air"]
        .compute()
        .isel(time=slice(0, 3))
        .sortby(["lat", "lon"])
        .values
    )
    np.testing.assert_array_equal(actual, expected)


def test_lazy_outer_indexer_array(air_dataset_small):
    """Fancy index along one dim works (IN-equivalent pushdown)."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    lazy = ctx.sql("SELECT * FROM air").to_dataset()
    eager = lazy.compute()
    indices = [0, 3, 5]
    np.testing.assert_array_equal(
        lazy["air"].isel(lat=indices).values,
        eager["air"].isel(lat=indices).values,
    )


def test_lazy_compute_returns_eager(air_dataset_small):
    """``.compute()`` returns an in-memory Dataset matching the source."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset().compute()
    np.testing.assert_array_equal(
        out.sortby(["time", "lat", "lon"])["air"].values,
        air_dataset_small.compute()
        .sortby(["time", "lat", "lon"])["air"]
        .values,
    )


def test_vectorized_indexer_falls_back_via_xarray_adapter(
    air_dataset_small,
):
    """VectorizedIndexer paths through xarray's adapter to outer + gather.

    Our SQLBackendArray declares ``IndexingSupport.OUTER``, so xarray's
    ``explicit_indexing_adapter`` converts vectorized indexers into a
    series of outer reads followed by an in-memory numpy gather. The
    public contract: values match the eager-computed equivalent.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    lazy = ctx.sql("SELECT * FROM air").to_dataset()
    eager = lazy.compute()

    points_t = xr.DataArray([0, 3, 1], dims="point")
    points_lat = xr.DataArray([2, 0, 5], dims="point")
    np.testing.assert_array_equal(
        lazy["air"].isel(time=points_t, lat=points_lat).values,
        eager["air"].isel(time=points_t, lat=points_lat).values,
    )


# ---------------------------------------------------------------------------
# Sparsity handling and fill_value
# ---------------------------------------------------------------------------


def test_sparsity_result_default_filters_lazy(air_dataset_small):
    """Default sparsity='result' keeps only filtered coords (lazy path)."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    threshold = float(air_dataset_small["lat"].values[5])
    out = ctx.sql(f"SELECT * FROM air WHERE lat > {threshold}").to_dataset()
    assert (out["lat"].values > threshold).all()
    assert out.sizes["lat"] < air_dataset_small.sizes["lat"]


def test_sparsity_template_full_grid(air_dataset_small):
    """sparsity='template' reindexes to the full grid with NaN fills."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    threshold = float(air_dataset_small["lat"].values[5])
    out = ctx.sql(f"SELECT * FROM air WHERE lat > {threshold}").to_dataset(
        sparsity="template"
    )
    assert out.sizes["lat"] == air_dataset_small.sizes["lat"]
    lat_vals = out["lat"].values
    below_mask = lat_vals <= threshold
    above_mask = lat_vals > threshold
    below = out["air"].isel(lat=np.where(below_mask)[0])
    above = out["air"].isel(lat=np.where(above_mask)[0])
    assert np.isnan(below.values).all()
    assert not np.isnan(above.values).any()


def test_sparsity_template_requires_template(air_dataset_small):
    """No resolvable template -> sparsity='template' raises."""
    other = air_dataset_small.copy()
    ctx = XarrayContext()
    ctx.from_dataset("a", air_dataset_small)
    ctx.from_dataset("b", other)
    with pytest.raises(ValueError, match="requires template= to be supplied"):
        ctx.sql("SELECT * FROM a").to_dataset(
            dims=["time", "lat", "lon"],
            sparsity="template",
        )


def test_sparsity_invalid_value_raises(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="sparsity must be"):
        ctx.sql("SELECT * FROM air").to_dataset(
            dims=["time", "lat", "lon"],
            sparsity="bogus",  # type: ignore[arg-type]
        )


def test_sparsity_template_with_aggregation(air_dataset_small):
    """sparsity='template' on an aggregation respects dimension_columns subset."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    threshold = float(air_dataset_small["lat"].values[5])
    out = ctx.sql(
        f"""
        SELECT lat, lon, AVG(air) AS air_avg
        FROM air
        WHERE lat > {threshold}
        GROUP BY lat, lon
        """
    ).to_dataset(dims=["lat", "lon"], sparsity="template")
    assert out.sizes["lat"] == air_dataset_small.sizes["lat"]
    assert "time" not in out.dims
    below_mask = out["lat"].values <= threshold
    below = out["air_avg"].isel(lat=np.where(below_mask)[0])
    assert np.isnan(below.values).all()


def test_fill_value_int_upcasts_to_float():
    """fill_value=NaN forces float upcast on int columns -- documented."""
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.arange(6, dtype=np.int64).reshape(3, 2))},
        coords={"lat": [0, 1, 2], "lon": [10, 11]},
    ).chunk({"lat": 3})
    ctx = XarrayContext()
    ctx.from_dataset("t", ds)
    out = ctx.sql("SELECT * FROM t WHERE lat > 0").to_dataset(
        sparsity="template"
    )
    assert np.issubdtype(out["v"].dtype, np.floating)
    assert np.isnan(out["v"].sel(lat=0).values).all()


def test_fill_value_custom_preserves_int(air_dataset_small):
    """Passing a typed sentinel preserves the data var's int dtype."""
    source = xr.Dataset(
        {
            "v": (
                ("lat", "lon"),
                np.arange(6, dtype=np.int64).reshape(3, 2) + 1,
            ),
        },
        coords={"lat": [0, 1, 2], "lon": [10, 11]},
    ).chunk({"lat": 3})
    ctx = XarrayContext()
    ctx.from_dataset("t", source)
    out = ctx.sql("SELECT * FROM t WHERE lat > 0").to_dataset(
        sparsity="template", fill_value=-1
    )
    assert np.issubdtype(out["v"].dtype, np.integer)
    assert (out["v"].sel(lat=0).values == -1).all()
    assert out["v"].sel(lat=2, lon=11).item() == 6


def test_sparsity_template_then_metadata(air_dataset_small):
    """sparsity='template' composes with template metadata recovery."""
    ds = air_dataset_small.copy()
    ds.attrs = {"src": "tmpl"}
    ds["air"].attrs = {"units": "K"}
    ctx = XarrayContext()
    ctx.from_dataset("air", ds)
    threshold = float(ds["lat"].values[5])
    out = ctx.sql(f"SELECT * FROM air WHERE lat > {threshold}").to_dataset(
        sparsity="template"
    )
    assert out.attrs == {"src": "tmpl"}
    assert out["air"].attrs == {"units": "K"}
    assert out.sizes["lat"] == ds.sizes["lat"]
