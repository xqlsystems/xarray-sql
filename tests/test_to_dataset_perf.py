"""Peak-memory tests for the lazy SQL -> xarray round-trip.

Asserts the lazy backend honors its contract: a single-chunk access
peaks far below an eager whole-grid materialization, and a streaming
aggregation does not balloon past the source size.
"""

import gc
import tracemalloc

import numpy as np
import pytest
import xarray as xr

from xarray_sql import XarrayContext


def _peak_mb(fn):
    """Return ``(result, peak_memory_mb)`` for a single call to ``fn``."""
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak / 1e6


@pytest.fixture(scope="module")
def air_source():
    """NCEP ``air_temperature`` chunked along time, ~31 MB dense.

    Module-scoped so the pooch download (~3.5 MB) is amortized across
    tests in this file. Skips when the tutorial dataset is unreachable.
    """
    try:
        return xr.tutorial.open_dataset("air_temperature").chunk({"time": 24})
    except (OSError, ValueError, ImportError) as e:
        pytest.skip(f"air_temperature tutorial dataset unavailable: {e}")


def test_lazy_chunk_peak_memory_is_bounded(air_source):
    """``.sel(time=t0).load()`` materializes only one chunk, not the cube.

    Reference observation: lazy chunk peak is ~1.8 MB on a 31 MB
    dense source. A regression that quietly buffers the whole result
    would push past 10 MB. Eager ``to_dataset(chunks=None)`` is
    measured too as a sanity floor for the gap: the eager path should
    be at least an order of magnitude heavier than the lazy chunk,
    otherwise the lazy path isn't actually lazy.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_source, chunks={"time": 24})
    t0 = air_source["time"].values[0]

    out = ctx.sql('SELECT * FROM "air"').to_dataset()
    chunk, chunk_peak = _peak_mb(lambda: out["air"].sel(time=t0).load())
    assert chunk.size == air_source.sizes["lat"] * air_source.sizes["lon"]
    assert chunk_peak < 10.0, (
        f"lazy single-chunk access should stay under 10 MB on "
        f"air_temperature, got {chunk_peak:.2f} MB"
    )

    _, eager_peak = _peak_mb(
        lambda: ctx.sql('SELECT * FROM "air"').to_dataset(chunks=None)
    )
    assert eager_peak > 50.0, (
        f"eager whole-grid materialization should peak above 50 MB; "
        f"if it doesn't, the eager path may have silently gone lazy "
        f"and the lazy assertion above no longer means anything. "
        f"Got {eager_peak:.1f} MB"
    )
    assert eager_peak / max(chunk_peak, 0.1) > 10.0, (
        f"eager peak should be at least 10x the lazy chunk peak; "
        f"if it isn't, the lazy path isn't actually lazy. "
        f"Got eager={eager_peak:.1f} MB, lazy={chunk_peak:.2f} MB"
    )


def test_streaming_aggregation_does_not_explode(air_source):
    """A ``GROUP BY`` reducing the long axis streams in a single pass.

    Reduces 3.86M rows of ``air_temperature`` to ~1.3K group cells. The
    aggregation must stream the source once and emit a tiny result, not
    buffer the entire row set into memory. The engine streams partitions
    concurrently, so the tracemalloc peak varies run to run (~1.2x-4x the
    source, scaling with the number of in-flight partitions); we take the
    best of a few samples for a stable floor (~1.5x source), well below a
    "buffer the whole row set" regression, which would keep every sample
    high. Threshold is 4x source size so transient buffers fit.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_source, chunks={"time": 24})
    source_mb = air_source.nbytes / 1e6

    def run_agg():
        return ctx.sql(
            'SELECT lat, lon, AVG(air) AS air_avg FROM "air" GROUP BY lat, lon'
        ).to_dataset(dims=["lat", "lon"])

    peaks = []
    for _ in range(3):
        agg, peak = _peak_mb(run_agg)
        peaks.append(peak)
    agg_peak = min(peaks)

    assert agg.sizes["lat"] * agg.sizes["lon"] == (
        air_source.sizes["lat"] * air_source.sizes["lon"]
    )
    assert agg_peak < 4 * source_mb, (
        f"GROUP BY reduction should not balloon past 4x source size; got "
        f"best-of-{len(peaks)} peak={agg_peak:.1f} MB on a "
        f"{source_mb:.1f} MB source"
    )

    # Sanity: values agree with the xarray-native reduction.
    ref = (
        air_source.compute()
        .mean(dim="time")["air"]
        .sortby(["lat", "lon"])
        .values
    )
    got = agg.sortby(["lat", "lon"])["air_avg"].values
    np.testing.assert_allclose(got, ref, rtol=1e-5)
