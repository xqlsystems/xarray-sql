import uuid

import pyarrow as pa
import xarray as xr
from datafusion import SessionContext
from datafusion.catalog import Schema
from types import ModuleType

from . import cftime as cft
from .df import (
    Chunks,
    TableNames,
    group_vars_by_dims,
    resolve_table_names,
    shared_coord_arrays,
)
from .ds import XarrayDataFrame
from .reader import _in_memory_partitions, read_xarray_table

_proj: ModuleType | None
try:  # pyproj is an optional dependency (`pip install xarray-sql[geo]`).
    from . import proj

    _proj = proj
except ImportError:  # pragma: no cover - depends on the environment
    _proj = None


class Context(SessionContext):
    """A datafusion `SessionContext` that also supports `xarray.Dataset`s.

    Also available under its original name, ``XarrayContext``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track registered xarray Datasets so XarrayDataFrame can recover
        # defaults (dimension_columns) and metadata (var/dataset attrs,
        # non-dim coords, dim-coord dtype) that the forward pivot drops.
        # Keys are the fully-qualified table names users will reference
        # in SQL (e.g. ``"air"`` for a uniform-dim Dataset, or
        # ``"era5.surface"`` for one entry from a multi-dim-group split).
        self._registered_datasets: dict[str, xr.Dataset] = {}
        # With pyproj installed, every context speaks CRS out of the box:
        # reproject(x, y, src_crs, dst_crs), à la PostGIS ST_Transform.
        if _proj is not None:
            _proj.register(self)

    def from_dataset(
        self,
        name: str,
        input_table: xr.Dataset,
        *,
        table_names: TableNames = None,
        chunks: Chunks = None,
    ):
        """Register an xarray Dataset as one or more queryable SQL tables.

        When all data variables share the same dimensions, the dataset is
        registered as a single table named ``name``. When variables have
        differing dimensions (e.g. some on a 3D grid and others on a 4D
        grid), the dataset is split into one table per dimension group.
        The tables are registered under a SQL schema (namespace) named
        ``name`` and named ``<dim1>_<dim2>_...`` by default::

            ctx.from_dataset('era5', ds, chunks={'time': 24})
            # registers tables: 'era5.time_lat_lon' and
            #                   'era5.time_lat_lon_level'
            ctx.sql('SELECT AVG(temperature_2m) FROM era5.time_lat_lon')

        Use ``table_names`` to override the name for specific dimension
        tuples::

            ctx.from_dataset(
                'era5', ds,
                table_names={('time', 'lat', 'lon'): 'surface'},
            )
            ctx.sql('SELECT * FROM era5.surface')

        A Dataset is read lazily, chunk by chunk, as queries run, unless it
        is already in memory (NumPy arrays, not Dask or a lazily opened
        file) and at most 64 MiB as a table. Such a Dataset is registered
        as a native DataFusion in-memory table, which is faster to scan
        many times, as a query does when it reads a table in several
        places. The table shares its data variables' memory with the
        Dataset rather than copying it, so, as with a lazy table, changing
        the Dataset's values in place changes what queries return. The one
        exception is NaN, which a query reads as NULL: the table records
        which values are NaN when it is registered. So an in-place write
        must keep the NaNs where they were, or the Dataset must be
        registered again. The table adds a column per dimension, one value
        per row.

        For datasets with non-Gregorian cftime coordinates (e.g. 360_day,
        julian), a ``cftime()`` scalar UDF is automatically registered so
        you can write ergonomic SQL filters::

            ctx.from_dataset("ds360", ds, chunks={"time": 6})
            ctx.sql("SELECT * FROM ds360 WHERE time >= cftime('2000-07-01')")

        .. note::

            Only one ``cftime()`` UDF is registered per context, using the
            units and calendar of the *first* non-Gregorian coordinate
            encountered. If you register multiple datasets with *different*
            non-Gregorian calendars (e.g. one 360_day and one julian), the
            UDF from the first registration will be used for all subsequent
            ``cftime()`` calls and may produce incorrect offsets for the
            other dataset. In that case, create a separate ``Context``
            for each calendar.

        Args:
            name: The SQL identifier under which the dataset is registered.
                For datasets with uniform dimensions, this is the table
                name. For datasets with mixed dimensions, this is the name
                of a SQL schema (namespace) containing one table per
                dimension group.
            input_table: An xarray Dataset.
            table_names: Optional mapping from dimension tuples to custom
                table names within the schema, used when the dataset has
                variables with differing dimensions.
            chunks: Xarray-like chunks specification. If not provided, uses
                the Dataset's existing chunks, or for an unchunked Dataset
                its storage chunks (see [register][xarray_sql.register]).

        Returns:
            self, to allow chaining.
        """
        groups = group_vars_by_dims(input_table)
        names = resolve_table_names(input_table, table_names)

        # Materialise dim coordinates once and share across every sub-table.
        # For Zarr-backed parents (e.g. ARCO-ERA5 on GCS) this saves one
        # network round-trip per dim per dim-group.
        coord_arrays = shared_coord_arrays(input_table)

        if len(groups) <= 1:
            self._registered_datasets[name] = input_table
            return self._from_dataset(
                name, input_table, chunks, coord_arrays=coord_arrays
            )

        schema = Schema.memory_schema(self)
        self.catalog().register_schema(name, schema)

        for dims, var_names in groups.items():
            sub_name = names[dims]
            sub_ds = input_table[var_names]
            self._from_dataset(
                sub_name,
                sub_ds,
                chunks,
                schema=schema,
                coord_arrays=coord_arrays,
            )
            # Track the fully-qualified name so XarrayDataFrame metadata
            # recovery can find this Dataset on round-trip.
            self._registered_datasets[f"{name}.{sub_name}"] = sub_ds

        return self

    def _from_dataset(
        self,
        table_name: str,
        input_table: xr.Dataset,
        chunks: Chunks = None,
        schema: Schema | None = None,
        coord_arrays: dict | None = None,
    ):
        """Register a Dataset as a single SQL table.

        Registers a top-level table by default, or a table inside ``schema``
        (a SQL namespace) when one is given.
        """
        register = (
            self.register_table if schema is None else schema.register_table
        )
        partitions = _in_memory_partitions(input_table, chunks)
        if partitions is None:
            table = read_xarray_table(
                input_table, chunks, coord_arrays=coord_arrays
            )
        else:
            table = self._in_memory_table(partitions)
        register(table_name, table)
        self._maybe_register_cftime_udf(input_table)
        return self

    def _in_memory_table(self, partitions: list[list[pa.RecordBatch]]):
        """A native DataFusion in-memory table of ``partitions``."""
        # The in-memory table is reachable from Python only through a
        # registered name, so register it under a placeholder, take the
        # table, and drop the placeholder. The table, not a view of it, is
        # what the caller registers, so a query's plan scans the caller's
        # name, which ddx needs to find the table it differentiates.
        placeholder = f"_xarray_sql_{uuid.uuid4().hex}"
        self.register_record_batches(placeholder, partitions)
        table = self.catalog().schema().table(placeholder)
        self.deregister_table(placeholder)
        return table

    def _maybe_register_cftime_udf(self, ds: xr.Dataset) -> None:
        """Auto-register a cftime() UDF for non-Gregorian cftime coordinates."""
        for coord_name in ds.dims:
            if cft.is_cftime_index(ds, coord_name):
                units, cal = cft.encoding(ds, coord_name)
                if not cft.is_gregorian_like(cal):
                    self.register_udf(cft.make_cftime_udf(units, cal))
                    break  # One UDF per context is enough.

    def sql(self, query: str, *args, **kwargs) -> XarrayDataFrame:
        """Run a SQL query, returning an [XarrayDataFrame][xarray_sql.ds.XarrayDataFrame] wrapper.

        Identical to ``datafusion.SessionContext.sql`` except the returned
        object wraps the DataFusion DataFrame. The wrapper exposes
        ``.to_pandas()`` (unchanged), forwards every other DataFusion
        method via ``__getattr__``, and adds
        ``.to_dataset(dimension_columns=[...])`` for round-tripping the
        result back to an ``xr.Dataset``.

        Args:
            query: A SQL query string.
            *args: Forwarded to ``SessionContext.sql``.
            **kwargs: Forwarded to ``SessionContext.sql``.

        Returns:
            An [XarrayDataFrame][xarray_sql.ds.XarrayDataFrame] wrapping the DataFusion DataFrame.
        """
        inner = super().sql(query, *args, **kwargs)
        return XarrayDataFrame(inner, templates=self._registered_datasets)


XarrayContext = Context
"""The original name of [Context][xarray_sql.Context], kept for compatibility."""
