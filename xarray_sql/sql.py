import time
import uuid

import numpy as np
import pyarrow as pa
import xarray as xr
from datafusion import SessionConfig, SessionContext
from datafusion.catalog import Schema
from datafusion.expr import TableScan
from datafusion.plan import ExecutionPlan, LogicalPlan
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

    def __init__(self, config: SessionConfig | None = None, *args, **kwargs):
        if config is None:
            # Split a query's work across partitions only where enough rows
            # reach it, by DataFusion's estimate. DataFusion otherwise splits
            # every input whose row count is an estimate, as it is after a
            # join, a filter or an aggregate, however few rows that is, which
            # makes a query of a few thousand rows two to three times slower
            # to run. A config given is used as it is.
            #
            # A bare SessionConfig turns off the information schema, which
            # SessionContext() turns on; otherwise the two are the same.
            config = (
                SessionConfig()
                .with_information_schema(True)
                .set(
                    "datafusion.execution."
                    "use_row_number_estimates_to_optimize_partitioning",
                    "true",
                )
            )
        super().__init__(config, *args, **kwargs)
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

        A Dataset is read lazily, chunk by chunk, as queries run. One
        already in memory (NumPy, not Dask or a lazily opened file) and at
        most 64 MiB as a table is instead a DataFusion in-memory table,
        which is faster to scan many times. Either way, changing values in
        place changes what queries return, as long as no value becomes or
        stops being NaN (which reads as NULL) in an in-memory table.

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
        return self._wrap(super().sql(query, *args, **kwargs))

    def deregister_table(self, name: str) -> None:
        """Remove a table from the session, and forget the Dataset
        registered under its name."""
        super().deregister_table(name)
        self._registered_datasets.pop(name, None)

    def prepare(self, query: str) -> "PreparedQuery":
        """Plan a query once, to run it many times.

        A training loop runs the same query on every step, and only the
        values in its tables change. DataFusion plans a query each time it
        runs, and for a query over small tables, planning can take far
        longer than running. ``prepare`` plans the query once, and calling
        the result runs that plan, as ``jax.jit`` traces a function once
        and runs the compiled function after that. A call can first write
        new values into the query's tables::

            step = ctx.prepare(
                "SELECT x.i, SUM(x.v * w.v) AS y "
                "FROM x JOIN w ON x.j = w.j GROUP BY x.i"
            )
            for weights in trajectory:  # Datasets shaped like w's
                rows = step(w=weights)

        See [PreparedQuery][xarray_sql.PreparedQuery] for what a call
        can write.

        Args:
            query: A SQL query string.

        Returns:
            A [PreparedQuery][xarray_sql.PreparedQuery]; call it to run
            the query.
        """
        return PreparedQuery(self, query)

    def _write(self, writes: dict[str, xr.Dataset]) -> None:
        """Write each Dataset's values into the arrays of the Dataset
        registered under its name, all or none of them."""
        copies = []
        for name, ds in writes.items():
            registered = self._registered_datasets[name]
            if set(ds.data_vars) != set(registered.data_vars):
                raise ValueError(
                    f"{name!r} has the data variables "
                    f"{sorted(registered.data_vars)}, and the new Dataset "
                    f"has {sorted(ds.data_vars)}"
                )
            for dim in registered.dims:
                if dim not in ds.dims or not ds[dim].equals(registered[dim]):
                    raise ValueError(
                        f"the new Dataset for {name!r} has other coordinates "
                        f"along {dim!r}; register it again, and prepare the "
                        "query again"
                    )
            for var in registered.data_vars:
                old, new = registered[var].variable, ds[var].variable
                # Dims are matched by name, as xarray matches them.
                if set(new.dims) != set(old.dims) or new.dtype != old.dtype:
                    raise ValueError(
                        f"{name}.{var} is {old.dtype} over {old.dims}, and "
                        f"the new values are {new.dtype} over {new.dims}"
                    )
                target = old.data
                # A table reads a NumPy array's memory, shared or on every
                # scan; a Dask array or a lazily opened file has none to
                # write into.
                if not (
                    isinstance(target, np.ndarray) and target.flags.writeable
                ):
                    raise ValueError(
                        f"{name}.{var} isn't a writable NumPy array; register "
                        "the new Dataset instead, and prepare the query again"
                    )
                values = new.transpose(*old.dims).values
                # A table works out which values are NULL (NaN or NaT) when
                # it is registered, so a write must keep them in place.
                missing = _missing(values)
                if missing is not None and not np.array_equal(
                    missing, _missing(target)
                ):
                    raise ValueError(
                        f"the new values of {name}.{var} have NaN or NaT in "
                        "other places; xarray-sql reads them as NULL, so "
                        "register the new Dataset instead, and prepare the "
                        "query again"
                    )
                copies.append((registered[var], ds[var]))
        # Assigning into a registered DataArray writes into its array, which
        # the table shares.
        for target, values in copies:
            target[...] = values

    def _wrap(self, inner) -> XarrayDataFrame:
        """Wrap a DataFusion DataFrame so it can round-trip to xarray."""
        return XarrayDataFrame(inner, templates=self._registered_datasets)


class PreparedQuery:
    """A query planned once by [Context.prepare][xarray_sql.Context.prepare].

    ``query()`` runs it on the values its tables hold now, and returns its
    rows as a ``pyarrow.Table``. ``query(w=ds)`` first writes ``ds``'s
    values into the Dataset registered as ``w``, as
    ``w_ds["v"][...] = ds["v"]`` would, so other queries of ``w`` see them
    too. ``ds`` must match ``w_ds`` in all but its values: the same data
    variables, dtypes, dims and coordinates, and NaN or NaT in the same
    places, since xarray-sql reads them as NULL.

    To change anything else, register the new Dataset and prepare the
    query again. A call raises if a table the query reads has been
    registered again since it was prepared. Functions such as ``now()``
    are evaluated once, when the query is prepared.
    """

    def __init__(self, ctx: Context, query: str):
        self._ctx = ctx
        self._query = query
        start = time.perf_counter()
        df = SessionContext.sql(ctx, query)
        plan = df.execution_plan()
        planning = time.perf_counter() - start
        self._schema = df.schema()
        # The Datasets the query reads, as they are registered now.
        self._registered = {
            name: ctx._registered_datasets[name]
            for name in _tables_read(df.logical_plan())
            if name in ctx._registered_datasets
        }
        # A plan keeps state from a run (a hash join keeps the table it
        # built), so each call runs a fresh copy of it, decoded from its
        # serialized form. Copying a plan copies its in-memory tables, so
        # over large ones it's slower than planning again; a plan that reads
        # a lazy table, or is nested deeper than protobuf decodes, can't be
        # copied at all. Either way, each call plans the query again.
        self._plan: ExecutionPlan | None = None
        start = time.perf_counter()
        try:
            _copy(ctx, plan)
        except Exception:  # noqa: BLE001 - DataFusion raises a bare Exception
            return
        if time.perf_counter() - start < planning:
            self._plan = plan

    def __call__(self, **tables: xr.Dataset) -> pa.Table:
        """Run the query, after writing each Dataset given into the table
        of its name.

        Args:
            **tables: A Dataset for each table to write new values into.

        Returns:
            The query's rows.
        """
        for name in tables:
            if name not in self._registered:
                raise ValueError(
                    f"the query reads no Dataset registered as {name!r} "
                    "when it was prepared"
                )
        for name, registered in self._registered.items():
            if self._ctx._registered_datasets.get(name) is not registered:
                raise ValueError(
                    f"{name!r} has been registered again since the query was "
                    "prepared; prepare the query again"
                )
        self._ctx._write(tables)
        if self._plan is None:
            return SessionContext.sql(self._ctx, self._query).to_arrow_table()
        plan = _copy(self._ctx, self._plan)
        batches = [
            batch.to_pyarrow()
            for partition in range(plan.partition_count)
            for batch in self._ctx.execute(plan, partition)
        ]
        return pa.Table.from_batches(batches, self._schema)


def _copy(ctx: SessionContext, plan: ExecutionPlan) -> ExecutionPlan:
    """A copy of ``plan`` with none of its state from a run, reading its
    in-memory tables' values as they are now."""
    # TODO(#283): reset the plan's state with datafusion-python's binding of
    # reset_plan_states, once it has one, rather than decode it again.
    # `plan` itself never runs, so it has no state to leave behind.
    return ExecutionPlan.from_bytes(ctx, plan.to_bytes())


def _tables_read(plan: LogicalPlan) -> set[str]:
    """The names of the tables ``plan`` scans, through any views."""
    names = set()
    stack = [plan]
    while stack:
        node = stack.pop()
        variant = node.to_variant()
        if isinstance(variant, TableScan):
            names.add(variant.table_name())
        stack.extend(node.inputs())
    return names


def _missing(values: np.ndarray) -> np.ndarray | None:
    """Where ``values`` holds NaN or NaT, which xarray-sql reads as NULL, or
    ``None`` for a dtype that holds neither."""
    if values.dtype.kind in "fc":
        return np.isnan(values)
    if values.dtype.kind in "mM":
        return np.isnat(values)
    return None


XarrayContext = Context
"""The original name of [Context][xarray_sql.Context], kept for compatibility."""
