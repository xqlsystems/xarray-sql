import copy
import re
import time
import uuid
from collections.abc import Callable, Mapping
from types import MappingProxyType, ModuleType

import numpy as np
import pyarrow as pa
import xarray as xr
from datafusion import (
    AggregateUDF,
    ScalarUDF,
    SessionConfig,
    SessionContext,
    WindowUDF,
)
from datafusion.catalog import Schema
from datafusion.plan import ExecutionPlan, LogicalPlan

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

_ad: ModuleType | None
try:  # ddxdb is an optional dependency (`pip install xarray-sql[ml]`).
    from . import ad

    _ad = ad
except ImportError:  # pragma: no cover - depends on the environment
    _ad = None


class Context(SessionContext):
    """A datafusion `SessionContext` that also supports `xarray.Dataset`s.

    Also available under its original name, ``XarrayContext``.
    ``threads=`` sets DataFusion's target parallel partitions, overriding
    that setting in an explicit ``config``. The default is the CPU count.
    It is a concurrency target, rather than a fixed number of OS threads.
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        *args,
        threads: int | None = None,
        **kwargs,
    ):
        if threads is not None and threads < 1:
            raise ValueError("threads must be positive")
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
        if threads is not None:
            # SessionConfig's with_ methods set the setting on the config
            # they're called on, so set it on a copy: the caller's config is
            # theirs to use again. The copy shares the Rust config, which
            # with_target_partitions replaces rather than changes.
            config = copy.copy(config).with_target_partitions(threads)
        super().__init__(config, *args, **kwargs)
        # Track registered xarray Datasets so XarrayDataFrame can recover
        # defaults (dimension_columns) and metadata (var/dataset attrs,
        # non-dim coords, dim-coord dtype) that the forward pivot drops.
        # Keys are the fully-qualified table names users will reference
        # in SQL (e.g. ``"air"`` for a uniform-dim Dataset, or
        # ``"era5.surface"`` for one entry from a multi-dim-group split).
        self._registered_datasets: dict[str, xr.Dataset] = {}
        # How many times each name's table or view has been removed or
        # replaced. DataFusion refuses to register a name that's taken, so a
        # table registered again from Python was deregistered first; SQL
        # that replaces or drops a table or a view counts too (see sql()).
        self._replacements: dict[str, int] = {}
        # The function last registered under each name through this
        # Context's register_ methods, which pyproj's reproject() below is.
        self._functions: dict[str, ScalarUDF | AggregateUDF | WindowUDF] = {}
        # With pyproj installed, every context speaks CRS out of the box:
        # reproject(x, y, src_crs, dst_crs), à la PostGIS ST_Transform.
        if _proj is not None:
            _proj.register(self)
        # With ddxdb installed, .sql() also takes gradients: grad(), vjp()
        # and jvp() in a select list, and ddx_stop_gradient().
        if _ad is not None:
            _ad.register(self)

    def with_extensions(self, *extensions) -> "Context":
        """Install DataFusion extension bundles, retaining xarray support.

        Returns another handle on the same session, with this Context's
        Dataset metadata and prepared-query invalidation state shared.
        Extension codecs belong to the returned handle, so use it for
        subsequent queries, as with DataFusion's own method.
        """
        session = super().with_extensions(*extensions)
        derived = object.__new__(type(self))
        derived.__dict__.update(self.__dict__)
        derived.ctx = session.ctx
        return derived

    @property
    def datasets(self) -> Mapping[str, xr.Dataset]:
        """The Dataset registered under each table's name, read-only.

        Each value is the very Dataset the table reads, not a copy:
        [write][xarray_sql.Context.write] updates it in place, so it holds
        the new values as soon as ``write`` returns. Changing its arrays
        any other way skips ``write``'s checks. A Dataset split into a
        table per group of dims is listed under each table's qualified
        name, such as ``"era5.surface"``. Tables registered any other way,
        such as with ``SessionContext.register_record_batches``, aren't
        listed.
        """
        return MappingProxyType(self._registered_datasets)

    @property
    def replacements(self) -> Mapping[str, int]:
        """How many times each table or view has been replaced or dropped,
        by name, read-only.

        It counts [deregister_table][xarray_sql.Context.deregister_table],
        which registering a name again needs first, and a statement run
        with [sql][xarray_sql.Context.sql] that replaces or drops a table
        or a view: ``CREATE OR REPLACE TABLE``, ``CREATE OR REPLACE VIEW``,
        ``DROP TABLE`` or ``DROP VIEW``. A name never replaced isn't in it.
        Names leave out DataFusion's default catalog and schema, as
        [tables_read][xarray_sql.Context.tables_read] does. Writing new
        values with [write][xarray_sql.Context.write] doesn't count.

        A change made through ``SessionContext``'s own methods, such as
        ``SessionContext.sql(ctx, ...)``, or from Rust, isn't counted.
        """
        return MappingProxyType(self._replacements)

    @property
    def registered_functions(
        self,
    ) -> Mapping[str, ScalarUDF | AggregateUDF | WindowUDF]:
        """The function last registered under each name through this
        Context, read-only.

        It holds each ``ScalarUDF``, ``AggregateUDF`` and ``WindowUDF``
        registered with [register_udf][xarray_sql.Context.register_udf],
        [register_udaf][xarray_sql.Context.register_udaf] or
        [register_udwf][xarray_sql.Context.register_udwf], less those
        deregistered since. Registering a name again replaces its value
        with the new object, so comparing objects shows whether a name was
        registered again. Unlike DataFusion's ``udfs()``, it leaves out the
        built-in functions, so it shows when a name such as ``sum`` means a
        function of the user's.

        Keys are each function's name as it was registered. SQL lowers an
        unquoted name before looking it up, so compare lowered names to
        match what a query calls.

        It doesn't see a function registered from Rust: by an extension
        bundle in [with_extensions][xarray_sql.Context.with_extensions],
        by ``enable_spark_functions``, or by ddx, the package that
        differentiates SQL, whose ``ddx_stop_gradient`` every Context has
        with ``pip install "xarray-sql[ml]"``. Nor does it see one
        registered through
        ``SessionContext``'s own methods.
        """
        return MappingProxyType(self._functions)

    def register_udf(self, udf: ScalarUDF) -> None:
        """Register a scalar function, as ``SessionContext.register_udf``
        does, and record it in
        [registered_functions][xarray_sql.Context.registered_functions]."""
        super().register_udf(udf)
        self._functions[udf.name] = udf

    def register_udaf(self, udaf: AggregateUDF) -> None:
        """Register an aggregate function, as
        ``SessionContext.register_udaf`` does, and record it in
        [registered_functions][xarray_sql.Context.registered_functions]."""
        super().register_udaf(udaf)
        self._functions[udaf.name] = udaf

    def register_udwf(self, udwf: WindowUDF) -> None:
        """Register a window function, as ``SessionContext.register_udwf``
        does, and record it in
        [registered_functions][xarray_sql.Context.registered_functions]."""
        super().register_udwf(udwf)
        self._functions[udwf.name] = udwf

    def deregister_udf(self, name: str) -> None:
        """Remove a scalar function, and forget it in
        [registered_functions][xarray_sql.Context.registered_functions]."""
        super().deregister_udf(name)
        self._forget_function(name, ScalarUDF)

    def deregister_udaf(self, name: str) -> None:
        """Remove an aggregate function, and forget it in
        [registered_functions][xarray_sql.Context.registered_functions]."""
        super().deregister_udaf(name)
        self._forget_function(name, AggregateUDF)

    def deregister_udwf(self, name: str) -> None:
        """Remove a window function, and forget it in
        [registered_functions][xarray_sql.Context.registered_functions]."""
        super().deregister_udwf(name)
        self._forget_function(name, WindowUDF)

    def _forget_function(self, name: str, kind: type) -> None:
        """Forget the function of ``kind`` recorded as ``name``, which a
        function of another kind of the same name leaves alone."""
        if isinstance(self._functions.get(name), kind):
            del self._functions[name]

    def tables_read(self, plan: LogicalPlan) -> set[str]:
        """The names of the tables and views a query's plan reads.

        It finds them in the plan's subqueries too (``IN``, ``EXISTS`` and
        scalar subqueries), and lists both a view's own name and the tables
        under it. Names leave out DataFusion's default catalog and schema,
        so a query of ``public.air`` reads ``"air"``, the key it has in
        [datasets][xarray_sql.Context.datasets].

        Views are told apart from names given with ``AS`` by looking them
        up in this Context's catalog as it is now, so call this before a
        view the plan reads is dropped. A name given with ``AS`` that's
        also a view's is listed too.

        Args:
            plan: A query's logical plan, such as
                ``ctx.sql(query).logical_plan()``.

        Returns:
            The names of the tables and views ``plan`` reads.
        """
        return {
            _unqualified(name) for name in _SCAN.findall(plan.display_indent())
        } | self._views_read(plan)

    def _views_read(self, plan: LogicalPlan) -> set[str]:
        """The names of the views ``plan`` reads, in its subqueries too,
        without DataFusion's default catalog and schema."""
        # DataFusion plans a view as its query, under its name, as it plans
        # ``FROM t AS x`` under ``x``; only the catalog tells them apart.
        return {
            _unqualified(name)
            for name in _ALIAS.findall(plan.display_indent())
            if self._is_view(name)
        }

    def _is_view(self, name: str) -> bool:
        """Whether ``name``, as a plan's display prints it, is a view in
        this Context's catalog."""
        *path, table = name.split(".")
        try:
            catalog = self.catalog(*path[:-1])
            schema = catalog.schema(*path[-1:])
            return bool(schema.table(table).kind == "view")
        except Exception:  # noqa: BLE001 - DataFusion raises a bare Exception
            return False

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
            self._from_dataset(
                name, input_table, chunks, coord_arrays=coord_arrays
            )
            # Recorded once DataFusion has registered the table, which it
            # refuses to do under a name that's taken.
            self._registered_datasets[name] = input_table
            return self

        schema = Schema.memory_schema(self)
        # register_schema replaces a schema of the same name, tables and
        # all, so each of its tables counts as replaced.
        catalog = self.catalog()
        if name in catalog.schema_names():
            for table in catalog.schema(name).table_names():
                self._forget(f"{name}.{table}")
        catalog.register_schema(name, schema)

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
            register(table_name, table)
        elif schema is None:
            # Uniform Datasets can register their shared Arrow buffers
            # directly, without a temporary catalog entry and lookup.
            self.register_record_batches(table_name, partitions)
        else:
            register(table_name, self._in_memory_table(partitions))
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
        # A placeholder isn't a table anyone replaced.
        SessionContext.deregister_table(self, placeholder)
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

        With the ml extension installed (``pip install xarray-sql[ml]``),
        the query may also call ``grad``, ``vjp`` and ``jvp`` (see
        [xarray_sql.ad][]).

        Args:
            query: A SQL query string.
            *args: Forwarded to ``SessionContext.sql``.
            **kwargs: Forwarded to ``SessionContext.sql``.

        Returns:
            An [XarrayDataFrame][xarray_sql.ds.XarrayDataFrame] wrapping the DataFusion DataFrame.
        """
        # DataFusion runs a statement that replaces or drops a table or a
        # view as it plans it, without deregistering it.
        replaced = _table_replaced(query)
        existed = replaced is not None and self.table_exist(replaced)
        if _ad is not None and _ad.calls_ad(query):
            if args or kwargs:
                raise TypeError(
                    "a query that calls grad, vjp or jvp takes no other "
                    "arguments"
                )
            df = _ad.sql(self, query)
        else:
            df = super().sql(query, *args, **kwargs)
        if replaced is not None and existed:
            self._forget(_resolved(replaced))
        return self._wrap(df)

    def deregister_table(self, name: str) -> None:
        """Remove a table from the session, and forget the Dataset
        registered under its name."""
        super().deregister_table(name)
        self._forget(name)

    def _forget(self, name: str) -> None:
        """Forget the Dataset registered as ``name``, whose table has been
        removed or replaced."""
        name = _unqualified(name)
        self._registered_datasets.pop(name, None)
        self._replacements[name] = self._replacements.get(name, 0) + 1

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

        A query that calls ``grad``, ``vjp`` or ``jvp`` keeps the programs
        ddx builds for them instead. See
        [PreparedQuery][xarray_sql.PreparedQuery] for what a call can
        write.

        Args:
            query: A SQL query string.

        Returns:
            A [PreparedQuery][xarray_sql.PreparedQuery]; call it to run
            the query.
        """
        return PreparedQuery(self, query)

    def write(self, tables: Mapping[str, xr.Dataset]) -> None:
        """Write new values into registered Datasets' arrays, in place.

        ``ctx.write({"w": ds})`` writes ``ds``'s values into the Dataset
        registered as ``w``, as ``w_ds["v"][...] = ds["v"]`` would, so every
        query of ``w`` reads them from then on, and so does
        ``ctx.datasets["w"]``. It checks every table before it writes any,
        and writes all of them or none. A Dataset must match the one it's
        written into in all but its values: the same data variables, dtypes,
        dims and coordinates, and NaN or NaT in the same places, since
        xarray-sql reads them as NULL. The registered arrays must be
        writable NumPy arrays. To change anything else, register the new
        Dataset instead.

        Writing doesn't count in
        [replacements][xarray_sql.Context.replacements], and
        ``write({})`` does nothing.

        Args:
            tables: A Dataset of new values for each table, by the name it
                has in [datasets][xarray_sql.Context.datasets].

        Raises:
            ValueError: If no Dataset is registered under a name, or a
                Dataset doesn't match the one registered under its name.
        """
        copies = []
        for name, ds in tables.items():
            registered = self._registered_datasets.get(name)
            if registered is None:
                raise ValueError(f"no Dataset is registered as {name!r}")
            if set(ds.data_vars) != set(registered.data_vars):
                raise ValueError(
                    f"{name!r} has the data variables "
                    f"{sorted(registered.data_vars)}, and the new Dataset "
                    f"has {sorted(ds.data_vars)}"
                )
            for coord in set(registered.coords) | set(ds.coords):
                if coord not in ds.coords or not ds[coord].equals(
                    registered.coords.get(coord)
                ):
                    raise ValueError(
                        f"the new Dataset for {name!r} has other coordinates "
                        f"({coord!r}); register it again, and prepare the "
                        "query again"
                    )
            for var in registered.data_vars:
                old, new = registered[var].variable, ds[var].variable
                # Dims are matched by name, as xarray matches them.
                if new.sizes != old.sizes or new.dtype != old.dtype:
                    raise ValueError(
                        f"{name}.{var} is {old.dtype} over "
                        f"{dict(old.sizes)}, and the new values are "
                        f"{new.dtype} over {dict(new.sizes)}"
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
    ``ctx.write({"w": ds})`` does, so other queries of ``w`` see them too.
    [Context.write][xarray_sql.Context.write] says what ``ds`` must match.

    To change anything else, register the new Dataset and prepare the
    query again. A call raises if a table the query reads, in a subquery
    too, has been registered again since it was prepared, or a view it
    reads has been replaced or dropped. Functions such as ``now()``
    are evaluated once, when the query is prepared.

    A query that calls ``grad``, ``vjp`` or ``jvp`` keeps the programs ddx
    builds for those calls, rather than a plan. Preparing it runs it once,
    to build them and to find the tables it reads. A table it reads may also
    be registered again under the same name, with the same columns and types.
    A call raises if a view it reads has been replaced or dropped.
    """

    def __init__(self, ctx: Context, query: str):
        self._ctx = ctx
        self._gradient: Callable[[Callable[[], None]], pa.Table] | None = None
        if _ad is not None and _ad.calls_ad(query):
            self._gradient, plans = _ad._prepare(ctx, query)
            # A call may write any Dataset the query reads, which leaves out
            # the tables ddx wrote. It finds each by name, so one registered
            # again is written too.
            self._writable = set().union(*map(ctx.tables_read, plans)) & set(
                ctx.datasets
            )
            # ddx built its programs from each view's SQL as it was then.
            self._read = {
                name: ctx.replacements.get(name, 0)
                for name in set().union(*map(ctx._views_read, plans))
            }
            return
        start = time.perf_counter()
        df = SessionContext.sql(ctx, query)
        plan = df.execution_plan()
        planning = time.perf_counter() - start
        self._schema = df.schema()
        # Planned again from its optimized form, the query reads the tables
        # it was prepared with, and functions such as now() keep the values
        # they were folded into.
        self._optimized = df.optimized_logical_plan()
        # Each table and view the query reads, by how often its name had
        # been replaced; and of those, the Datasets a call may write into.
        read = ctx.tables_read(df.logical_plan())
        self._read = {name: ctx.replacements.get(name, 0) for name in read}
        self._writable = read & set(ctx.datasets)
        # A plan keeps state from a run (a hash join keeps the table it
        # built), so each call runs a fresh copy of it, decoded from its
        # serialized form. Copying a plan copies its in-memory tables, so
        # over large ones it's slower than planning again. A plan that reads
        # a lazy table, has an extension's nodes, or is nested deeper than
        # protobuf decodes can't be copied at all: DataFusion's own codec
        # serializes it, since an extension's codec may keep state from a
        # run. Either way, each call plans the query again.
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
            if name not in self._writable:
                raise ValueError(
                    f"the query reads no Dataset registered as {name!r} "
                    "when it was prepared"
                )
        for name, replacements in self._read.items():
            if self._ctx.replacements.get(name, 0) != replacements:
                raise ValueError(
                    f"{name!r} has been registered again, replaced or "
                    "dropped since the query was prepared; prepare the "
                    "query again"
                )
        if self._gradient is not None:
            return self._gradient(lambda: self._ctx.write(tables))
        self._ctx.write(tables)
        if self._plan is None:
            df = self._ctx.create_dataframe_from_logical_plan(self._optimized)
            return df.to_arrow_table()
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
    # TODO(xqlsystems/einfold#159): serialize with the session's codecs
    # too, once EinFold's copies read their tables' current values.
    return ExecutionPlan.from_bytes(ctx, plan.to_bytes())


# A plan's display prints each of its nodes on a line of its own, with each
# subquery's plan under the node whose expression holds it. The display is
# read, rather than the nodes, since a node's inputs leave out its
# subqueries, and datafusion-python's expressions don't reach every
# subquery's plan (``EXISTS``'s, for one). DataFusion prints a scan's
# options after its table's name, and a view's plan under its name.
_SCAN = re.compile(
    r"^ *TableScan: (.+?)"
    r"(?: (?:projection|partial_filters|full_filters|fetch)=.*)?$",
    re.MULTILINE,
)
_ALIAS = re.compile(r"^ *SubqueryAlias: (.+)$", re.MULTILINE)


# Whitespace or comments, which SQL reads between any two words.
_GAP = r"(?:\s|--[^\n]*|/\*.*?\*/)"
# A name's part: quoted, which keeps its case, or not, which DataFusion
# lowers.
_PART = r'"[^"]+"|\w+'
_TABLE_DDL = re.compile(
    rf"{_GAP}*"
    rf"(?:CREATE{_GAP}+OR{_GAP}+REPLACE{_GAP}+(?:(?:EXTERNAL{_GAP}+)?TABLE|VIEW)"
    rf"|DROP{_GAP}+(?:TABLE|VIEW))"
    rf"{_GAP}+(?:IF{_GAP}+EXISTS{_GAP}+)?((?:{_PART})(?:\.(?:{_PART}))*)",
    re.IGNORECASE | re.DOTALL,
)


def _table_replaced(query: str) -> str | None:
    """The name of the table or view ``query`` replaces or drops, as it's
    written, or ``None`` for any other statement."""
    match = _TABLE_DDL.match(query)
    return None if match is None else match.group(1)


def _resolved(name: str) -> str:
    """``name``, written in SQL, as a plan's display prints it: each part
    unquoted, or lowered if it wasn't quoted, joined by dots."""
    return ".".join(
        part[1:-1] if part.startswith('"') else part.lower()
        for part in re.findall(_PART, name)
    )


def _unqualified(name: str) -> str:
    """``name`` without DataFusion's default catalog and schema, which a
    query may name or leave out: ``datafusion.public.v`` and ``public.v``
    are both ``v``."""
    for prefix in ("datafusion.public.", "public."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


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
