//! An Arrow Flight SQL endpoint over lazily registered xarray tables.
//!
//! Remote clients (the ADBC Flight SQL driver, the Flight SQL JDBC/ODBC
//! drivers, and anything else that speaks the protocol) send SQL; a
//! native DataFusion session plans it against the same
//! `PrunableStreamingTable` providers the in-process engine uses, so
//! partition pruning, projection pushdown, and exact statistics carry
//! over unchanged. Results stream back as Arrow record batches, and
//! source chunks are read only while a query executes.
//!
//! Plain Arrow Flight clients (e.g. ClickHouse's `arrowFlight` table
//! function) are served too: a path descriptor names a table, as in
//! `["weather"]` or `["era5.surface"]`, or is itself a `SELECT`/`WITH`
//! query, which is how such clients get pruning they cannot push down.
//!
//! The service is stateless: a statement's ticket and a prepared
//! statement's handle are the SQL text itself, re-planned on use. The
//! session is read-only: DDL (which could read or write the server's
//! filesystem through `CREATE EXTERNAL TABLE` / `COPY`), DML, and other
//! statements are rejected before planning.

use std::net::TcpListener;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::thread::JoinHandle;
use std::time::Duration;

use arrow::datatypes::Schema;
use arrow::ipc::writer::IpcWriteOptions;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::error::FlightError;
use arrow_flight::flight_descriptor::DescriptorType;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::sql::metadata::{SqlInfoData, SqlInfoDataBuilder};
use arrow_flight::sql::server::FlightSqlService;
use arrow_flight::sql::{
    ActionClosePreparedStatementRequest, ActionCreatePreparedStatementRequest,
    ActionCreatePreparedStatementResult, Any, Command, CommandGetCatalogs, CommandGetDbSchemas,
    CommandGetSqlInfo, CommandGetTables, CommandPreparedStatementQuery, CommandStatementQuery,
    ProstMessageExt, SqlInfo, TicketStatementQuery,
};
use arrow_flight::{
    Action, Criteria, Empty, FlightData, FlightDescriptor, FlightEndpoint, FlightInfo,
    HandshakeRequest, IpcMessage, PollInfo, SchemaAsIpc, SchemaResult, Ticket,
};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{MemorySchemaProvider, SchemaProvider};
use datafusion::error::DataFusionError;
use datafusion::execution::context::SQLOptions;
use datafusion::prelude::{DataFrame, SessionContext};
use datafusion::sql::TableReference;
use futures::{stream, Stream, TryStreamExt};
use prost::Message;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::sync::oneshot;
use tonic::transport::server::TcpIncoming;
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

use crate::LazyArrowStreamTable;

type DoGetStream = Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;

static SQL_INFO: LazyLock<SqlInfoData> = LazyLock::new(|| {
    let mut builder = SqlInfoDataBuilder::new();
    builder.append(SqlInfo::FlightSqlServerName, "xarray-sql");
    builder.append(SqlInfo::FlightSqlServerVersion, env!("CARGO_PKG_VERSION"));
    builder.append(SqlInfo::FlightSqlServerArrowVersion, "1.3");
    builder.append(SqlInfo::FlightSqlServerReadOnly, true);
    builder.build().expect("static SqlInfo is valid")
});

fn read_only() -> SQLOptions {
    SQLOptions::new()
        .with_allow_ddl(false)
        .with_allow_dml(false)
        .with_allow_statements(false)
}

fn plan_error(e: DataFusionError) -> Status {
    Status::invalid_argument(e.to_string())
}

fn internal_error(e: impl std::fmt::Display) -> Status {
    Status::internal(e.to_string())
}

fn utf8(bytes: &[u8]) -> Result<&str, Status> {
    std::str::from_utf8(bytes).map_err(|e| Status::invalid_argument(e.to_string()))
}

/// Encode one metadata batch (catalog, schema, and table listings) as a
/// Flight data stream.
fn single_batch_stream(
    schema: Arc<Schema>,
    batch: Result<RecordBatch, FlightError>,
) -> DoGetStream {
    let stream = FlightDataEncoderBuilder::new()
        .with_schema(schema)
        .build(stream::once(async move { batch }))
        .map_err(Status::from);
    Box::pin(stream)
}

/// A FlightInfo whose single endpoint redeems ``ticket`` on this server.
fn flight_info(
    schema: &Schema,
    ticket: impl ProstMessageExt,
    descriptor: FlightDescriptor,
) -> Result<Response<FlightInfo>, Status> {
    let ticket = Ticket::new(ticket.as_any().encode_to_vec());
    let info = FlightInfo::new()
        .try_with_schema(schema)
        .map_err(internal_error)?
        .with_endpoint(FlightEndpoint::new().with_ticket(ticket))
        .with_descriptor(descriptor);
    Ok(Response::new(info))
}

/// The Flight SQL service over one DataFusion session.
struct XarrayFlightSql {
    ctx: SessionContext,
}

impl XarrayFlightSql {
    async fn plan(&self, sql: &str) -> Result<DataFrame, Status> {
        self.ctx
            .sql_with_options(sql, read_only())
            .await
            .map_err(plan_error)
    }

    async fn result_schema(&self, sql: &str) -> Result<Arc<Schema>, Status> {
        Ok(Arc::clone(self.plan(sql).await?.schema().inner()))
    }

    async fn execute(&self, sql: &str) -> Result<Response<DoGetStream>, Status> {
        let frame = self.plan(sql).await?;
        let schema = Arc::clone(frame.schema().inner());
        let batches = frame.execute_stream().await.map_err(internal_error)?;
        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(batches.map_err(|e| FlightError::ExternalError(Box::new(e))))
            .map_err(Status::from);
        Ok(Response::new(Box::pin(stream)))
    }

    /// Every (catalog, schema) pair in the session.
    fn schemas(&self) -> Vec<(String, String, Arc<dyn SchemaProvider>)> {
        let mut out = Vec::new();
        for catalog_name in self.ctx.catalog_names() {
            let Some(catalog) = self.ctx.catalog(&catalog_name) else {
                continue;
            };
            for schema_name in catalog.schema_names() {
                if let Some(schema) = catalog.schema(&schema_name) {
                    out.push((catalog_name.clone(), schema_name, schema));
                }
            }
        }
        out
    }
}

#[tonic::async_trait]
impl FlightSqlService for XarrayFlightSql {
    type FlightService = XarrayFlightSql;

    async fn get_flight_info_statement(
        &self,
        query: CommandStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let schema = self.result_schema(&query.query).await?;
        let ticket = TicketStatementQuery {
            statement_handle: query.query.into_bytes().into(),
        };
        flight_info(&schema, ticket, request.into_inner())
    }

    async fn do_get_statement(
        &self,
        ticket: TicketStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.execute(utf8(&ticket.statement_handle)?).await
    }

    async fn do_action_create_prepared_statement(
        &self,
        query: ActionCreatePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<ActionCreatePreparedStatementResult, Status> {
        let schema = self.result_schema(&query.query).await?;
        let IpcMessage(dataset_schema) = SchemaAsIpc::new(&schema, &IpcWriteOptions::default())
            .try_into()
            .map_err(internal_error)?;
        Ok(ActionCreatePreparedStatementResult {
            prepared_statement_handle: query.query.into_bytes().into(),
            dataset_schema,
            parameter_schema: Default::default(),
        })
    }

    async fn do_action_close_prepared_statement(
        &self,
        _query: ActionClosePreparedStatementRequest,
        _request: Request<Action>,
    ) -> Result<(), Status> {
        // Handles are the SQL text; there is nothing to release.
        Ok(())
    }

    async fn get_flight_info_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let schema = self
            .result_schema(utf8(&query.prepared_statement_handle)?)
            .await?;
        flight_info(&schema, query, request.into_inner())
    }

    async fn do_get_prepared_statement(
        &self,
        query: CommandPreparedStatementQuery,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        self.execute(utf8(&query.prepared_statement_handle)?).await
    }

    async fn get_flight_info_catalogs(
        &self,
        query: CommandGetCatalogs,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let schema = query.into_builder().schema();
        flight_info(&schema, query, request.into_inner())
    }

    async fn do_get_catalogs(
        &self,
        query: CommandGetCatalogs,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        let mut builder = query.into_builder();
        for catalog in self.ctx.catalog_names() {
            builder.append(catalog);
        }
        Ok(Response::new(single_batch_stream(
            builder.schema(),
            builder.build(),
        )))
    }

    async fn get_flight_info_schemas(
        &self,
        query: CommandGetDbSchemas,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let schema = query.clone().into_builder().schema();
        flight_info(&schema, query, request.into_inner())
    }

    async fn do_get_schemas(
        &self,
        query: CommandGetDbSchemas,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        let mut builder = query.into_builder();
        for (catalog, schema, _) in self.schemas() {
            builder.append(catalog, schema);
        }
        Ok(Response::new(single_batch_stream(
            builder.schema(),
            builder.build(),
        )))
    }

    async fn get_flight_info_tables(
        &self,
        query: CommandGetTables,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let schema = query.clone().into_builder().schema();
        flight_info(&schema, query, request.into_inner())
    }

    async fn do_get_tables(
        &self,
        query: CommandGetTables,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        let mut builder = query.into_builder();
        for (catalog, schema_name, schema) in self.schemas() {
            for table_name in schema.table_names() {
                let Some(table) = schema.table(&table_name).await.map_err(internal_error)? else {
                    continue;
                };
                builder
                    .append(
                        &catalog,
                        &schema_name,
                        &table_name,
                        "TABLE",
                        &table.schema(),
                    )
                    .map_err(internal_error)?;
            }
        }
        Ok(Response::new(single_batch_stream(
            builder.schema(),
            builder.build(),
        )))
    }

    async fn get_flight_info_sql_info(
        &self,
        query: CommandGetSqlInfo,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let schema = query.clone().into_builder(&SQL_INFO).schema();
        flight_info(&schema, query, request.into_inner())
    }

    async fn do_get_sql_info(
        &self,
        query: CommandGetSqlInfo,
        _request: Request<Ticket>,
    ) -> Result<Response<DoGetStream>, Status> {
        let builder = query.into_builder(&SQL_INFO);
        Ok(Response::new(single_batch_stream(
            builder.schema(),
            builder.build(),
        )))
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}

/// Serves plain Arrow Flight path descriptors alongside Flight SQL.
///
/// arrow-flight's `FlightService` implementation for a `FlightSqlService`
/// decodes every descriptor as a Flight SQL command and leaves
/// `GetSchema` unimplemented. This wrapper answers path descriptors and
/// `GetSchema` itself and hands everything else to the Flight SQL
/// service. A path's ticket is an ordinary statement ticket, so `DoGet`
/// needs no special handling.
struct FlightRouter {
    sql: XarrayFlightSql,
}

impl FlightRouter {
    /// The SQL a path descriptor stands for, or `None` for a command.
    fn path_query(&self, descriptor: &FlightDescriptor) -> Result<Option<String>, Status> {
        if descriptor.r#type() != DescriptorType::Path {
            return Ok(None);
        }
        let reference = match descriptor.path.as_slice() {
            [name] => {
                let first = name.split_whitespace().next().unwrap_or_default();
                if first.eq_ignore_ascii_case("select") || first.eq_ignore_ascii_case("with") {
                    return Ok(Some(name.clone()));
                }
                // A registered name matches exactly, as the two- and
                // three-part forms do. Otherwise it is parsed like a
                // table name in SQL: `era5.surface` is schema-qualified.
                let exact = TableReference::bare(name.as_str());
                if self.sql.ctx.table_exist(exact.clone()).unwrap_or(false) {
                    exact
                } else {
                    TableReference::parse_str(name)
                }
            }
            [schema, table] => TableReference::partial(schema.as_str(), table.as_str()),
            [catalog, schema, table] => {
                TableReference::full(catalog.as_str(), schema.as_str(), table.as_str())
            }
            _ => {
                return Err(Status::invalid_argument(
                    "a Flight descriptor path is a table name or a SQL query",
                ))
            }
        };
        Ok(Some(format!(
            "SELECT * FROM {}",
            reference.to_quoted_string()
        )))
    }

    /// The SQL behind a descriptor: a path, or a Flight SQL statement.
    fn descriptor_query(&self, descriptor: &FlightDescriptor) -> Result<String, Status> {
        if let Some(sql) = self.path_query(descriptor)? {
            return Ok(sql);
        }
        let message = Any::decode(&*descriptor.cmd).map_err(internal_error)?;
        match Command::try_from(message).map_err(internal_error)? {
            Command::CommandStatementQuery(query) => Ok(query.query),
            Command::CommandPreparedStatementQuery(query) => {
                Ok(utf8(&query.prepared_statement_handle)?.to_string())
            }
            other => Err(Status::unimplemented(format!(
                "GetSchema is not supported for {}",
                other.type_url()
            ))),
        }
    }
}

#[tonic::async_trait]
impl FlightService for FlightRouter {
    type HandshakeStream = <XarrayFlightSql as FlightService>::HandshakeStream;
    type ListFlightsStream = <XarrayFlightSql as FlightService>::ListFlightsStream;
    type DoGetStream = <XarrayFlightSql as FlightService>::DoGetStream;
    type DoPutStream = <XarrayFlightSql as FlightService>::DoPutStream;
    type DoExchangeStream = <XarrayFlightSql as FlightService>::DoExchangeStream;
    type DoActionStream = <XarrayFlightSql as FlightService>::DoActionStream;
    type ListActionsStream = <XarrayFlightSql as FlightService>::ListActionsStream;

    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let Some(sql) = self.path_query(request.get_ref())? else {
            return FlightService::get_flight_info(&self.sql, request).await;
        };
        let schema = self.sql.result_schema(&sql).await?;
        let ticket = TicketStatementQuery {
            statement_handle: sql.into_bytes().into(),
        };
        flight_info(&schema, ticket, request.into_inner())
    }

    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let sql = self.descriptor_query(request.get_ref())?;
        let schema = self.sql.result_schema(&sql).await?;
        let result = SchemaAsIpc::new(&schema, &IpcWriteOptions::default())
            .try_into()
            .map_err(internal_error)?;
        Ok(Response::new(result))
    }

    async fn handshake(
        &self,
        request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        FlightService::handshake(&self.sql, request).await
    }

    async fn list_flights(
        &self,
        request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        FlightService::list_flights(&self.sql, request).await
    }

    async fn poll_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        FlightService::poll_flight_info(&self.sql, request).await
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        FlightService::do_get(&self.sql, request).await
    }

    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        FlightService::do_put(&self.sql, request).await
    }

    async fn do_exchange(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        FlightService::do_exchange(&self.sql, request).await
    }

    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        FlightService::do_action(&self.sql, request).await
    }

    async fn list_actions(
        &self,
        request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        FlightService::list_actions(&self.sql, request).await
    }
}

/// How long in-flight queries may run once a dropped server stops.
const DROP_GRACE_PERIOD: Duration = Duration::from_secs(5);

/// Handles of a server that is accepting connections.
struct Running {
    /// Starts shutdown; carries how long in-flight queries may run on.
    shutdown: oneshot::Sender<Duration>,
    /// Ends with the error that stopped the server, if one did.
    thread: JoinHandle<Result<(), String>>,
}

/// A Flight SQL server over a native DataFusion session.
///
/// Register tables first, then call ``serve``. The server runs on its own
/// thread with a multi-threaded Tokio runtime; partitions acquire the GIL
/// only for each Python call, as they do in-process.
#[pyclass(name = "FlightSqlServer")]
pub(crate) struct FlightSqlServer {
    ctx: SessionContext,
    running: Option<Running>,
}

fn runtime_error(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pymethods]
impl FlightSqlServer {
    #[new]
    fn new() -> Self {
        Self {
            ctx: SessionContext::new(),
            running: None,
        }
    }

    /// Register ``table`` as ``name``, inside the SQL schema ``schema``
    /// when one is given (created on first use).
    #[pyo3(signature = (name, table, schema=None))]
    fn register_table(
        &self,
        name: &str,
        table: PyRef<'_, LazyArrowStreamTable>,
        schema: Option<&str>,
    ) -> PyResult<()> {
        let provider = table.table.clone();
        let Some(schema_name) = schema else {
            // Bare, not parsed as SQL: a `&str` would fold `Weather` to
            // `weather`, and a quoted `"Weather"` could never find it.
            self.ctx
                .register_table(TableReference::bare(name), provider)
                .map_err(runtime_error)?;
            return Ok(());
        };
        let catalog = self
            .ctx
            .catalog("datafusion")
            .ok_or_else(|| runtime_error("the default catalog is missing"))?;
        let schema_provider = match catalog.schema(schema_name) {
            Some(existing) => existing,
            None => {
                let created: Arc<dyn SchemaProvider> = Arc::new(MemorySchemaProvider::new());
                catalog
                    .register_schema(schema_name, Arc::clone(&created))
                    .map_err(runtime_error)?;
                created
            }
        };
        schema_provider
            .register_table(name.to_string(), provider)
            .map_err(runtime_error)?;
        Ok(())
    }

    /// Start accepting connections on ``host:port``; returns the bound
    /// port (useful with ``port=0``, which picks a free one).
    fn serve(&mut self, host: &str, port: u16) -> PyResult<u16> {
        if self.is_running() {
            return Err(runtime_error("the server is already running"));
        }
        let listener = TcpListener::bind((host, port))?;
        listener.set_nonblocking(true)?;
        let bound = listener.local_addr()?.port();

        let service = XarrayFlightSql {
            ctx: self.ctx.clone(),
        };
        let (shutdown, shutdown_rx) = oneshot::channel::<Duration>();
        let (ready, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        let thread = std::thread::Builder::new()
            .name("xarray-sql-flight-sql".to_string())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(e) => {
                        let _ = ready.send(Err(e.to_string()));
                        return Ok(());
                    }
                };
                let result = runtime.block_on(async move {
                    let listener = match tokio::net::TcpListener::from_std(listener) {
                        Ok(listener) => listener,
                        Err(e) => {
                            let _ = ready.send(Err(e.to_string()));
                            return Ok(());
                        }
                    };
                    let incoming = TcpIncoming::from(listener).with_nodelay(Some(true));
                    let _ = ready.send(Ok(()));
                    // Graceful shutdown waits for every open response
                    // stream, including ones a client stopped reading, so
                    // it gets a deadline after which the server is dropped.
                    let (grace, grace_rx) = oneshot::channel::<Duration>();
                    let server = Server::builder()
                        .add_service(FlightServiceServer::new(FlightRouter { sql: service }))
                        .serve_with_incoming_shutdown(incoming, async move {
                            let period = shutdown_rx.await.unwrap_or(Duration::ZERO);
                            let _ = grace.send(period);
                        });
                    let deadline = async move {
                        match grace_rx.await {
                            Ok(period) => tokio::time::sleep(period).await,
                            Err(_) => std::future::pending::<()>().await,
                        }
                    };
                    tokio::select! {
                        served = server => served.map_err(|e| e.to_string()),
                        _ = deadline => Ok(()),
                    }
                });
                // Cancels whatever the deadline cut off, closing its
                // connections, without waiting on it.
                runtime.shutdown_background();
                result
            })?;

        match ready_rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(runtime_error(e)),
            Err(_) => return Err(runtime_error("the server thread exited during startup")),
        }
        self.running = Some(Running { shutdown, thread });
        Ok(bound)
    }

    /// Whether the server thread is still accepting connections.
    fn is_running(&self) -> bool {
        self.running
            .as_ref()
            .is_some_and(|running| !running.thread.is_finished())
    }

    /// Stop accepting connections, give in-flight queries up to
    /// ``timeout`` seconds to finish, then close the remaining connections.
    /// Raises the error that stopped the server, if one did.
    #[pyo3(signature = (timeout=5.0))]
    fn shutdown(&mut self, py: Python<'_>, timeout: f64) -> PyResult<()> {
        let grace = Duration::try_from_secs_f64(timeout)
            .map_err(|e| PyValueError::new_err(format!("invalid timeout {timeout}: {e}")))?;
        if let Some(running) = self.running.take() {
            let _ = running.shutdown.send(grace);
            // In-flight partitions may need the GIL to finish.
            py.detach(|| running.thread.join())
                .map_err(|_| runtime_error("the server thread panicked"))?
                .map_err(|e| runtime_error(format!("the server stopped: {e}")))?;
        }
        Ok(())
    }
}

impl Drop for FlightSqlServer {
    fn drop(&mut self) {
        // Signal only: joining here could wait on the GIL this thread holds.
        if let Some(running) = self.running.take() {
            let _ = running.shutdown.send(DROP_GRACE_PERIOD);
        }
    }
}
