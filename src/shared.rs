//! Parse an explicitly staged SELECT with the same SQL parser as DataFusion.

use std::ops::ControlFlow;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::pyarrow::FromPyArrow;
use datafusion::catalog::TableProvider;
use datafusion::common::Column;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::Expr as DFExpr;
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use sqlparser::ast::{Expr, ObjectName, Query, Statement, Visit, Visitor};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

#[derive(Default)]
struct References {
    tables: Vec<String>,
    functions: Vec<String>,
    nested_with: bool,
}

impl Visitor for References {
    type Break = ();

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<()> {
        self.nested_with |= query.with.is_some();
        ControlFlow::Continue(())
    }

    fn pre_visit_relation(&mut self, name: &ObjectName) -> ControlFlow<()> {
        if name.0.len() == 1 {
            if let Some(id) = name.0[0].as_ident() {
                self.tables.push(if id.quote_style.is_some() {
                    id.value.clone()
                } else {
                    id.value.to_lowercase()
                });
            }
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if let Expr::Function(f) = expr {
            self.functions.push(f.name.to_string().to_lowercase());
        }
        ControlFlow::Continue(())
    }
}

type Part = (String, String, Vec<String>, Vec<String>);

/// Return top-level CTEs and the final SELECT, with their direct references.
#[pyfunction]
#[pyo3(signature = (sql, allow_nested=false))]
pub(crate) fn shared_ctes(sql: &str, allow_nested: bool) -> PyResult<Vec<Part>> {
    let mut statements = Parser::parse_sql(&GenericDialect {}, sql)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if statements.len() != 1 {
        return Err(PyValueError::new_err(
            "shared execution requires one SELECT",
        ));
    }
    let Statement::Query(mut query) = statements.remove(0) else {
        return Err(PyValueError::new_err("shared execution requires a SELECT"));
    };
    let mut parts = Vec::new();
    if let Some(with) = query.with.take() {
        if with.recursive {
            return Err(PyValueError::new_err("recursive CTEs cannot be staged"));
        }
        for cte in with.cte_tables {
            if !cte.alias.columns.is_empty() {
                return Err(PyValueError::new_err("CTE column aliases cannot be staged"));
            }
            let id = cte.alias.name;
            let name = if id.quote_style.is_some() {
                id.value
            } else {
                id.value.to_lowercase()
            };
            parts.push(part(name, &cte.query, allow_nested)?);
        }
    }
    parts.push(part(String::new(), &query, allow_nested)?);
    Ok(parts)
}

fn part(name: String, query: &Query, allow_nested: bool) -> PyResult<Part> {
    let mut refs = References::default();
    let _ = query.visit(&mut refs);
    if refs.nested_with && !allow_nested {
        return Err(PyValueError::new_err("nested WITH scopes cannot be staged"));
    }
    Ok((name, query.to_string(), refs.tables, refs.functions))
}

/// An internal Arrow stage with a verified ascending, NULL-last key order.
/// The Python stage executor verifies or sorts these keys before registration
/// and never mutates key buffers while a plan reads this provider.
#[pyclass(module = "xarray_sql._native", name = "_SortedArrowTable")]
pub(crate) struct SortedArrowTable {
    table: Arc<dyn TableProvider>,
}

#[pymethods]
impl SortedArrowTable {
    #[new]
    fn new(batch: &Bound<'_, PyAny>, keys: Vec<String>) -> PyResult<Self> {
        let batch = RecordBatch::from_pyarrow_bound(batch)?;
        let ordering = keys
            .into_iter()
            .map(|key| DFExpr::Column(Column::from_name(key)).sort(true, false))
            .collect();
        let table = MemTable::try_new(batch.schema(), vec![vec![batch]])
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_sort_order(vec![ordering]);
        Ok(Self {
            table: Arc::new(table),
        })
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = crate::ffi_logical_codec_from_pycapsule(session)?;
        let provider = FFI_TableProvider::new_with_ffi_codec(self.table.clone(), true, None, codec);
        PyCapsule::new_with_value(py, provider, c"datafusion_table_provider")
    }
}
