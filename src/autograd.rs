//! Symbolic differentiation of DataFusion logical [`Expr`] trees.
//!
//! This is the autograd engine for xarray-sql. Given an [`Expr`] and the name
//! of a column to differentiate with respect to, [`differentiate`] returns a
//! new [`Expr`] for the (symbolic) partial derivative, built entirely from
//! ordinary DataFusion expressions so the result can be planned and evaluated
//! by DataFusion like any other SQL expression.
//!
//! ## Design
//!
//! The approach mirrors JAX's per-primitive rule registry (`defjvp` and
//! friends in `jax/_src/interpreters/ad.py`): every expression node has a
//! differentiation rule, and the chain rule composes them as the tree is
//! walked. Because each row of a relational table is an independent evaluation
//! point, differentiating a column expression and letting DataFusion evaluate
//! it row-by-row is the moral equivalent of `jax.vmap(jax.grad(f))` — the rows
//! *are* the batch dimension.
//!
//! A small simplifier folds the `0`/`1` constants that differentiation
//! produces in abundance (e.g. `d/dx (c) = 0`, `d/dx (x) = 1`), keeping output
//! expressions compact. This plays the role of JAX's `Zero` tangents and
//! `add_tangents`: a `0` derivative short-circuits products and drops out of
//! sums, and a `1` factor drops out of products.
//!
//! ## Surface
//!
//! Three scalar operations, all rewritten away before execution:
//!
//! * `grad(expr, column)` — the partial derivative `d(expr)/d(column)`.
//! * `jvp(expr, column, tangent)` — forward-mode directional derivative,
//!   `d(expr)/d(column) * tangent` (seed a tangent on an input).
//! * `vjp(expr, column, cotangent)` — reverse-mode pullback,
//!   `cotangent * d(expr)/d(column)` (seed a cotangent on the output).
//!
//! All three return a scalar per row, staying in the long/tidy data model. A
//! full gradient or Jacobian is expressed as several scalar columns (e.g.
//! `grad(f, x) AS dfdx, grad(f, y) AS dfdy`) rather than a nested array, which
//! would break the one-value-per-coordinate model.
//!
//! Calls nest, giving higher-order derivatives for free: the rewrite walks
//! bottom-up, so the inner call in `grad(grad(f, x), x)` is differentiated
//! first and the outer call differentiates that result.
//!
//! Differentiation through an aggregate is just linearity and needs no special
//! handling: write the `grad` *inside* the aggregate, e.g. `SUM(grad(f, x))` or
//! `AVG(grad(loss, theta))`. Because the marker is rewritten to plain SQL
//! before the aggregate runs (and the column is in scope there), this is the
//! relational `d/dθ Σ f = Σ ∂f/∂θ` — enough to run gradient descent in SQL.
//! (The transposed form `grad(SUM(f), x)` is rejected by SQL's own scoping,
//! since `x` is gone after aggregation.)

#![allow(dead_code)]

use std::collections::HashMap;
use std::f64::consts::{LN_10, LN_2};
use std::ops::ControlFlow;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{DFSchema, DataFusionError, Result, ScalarValue, TableReference};
use datafusion::functions::math::expr_fn;
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{
    lit, BinaryExpr, Cast, ColumnarValue, Expr, LogicalPlan, Operator, ScalarFunctionArgs,
    ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::prelude::SessionContext;
use datafusion::sql::sqlparser::ast::{Expr as SqlExpr, Visit, VisitMut, Visitor, VisitorMut};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::unparser::expr_to_sql;

// ---------------------------------------------------------------------------
// Constant helpers and the 0/1-folding builders
// ---------------------------------------------------------------------------

/// The constant `0.0`, used as the derivative of anything not depending on the
/// differentiation variable.
fn zero() -> Expr {
    lit(0.0_f64)
}

/// The constant `1.0`, used as the derivative of the differentiation variable.
fn one() -> Expr {
    lit(1.0_f64)
}

/// Interpret a [`ScalarValue`] as `f64` if it is a (non-null) numeric scalar.
fn scalar_as_f64(sv: &ScalarValue) -> Option<f64> {
    match sv {
        ScalarValue::Float64(Some(v)) => Some(*v),
        ScalarValue::Float32(Some(v)) => Some(*v as f64),
        ScalarValue::Int64(Some(v)) => Some(*v as f64),
        ScalarValue::Int32(Some(v)) => Some(*v as f64),
        ScalarValue::Int16(Some(v)) => Some(*v as f64),
        ScalarValue::Int8(Some(v)) => Some(*v as f64),
        ScalarValue::UInt64(Some(v)) => Some(*v as f64),
        ScalarValue::UInt32(Some(v)) => Some(*v as f64),
        ScalarValue::UInt16(Some(v)) => Some(*v as f64),
        ScalarValue::UInt8(Some(v)) => Some(*v as f64),
        _ => None,
    }
}

/// Return the constant `f64` value of a literal expression, if it is one.
fn as_const(e: &Expr) -> Option<f64> {
    match e {
        Expr::Literal(sv, _) => scalar_as_f64(sv),
        _ => None,
    }
}

/// True if the expression is a numeric literal exactly equal to zero.
fn is_zero(e: &Expr) -> bool {
    matches!(as_const(e), Some(v) if v == 0.0)
}

/// True if the expression is a numeric literal exactly equal to one.
fn is_one(e: &Expr) -> bool {
    matches!(as_const(e), Some(v) if v == 1.0)
}

fn binary(left: Expr, op: Operator, right: Expr) -> Expr {
    Expr::BinaryExpr(BinaryExpr::new(Box::new(left), op, Box::new(right)))
}

/// `a + b`, dropping a zero operand.
fn add(a: Expr, b: Expr) -> Expr {
    if is_zero(&a) {
        b
    } else if is_zero(&b) {
        a
    } else {
        binary(a, Operator::Plus, b)
    }
}

/// `a - b`, dropping a zero right operand and turning `0 - b` into `-b`.
fn sub(a: Expr, b: Expr) -> Expr {
    if is_zero(&b) {
        a
    } else if is_zero(&a) {
        neg(b)
    } else {
        binary(a, Operator::Minus, b)
    }
}

/// `a * b`, folding `0 * _ = 0` and `1 * b = b` (and the mirror cases).
fn mul(a: Expr, b: Expr) -> Expr {
    if is_zero(&a) || is_zero(&b) {
        zero()
    } else if is_one(&a) {
        b
    } else if is_one(&b) {
        a
    } else {
        binary(a, Operator::Multiply, b)
    }
}

/// `a / b`, folding `0 / _ = 0` and `a / 1 = a`.
fn div(a: Expr, b: Expr) -> Expr {
    if is_zero(&a) {
        zero()
    } else if is_one(&b) {
        a
    } else {
        binary(a, Operator::Divide, b)
    }
}

/// `-a`, folding `-0 = 0`.
fn neg(a: Expr) -> Expr {
    if is_zero(&a) {
        zero()
    } else {
        Expr::Negative(Box::new(a))
    }
}

/// `e * e`.
fn square(e: Expr) -> Expr {
    mul(e.clone(), e)
}

// ---------------------------------------------------------------------------
// The differentiation engine (forward-mode linearization)
// ---------------------------------------------------------------------------

/// A *leaf rule*: the tangent of a column, i.e. the seed assigned to each input
/// during forward-mode differentiation.
///
/// `grad` uses a one-hot leaf (`1` for the differentiation variable, `0`
/// otherwise); `jvp` uses an arbitrary seed per input. Everything above the
/// leaves — the chain rule — is shared.
type Leaf<'a> = dyn Fn(&str) -> Expr + 'a;

/// Linearize `expr`: push tangents from the leaves (per `leaf`) up through the
/// expression via the chain rule, returning the tangent of `expr`.
///
/// This is forward-mode automatic differentiation. `differentiate` (a single
/// partial derivative) and `jvp` (a directional derivative) are both thin
/// wrappers that only differ in their leaf rule. Returns a
/// [`DataFusionError::NotImplemented`] for nodes or functions without a rule,
/// so callers surface a clear error rather than a silently-wrong derivative.
fn linearize(expr: &Expr, leaf: &Leaf) -> Result<Expr> {
    match expr {
        // The leaf rule decides a column's tangent.
        Expr::Column(c) => Ok(leaf(&c.name)),

        // Constants have zero tangent.
        Expr::Literal(_, _) => Ok(zero()),

        // An alias is transparent; the surrounding query re-applies any naming.
        Expr::Alias(a) => linearize(&a.expr, leaf),

        // A numeric cast is (locally) linear: tangent of cast(u) = cast(du).
        Expr::Cast(c) => {
            let du = linearize(&c.expr, leaf)?;
            Ok(Expr::Cast(Cast::new(
                Box::new(du),
                c.field.data_type().clone(),
            )))
        }

        // tangent of -u = -(du).
        Expr::Negative(inner) => Ok(neg(linearize(inner, leaf)?)),

        Expr::BinaryExpr(be) => linearize_binary(be, leaf),

        Expr::ScalarFunction(sf) => linearize_scalar_function(sf, leaf),

        other => Err(DataFusionError::NotImplemented(format!(
            "grad: differentiation is not implemented for this expression: {other}"
        ))),
    }
}

/// Differentiate `expr` with respect to the column named `wrt`.
///
/// Forward-mode with a one-hot seed: `1` on `wrt`, `0` on every other column.
pub fn differentiate(expr: &Expr, wrt: &str) -> Result<Expr> {
    linearize(expr, &|name| if name == wrt { one() } else { zero() })
}

/// Forward-mode directional derivative: the tangent of `expr` given a tangent
/// (`seeds[col]`) for each seeded input column; unseeded columns are constant.
fn jvp(expr: &Expr, seeds: &HashMap<String, Expr>) -> Result<Expr> {
    linearize(expr, &|name| seeds.get(name).cloned().unwrap_or_else(zero))
}

/// Linearize a binary arithmetic expression via the sum/product/quotient rules.
fn linearize_binary(be: &BinaryExpr, leaf: &Leaf) -> Result<Expr> {
    let a = be.left.as_ref();
    let b = be.right.as_ref();
    let da = linearize(a, leaf)?;
    let db = linearize(b, leaf)?;

    match be.op {
        // tangent of (a + b) = da + db
        Operator::Plus => Ok(add(da, db)),
        // tangent of (a - b) = da - db
        Operator::Minus => Ok(sub(da, db)),
        // tangent of (a * b) = da*b + a*db   (product rule)
        Operator::Multiply => Ok(add(mul(da, b.clone()), mul(a.clone(), db))),
        // tangent of (a / b) = (da*b - a*db) / b^2   (quotient rule)
        Operator::Divide => {
            let numerator = sub(mul(da, b.clone()), mul(a.clone(), db));
            Ok(div(numerator, square(b.clone())))
        }
        op => Err(DataFusionError::NotImplemented(format!(
            "grad: operator '{op}' is not differentiable"
        ))),
    }
}

/// Linearize a scalar-function call via the chain rule.
///
/// For a unary primitive `f(u)`, the tangent is `f'(u) * du`. For `power`,
/// which is binary, we handle the constant-exponent and constant-base cases.
fn linearize_scalar_function(sf: &ScalarFunction, leaf: &Leaf) -> Result<Expr> {
    let name = sf.func.name();
    let args = &sf.args;

    // `power(base, exponent)` is the one binary primitive we linearize.
    if name == "power" {
        return linearize_power(args, leaf);
    }

    if args.len() != 1 {
        return Err(DataFusionError::NotImplemented(format!(
            "grad: no derivative rule for function '{name}' with {} arguments",
            args.len()
        )));
    }

    let u = &args[0];
    let du = linearize(u, leaf)?;
    // Chain rule short-circuit: if du is 0, the whole tangent is 0 and we avoid
    // emitting the (dead) outer derivative term entirely.
    if is_zero(&du) {
        return Ok(zero());
    }

    let outer = match name {
        // Trigonometric.
        "sin" => expr_fn::cos(u.clone()),
        "cos" => neg(expr_fn::sin(u.clone())),
        "tan" => div(one(), square(expr_fn::cos(u.clone()))),
        // Inverse trigonometric.
        "asin" => div(one(), expr_fn::sqrt(sub(one(), square(u.clone())))),
        "acos" => neg(div(one(), expr_fn::sqrt(sub(one(), square(u.clone()))))),
        "atan" => div(one(), add(one(), square(u.clone()))),
        // Exponential / logarithmic.
        "exp" => expr_fn::exp(u.clone()),
        "ln" => div(one(), u.clone()),
        "log2" => div(one(), mul(u.clone(), lit(LN_2))),
        "log10" => div(one(), mul(u.clone(), lit(LN_10))),
        "sqrt" => div(one(), mul(lit(2.0_f64), expr_fn::sqrt(u.clone()))),
        // Hyperbolic.
        "sinh" => expr_fn::cosh(u.clone()),
        "cosh" => expr_fn::sinh(u.clone()),
        "tanh" => sub(one(), square(expr_fn::tanh(u.clone()))),
        // Piecewise-linear: derivative is the sign (undefined at 0, like JAX).
        "abs" => expr_fn::signum(u.clone()),
        _ => {
            return Err(DataFusionError::NotImplemented(format!(
                "grad: no derivative rule for function '{name}'"
            )))
        }
    };

    Ok(mul(outer, du))
}

/// Linearize `power(base, exponent)`.
///
/// * Constant exponent `c`: tangent = `c * base^(c-1) * d(base)`.
/// * Constant base `a`: tangent = `a^u * ln(a) * d(u)`.
/// * Both variable (`u^v`): not supported yet.
fn linearize_power(args: &[Expr], leaf: &Leaf) -> Result<Expr> {
    if args.len() != 2 {
        return Err(DataFusionError::NotImplemented(
            "grad: power() expects exactly two arguments".to_string(),
        ));
    }
    let base = &args[0];
    let exponent = &args[1];

    match (as_const(base), as_const(exponent)) {
        // Constant exponent (covers the common x^2, x^0.5, ... cases).
        (_, Some(c)) => {
            let dbase = linearize(base, leaf)?;
            if is_zero(&dbase) {
                return Ok(zero());
            }
            let outer = mul(lit(c), expr_fn::power(base.clone(), lit(c - 1.0)));
            Ok(mul(outer, dbase))
        }
        // Constant base, variable exponent.
        (Some(a), None) => {
            let dexp = linearize(exponent, leaf)?;
            if is_zero(&dexp) {
                return Ok(zero());
            }
            let outer = mul(expr_fn::power(base.clone(), exponent.clone()), lit(a.ln()));
            Ok(mul(outer, dexp))
        }
        // General u^v requires the exp/log trick; deferred for now.
        (None, None) => Err(DataFusionError::NotImplemented(
            "grad: power(base, exponent) where both depend on the \
             differentiation variable is not yet supported"
                .to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// The `grad` / `jacobian` marker UDFs and the plan-level rewrite
// ---------------------------------------------------------------------------

/// A no-op placeholder UDF for the autograd surface functions.
///
/// `grad`, `jvp`, and `vjp` are *markers*: they carry the differentiation
/// request intact through SQL parsing, logical planning, and Substrait
/// serialization. They are always rewritten away by [`rewrite_grad_calls`]
/// before execution, so `invoke` is never reached in normal use (and
/// deliberately errors if it somehow is, rather than returning a wrong value).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct MarkerUdf {
    name: String,
    signature: Signature,
}

impl MarkerUdf {
    fn new(name: &str, arity: usize) -> Self {
        Self {
            name: name.to_string(),
            signature: Signature::any(arity, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for MarkerUdf {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // Every autograd marker rewrites to a scalar derivative expression.
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Err(DataFusionError::Execution(format!(
            "{}() marker reached execution without being rewritten; this is \
             an internal xarray-sql autograd error",
            self.name
        )))
    }
}

/// The `grad(expr, column)` marker: scalar partial derivative `d(expr)/dcolumn`.
pub fn grad_marker() -> ScalarUDF {
    ScalarUDF::from(MarkerUdf::new("grad", 2))
}

/// The `jvp(expr, column, tangent)` marker: forward-mode directional derivative.
pub fn jvp_marker() -> ScalarUDF {
    ScalarUDF::from(MarkerUdf::new("jvp", 3))
}

/// The `vjp(expr, column, cotangent)` marker: reverse-mode pullback to an input.
pub fn vjp_marker() -> ScalarUDF {
    ScalarUDF::from(MarkerUdf::new("vjp", 3))
}

/// Rewrite every `grad`/`jvp`/`vjp` call anywhere in a logical plan into its
/// symbolic derivative, leaving everything else untouched. The plan's schema is
/// recomputed afterwards because replacing a marker can change an expression's
/// name or type.
pub fn rewrite_grad_calls(plan: LogicalPlan) -> Result<LogicalPlan> {
    let rewritten = plan
        .transform_up(|node| node.map_expressions(rewrite_grad_in_expr))?
        .data;
    rewritten.recompute_schema()
}

/// Replace any `grad`/`jvp`/`vjp` calls nested anywhere inside a single
/// expression.
fn rewrite_grad_in_expr(expr: Expr) -> Result<Transformed<Expr>> {
    expr.transform_up(|e| {
        let Expr::ScalarFunction(sf) = &e else {
            return Ok(Transformed::no(e));
        };
        match sf.func.name() {
            "grad" => Ok(Transformed::yes(rewrite_grad(&sf.args)?)),
            "jvp" => Ok(Transformed::yes(rewrite_jvp(&sf.args)?)),
            "vjp" => Ok(Transformed::yes(rewrite_vjp(&sf.args)?)),
            _ => Ok(Transformed::no(e)),
        }
    })
}

/// Read a bare column name from a marker argument, or report a clear error.
fn column_arg(func: &str, arg: &Expr) -> Result<String> {
    match arg {
        Expr::Column(c) => Ok(c.name.clone()),
        other => Err(DataFusionError::Plan(format!(
            "{func}(): the column argument must be a bare column to \
             differentiate with respect to, got: {other}"
        ))),
    }
}

/// `grad(expr, column)` -> `d(expr)/d(column)`.
fn rewrite_grad(args: &[Expr]) -> Result<Expr> {
    if args.len() != 2 {
        return Err(DataFusionError::Plan(format!(
            "grad() expects two arguments grad(expr, column), got {}",
            args.len()
        )));
    }
    let wrt = column_arg("grad", &args[1])?;
    differentiate(&args[0], &wrt)
}

/// `jvp(expr, column, tangent)` -> forward-mode tangent: seed `tangent` on
/// `column` and push it through `expr`, yielding `d(expr)/d(column) * tangent`.
///
/// A directional derivative over several inputs is the sum of per-input jvps,
/// e.g. `jvp(f, x, dx) + jvp(f, y, dy)`, since each treats the other inputs as
/// having zero tangent.
fn rewrite_jvp(args: &[Expr]) -> Result<Expr> {
    if args.len() != 3 {
        return Err(DataFusionError::Plan(format!(
            "jvp() expects three arguments jvp(expr, column, tangent), got {}",
            args.len()
        )));
    }
    let wrt = column_arg("jvp", &args[1])?;
    let seeds = HashMap::from([(wrt, args[2].clone())]);
    jvp(&args[0], &seeds)
}

/// `vjp(expr, column, cotangent)` -> reverse-mode pullback: the sensitivity that
/// an output cotangent induces on `column`, i.e. `cotangent * d(expr)/d(column)`.
///
/// For a single scalar output this equals the matching `jvp` (both contract the
/// same partial derivative); the surfaces differ in where the seed lives — `jvp`
/// seeds an input tangent, `vjp` seeds an output cotangent.
fn rewrite_vjp(args: &[Expr]) -> Result<Expr> {
    if args.len() != 3 {
        return Err(DataFusionError::Plan(format!(
            "vjp() expects three arguments vjp(expr, column, cotangent), got {}",
            args.len()
        )));
    }
    let wrt = column_arg("vjp", &args[1])?;
    let derivative = differentiate(&args[0], &wrt)?;
    Ok(mul(args[2].clone(), derivative))
}

// ---------------------------------------------------------------------------
// SQL source-to-source rewrite
// ---------------------------------------------------------------------------

/// Rewrite every `grad`/`jvp`/`vjp` call in a SQL statement into its symbolic
/// derivative, returning the rewritten SQL text.
///
/// Unlike a logical-plan rewrite, this is a pure source-to-source transform run
/// *before* the query is planned, so it works for any query shape the SQL parser
/// accepts — recursive CTEs, DML, and subqueries included. Each marker call is
/// parsed into a DataFusion [`Expr`], differentiated by the engine in this
/// module, and rendered back to SQL in place. Columns are taken from the call's
/// own identifiers (all treated as `Float64`; types don't affect the symbolic
/// result), so no catalog or table schema is needed.
pub fn rewrite_grad_in_sql(sql: &str) -> Result<String> {
    let dialect = GenericDialect {};
    let mut statements = Parser::parse_sql(&dialect, sql)
        .map_err(|e| DataFusionError::Plan(format!("grad: failed to parse SQL: {e}")))?;

    // A throwaway context that only needs the marker UDFs registered so the
    // calls parse into `ScalarFunction` nodes the engine can dispatch on.
    let ctx = SessionContext::new();
    ctx.register_udf(grad_marker());
    ctx.register_udf(jvp_marker());
    ctx.register_udf(vjp_marker());

    let mut rewriter = GradSqlRewriter { ctx: &ctx };
    for stmt in &mut statements {
        if let ControlFlow::Break(msg) = stmt.visit(&mut rewriter) {
            return Err(DataFusionError::Plan(msg));
        }
    }

    Ok(statements
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; "))
}

/// True if `name` is one of the autograd marker functions (case-insensitive).
fn is_marker_name(name: &str) -> bool {
    matches!(name.to_lowercase().as_str(), "grad" | "jvp" | "vjp")
}

/// Walks a SQL AST and replaces each `grad`/`jvp`/`vjp` call with its derivative.
struct GradSqlRewriter<'a> {
    ctx: &'a SessionContext,
}

impl VisitorMut for GradSqlRewriter<'_> {
    type Break = String;

    fn pre_visit_expr(&mut self, expr: &mut SqlExpr) -> ControlFlow<String> {
        let is_marker = matches!(
            expr,
            SqlExpr::Function(f) if is_marker_name(&f.name.to_string())
        );
        if !is_marker {
            return ControlFlow::Continue(());
        }
        match self.rewrite_call(expr) {
            Ok(()) => ControlFlow::Continue(()),
            Err(e) => ControlFlow::Break(e),
        }
    }
}

impl GradSqlRewriter<'_> {
    /// Differentiate a single marker call in place. The replacement is wrapped
    /// in parentheses so it keeps the call's precedence in the surrounding SQL.
    fn rewrite_call(&self, expr: &mut SqlExpr) -> std::result::Result<(), String> {
        let schema = call_schema(expr)?;
        let text = expr.to_string();
        let parsed = self
            .ctx
            .parse_sql_expr(&text, &schema)
            .map_err(|e| format!("grad: failed to parse '{text}': {e}"))?;
        let derivative = rewrite_grad_in_expr(parsed)
            .map_err(|e| format!("grad: failed to differentiate '{text}': {e}"))?
            .data;
        let rendered = expr_to_sql(&derivative)
            .map_err(|e| format!("grad: failed to render derivative for '{text}': {e}"))?;
        *expr = SqlExpr::Nested(Box::new(rendered));
        Ok(())
    }
}

/// Build a `Float64` schema covering every column identifier referenced inside a
/// marker call, so the call's argument expression can be parsed standalone.
fn call_schema(call: &SqlExpr) -> std::result::Result<DFSchema, String> {
    let mut collector = ColumnCollector::default();
    let _ = call.visit(&mut collector);
    let fields = collector
        .cols
        .into_iter()
        .map(|(qualifier, name)| {
            let qualifier = qualifier.map(TableReference::bare);
            (
                qualifier,
                Arc::new(Field::new(name, DataType::Float64, true)),
            )
        })
        .collect();
    DFSchema::new_with_metadata(fields, HashMap::new())
        .map_err(|e| format!("grad: failed to build schema for differentiation: {e}"))
}

/// Collects the (optional qualifier, name) of every column identifier in a SQL
/// expression tree.
#[derive(Default)]
struct ColumnCollector {
    cols: Vec<(Option<String>, String)>,
}

impl Visitor for ColumnCollector {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &SqlExpr) -> ControlFlow<()> {
        let pair = match expr {
            SqlExpr::Identifier(ident) => Some((None, ident.value.clone())),
            SqlExpr::CompoundIdentifier(parts) => parts.last().map(|last| {
                let qualifier = (parts.len() >= 2).then(|| parts[parts.len() - 2].value.clone());
                (qualifier, last.value.clone())
            }),
            _ => None,
        };
        if let Some(pair) = pair {
            if !self.cols.contains(&pair) {
                self.cols.push(pair);
            }
        }
        ControlFlow::Continue(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::col;

    use super::*;

    #[test]
    fn constant_has_zero_derivative() {
        assert_eq!(differentiate(&lit(3.0_f64), "x").unwrap(), zero());
    }

    #[test]
    fn variable_has_unit_derivative() {
        assert_eq!(differentiate(&col("x"), "x").unwrap(), one());
    }

    #[test]
    fn other_variable_has_zero_derivative() {
        assert_eq!(differentiate(&col("y"), "x").unwrap(), zero());
    }

    #[test]
    fn sum_rule_folds_constants() {
        // d/dx (x + y) = 1 + 0 = 1
        let e = add(col("x"), col("y"));
        assert_eq!(differentiate(&e, "x").unwrap(), one());
    }

    #[test]
    fn product_rule() {
        // d/dx (x * x) = 1*x + x*1 = x + x
        let e = binary(col("x"), Operator::Multiply, col("x"));
        let expected = add(col("x"), col("x"));
        assert_eq!(differentiate(&e, "x").unwrap(), expected);
    }

    #[test]
    fn quotient_rule() {
        // d/dx (x / y) = (1*y - x*0) / (y*y) = y / (y*y)
        let e = binary(col("x"), Operator::Divide, col("y"));
        let expected = div(col("y"), square(col("y")));
        assert_eq!(differentiate(&e, "x").unwrap(), expected);
    }

    #[test]
    fn chain_rule_sin() {
        // d/dx sin(x) = cos(x) * 1 = cos(x)
        let d = differentiate(&expr_fn::sin(col("x")), "x").unwrap();
        assert_eq!(d, expr_fn::cos(col("x")));
        // Readable, precedence-free rendering.
        assert_eq!(d.to_string(), "cos(x)");
    }

    #[test]
    fn composite_sin_times_x() {
        // d/dx (sin(x) * x) = cos(x)*x + sin(x)
        let e = binary(expr_fn::sin(col("x")), Operator::Multiply, col("x"));
        let d = differentiate(&e, "x").unwrap();
        assert_eq!(d.to_string(), "cos(x) * x + sin(x)");
    }

    #[test]
    fn power_constant_exponent() {
        // d/dx power(x, 2) = 2 * power(x, 1) * 1 = 2 * power(x, 1)
        let e = expr_fn::power(col("x"), lit(2.0_f64));
        let expected = mul(lit(2.0_f64), expr_fn::power(col("x"), lit(1.0_f64)));
        assert_eq!(differentiate(&e, "x").unwrap(), expected);
    }

    #[test]
    fn unsupported_operator_errors() {
        let e = binary(col("x"), Operator::Modulo, col("y"));
        assert!(differentiate(&e, "x").is_err());
    }

    #[test]
    fn unsupported_function_errors() {
        // atan2 is binary and has no rule yet.
        let e = expr_fn::atan2(col("x"), col("y"));
        assert!(differentiate(&e, "x").is_err());
    }

    #[test]
    fn higher_order_derivative() {
        // Differentiation composes: d2/dx2 sin(x) = -sin(x).
        let d1 = differentiate(&expr_fn::sin(col("x")), "x").unwrap();
        let d2 = differentiate(&d1, "x").unwrap();
        assert_eq!(d2, neg(expr_fn::sin(col("x"))));
    }

    #[test]
    fn jvp_seeds_a_tangent_on_one_input() {
        // jvp(x*y, {x: dx}) = product rule with tangent(x)=dx, tangent(y)=0
        //                   = dx*y + x*0 = dx*y
        let f = binary(col("x"), Operator::Multiply, col("y"));
        let seeds = HashMap::from([("x".to_string(), col("dx"))]);
        let t = jvp(&f, &seeds).unwrap();
        assert_eq!(t, mul(col("dx"), col("y")));
    }

    #[test]
    fn jvp_with_unit_seed_matches_grad() {
        // A one-hot tangent reproduces the partial derivative.
        let f = expr_fn::sin(col("x"));
        let seeds = HashMap::from([("x".to_string(), one())]);
        assert_eq!(jvp(&f, &seeds).unwrap(), differentiate(&f, "x").unwrap());
    }

    #[test]
    fn vjp_equals_cotangent_times_grad() {
        // rewrite_vjp(sin(x), x, w) = w * cos(x)
        let f = expr_fn::sin(col("x"));
        let got = rewrite_vjp(&[f.clone(), col("x"), col("w")]).unwrap();
        assert_eq!(got, mul(col("w"), expr_fn::cos(col("x"))));
    }

    #[test]
    fn jvp_and_vjp_agree_for_unit_seed() {
        // With matching unit seed/cotangent, forward and reverse coincide.
        let f = binary(expr_fn::sin(col("x")), Operator::Multiply, col("x"));
        let fwd = rewrite_jvp(&[f.clone(), col("x"), one()]).unwrap();
        let rev = rewrite_vjp(&[f, col("x"), one()]).unwrap();
        assert_eq!(fwd, rev);
    }

    #[test]
    fn sql_rewrite_replaces_grad_call() {
        // grad(sin(x), x) -> cos(x); the surrounding SELECT is preserved.
        let out = rewrite_grad_in_sql("SELECT grad(sin(x), x) AS d FROM t").unwrap();
        assert_eq!(out, "SELECT (cos(x)) AS d FROM t");
    }

    #[test]
    fn sql_rewrite_leaves_non_grad_queries_intact() {
        // A query with no marker is still parsed and re-emitted unchanged in
        // meaning (the caller only invokes the rewrite when a marker is present).
        let out = rewrite_grad_in_sql("SELECT a + b FROM t").unwrap();
        assert_eq!(out, "SELECT a + b FROM t");
    }

    #[test]
    fn sql_rewrite_fires_inside_recursive_cte() {
        // The #197 capability: a marker inside a recursive term is rewritten,
        // a query shape the Substrait bridge could never carry. d/dx(x*x) = x+x.
        let out = rewrite_grad_in_sql(
            "WITH RECURSIVE r AS (SELECT 1.0 AS x UNION ALL \
             SELECT x - grad(x * x, x) FROM r WHERE x < 10) SELECT x FROM r",
        )
        .unwrap();
        assert!(out.contains("(x + x)"), "unexpected rewrite: {out}");
        assert!(
            !out.to_lowercase().contains("grad("),
            "marker left behind: {out}"
        );
    }

    #[test]
    fn sql_rewrite_handles_nested_higher_order_grad() {
        // grad(grad(power(x, 3), x), x) -> d2/dx2 (x^3) = 6x; bottom-up so the
        // inner call is differentiated before the outer one.
        let out = rewrite_grad_in_sql("SELECT grad(grad(power(x, 3), x), x) AS d FROM t").unwrap();
        assert!(
            !out.to_lowercase().contains("grad("),
            "marker left behind: {out}"
        );
    }
}
