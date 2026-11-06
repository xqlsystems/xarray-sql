//! Splitting a query so each CTE or view it reads more than once runs once.
//!
//! DataFusion plans a CTE or a view in full at every place a query reads it,
//! so a model whose layers each read the one before several times plans and
//! runs the first layers many times over. [`share`] rewrites the query into
//! statements: one for each CTE or view read more than once, which the caller
//! stores as a table under the name the statement gives, and then the query
//! itself, reading those tables.

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;

use pyo3::prelude::*;
use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::ast::{
    Cte, Ident, ObjectName, ObjectNamePart, Query, Statement, TableAlias, TableAliasColumnDef,
    TableFactor, VisitMut, VisitorMut, With,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

/// A CTE or view, hoisted out of the query that defined it.
struct Def {
    query: Query,
    columns: Vec<TableAliasColumnDef>,
    /// The definitions it reads, once for each place it reads them.
    reads: Vec<usize>,
}

/// Why a query is left as it is.
struct Unsupported;

struct Hoister<'a> {
    prefix: &'a str,
    views: &'a HashMap<String, String>,
    defs: Vec<Def>,
    view_defs: HashMap<String, usize>,
    expanding: HashSet<String>,
}

/// The name a query's identifier resolves to: unquoted identifiers are
/// lowercased, as DataFusion does.
fn normalized(ident: &Ident) -> String {
    match ident.quote_style {
        Some(_) => ident.value.clone(),
        None => ident.value.to_lowercase(),
    }
}

impl Hoister<'_> {
    fn name(&self, index: usize) -> String {
        format!("{}{index}", self.prefix)
    }

    /// Hoists `query`'s own CTEs into definitions, and renames each table
    /// it reads that is a CTE in `scope` or a view to that definition's name.
    /// Returns the definitions it reads.
    fn hoist(
        &mut self,
        query: &mut Query,
        scope: &HashMap<String, usize>,
    ) -> Result<Vec<usize>, Unsupported> {
        let mut scope = scope.clone();
        if let Some(with) = query.with.take() {
            if with.recursive {
                return Err(Unsupported);
            }
            for cte in with.cte_tables {
                if cte.materialized.is_some() || cte.from.is_some() {
                    return Err(Unsupported);
                }
                let mut body = *cte.query;
                let reads = self.hoist(&mut body, &scope)?;
                self.defs.push(Def {
                    query: body,
                    columns: cte.alias.columns,
                    reads,
                });
                scope.insert(normalized(&cte.alias.name), self.defs.len() - 1);
            }
        }
        let mut renamer = Renamer {
            hoister: self,
            scope: &scope,
            top: true,
            reads: Vec::new(),
        };
        match query.visit(&mut renamer) {
            ControlFlow::Break(e) => Err(e),
            ControlFlow::Continue(()) => Ok(renamer.reads),
        }
    }

    /// The definition of the view `name`, hoisting it the first time.
    fn view(&mut self, name: &str) -> Result<usize, Unsupported> {
        if let Some(&index) = self.view_defs.get(name) {
            return Ok(index);
        }
        if !self.expanding.insert(name.to_string()) {
            return Err(Unsupported);
        }
        let sql = &self.views[name];
        let mut statements = Parser::parse_sql(&GenericDialect {}, sql).map_err(|_| Unsupported)?;
        let Some(Statement::CreateView(view)) = statements.pop() else {
            return Err(Unsupported);
        };
        if !statements.is_empty() || !view.columns.is_empty() {
            return Err(Unsupported);
        }
        let mut body = *view.query;
        // A view reads the catalog, not the CTEs of the query that reads it.
        let reads = self.hoist(&mut body, &HashMap::new())?;
        self.defs.push(Def {
            query: body,
            columns: Vec::new(),
            reads,
        });
        self.expanding.remove(name);
        let index = self.defs.len() - 1;
        self.view_defs.insert(name.to_string(), index);
        Ok(index)
    }
}

/// Renames the tables a query reads that are CTEs in scope, or views.
struct Renamer<'a, 'b> {
    hoister: &'a mut Hoister<'b>,
    scope: &'a HashMap<String, usize>,
    /// Whether the next query visited is the one being renamed, whose CTEs
    /// were hoisted already, rather than a subquery of it.
    top: bool,
    reads: Vec<usize>,
}

impl VisitorMut for Renamer<'_, '_> {
    type Break = Unsupported;

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<Unsupported> {
        // A subquery's own CTEs would shadow names in scope; leave such
        // queries to DataFusion.
        if !std::mem::take(&mut self.top) && query.with.is_some() {
            return ControlFlow::Break(Unsupported);
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<Unsupported> {
        let TableFactor::Table {
            name, alias, args, ..
        } = factor
        else {
            return ControlFlow::Continue(());
        };
        let [ObjectNamePart::Identifier(ident)] = name.0.as_slice() else {
            return ControlFlow::Continue(());
        };
        if args.is_some() {
            return ControlFlow::Continue(());
        }
        let key = normalized(ident);
        let index = match self.scope.get(&key) {
            Some(&index) => index,
            None if self.hoister.views.contains_key(&key) => match self.hoister.view(&key) {
                Ok(index) => index,
                Err(e) => return ControlFlow::Break(e),
            },
            None => return ControlFlow::Continue(()),
        };
        // Columns qualified by the name read, as in `x0.val`, still resolve.
        if alias.is_none() {
            *alias = Some(TableAlias {
                explicit: true,
                name: ident.clone(),
                columns: Vec::new(),
                at: None,
            });
        }
        *name = ObjectName::from(vec![Ident::new(self.hoister.name(index))]);
        self.reads.push(index);
        ControlFlow::Continue(())
    }
}

/// `query` with a `WITH` of the definitions it reads that aren't `shared`,
/// read through those that aren't either.
fn with_inlined(
    mut query: Query,
    reads: &[usize],
    defs: &[Def],
    shared: &[bool],
    hoister: &Hoister,
) -> Query {
    let mut needed = HashSet::new();
    let mut stack: Vec<usize> = reads.iter().copied().filter(|&i| !shared[i]).collect();
    while let Some(i) = stack.pop() {
        if needed.insert(i) {
            stack.extend(defs[i].reads.iter().copied().filter(|&j| !shared[j]));
        }
    }
    let mut needed: Vec<usize> = needed.into_iter().collect();
    // A definition only reads those made before it.
    needed.sort_unstable();
    if !needed.is_empty() {
        query.with = Some(With {
            with_token: AttachedToken::empty(),
            recursive: false,
            cte_tables: needed
                .into_iter()
                .map(|i| Cte {
                    alias: TableAlias {
                        explicit: false,
                        name: Ident::new(hoister.name(i)),
                        columns: defs[i].columns.clone(),
                        at: None,
                    },
                    query: Box::new(defs[i].query.clone()),
                    from: None,
                    materialized: None,
                    closing_paren_token: AttachedToken::empty(),
                })
                .collect(),
        });
    }
    query
}

/// Splits `query` so that each CTE or view it reads more than once runs once.
///
/// `views` maps the name of each view in the default schema to the SQL that
/// created it. Returns `(steps, query)`: each step is a table name and the
/// SQL whose rows that table must hold before the next step runs, and
/// `query` reads them. Names start with `prefix`. Returns `None` when
/// nothing is read twice, or for SQL this doesn't rewrite: a recursive CTE,
/// a subquery with CTEs of its own, or more than one statement.
#[pyfunction]
pub fn share(
    query: &str,
    views: HashMap<String, String>,
    prefix: &str,
) -> Option<(Vec<(String, String)>, String)> {
    let mut statements = Parser::parse_sql(&GenericDialect {}, query).ok()?;
    let Some(Statement::Query(mut query)) = statements.pop() else {
        return None;
    };
    if !statements.is_empty() {
        return None;
    }
    let mut hoister = Hoister {
        prefix,
        views: &views,
        defs: Vec::new(),
        view_defs: HashMap::new(),
        expanding: HashSet::new(),
    };
    let reads = hoister.hoist(&mut query, &HashMap::new()).ok()?;

    // How often each definition runs if every other is inlined: once for
    // each place it's read, counting only places in definitions that run.
    let defs = std::mem::take(&mut hoister.defs);
    let mut used = vec![false; defs.len()];
    let mut stack = reads.clone();
    while let Some(i) = stack.pop() {
        if !std::mem::replace(&mut used[i], true) {
            stack.extend(&defs[i].reads);
        }
    }
    let mut times = vec![0usize; defs.len()];
    for i in reads.iter().copied().chain(
        (0..defs.len())
            .filter(|&i| used[i])
            .flat_map(|i| defs[i].reads.iter().copied()),
    ) {
        times[i] += 1;
    }
    // Stored once, a definition read twice runs once; every other is then
    // inlined into the one place it's read, which runs once too.
    let shared: Vec<bool> = times.iter().map(|&n| n > 1).collect();
    if !shared.contains(&true) {
        return None;
    }

    let steps = (0..defs.len())
        .filter(|&i| shared[i])
        .map(|i| {
            let name = hoister.name(i);
            let mut not_shared = shared.clone();
            not_shared[i] = false;
            let select = Parser::parse_sql(&GenericDialect {}, &format!("SELECT * FROM {name}"))
                .ok()
                .and_then(|mut s| match s.pop() {
                    Some(Statement::Query(q)) => Some(*q),
                    _ => None,
                })
                .expect("a SELECT of one table parses");
            let step = with_inlined(select, &[i], &defs, &not_shared, &hoister);
            (name, step.to_string())
        })
        .collect();
    let query = with_inlined(*query, &reads, &defs, &shared, &hoister);
    Some((steps, query.to_string()))
}
