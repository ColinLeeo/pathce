use crate::error::{GCardError, GCardResult};
use crate::parser::{Op, Pattern, PredTerm, Value};
use duckdb::Connection;

type NodeId = i64;

fn check_single_predicate(pred: &PredTerm, row: &duckdb::Row, col_idx: usize) -> GCardResult<bool> {
    let matches = match (&pred.op, &pred.value) {
        (Op::Eq, Value::Int(v)) => {
            let cv: i64 = row.get(col_idx)?;
            cv == *v
        }
        (Op::Eq, Value::Str(v)) => {
            let cv: String = row.get(col_idx)?;
            cv == *v
        }
        (Op::Eq, Value::Bool(v)) => {
            let cv: bool = row.get(col_idx)?;
            cv == *v
        }
        (Op::Ge, Value::Int(v)) => {
            let cv: i64 = row.get(col_idx)?;
            cv >= *v
        }
        (Op::Le, Value::Int(v)) => {
            let cv: i64 = row.get(col_idx)?;
            cv <= *v
        }
        (Op::Gt, Value::Int(v)) => {
            let cv: i64 = row.get(col_idx)?;
            cv > *v
        }
        (Op::Lt, Value::Int(v)) => {
            let cv: i64 = row.get(col_idx)?;
            cv < *v
        }
        // 字符串和布尔值的比较操作可以根据需要添加
        _ => {
            return Err(GCardError::InvalidData(format!(
                "Unsupported operation for value type"
            )))
        }
    };

    Ok(matches)
}

fn check_node_predicates(
    pattern: &Pattern,
    var: &str,
    row: &duckdb::Row,
    column_names: &[String],
) -> GCardResult<bool> {
    if let Some(node_pat) = pattern.nodes.get(var) {
        for pred in &node_pat.preds {
            let col_idx = column_names
                .iter()
                .position(|n| n == &pred.key)
                .ok_or_else(|| {
                    GCardError::InvalidData(format!("Column '{}' not found in table", pred.key))
                })?;

            if !check_single_predicate(pred, row, col_idx)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn check_edge_predicates(
    pattern: &Pattern,
    var: &str,
    row: &duckdb::Row,
    column_names: &[String],
) -> GCardResult<bool> {
    if let Some(edge_pat) = pattern.edges.get(var) {
        for pred in &edge_pat.preds {
            let col_idx = column_names
                .iter()
                .position(|n| n.to_lowercase().as_str() == &pred.key.to_lowercase())
                .ok_or_else(|| {
                    GCardError::InvalidData(format!("Column '{}' not found in table", pred.key))
                })?;

            if !check_single_predicate(pred, row, col_idx)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn get_column_names(conn: &Connection, table_name: &str) -> GCardResult<Vec<String>> {
    // 使用 DESCRIBE 或 PRAGMA 获取列信息
    let sql = format!("DESCRIBE \"{}\"", table_name);
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| GCardError::InvalidData(format!("SQL prepare error: {}", e)))?;
    
    let mut rows = stmt
        .query([])
        .map_err(|e| GCardError::InvalidData(format!("SQL query error: {}", e)))?;
    
    let mut names = Vec::new();
    while let Some(row) = rows.next().map_err(|e| GCardError::InvalidData(format!("SQL row error: {}", e)))? {
        // DESCRIBE 返回的第一列是 column_name
        let col_name: String = row.get(0)?;
        names.push(col_name);
    }
    
    Ok(names)
}
fn wander_join_sample(
    conn: &Connection,
    pattern: &Pattern,
    expr_tables: &[String],
    expr_vars: &[String],
) -> GCardResult<Option<bool>> {
    // expr_tables: [Node, Edge, Node, Edge, Node, ...]
    // expr_vars: [node_var, edge_var, node_var, edge_var, node_var, ...]
    let start_table = &expr_tables[0];
    let start_var = &expr_vars[0];

    let column_names = get_column_names(conn, start_table)?;

    let sql = format!(
        "SELECT * FROM \"{}\" ORDER BY RANDOM() LIMIT 1",
        start_table
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| GCardError::InvalidData(format!("SQL prepare error: {}", e)))?;

    let mut rows = stmt
        .query([])
        .map_err(|e| GCardError::InvalidData(format!("SQL query error: {}", e)))?;

    let start_row = match rows
        .next()
        .map_err(|e| GCardError::InvalidData(format!("SQL row error: {}", e)))?
    {
        Some(row) => row,
        None => return Ok(None),
    };

    let mut all_predicates_ok = check_node_predicates(pattern, start_var, &start_row, &column_names)?;

    let id_idx = column_names
        .iter()
        .position(|n| n == "id")
        .ok_or_else(|| GCardError::InvalidData("Column 'id' not found".to_string()))?;
    let mut current_node_id: NodeId = start_row.get(id_idx)?;

    for i in (1..expr_tables.len()).step_by(2) {
        if i + 1 >= expr_tables.len() {
            break;
        }

        let edge_table = &expr_tables[i];
        let edge_var = &expr_vars[i];

        let edge_column_names = get_column_names(conn, edge_table)?;

        let sql = format!(
            "SELECT * FROM \"{}\" WHERE src = {} ORDER BY RANDOM() LIMIT 1",
            edge_table, current_node_id
        );

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| GCardError::InvalidData(format!("SQL prepare error: {}", e)))?;

        let mut rows = stmt
            .query([])
            .map_err(|e| GCardError::InvalidData(format!("SQL query error: {}", e)))?;

        let edge_row = match rows
            .next()
            .map_err(|e| GCardError::InvalidData(format!("SQL row error: {}", e)))?
        {
            Some(row) => row,
            None => return Ok(None),
        };

        let edge_pred_ok = check_edge_predicates(pattern, edge_var, &edge_row, &edge_column_names)?;
        if !edge_pred_ok {
            all_predicates_ok = false;
        }

        let dst_idx = edge_column_names
            .iter()
            .position(|n| n == "dst")
            .ok_or_else(|| GCardError::InvalidData("Column 'dst' not found".to_string()))?;
        let next_node_id: NodeId = edge_row.get(dst_idx)?;

        let next_node_table = &expr_tables[i + 1];
        let next_node_var = &expr_vars[i + 1];

        let node_column_names = get_column_names(conn, next_node_table)?;

        let sql = format!(
            "SELECT * FROM \"{}\" WHERE id = {}",
            next_node_table, next_node_id
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| GCardError::InvalidData(format!("SQL prepare error: {}", e)))?;

        let mut rows = stmt
            .query([])
            .map_err(|e| GCardError::InvalidData(format!("SQL query error: {}", e)))?;

        let next_node_row = match rows
            .next()
            .map_err(|e| GCardError::InvalidData(format!("SQL row error: {}", e)))?
        {
            Some(row) => row,
            None => return Ok(None),
        };

        let next_node_pred_ok = check_node_predicates(pattern, next_node_var, &next_node_row, &node_column_names)?;
        if !next_node_pred_ok {
            all_predicates_ok = false;
        }

        current_node_id = next_node_id;
    }


    if all_predicates_ok {
        Ok(Some(true))
    } else {
        Ok(Some(false))
    }
}

pub fn wander_join_selectivity(
    conn: &Connection,
    pattern: &Pattern,
    expr_tables: &[String],
    expr_vars: &[String],
    num_samples: usize,
) -> GCardResult<f64> {
    if expr_tables.len() != expr_vars.len() {
        return Err(GCardError::InvalidData(format!(
            "expr_tables and expr_vars must have the same length: {} != {}",
            expr_tables.len(),
            expr_vars.len()
        )));
    }

    if expr_tables.is_empty() {
        return Err(GCardError::InvalidData(
            "expr_tables cannot be empty".to_string(),
        ));
    }

    let mut structure_match_count = 0;
    let mut predicate_match_count = 0;

    for _ in 0..num_samples {
        match wander_join_sample(conn, pattern, expr_tables, expr_vars) {
            Ok(None) => {}
            Ok(Some(false)) => {
                structure_match_count += 1;
            }
            Ok(Some(true)) => {
                structure_match_count += 1;
                predicate_match_count += 1;
            }
            Err(e) => return Err(e),
        }
    }

    if structure_match_count == 0 {
        return Ok(0.0);
    }

    Ok(predicate_match_count as f64 / structure_match_count as f64)
}
