use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use clap::Args;
use duckdb::Connection;
use serde::{Deserialize, Serialize};

#[derive(Debug, Args)]
pub struct CreateCatalogArgs {
    /// Specify the input DuckDB database file path (e.g., /path/to/database.duckdb).
    #[arg(short, long, value_name = "DATABASE_FILE")]
    database: PathBuf,
    /// Specify the output bincode file.
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output: PathBuf,
    /// Maximum number of edges in a path combination (default: 3).
    #[arg(short = 'k', long, default_value = "3", value_name = "MAX_EDGES")]
    max_k: usize,
}

type PathPattern = (Vec<String>, Vec<String>);

type NodeId = i64;
type PathsByLen = HashMap<usize, HashSet<PathPattern>>;

type DegreeMap = HashMap<NodeId, u64>;

type PatternDegCache = HashMap<PathPattern, HashMap<String, DegreeMap>>;

type AdjCache = HashMap<(String, String, String), Arc<HashMap<NodeId, Vec<NodeId>>>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AltKey(pub Vec<String>);

impl AltKey {
    pub fn sorted(&self) -> AltKey {
        let mut v = self.0.clone();
        v.sort();
        AltKey(v)
    }
}

pub type BinCatalog = HashMap<AltKey, HashMap<String, Vec<u64>>>;

pub fn create_catalog(args: CreateCatalogArgs) {
    println!("{:#?}", args);

    // let start = Instant::now();

    let conn = match Connection::open(&args.database) {
        Ok(conn) => {
            println!(
                "Successfully connected to database file: {:?}",
                args.database
            );
            conn
        }
        Err(e) => {
            eprintln!(
                "Error connecting to database file {:?}: {}",
                args.database, e
            );
            std::process::exit(1);
        }
    };

    if let Err(e) = conn.execute_batch("SET max_expression_depth = 9999999") {
        eprintln!("Warning: Failed to set max_expression_depth: {}", e);
    }

    let tables = match get_tables(&conn) {
        Ok(tables) => {
            println!("Found {} tables in database", tables.len());
            for table in &tables {
                println!("  - {}", table);
            }
            tables
        }
        Err(e) => {
            eprintln!("Error querying tables: {}", e);
            std::process::exit(1);
        }
    };

    if tables.is_empty() {
        eprintln!("Warning: No tables found in database");
        return;
    }

    let edges = match parse_edge_tables(&tables) {
        Ok(edges) => {
            println!("\nParsed {} edge tables:", edges.len());
            for (edge_name, edge) in &edges {
                println!(
                    "  {}: {} --[{}]--> {}",
                    edge_name, edge.src_label, edge.edge_name, edge.dst_label
                );
            }
            edges
        }
        Err(e) => {
            eprintln!("Error parsing edge tables: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "\nEnumerating edge combinations with max_k = {}...",
        args.max_k
    );

    let schema_path = enumerate_all_paths_walks_in_schema(&edges, args.max_k);

    // 打印 schema_path 信息
    println!("\n=== Schema Path Patterns ===");
    for (len, patterns) in &schema_path {
        println!("\n长度: {}, 数量: {}", len, patterns.len());

        for (node_seq, edge_seq) in patterns {
            // 交替打印节点和边，形成链表格式
            let mut parts = Vec::new();
            for i in 0..edge_seq.len() {
                parts.push(node_seq[i].clone());
                parts.push(edge_seq[i].clone());
            }
            // 添加最后一个节点
            if let Some(last_node) = node_seq.last() {
                parts.push(last_node.clone());
            }
            println!("  {}", parts.join("--"));
        }
    }
    println!();

    let mut cache: PatternDegCache = HashMap::new();
    let mut adj_cache: AdjCache = HashMap::new();
    for len in 1..=args.max_k {
        let pattern = schema_path.get(&len).unwrap();
        if len == 1 {
            compute_len1_degrees(&conn, pattern, &mut cache).unwrap()
        } else {
            compute_len_ge2_degrees(&conn, &mut adj_cache, pattern, &mut cache).unwrap()
        }
    }

    // let serialized = build_serialized_catalog(&cache);
    //
    // write_catalog_json(&serialized, &args.output).expect("failed to write catalog json");

    let bin = build_bin_catalog(&cache);
    write_bincode(&args.output, &bin).expect("Error writing bincode");
}

fn get_tables(conn: &Connection) -> Result<Vec<String>, duckdb::Error> {
    let mut stmt = conn.prepare(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
    )?;
    let tables = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(tables)
}

fn load_degree_groupby(
    conn: &Connection,
    table: &str,
    col: &str,
) -> Result<DegreeMap, duckdb::Error> {
    let sql = format!(
        "SELECT {c} as id, count(*) as deg from \"{t}\" group by {c}",
        c = col,
        t = table
    );

    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out: DegreeMap = HashMap::new();
    while let Some(r) = rows.next()? {
        let id: NodeId = r.get(0)?;
        let deg: u64 = r.get::<_, u64>(1)?;
        out.insert(id, deg);
    }
    Ok(out)
}

fn load_adj(
    conn: &Connection,
    table: &str,
    left_col: &str,
    right_col: &str,
) -> Result<HashMap<NodeId, Vec<NodeId>>, duckdb::Error> {
    let sql = format!(
        "SELECT {l} as l, {r} as r from \"{t}\"",
        l = left_col,
        r = right_col,
        t = table
    );
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    while let Some(r) = rows.next()? {
        let l: NodeId = r.get(0)?;
        let r: NodeId = r.get(1)?;
        adj.entry(l).or_default().push(r);
    }
    Ok(adj)
}

fn get_adj_left_right(
    conn: &Connection,
    adj_cache: &mut AdjCache,
    table: &str,
    left: bool,
) -> Result<Arc<HashMap<NodeId, Vec<NodeId>>>, duckdb::Error> {
    let left_col = if left { "src" } else { "dst" };
    let right_col = if left { "dst" } else { "src" };
    let key = (
        table.to_string(),
        left_col.to_string(),
        right_col.to_string(),
    );
    if !adj_cache.contains_key(&key) {
        let adj = load_adj(conn, table, left_col, right_col)?;
        adj_cache.insert(key.clone(), Arc::new(adj));
    }
    Ok(Arc::clone(adj_cache.get(&key).unwrap()))
}

fn dp_extend(adj: Arc<HashMap<NodeId, Vec<NodeId>>>, suffix_deg: &DegreeMap) -> DegreeMap {
    let mut out = HashMap::new();
    for (u, nbrs) in adj.iter() {
        let mut sum = 0;
        for v in nbrs {
            if let Some(d) = suffix_deg.get(&v) {
                sum += *d;
            }
        }
        out.insert(*u, sum);
    }
    out
}

fn compute_len1_degrees(
    conn: &Connection,
    patterns: &HashSet<PathPattern>,
    cache: &mut PatternDegCache,
) -> Result<(), duckdb::Error> {
    for (v_seq, e_seq) in patterns {
        let table = &e_seq[0];
        let src_name = &v_seq[0];
        let end_name = &v_seq[1];
        let edge_info = parse_edge_table(table).unwrap();
        let left_deg = load_degree_groupby(conn, table, "src")?;
        let right_deg = load_degree_groupby(conn, table, "dst")?;
        if src_name == &edge_info.src_label.to_string() {
            cache
                .entry((v_seq.clone(), e_seq.clone()))
                .or_default()
                .insert(src_name.to_string(), left_deg.clone());
            cache
                .entry((v_seq.clone(), e_seq.clone()))
                .or_default()
                .insert(end_name.to_string(), right_deg.clone());
        } else {
            cache
                .entry((v_seq.clone(), e_seq.clone()))
                .or_default()
                .insert(src_name.to_string(), right_deg.clone());
            cache
                .entry((v_seq.clone(), e_seq.clone()))
                .or_default()
                .insert(end_name.to_string(), left_deg.clone());
        }
    }
    Ok(())
}

fn compute_len_ge2_degrees(
    conn: &Connection,
    adj_cache: &mut AdjCache,
    patterns: &HashSet<PathPattern>,
    cache: &mut PatternDegCache,
) -> Result<(), duckdb::Error> {
    for pattern in patterns {
        let (v_seq, e_seq) = pattern;
        let suffix = (v_seq[1..].to_vec(), e_seq[1..].to_vec());
        let left_node = v_seq[0].to_string();
        let right_node = v_seq.last().unwrap().to_string();
        let suffix_left_deg = cache[&suffix].get(&suffix.0[0].to_string()).unwrap();
        let edge_info = parse_edge_table(e_seq[0].as_str()).unwrap();
        let adj0 =
            get_adj_left_right(conn, adj_cache, &e_seq[0], edge_info.src_label == left_node)?;
        let left_deg = dp_extend(adj0, &suffix_left_deg);
        let rp: PathPattern = (
            v_seq.iter().cloned().rev().collect::<Vec<_>>(),
            e_seq.iter().cloned().rev().collect::<Vec<_>>(),
        );

        let rsuffix = (rp.0[1..].to_vec(), rp.1[1..].to_vec());
        let rsuffix_left_deg = cache[&rsuffix].get(&rsuffix.0[0].to_string()).unwrap();
        let edge_info = parse_edge_table(rp.1[0].as_str()).unwrap();
        let radj0 =
            get_adj_left_right(conn, adj_cache, &rp.1[0], right_node == edge_info.src_label)?;
        let right_deg = dp_extend(radj0, &rsuffix_left_deg);
        cache
            .entry(pattern.clone())
            .or_default()
            .insert(left_node, left_deg.clone());
        cache
            .entry(pattern.clone())
            .or_default()
            .insert(right_node, right_deg.clone());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EdgeInfo {
    edge_name: String,
    src_label: String,
    dst_label: String,
}

fn parse_edge_table(table_name: &str) -> Option<EdgeInfo> {
    if let Some(last_underscore) = table_name.rfind('_') {
        let dst_label = &table_name[last_underscore + 1..];

        let before_last = &table_name[..last_underscore];
        if let Some(second_last_underscore) = before_last.rfind('_') {
            let src_label = &table_name[..second_last_underscore];

            return Some(EdgeInfo {
                edge_name: table_name.to_string(),
                src_label: src_label.to_string(),
                dst_label: dst_label.to_string(),
            });
        }
    }

    None
}

fn parse_edge_tables(tables: &[String]) -> Result<HashMap<String, EdgeInfo>, String> {
    let mut edges = HashMap::new();

    for table in tables {
        if let Some(edge_info) = parse_edge_table(table) {
            edges.insert(edge_info.edge_name.clone(), edge_info);
        }
    }

    if edges.is_empty() {
        return Err("No valid edge tables found".to_string());
    }

    Ok(edges)
}

fn collect_vertex_and_edge_types(
    edges: &HashMap<String, EdgeInfo>,
) -> (HashSet<String>, HashSet<String>) {
    let mut vertices = HashSet::new();
    let mut edge_type = HashSet::new();
    for (_k, e) in edges.iter() {
        vertices.insert(e.src_label.clone());
        vertices.insert(e.dst_label.clone());
        edge_type.insert(e.edge_name.clone());
    }
    (vertices, edge_type)
}

fn build_undirected_adj(
    edges: &HashMap<String, EdgeInfo>,
) -> HashMap<String, Vec<(String, String)>> {
    let mut adj: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (_k, e) in edges {
        adj.entry(e.src_label.clone())
            .or_default()
            .push((e.edge_name.clone(), e.dst_label.clone()));
        adj.entry(e.dst_label.clone())
            .or_default()
            .push((e.edge_name.clone(), e.src_label.clone()));
    }
    adj
}

fn enumerate_all_paths_walks_in_schema(
    edges: &HashMap<String, EdgeInfo>,
    max_len: usize,
) -> PathsByLen {
    let (vertex_types, _edge_types) = collect_vertex_and_edge_types(edges);
    let adj = build_undirected_adj(edges);

    let mut out: PathsByLen = HashMap::new();

    fn dfs(
        adj: &HashMap<String, Vec<(String, String)>>,
        max_len: usize,
        node_seq: &mut Vec<String>,
        edge_seq: &mut Vec<String>,
        out: &mut PathsByLen,
    ) {
        let cur_len = edge_seq.len();
        if cur_len > 0 {
            out.entry(cur_len)
                .or_default()
                .insert((node_seq.clone(), edge_seq.clone()));
        }

        if cur_len == max_len {
            return;
        }

        let cur_node = node_seq.last().unwrap().clone();
        if let Some(nbrs) = adj.get(&cur_node) {
            for (edge_name, next_node) in nbrs.iter() {
                edge_seq.push(edge_name.clone());
                node_seq.push(next_node.clone());
                dfs(adj, max_len, node_seq, edge_seq, out);

                node_seq.pop();
                edge_seq.pop();
            }
        }
    }

    for start in vertex_types.iter() {
        let mut node_seq = vec![start.clone()];
        let mut edge_seq: Vec<String> = Vec::new();
        dfs(&adj, max_len, &mut node_seq, &mut edge_seq, &mut out);
    }

    out
}

#[derive(Serialize)]
struct DegreeVec {
    degrees: Vec<u64>,
}

type SerializedPattern = HashMap<String, DegreeVec>;
type SerializedCatalog = HashMap<String, SerializedPattern>;
fn clean_degree_map(deg: &DegreeMap) -> Vec<u64> {
    let mut v: Vec<u64> = deg.values().copied().filter(|&d| d > 0).collect();

    v.sort_unstable_by(|a, b| b.cmp(a));

    v
}

fn build_serialized_catalog(cache: &PatternDegCache) -> SerializedCatalog {
    let mut out: SerializedCatalog = HashMap::new();

    for ((node_seq, edge_seq), degree_map) in cache {
        let edge_key = edge_seq.join(",");

        let start_label = node_seq.first().unwrap().clone();
        let end_label = node_seq.last().unwrap().clone();
        let start_node = node_seq.first().unwrap().clone();
        let end_node = node_seq.last().unwrap().clone();

        let start_vec = clean_degree_map(degree_map.get(&start_node.clone()).unwrap());
        let end_vec = clean_degree_map(degree_map.get(&end_node.clone()).unwrap());
        if edge_seq.len() == 2 && (start_vec.is_empty() || end_vec.is_empty()) {
            println!("No edge found for {}", edge_key);
        }

        let entry = out.entry(edge_key).or_insert_with(HashMap::new);

        entry.insert(start_label, DegreeVec { degrees: start_vec });
        entry.insert(end_label, DegreeVec { degrees: end_vec });
    }

    out
}

fn build_bin_catalog(cache: &PatternDegCache) -> BinCatalog {
    let mut out: BinCatalog = HashMap::new();
    for ((node_seq, edge_seq), degree_map) in cache {
        let key = make_alt_key(node_seq, edge_seq);

        let start_node = node_seq.first().unwrap().clone();
        let end_node = node_seq.last().unwrap().clone();
        let start_vec = clean_degree_map(degree_map.get(&start_node.clone()).unwrap());
        let end_vec = clean_degree_map(degree_map.get(&end_node.clone()).unwrap());
        if out.contains_key(&key) {
            let obj = out.get_mut(&key).unwrap();
            let vec_start = obj.get(&start_node).unwrap();
            let vec_end = obj.get(&end_node).unwrap();
            if (*vec_start!= start_vec) {
                println!("conflict!!!")
            }
            if (*vec_end != end_vec) {
                println!("conflict!!!")
            }
            continue;
        }
        let entry = out.entry(key).or_insert_with(HashMap::new);
        entry.insert(start_node, start_vec);
        entry.insert(end_node, end_vec);
    }
    out
}

fn write_bincode(path: &PathBuf, catalog: &BinCatalog) -> Result<(), std::io::Error> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    bincode::serialize_into(&mut w, catalog).expect("serailize failed");
    Ok(())
}

fn make_alt_key(node_seq: &[String], edge_seq: &[String]) -> AltKey {
    let mut out = Vec::with_capacity(node_seq.len() + edge_seq.len());
    for i in 0..node_seq.len() {
        out.push(node_seq[i].clone());
        if i < edge_seq.len() {
            out.push(edge_seq[i].clone());
        }
    }
    AltKey(out)
}

use std::fs::File;
use std::io::BufWriter;

fn write_catalog_json(
    catalog: &SerializedCatalog,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, catalog)?;
    Ok(())
}
