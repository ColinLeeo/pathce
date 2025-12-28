use clap::Args;
use duckdb::Connection;
use gcard_es::degreepiecewise::{FastCompressor, PiecewiseConstantFunction};
use gcard_es::{make_alt_key, AltKey, CompressedDegreeSeq, DegreeSeqGraphCompressed};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Args)]
pub struct CreateCatalogArgs {
    /// Specify the input DuckDB database file path (e.g., /path/to/database.duckdb).
    #[arg(short, long, value_name = "DATABASE_FILE")]
    database: PathBuf,
    /// Specify the output bincode file.
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output: PathBuf,
    /// Maximum number of edges in a path combination (default: 3).
    #[arg(short = 'k', long, default_value = "2", value_name = "MAX_EDGES")]
    max_k: usize,
    #[arg(short = 't', long, default_value = "1", value_name = "THREAD_NUM")]
    threads: usize,
    /// Compression method: SafeBound or FastCompressor.
    #[arg(short = 'm', long, value_name = "METHOD")]
    method: CompressionMethod,
    /// Base value for FastCompressor (default: 2).
    #[arg(long, default_value = "2")]
    base: u64,
    /// Whether to export full graph statistics.
    #[arg(long)]
    export_graph_data: bool,
    /// Epsilon value for SafeBound compression (default: 0.01).
    #[arg(long, default_value = "0.01")]
    epsilon: f64,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum CompressionMethod {
    SafeBound,
    FastCompressor,
}
// In GCard: a_(p)_b_(k)_c is equal c_(k)_b_(p)_a
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathPattern {
    vs: Vec<String>,
    es: Vec<String>,
}

impl PathPattern {
    pub fn new(vs: Vec<String>, es: Vec<String>) -> PathPattern {
        assert_eq!(vs.len(), es.len() + 1);
        let mut rvs = vs.clone();
        rvs.reverse();
        let mut res = es.clone();
        res.reverse();
        if (&vs, &es) <= (&rvs, &res) {
            PathPattern { vs, es }
        } else {
            PathPattern { vs: rvs, es: res }
        }
    }

    pub fn new_without_reverse(vs: Vec<String>, es: Vec<String>) -> PathPattern {
        assert_eq!(vs.len(), es.len() + 1);
        PathPattern { vs, es }
    }

    pub fn sort(&self) -> Self {
        let mut vs = self.vs.clone();
        vs.reverse();
        let mut es = self.es.clone();
        es.reverse();
        if (&vs, &es) <= (&self.vs, &self.es) {
            PathPattern { vs, es }
        } else {
            PathPattern { vs: self.vs.clone(), es: self.es.clone() }
        }
    }
}

impl PartialEq for PathPattern {
    fn eq(&self, other: &Self) -> bool {
        self.vs == other.vs && self.es == other.es
    }
}
impl Eq for PathPattern {}

impl Hash for PathPattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vs.hash(state);
        self.es.hash(state);
    }
}

type NodeId = i64;
type PathsByLen = HashMap<usize, HashSet<PathPattern>>;

type DegreeMap = HashMap<NodeId, u64>;

type PatternDegCache = HashMap<PathPattern, HashMap<String, DegreeMap>>;

type AdjCache = HashMap<(String, String, String), Arc<HashMap<NodeId, Vec<NodeId>>>>;

pub type BinCatalog = HashMap<AltKey, HashMap<String, Vec<u64>>>;

pub fn create_catalog(args: CreateCatalogArgs) {
    println!("{:#?}", args);

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

    println!("\n=== Schema Path Patterns ===");
    for (len, patterns) in &schema_path {
        println!("\n长度: {}, 数量: {}", len, patterns.len());

        for pattern in patterns {
            let mut parts = Vec::new();
            for i in 0..pattern.es.len() {
                parts.push(pattern.vs[i].clone());
                parts.push(pattern.es[i].clone());
            }
            if let Some(last_node) = pattern.vs.last() {
                parts.push(last_node.clone());
            }
            println!("  {}", parts.join("--"));
        }
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();
    let start = Instant::now();
    let mut cache: PatternDegCache = HashMap::new();
    let mut adj_cache: AdjCache = HashMap::new();
    for len in 1..=args.max_k {
        let pattern = schema_path.get(&len).unwrap();
        if len == 1 {
            compute_len1_degrees(&args.database, pattern, &mut cache).unwrap()
        } else {
            compute_len_ge2_degrees(&conn, &mut adj_cache, pattern, &mut cache).unwrap()
        }
    }

    let finish = Instant::now();
    println!("use time: {}", (finish - start).as_secs_f32());

    if args.export_graph_data {
        write_full_statistic(&args.output, &cache).expect("Error writing output");
    }

    let bin = build_bin_catalog(&cache);

    println!("\nCompressing graph statistics...");
    println!("Compression method: {:?}", args.method);
    if matches!(args.method, CompressionMethod::FastCompressor) {
        println!("Base value: {}", args.base);
    }

    let compress_start = Instant::now();
    let compressed_graph =
        compress_catalog(&bin, &args.method, args.base, args.epsilon, args.threads);
    let compress_finish = Instant::now();
    println!(
        "Compression time: {:.3}s",
        (compress_finish - compress_start).as_secs_f64()
    );

    compressed_graph
        .export_bincode(&args.output)
        .expect("Error writing compressed bincode");
    println!("Successfully saved compressed graph to {:?}", args.output);
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

#[allow(dead_code)]
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
    db_path: &PathBuf,
    patterns: &HashSet<PathPattern>,
    cache: &mut PatternDegCache,
) -> Result<(), duckdb::Error> {
    if patterns.is_empty() {
        return Ok(());
    }
    let partials: Vec<(PathPattern, HashMap<String, DegreeMap>)> = patterns
        .par_iter()
        .map_init(
            || Connection::open(db_path).expect("failed to open db"),
            |conn, pattern| {
                let table = &pattern.es[0];
                let src_name = &pattern.vs[0];
                let end_name = &pattern.vs[1];
                let edge_info = parse_edge_table(table).unwrap();
                let left_deg = load_degree_groupby(conn, table, "src")?;
                let right_deg = load_degree_groupby(conn, table, "dst")?;
                let mut m = HashMap::new();
                if src_name == &edge_info.src_label.to_string() {
                    m.insert(src_name.to_string(), left_deg);
                    m.insert(end_name.to_string(), right_deg);
                } else {
                    m.insert(src_name.to_string(), right_deg);
                    m.insert(end_name.to_string(), left_deg);
                }
                Ok::<_, duckdb::Error>((pattern.clone(), m))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    for (pattern, m) in partials {
        cache.entry(pattern).or_default().extend(m);
    }
    Ok(())
}

fn compute_len_ge2_degrees(
    conn: &Connection,
    adj_cache: &mut AdjCache,
    patterns: &HashSet<PathPattern>,
    cache: &mut PatternDegCache,
) -> Result<(), duckdb::Error> {
    let mut needed: HashSet<(String, String, String)> = HashSet::new();
    for pattern in patterns {
        let v_seq = &pattern.vs;
        let e_seq = &pattern.es;
        let left_node = &v_seq[0];
        let right_node = v_seq.last().unwrap();
        {
            let edge_info = parse_edge_table(e_seq[0].as_str()).unwrap();
            let left = edge_info.src_label == *left_node;
            let (left_col, right_col) = if left { ("src", "dst") } else { ("dst", "src") };
            needed.insert((
                e_seq[0].clone(),
                left_col.to_string(),
                right_col.to_string(),
            ));
            // println!(
            //     "need add {}-{}-{}",
            //     e_seq[0].clone(),
            //     left_col.to_string(),
            //     right_col.to_string()
            // );
        }

        {
            let rp_e0 = e_seq.last().unwrap();
            let edge_info = parse_edge_table(rp_e0.as_str()).unwrap();
            let left = *right_node == edge_info.src_label;
            let (left_col, right_col) = if left { ("src", "dst") } else { ("dst", "src") };
            needed.insert((rp_e0.clone(), left_col.to_string(), right_col.to_string()));
            // println!(
            //     "need add {}-{}-{}",
            //     rp_e0,
            //     left_col.to_string(),
            //     right_col.to_string()
            // );
        }
    }
    // for key in needed.iter() {
    //     println!(" to load in need {}-{}-{}", key.0, key.1, key.2);
    // }

    for key in needed {
        if !adj_cache.contains_key(&key) {
            // println!("load {}-{}-{} into adj_cache", key.0, key.1, key.2);
            let (table, left_col, right_col) = (&key.0, &key.1, &key.2);
            let adj = load_adj(conn, table, left_col, right_col)?;
            adj_cache.insert(key, Arc::new(adj));
        }
    }

    let adj_cache_ro: &AdjCache = &*adj_cache;
    let cache_ro: &PatternDegCache = &*cache;

    let computed: Vec<(PathPattern, String, DegreeMap, String, DegreeMap)> = patterns
        .par_iter()
        .map(|pattern| {
            let v_seq = &pattern.vs;
            let e_seq = &pattern.es;

            let suffix = PathPattern::new_without_reverse(v_seq[1..].to_vec(), e_seq[1..].to_vec());
            let left_node = v_seq[0].to_string();
            let right_node = v_seq.last().unwrap().to_string();

            let suffix_left_key = suffix.vs[0].to_string();
            let suffix_left_deg = cache_ro
                .get(&suffix.sort())
                .expect("not null")
                .get(&suffix_left_key)
                .expect("suffix left deg missing");

            let edge_info = parse_edge_table(e_seq[0].as_str()).unwrap();
            let left = edge_info.src_label == left_node;
            let (left_col, right_col) = if left { ("src", "dst") } else { ("dst", "src") };
            let key0 = (
                e_seq[0].clone(),
                left_col.to_string(),
                right_col.to_string(),
            );
            let adj0 = Arc::clone(adj_cache_ro.get(&key0).expect("adj0 missing in adj_cache"));

            let left_deg = dp_extend(adj0, suffix_left_deg);

            let rp = PathPattern::new_without_reverse(
                v_seq.iter().cloned().rev().collect(),
                e_seq.iter().cloned().rev().collect(),
            );
            let rsuffix = PathPattern::new_without_reverse(rp.vs[1..].to_vec(), rp.es[1..].to_vec());

            let rsuffix_left_key = rsuffix.vs[0].to_string();
            let rsuffix_left_deg = cache_ro
                .get(&rsuffix.sort())
                .expect("rsuffix left deg missing")
                .get(&rsuffix_left_key)
                .expect("rsuffix left deg missing");

            let rp_e0 = rp.es[0].clone();
            let edge_info = parse_edge_table(rp_e0.as_str()).unwrap();
            let left = right_node == edge_info.src_label;
            let (left_col, right_col) = if left { ("src", "dst") } else { ("dst", "src") };
            let rkey0 = (rp_e0, left_col.to_string(), right_col.to_string());
            let radj0 = Arc::clone(
                adj_cache_ro
                    .get(&rkey0)
                    .expect("radj0 missing in adj_cache"),
            );

            let right_deg = dp_extend(radj0, rsuffix_left_deg);

            Ok::<_, duckdb::Error>((pattern.clone(), left_node, left_deg, right_node, right_deg))
        })
        .collect::<Result<Vec<_>, duckdb::Error>>()?;

    for (pattern, left_node, left_deg, right_node, right_deg) in computed {
        let entry = cache.entry(pattern).or_default();
        entry.insert(left_node, left_deg);
        entry.insert(right_node, right_deg);
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
                .insert(PathPattern::new(node_seq.clone(), edge_seq.clone()));
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

fn clean_degree_map(deg: &DegreeMap) -> Vec<u64> {
    let v: Vec<u64> = deg.values().copied().filter(|&d| d > 0).collect();
    v
}

fn build_bin_catalog(cache: &PatternDegCache) -> BinCatalog {
    let mut out: BinCatalog = HashMap::new();
    for (path_pattern, degree_map) in cache {
        let key = make_alt_key(path_pattern.vs.as_slice(), path_pattern.es.as_slice()).sorted();

        let start_node = path_pattern.vs.first().unwrap().clone();
        let end_node = path_pattern.vs.last().unwrap().clone();
        let start_vec = clean_degree_map(degree_map.get(&start_node.clone()).unwrap());
        let end_vec = clean_degree_map(degree_map.get(&end_node.clone()).unwrap());
        if end_vec.is_empty() || start_vec.is_empty() {
            println!("Error")
        }
        if out.contains_key(&key) {
            let obj = out.get_mut(&key).unwrap();
            let vec_start = obj.get(&start_node).unwrap();
            let vec_end = obj.get(&end_node).unwrap();
            if *vec_start != start_vec {
                println!("conflict!!!")
            }
            if *vec_end != end_vec {
                println!("conflict!!!")
            }
            continue;
        }
        if out.contains_key(&key) {
            println!("conflict!!! in key")
        }
        let entry = out.entry(key).or_insert_with(HashMap::new);
        entry.insert(start_node, start_vec);
        entry.insert(end_node, end_vec);
    }
    out
}

#[allow(dead_code)]
fn write_bincode(path: &PathBuf, catalog: &BinCatalog) -> Result<(), std::io::Error> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    bincode::serialize_into(&mut w, catalog).expect("serailize failed");
    Ok(())
}

fn write_full_statistic(path: &PathBuf, catalog: &PatternDegCache) -> Result<(), std::io::Error> {
    let f = File::create(path.with_extension(format!(
        "{}.full",
        path.extension().unwrap_or_default().to_string_lossy()
    )))?;

    let mut w = BufWriter::new(f);

    bincode::serialize_into(&mut w, catalog).expect("serailize failed");
    Ok(())
}

fn compress_catalog(
    catalog: &BinCatalog,
    method: &CompressionMethod,
    base: u64,
    epsilon: f64,
    threads: usize,
) -> DegreeSeqGraphCompressed {
    let mut compressed_graph = DegreeSeqGraphCompressed::new();
    let compressed_items: Vec<_> = catalog
        .par_iter()
        .map(|(path, endpoints)| {
            let mut compressed_endpoints = HashMap::new();
            for (node_type, degree_seq) in endpoints {
                let compressed = match method {
                    CompressionMethod::SafeBound => {
                        let mut deg = degree_seq.clone();
                        deg.sort_unstable_by(|a, b| b.cmp(a));
                        let func =
                            PiecewiseConstantFunction::from_degree_sequence(&deg, epsilon, true)
                                .unwrap();
                        CompressedDegreeSeq::SafeBound {
                            function: func.clone(),
                        }
                    }
                    CompressionMethod::FastCompressor => {
                        let mut compressor = FastCompressor::new(base, threads);
                        compressor.compress(degree_seq);
                        let (len, base, counts) = compressor.get_result();
                        CompressedDegreeSeq::FastCompressor { len, base, counts }
                    }
                };
                compressed_endpoints.insert(node_type.clone(), compressed);
            }
            (path.clone(), compressed_endpoints)
        })
        .collect();

    for (path, compressed_endpoints) in compressed_items {
        if !compressed_endpoints.is_empty() {
            compressed_graph
                .edge_set_to_endpoints
                .insert(path, compressed_endpoints);
        }
    }

    compressed_graph
}
