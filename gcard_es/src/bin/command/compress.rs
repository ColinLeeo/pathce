use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use clap::Args;
use gcard_es::degreepiecewise::FastCompressor;
use gcard_es::degreepiecewise::PiecewiseConstantFunction;
use gcard_es::{AltKey, DegreeSeqGraph};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Args)]
pub struct CompressArgs {
    /// Specify the input bincode file path.
    #[arg(short, long, value_name = "INPUT_FILE")]
    input: PathBuf,
    /// Compression method: SafeBound or FastCompressor.
    #[arg(short, long, value_name = "METHOD")]
    method: CompressionMethod,
    /// Specify the output compressed file path.
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output: PathBuf,
    /// Base value for FastCompressor (default: 2.0).
    #[arg(long, default_value = "2.0")]
    base: f64,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum CompressionMethod {
    SafeBound,
    FastCompressor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressedDegreeSeq {
    SafeBound {
        function: PiecewiseConstantFunction,
    },
    FastCompressor {
        len: usize,
        base: f64,
        counts: Vec<u64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedDegreeSeqGraph {
    pub edge_set_to_endpoints: HashMap<AltKey, HashMap<String, CompressedDegreeSeq>>,
}


fn fmt(d: Duration) -> String {
    format!("{:.3}s", d.as_secs_f64())
}

impl CompressedDegreeSeqGraph {
    pub fn new() -> Self {
        Self {
            edge_set_to_endpoints: HashMap::new(),
        }
    }

    pub fn export_bincode<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> gcard_es::error::GCardResult<()> {
        use std::fs::File;
        use std::io::BufWriter;
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|e| {
            gcard_es::error::GCardError::InvalidData(format!("Failed to serialize: {}", e))
        })?;
        Ok(())
    }
}

pub fn compress(args: CompressArgs) {
    println!("Compressing graph statistics...");
    println!("Input file: {:?}", args.input);
    println!("Compression method: {:?}", args.method);
    println!("Output file: {:?}", args.output);
    if matches!(args.method, CompressionMethod::FastCompressor) {
        println!("Base value: {}", args.base);
    }
    let t_total0 = Instant::now();
    let graph = match DegreeSeqGraph::import_bincode(&args.input) {
        Ok(g) => {
            println!(
                "Successfully loaded graph with {} edge sets",
                g.num_edge_sets()
            );
            g
        }
        Err(e) => {
            eprintln!("Error reading bincode file: {}", e);
            std::process::exit(1);
        }
    };

    let mut compressed_graph = CompressedDegreeSeqGraph::new();

    let mut total_paths = 0;
    let mut total_sequences = 0;

    let t_comp0 = Instant::now();
    for (path, endpoints) in &graph.edge_set_to_endpoints {
        total_paths += 1;
        let mut compressed_endpoints = HashMap::new();

        for (node_type, degree_seq) in endpoints {
            total_sequences += 1;
            let compressed = match args.method {
                CompressionMethod::SafeBound => {
                    let func = PiecewiseConstantFunction::from_degree_sequence(
                        degree_seq, 0.01, true,
                    ).unwrap();
                    CompressedDegreeSeq::SafeBound {
                        function: func.clone(),
                    }
                }
                CompressionMethod::FastCompressor => {
                    let mut compressor = FastCompressor::new(args.base);
                    compressor.compress(degree_seq.clone());
                    let (len, base, counts) = compressor.get_result();
                    CompressedDegreeSeq::FastCompressor { len, base, counts }
                }
            };
            compressed_endpoints.insert(node_type.clone(), compressed);
        }

        if !compressed_endpoints.is_empty() {
            compressed_graph
                .edge_set_to_endpoints
                .insert(path.clone(), compressed_endpoints);
        }
    }

    let comp_dt = t_comp0.elapsed();

    println!(
        "Compressed {} paths with {} degree sequences (compress time: {})",
        total_paths,
        total_sequences,
        fmt(comp_dt)
    );

    println!(
        "Compressed {} paths with {} degree sequences",
        total_paths, total_sequences
    );

    match compressed_graph.export_bincode(&args.output) {
        Ok(_) => {
            println!("Successfully saved compressed graph to {:?}", args.output);
        }
        Err(e) => {
            eprintln!("Error saving compressed graph: {}", e);
            std::process::exit(1);
        }
    }
}
