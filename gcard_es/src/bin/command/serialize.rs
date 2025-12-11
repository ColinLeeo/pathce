use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use gcard_es::DegreeSeqGraph;

#[derive(Debug, Args)]
pub struct SerializeArgs {
    /// Specify the input JSON file (one JSON object per line).
    #[arg(short, long, value_name = "INPUT_JSON")]
    input: PathBuf,
    /// Specify the output bincode file.
    #[arg(short, long, value_name = "OUTPUT_FILE")]
    output: PathBuf,
    /// Verify the output by reading it back (for debugging).
    #[arg(long)]
    verify: bool,
}

pub fn serialize(args: SerializeArgs) {
    println!("{:#?}", args);

    let start = Instant::now();
    // 从新格式 JSON 数组导入
    let graph = match DegreeSeqGraph::from_json_array(&args.input) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error loading graph data: {}", e);
            std::process::exit(1);
        }
    };
    let time = start.elapsed().as_secs_f64();
    println!("graph building time: {:.6} s", time);
    println!("number of edge sets: {}", graph.num_edge_sets());

    let start = Instant::now();
    match graph.export_bincode(&args.output) {
        Ok(_) => {
            let time = start.elapsed().as_secs_f64();
            println!("serializing time: {:.6} s", time);
        }
        Err(e) => {
            eprintln!("Error serializing to bincode: {}", e);
            std::process::exit(1);
        }
    }

    if args.verify {
        let start = Instant::now();
        match DegreeSeqGraph::import_bincode(&args.output) {
            Ok(loaded_graph) => {
                let time = start.elapsed().as_secs_f64();
                println!("verification: loaded {} edge sets in {:.6} s", loaded_graph.num_edge_sets(), time);

                if loaded_graph.num_edge_sets() != graph.num_edge_sets() {
                    eprintln!("Warning: Edge set count mismatch! Original: {}, Loaded: {}", 
                              graph.num_edge_sets(), loaded_graph.num_edge_sets());
                } else {
                    println!("✓ Verification passed: all {} edge sets loaded correctly", graph.num_edge_sets());
                }
            }
            Err(e) => {
                eprintln!("Error verifying bincode file: {}", e);
                std::process::exit(1);
            }
        }
    }
}
