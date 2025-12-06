use std::path::PathBuf;

use clap::Args;
use gcard_es::DegreeSeqGraph;

#[derive(Debug, Args)]
pub struct ShowArgs {
    /// Specify the bincode file to read.
    #[arg(short, long, value_name = "BINCODE_FILE")]
    input: PathBuf,
    /// Show detailed information for each path.
    #[arg(long)]
    detailed: bool,
}

pub fn show(args: ShowArgs) {
    let graph = match DegreeSeqGraph::import_bincode(&args.input) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error reading bincode file: {}", e);
            std::process::exit(1);
        }
    };

    println!("Total paths: {}", graph.num_paths());
    println!();

    if args.detailed {
        for (idx, (key, degree_piecewise)) in graph.iter().enumerate() {
            println!("Path {}:", idx + 1);
            println!("  Source: {}", key.src_node_type);
            println!("  Destination: {}", key.dst_node_type);
            println!("  Path length: {}", key.path.steps.len());
            println!("  Steps:");
            for (step_idx, step) in key.path.steps.iter().enumerate() {
                println!(
                    "    {}: {} --[{}]--> {}",
                    step_idx + 1, step.src_type, step.edge_type, step.dst_type
                );
            }
            let degree_seq = degree_piecewise.get_degree_sequence();
            println!("  Degree sequence length: {}", degree_seq.len());
            println!("  Degree sequence (first 10): {:?}", 
                     &degree_seq[..degree_seq.len().min(10)]);
            if degree_seq.len() > 10 {
                println!("  ... ({} more values)", degree_seq.len() - 10);
            }
            println!("  Piecewise function segments: {}", 
                     degree_piecewise.get_piecewise_function().constants.len());
            println!("  Total rows: {:.2}", degree_piecewise.get_num_rows());
            println!();
        }
    } else {
        println!("Paths summary:");
        for (idx, key) in graph.path_keys().enumerate() {
            println!(
                "  {}: {} --[{} steps]--> {}",
                idx + 1,
                key.src_node_type,
                key.path.steps.len(),
                key.dst_node_type
            );
        }
    }
}

