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

    println!("Total edge sets: {}", graph.num_edge_sets());
    println!();

    // if args.detailed {
    //     for (idx, (edge_set, endpoints)) in graph.iter().enumerate() {
    //         println!("Edge Set {}:", idx + 1);
    //         println!("  Edge labels: {:?}", edge_set.iter().collect::<Vec<_>>());
    //         println!("  Number of endpoints: {}", endpoints.len());
    //         println!();
    //
    //         for (node_type, degree_seq) in endpoints {
    //             println!("  Endpoint: {}", node_type);
    //             println!("    Degree sequence length: {}", degree_seq.len());
    //
    //             // 显示前10个值
    //             if !degree_seq.is_empty() {
    //                 let preview_len = degree_seq.len().min(10);
    //                 println!("    Degree sequence (first {}): {:?}",
    //                          preview_len, &degree_seq[..preview_len]);
    //                 if degree_seq.len() > 10 {
    //                     println!("    ... ({} more values)", degree_seq.len() - 10);
    //                 }
    //             }
    //
    //             // 如果详细模式，也显示 piecewise function 信息
    //             if let Ok(degree_piecewise) = DegreePiecewise::from_degree_sequence_default(degree_seq.clone()) {
    //                 let pcf = degree_piecewise.get_piecewise_function();
    //                 println!("    Piecewise function segments: {}", pcf.constants.len());
    //                 println!("    Total rows: {:.2}", degree_piecewise.get_num_rows());
    //             }
    //             println!();
    //         }
    //     }
    // } else {
    //     println!("Edge sets summary:");
    //     for (idx, (edge_set, endpoints)) in graph.iter().enumerate() {
    //         let edge_labels: Vec<&str> = edge_set.iter().map(|s| s.as_str()).collect();
    //         let node_types: Vec<&str> = endpoints.keys().map(|s| s.as_str()).collect();
    //         println!(
    //             "  {}: {:?} -> endpoints: {:?}",
    //             idx + 1,
    //             edge_labels,
    //             node_types
    //         );
    //     }
    // }
}
