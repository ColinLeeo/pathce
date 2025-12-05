mod command;

use std::thread;

use clap::Parser;
use mimalloc::MiMalloc;

use crate::command::*;

#[global_allocator]
static ALLOC: MiMalloc = MiMalloc;

/// An integrated framework for cardinality estimation of subgraph queries.
#[derive(Parser)]
#[command(version, about)]
#[command(propagate_version = true)]
enum Command {
    /// Load the JSON graph dataset and serialize it into a (bincode) graph file.
    Serialize(SerializeArgs),
    /// Show the contents of a bincode graph file.
    Show(ShowArgs),
}

const STACK_SIZE: usize = 128 * 1024 * 1024;

fn main() {
    env_logger::init();
    let handle = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let command = Command::parse();
            match command {
                Command::Serialize(args) => serialize(args),
                Command::Show(args) => show(args),
            }
        })
        .unwrap();
    handle.join().unwrap()
}
