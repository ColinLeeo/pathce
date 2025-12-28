pub mod command;

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
    /// Create catalog from DuckDB database and export to bincode file.
    CreateCatalog(CreateCatalogArgs),
}

const STACK_SIZE: usize = 128 * 1024 * 1024;

fn main() {
    env_logger::init();
    let handle = thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let command = Command::parse();
            match command {
                Command::CreateCatalog(args) => create_catalog(args),
            }
        })
        .unwrap();
    handle.join().unwrap()
}
