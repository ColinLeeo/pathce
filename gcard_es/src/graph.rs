use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::degreepiecewise::PiecewiseConstantFunction;
use crate::error::{GCardError, GCardResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AltKey(pub Vec<String>);

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
pub struct NewPathData {
    pub edge_labels: Vec<String>,
    pub endpoints: HashMap<String, Vec<u64>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DegreeSeqGraph {
    pub edge_set_to_endpoints: HashMap<AltKey, HashMap<String, CompressedDegreeSeq>>,
}

impl DegreeSeqGraph {
    pub fn new() -> Self {
        Self {
            edge_set_to_endpoints: HashMap::new(),
        }
    }

    pub fn get_piece_func_by_path(
        &self,
        path: &AltKey,
        target_node: &str,
    ) -> PiecewiseConstantFunction {
        if let Some(endpoints) = self.edge_set_to_endpoints.get(&path) {
            let func = endpoints.get(target_node).cloned().expect("not found");
            match func {
                CompressedDegreeSeq::SafeBound { function } => function,
                CompressedDegreeSeq::FastCompressor { len, base, counts } => {
                    let total: usize = counts.iter().map(|&c| c as usize).sum();
                    let mut seq = Vec::with_capacity(total);
                    for (i, &c) in counts.iter().enumerate() {
                        if c == 0 {
                            continue;
                        }
                        let upper_f = base.powi((i as i32) + 1);
                        let upper_u64 = if !upper_f.is_finite() || upper_f >= (u64::MAX as f64) {
                            u64::MAX
                        } else {
                            upper_f.ceil() as u64
                        };
                        seq.extend(std::iter::repeat(upper_u64).take(c as usize));
                    }
                    PiecewiseConstantFunction::from_degree_sequence(seq.as_slice(), 0.01, true)
                        .unwrap()
                }
            }
        } else {
            PiecewiseConstantFunction::empty()
        }
    }

    pub fn num_edge_sets(&self) -> usize {
        self.edge_set_to_endpoints.len()
    }

    pub fn export_bincode<P: AsRef<Path>>(&self, path: P) -> GCardResult<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)
            .map_err(|e| GCardError::InvalidData(format!("Failed to serialize: {}", e)))?;
        Ok(())
    }

    pub fn import_bincode<P: AsRef<Path>>(path: P) -> GCardResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let graph = bincode::deserialize_from(reader)
            .map_err(|e| GCardError::InvalidData(format!("Failed to deserialize: {}", e)))?;
        Ok(graph)
    }
}
