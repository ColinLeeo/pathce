use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{GCardError, GCardResult};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AltKey(pub Vec<String>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewPathData {
    pub edge_labels: Vec<String>,
    pub endpoints: HashMap<String, Vec<u64>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DegreeSeqGraph {
    pub edge_set_to_endpoints: HashMap<AltKey, HashMap<String, Vec<u64>>>,
}

impl DegreeSeqGraph {
    pub fn new() -> Self {
        Self {
            edge_set_to_endpoints: HashMap::new(),
        }
    }

    pub fn get_degree_seq_vec_by_path(
        &self,
        path: &AltKey,
        target_node: &str,
    ) -> Option<Vec<u64>> {
        if let Some(endpoints) = self.edge_set_to_endpoints.get(&path) {
            return endpoints.get(target_node).cloned();
        }
        None
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

