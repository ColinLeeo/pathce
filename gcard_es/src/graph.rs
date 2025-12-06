use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{GCardError, GCardResult};
use crate::degreepiecewise::DegreePiecewise;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PathStep {
    pub edge_type: String,
    pub src_type: String,
    pub dst_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PathDefinition {
    pub steps: Vec<PathStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathData {
    pub src_node_type: String,
    pub dst_node_type: String,
    pub path_len: usize,
    #[serde(rename = "path")]
    pub path_def: PathDefinition,
    pub degree_seq: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PathKey {
    pub src_node_type: String,
    pub dst_node_type: String,
    pub path: PathDefinition,
}

impl PathKey {
    pub fn from_data(data: &PathData) -> Self {
        Self {
            src_node_type: data.src_node_type.clone(),
            dst_node_type: data.dst_node_type.clone(),
            path: data.path_def.clone(),
        }
    }

    pub fn from_simple_string(
        path_str: &str,
        graph: &DegreeSeqGraph,
    ) -> Option<Self> {
        let parts: Vec<&str> = path_str.split('-').collect();
        if parts.len() < 3 || parts.len() % 2 == 0 {
            return None;
        }

        let src_node_type = parts[0].to_string();
        let mut steps = Vec::new();
        let mut current_src = src_node_type.clone();

        for i in (1..parts.len()).step_by(2) {
            if i + 1 >= parts.len() {
                break;
            }
            let edge_type = parts[i].to_string();
            let dst_node_type = parts[i + 1].to_string();

            steps.push(PathStep {
                edge_type: edge_type.clone(),
                src_type: current_src.clone(),
                dst_type: dst_node_type.clone(),
            });

            current_src = dst_node_type;
        }

        let dst_node_type = current_src;
        let path_def = PathDefinition { steps };

        let candidate = Self {
            src_node_type,
            dst_node_type,
            path: path_def,
        };

        if graph.contains_path(&candidate) {
            Some(candidate)
        } else {
            Some(candidate)
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DegreeSeqGraph {
    path_degree_seqs: HashMap<PathKey, DegreePiecewise>,
}

impl DegreeSeqGraph {
    pub fn new() -> Self {
        Self {
            path_degree_seqs: HashMap::new(),
        }
    }

    pub fn from_json<P: AsRef<Path>>(path: P) -> GCardResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data: PathData = serde_json::from_reader(reader)?;
        
        let mut graph = Self::new();
        graph.add_path_data(data)?;
        Ok(graph)
    }

    pub fn from_json_lines<P: AsRef<Path>>(path: P) -> GCardResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut graph = Self::new();

        use std::io::BufRead;
        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let data: PathData = serde_json::from_str(&line)
                .map_err(|e| GCardError::InvalidData(format!("Line {}: {}", line_num + 1, e)))?;
            graph.add_path_data(data)?;
        }

        Ok(graph)
    }

    pub fn from_json_array<P: AsRef<Path>>(path: P) -> GCardResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data_array: Vec<PathData> = serde_json::from_reader(reader)?;
        
        let mut graph = Self::new();
        for (idx, data) in data_array.into_iter().enumerate() {
            graph.add_path_data(data)
                .map_err(|e| GCardError::InvalidData(format!("Item {}: {}", idx + 1, e)))?;
        }
        Ok(graph)
    }

    pub fn add_path_data(&mut self, data: PathData) -> GCardResult<()> {
        if data.path_len != data.path_def.steps.len() {
            return Err(GCardError::InvalidData(format!(
                "path_len ({}) does not match path.steps.len() ({})",
                data.path_len,
                data.path_def.steps.len()
            )));
        }

        if !data.path_def.steps.is_empty() {
            let first_step = &data.path_def.steps[0];
            let last_step = &data.path_def.steps[data.path_def.steps.len() - 1];

            if first_step.src_type != data.src_node_type {
                return Err(GCardError::InvalidData(format!(
                    "First step src_type ({}) does not match src_node_type ({})",
                    first_step.src_type, data.src_node_type
                )));
            }

            if last_step.dst_type != data.dst_node_type {
                return Err(GCardError::InvalidData(format!(
                    "Last step dst_type ({}) does not match dst_node_type ({})",
                    last_step.dst_type, data.dst_node_type
                )));
            }

            for i in 0..data.path_def.steps.len() - 1 {
                let current = &data.path_def.steps[i];
                let next = &data.path_def.steps[i + 1];
                if current.dst_type != next.src_type {
                    return Err(GCardError::InvalidData(format!(
                        "Path step {} dst_type ({}) does not match step {} src_type ({})",
                        i, current.dst_type, i + 1, next.src_type
                    )));
                }
            }
        }

        let key = PathKey::from_data(&data);
        let degree_piecewise = DegreePiecewise::from_degree_sequence_default(data.degree_seq)?;
        self.path_degree_seqs.insert(key, degree_piecewise);
        Ok(())
    }

    pub fn get_degree_seq(&self, key: &PathKey) -> Option<&DegreePiecewise> {
        self.path_degree_seqs.get(key)
    }

    pub fn path_keys(&self) -> impl Iterator<Item = &PathKey> {
        self.path_degree_seqs.keys()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&PathKey, &DegreePiecewise)> {
        self.path_degree_seqs.iter()
    }

    pub fn num_paths(&self) -> usize {
        self.path_degree_seqs.len()
    }

    pub fn contains_path(&self, key: &PathKey) -> bool {
        self.path_degree_seqs.contains_key(key)
    }

    pub fn export_bincode<P: AsRef<Path>>(&self, path: P) -> GCardResult<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    pub fn import_bincode<P: AsRef<Path>>(path: P) -> GCardResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let graph = bincode::deserialize_from(reader)?;
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_key() {
        let data = PathData {
            src_node_type: "Person".to_string(),
            dst_node_type: "University".to_string(),
            path_len: 2,
            path_def: PathDefinition {
                steps: vec![
                    PathStep {
                        edge_type: "knows".to_string(),
                        src_type: "Person".to_string(),
                        dst_type: "Person".to_string(),
                    },
                    PathStep {
                        edge_type: "studyAt".to_string(),
                        src_type: "Person".to_string(),
                        dst_type: "University".to_string(),
                    },
                ],
            },
            degree_seq: vec![9, 5, 4, 3, 3, 3, 3, 3, 3],
        };

        let key = PathKey::from_data(&data);
        assert_eq!(key.src_node_type, "Person");
        assert_eq!(key.dst_node_type, "University");
        assert_eq!(key.path.steps.len(), 2);
    }

    #[test]
    fn test_add_path_data() {
        let mut graph = DegreeSeqGraph::new();
        let data = PathData {
            src_node_type: "Person".to_string(),
            dst_node_type: "Comment".to_string(),
            path_len: 1,
            path_def: PathDefinition {
                steps: vec![PathStep {
                    edge_type: "likes".to_string(),
                    src_type: "Person".to_string(),
                    dst_type: "Comment".to_string(),
                }],
            },
            degree_seq: vec![10, 9, 9, 7],
        };

        assert!(graph.add_path_data(data).is_ok());
        assert_eq!(graph.num_paths(), 1);
    }
}

