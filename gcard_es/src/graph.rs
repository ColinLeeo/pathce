use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{GCardError, GCardResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewPathData {
    pub edge_labels: Vec<String>,
    pub endpoints: HashMap<String, Vec<u64>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DegreeSeqGraph {
    edge_set_to_endpoints: HashMap<BTreeSet<String>, HashMap<String, Vec<u64>>>,
}

impl DegreeSeqGraph {
    pub fn new() -> Self {
        Self {
            edge_set_to_endpoints: HashMap::new(),
        }
    }

    pub fn add_new_path_data(&mut self, data: NewPathData) -> GCardResult<()> {
        let edge_key: BTreeSet<String> = data.edge_labels.into_iter().collect();
        self.edge_set_to_endpoints.insert(edge_key, data.endpoints);
        Ok(())
    }

    pub fn from_json_array<P: AsRef<Path>>(path: P) -> GCardResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data_array: Vec<NewPathData> = serde_json::from_reader(reader)?;

        let mut graph = Self::new();
        for (idx, new_data) in data_array.into_iter().enumerate() {
            graph
                .add_new_path_data(new_data)
                .map_err(|e| GCardError::InvalidData(format!("Item {}: {}", idx + 1, e)))?;
        }

        Ok(graph)
    }

    pub fn get_degree_seq_vec_by_edges(
        &self,
        edges: &std::collections::HashSet<&str>,
        target_node: &str,
    ) -> Option<Vec<u64>> {
        let query_key: BTreeSet<String> = edges.iter().map(|s| s.to_string()).collect();

        if let Some(endpoints) = self.edge_set_to_endpoints.get(&query_key) {
            return endpoints.get(target_node).cloned();
        }

        None
    }

    pub fn num_edge_sets(&self) -> usize {
        self.edge_set_to_endpoints.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&BTreeSet<String>, &HashMap<String, Vec<u64>>)> {
        self.edge_set_to_endpoints.iter()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_add_new_path_data() {
        let mut graph = DegreeSeqGraph::new();
        let data = NewPathData {
            edge_labels: vec!["likes".to_string()],
            endpoints: {
                let mut map = HashMap::new();
                map.insert("Post".to_string(), vec![1, 2, 3, 3, 2, 2, 1]);
                map.insert("Person".to_string(), vec![2, 3, 4, 5, 2, 1]);
                map
            },
        };

        assert!(graph.add_new_path_data(data).is_ok());
        assert_eq!(graph.num_edge_sets(), 1);
    }

    #[test]
    fn test_get_degree_seq_vec_by_edges() {
        let mut graph = DegreeSeqGraph::new();
        let data = NewPathData {
            edge_labels: vec!["likes".to_string()],
            endpoints: {
                let mut map = HashMap::new();
                map.insert("Post".to_string(), vec![1, 2, 3, 3, 2, 2, 1]);
                map.insert("Person".to_string(), vec![2, 3, 4, 5, 2, 1]);
                map
            },
        };
        graph.add_new_path_data(data).unwrap();

        let edges: HashSet<&str> = ["likes"].iter().cloned().collect();
        let post_ds = graph.get_degree_seq_vec_by_edges(&edges, "Post");
        assert_eq!(post_ds, Some(vec![1, 2, 3, 3, 2, 2, 1]));

        let person_ds = graph.get_degree_seq_vec_by_edges(&edges, "Person");
        assert_eq!(person_ds, Some(vec![2, 3, 4, 5, 2, 1]));

        let missing_ds = graph.get_degree_seq_vec_by_edges(&edges, "Tag");
        assert_eq!(missing_ds, None);
    }

    #[test]
    fn test_multiple_edge_labels() {
        let mut graph = DegreeSeqGraph::new();
        let data = NewPathData {
            edge_labels: vec!["hasTag".to_string(), "likes".to_string()],
            endpoints: {
                let mut map = HashMap::new();
                map.insert("Tag".to_string(), vec![1, 2, 3]);
                map.insert("Person".to_string(), vec![4, 5, 6]);
                map
            },
        };
        graph.add_new_path_data(data).unwrap();

        let edges1: HashSet<&str> = ["hasTag", "likes"].iter().cloned().collect();
        let edges2: HashSet<&str> = ["likes", "hasTag"].iter().cloned().collect();

        let tag_ds1 = graph.get_degree_seq_vec_by_edges(&edges1, "Tag");
        let tag_ds2 = graph.get_degree_seq_vec_by_edges(&edges2, "Tag");
        assert_eq!(tag_ds1, tag_ds2);
        assert_eq!(tag_ds1, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_export_import_bincode() {
        let mut graph = DegreeSeqGraph::new();
        let data = NewPathData {
            edge_labels: vec!["likes".to_string()],
            endpoints: {
                let mut map = HashMap::new();
                map.insert("Post".to_string(), vec![1, 2, 3]);
                map
            },
        };
        graph.add_new_path_data(data).unwrap();

        let temp_path = std::env::temp_dir().join("test_graph.bincode");
        assert!(graph.export_bincode(&temp_path).is_ok());

        let loaded_graph = DegreeSeqGraph::import_bincode(&temp_path).unwrap();
        assert_eq!(loaded_graph.num_edge_sets(), 1);

        let edges: HashSet<&str> = ["likes"].iter().cloned().collect();
        let post_ds = loaded_graph.get_degree_seq_vec_by_edges(&edges, "Post");
        assert_eq!(post_ds, Some(vec![1, 2, 3]));

        let _ = std::fs::remove_file(&temp_path);
    }
}
