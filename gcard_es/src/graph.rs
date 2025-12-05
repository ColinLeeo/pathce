use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{GCardError, GCardResult};

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
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DegreeSeqGraph {
    path_degree_seqs: HashMap<PathKey, Vec<u64>>,
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

        // 验证路径的起始和结束类型
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

            // 验证路径的连续性（中间步骤的 dst_type 应该等于下一步的 src_type）
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
        self.path_degree_seqs.insert(key, data.degree_seq);
        Ok(())
    }

    /// 获取路径的度数序列
    pub fn get_degree_seq(&self, key: &PathKey) -> Option<&Vec<u64>> {
        self.path_degree_seqs.get(key)
    }

    /// 获取所有路径键
    pub fn path_keys(&self) -> impl Iterator<Item = &PathKey> {
        self.path_degree_seqs.keys()
    }

    /// 获取所有路径和对应的度数序列
    pub fn iter(&self) -> impl Iterator<Item = (&PathKey, &Vec<u64>)> {
        self.path_degree_seqs.iter()
    }

    /// 获取路径数量
    pub fn num_paths(&self) -> usize {
        self.path_degree_seqs.len()
    }

    /// 检查是否包含指定的路径
    pub fn contains_path(&self, key: &PathKey) -> bool {
        self.path_degree_seqs.contains_key(key)
    }

    /// 导出为 bincode 格式
    pub fn export_bincode<P: AsRef<Path>>(&self, path: P) -> GCardResult<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    /// 从 bincode 格式导入
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

