use std::collections::HashMap;
use std::rc::Rc;

use crate::degreepiecewise::Pcf;
use crate::error::GCardResult;
use crate::graph::{DegreeSeqGraph, PathKey};

#[derive(Debug, Clone)]
pub enum QueryExpr {
    PathKeyRef(PathKey),
    IntermediateRef(String),
    PcfRef(Rc<Pcf>),
}

impl QueryExpr {
    pub fn from_path_key(key: PathKey) -> Self {
        Self::PathKeyRef(key)
    }

    pub fn from_simple_path(path_str: &str, graph: &DegreeSeqGraph) -> Option<Self> {
        PathKey::from_simple_string(path_str, graph).map(Self::PathKeyRef)
    }

    pub fn from_intermediate(name: String) -> Self {
        Self::IntermediateRef(name)
    }

    pub fn from_pcf(pcf: Pcf) -> Self {
        Self::PcfRef(Rc::new(pcf))
    }

    pub fn from_pcf_ref(pcf: &Pcf) -> Self {
        Self::PcfRef(Rc::new(pcf.clone()))
    }

    pub fn resolve<'a>(
        &self,
        graph: &'a DegreeSeqGraph,
        intermediate_results: &'a HashMap<String, Rc<Pcf>>,
    ) -> GCardResult<Rc<Pcf>> {
        match self {
            Self::PathKeyRef(key) => graph
                .get_degree_seq(key)
                .map(|dp| Rc::new(dp.get_piecewise_function().clone()))
                .ok_or_else(|| {
                    crate::error::GCardError::InvalidData(format!(
                        "Path key not found in graph: {:?}",
                        key
                    ))
                }),
            Self::IntermediateRef(name) => intermediate_results
                .get(name)
                .map(|rc| Rc::clone(rc))
                .ok_or_else(|| {
                    crate::error::GCardError::InvalidData(format!(
                        "Intermediate result not found: {}",
                        name
                    ))
                }),
            Self::PcfRef(rc) => Ok(Rc::clone(rc)),
        }
    }
}

pub struct QueryPlanner<'a> {
    graph: &'a DegreeSeqGraph,
    intermediate_results: HashMap<String, Rc<Pcf>>,
}

impl<'a> QueryPlanner<'a> {
    pub fn new(graph: &'a DegreeSeqGraph) -> Self {
        Self {
            graph,
            intermediate_results: HashMap::new(),
        }
    }

    pub fn from_simple_path(&self, path_str: &str) -> GCardResult<QueryExpr> {
        PathKey::from_simple_string(path_str, self.graph)
            .map(QueryExpr::from_path_key)
            .ok_or_else(|| {
                crate::error::GCardError::InvalidData(format!(
                    "Invalid path string format: {}",
                    path_str
                ))
            })
    }

    pub fn save_result(&mut self, name: &str, pcf: Pcf) {
        self.intermediate_results
            .insert(name.to_string(), Rc::new(pcf));
    }

    pub fn get_result(&self, name: &str) -> Option<Rc<Pcf>> {
        self.intermediate_results.get(name).map(|rc| Rc::clone(rc))
    }

    fn resolve_exprs(&self, exprs: &[QueryExpr]) -> GCardResult<Vec<Rc<Pcf>>> {
        exprs
            .iter()
            .map(|expr| expr.resolve(self.graph, &self.intermediate_results))
            .collect()
    }

    pub fn alpha(&self, exprs: &[QueryExpr]) -> GCardResult<Pcf> {
        let pcfs_rc = self.resolve_exprs(exprs)?;
        let pcfs_refs: Vec<&Pcf> = pcfs_rc.iter().map(|rc| rc.as_ref()).collect();
        Ok(crate::degreepiecewise::alpha_refs(&pcfs_refs))
    }

    pub fn beta(&self, rx: &QueryExpr, sx: &QueryExpr, sy: &QueryExpr) -> GCardResult<Pcf> {
        let rx_pcf = rx.resolve(self.graph, &self.intermediate_results)?;
        let sx_pcf = sx.resolve(self.graph, &self.intermediate_results)?;
        let sy_pcf = sy.resolve(self.graph, &self.intermediate_results)?;
        Ok(
            crate::degreepiecewise::calculate_non_joining_column_frequency(
                rx_pcf.as_ref(),
                sx_pcf.as_ref(),
                sy_pcf.as_ref(),
            ),
        )
    }

    pub fn beta_two(
        &self,
        rx: &QueryExpr,
        ry: &QueryExpr,
        my: &QueryExpr,
        mz: &QueryExpr,
    ) -> GCardResult<(Pcf, Pcf)> {
        let rx_pcf = rx.resolve(self.graph, &self.intermediate_results)?;
        let py_pcf = ry.resolve(self.graph, &self.intermediate_results)?;
        let my_pcf = my.resolve(self.graph, &self.intermediate_results)?;
        let mz_pcf = mz.resolve(self.graph, &self.intermediate_results)?;

        let rx_result = crate::degreepiecewise::calculate_non_joining_column_frequency(
            rx_pcf.as_ref(),
            py_pcf.as_ref(),
            my_pcf.as_ref(),
        );

        let mz_result = crate::degreepiecewise::calculate_non_joining_column_frequency(
            mz_pcf.as_ref(),
            my_pcf.as_ref(),
            py_pcf.as_ref(),
        );

        Ok((rx_result, mz_result))
    }

    pub fn gamma(&self, path_pairs: &[(QueryExpr, QueryExpr)]) -> GCardResult<(Pcf, Pcf)> {
        let mut pcf_pairs_rc = Vec::new();
        for (a_expr, b_expr) in path_pairs {
            let a_pcf = a_expr.resolve(self.graph, &self.intermediate_results)?;
            let b_pcf = b_expr.resolve(self.graph, &self.intermediate_results)?;
            pcf_pairs_rc.push((a_pcf, b_pcf));
        }

        let ref_pairs: Vec<(&Pcf, &Pcf)> = pcf_pairs_rc
            .iter()
            .map(|(a, b)| (a.as_ref(), b.as_ref()))
            .collect();

        Ok(crate::degreepiecewise::gamma(ref_pairs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{PathDefinition, PathKey, PathStep};

    fn create_test_graph() -> DegreeSeqGraph {
        let mut graph = DegreeSeqGraph::new();

        // P1: Person --knows--> Person
        let p1_data = crate::graph::PathData {
            src_node_type: "Person".to_string(),
            dst_node_type: "Person".to_string(),
            path_len: 1,
            path_def: PathDefinition {
                steps: vec![PathStep {
                    edge_type: "knows".to_string(),
                    src_type: "Person".to_string(),
                    dst_type: "Person".to_string(),
                }],
            },
            degree_seq: vec![13, 7, 6, 5, 5, 4, 4, 4, 4, 3, 2, 2, 1, 1, 1, 0, 0, 0],
        };
        graph.add_path_data(p1_data).unwrap();

        // P2: Person --studyAt--> University
        let p2_data = crate::graph::PathData {
            src_node_type: "Person".to_string(),
            dst_node_type: "University".to_string(),
            path_len: 1,
            path_def: PathDefinition {
                steps: vec![PathStep {
                    edge_type: "studyAt".to_string(),
                    src_type: "Person".to_string(),
                    dst_type: "University".to_string(),
                }],
            },
            degree_seq: vec![
                2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            ],
        };
        graph.add_path_data(p2_data).unwrap();

        // P3: University --study--> Person
        let p3_data = crate::graph::PathData {
            src_node_type: "University".to_string(),
            dst_node_type: "Person".to_string(),
            path_len: 1,
            path_def: PathDefinition {
                steps: vec![PathStep {
                    edge_type: "studyAt".to_string(),
                    src_type: "University".to_string(),
                    dst_type: "Person".to_string(),
                }],
            },
            degree_seq: vec![
                10, 10, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        };
        graph.add_path_data(p3_data).unwrap();
        graph
    }

    #[test]
    fn test_alpha_operation() {
        let graph = create_test_graph();
        let planner = QueryPlanner::new(&graph);

        let p1_key = planner.from_simple_path("Person-knows-Person").unwrap();
        let p2_key = planner
            .from_simple_path("Person-studyAt-University")
            .unwrap();

        let result = planner.alpha(&[p1_key.clone(), p2_key.clone()]).unwrap();

        assert!(result.get_num_rows() > 0.0);
        println!("Alpha result: {} rows", result.get_num_rows());
    }
    #[test]
    fn test_beta_operation() {
        let graph = create_test_graph();
        let planner = QueryPlanner::new(&graph);

        let p1_key = planner.from_simple_path("Person-knows-Person").unwrap();
        let p2_key = planner
            .from_simple_path("Person-studyAt-University")
            .unwrap();

        let result = planner.beta(&p2_key, &p1_key, &p1_key).unwrap();
        println!("Beta result: {} rows", result.get_num_rows());
    }

    #[test]
    fn test_gamma_operation() {
        let graph = create_test_graph();
        let planner = QueryPlanner::new(&graph);
        let graph = create_test_graph();
        let planner = QueryPlanner::new(&graph);

        let p_k_p = planner.from_simple_path("Person-knows-Person").unwrap();
        let p_s_u = planner
            .from_simple_path("Person-studyAt-University")
            .unwrap();
        let u_s_p = planner
            .from_simple_path("University-studyAt-Person")
            .unwrap();

        let s1 = p_s_u.clone();
        let s2 = u_s_p.clone();
        let p_u = planner.beta(&p_s_u, &p_k_p, &p_k_p).unwrap();
        let u_p = planner.beta(&p_k_p, &p_s_u, &u_s_p).unwrap();
        let result = planner.gamma(&[
            (s1, s2),
            (
                QueryExpr::PcfRef(Rc::new(p_u)),
                QueryExpr::PcfRef(Rc::new(u_p)),
            ),
        ]).unwrap();

        assert!(result.0.get_num_rows() > 0.0);
        assert!(result.1.get_num_rows() > 0.0);
        assert_eq!(result.0.get_num_rows(), result.1.get_num_rows());
    }
}
