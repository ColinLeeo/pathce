use serde::{Deserialize, Serialize};

use crate::error::GCardResult;

use super::pointwise_function_mult;
use super::pointwise_function_min;
use super::PiecewiseConstantFunction;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DegreePiecewise {
    pub degree_sequence: Vec<u64>,
    pub piecewise_function: PiecewiseConstantFunction,
}

impl DegreePiecewise {
    pub fn from_degree_sequence(
        degree_sequence: Vec<u64>,
        relative_error_per_segment: f64,
        model_cdf: bool,
    ) -> GCardResult<Self> {
        let piecewise_function = PiecewiseConstantFunction::from_degree_sequence(
            &degree_sequence,
            relative_error_per_segment,
            model_cdf,
        )?;

        Ok(Self {
            degree_sequence,
            piecewise_function,
        })
    }

    pub fn from_degree_sequence_default(degree_sequence: Vec<u64>) -> GCardResult<Self> {
        Self::from_degree_sequence(
            degree_sequence,
            0.1,
            true,
        )
    }

    pub fn get_degree_sequence(&self) -> &[u64] {
        &self.degree_sequence
    }

    pub fn get_piecewise_function(&self) -> &PiecewiseConstantFunction {
        &self.piecewise_function
    }

    pub fn get_num_rows(&self) -> f64 {
        self.piecewise_function.get_num_rows()
    }

    pub fn calculate_value_at_point(&self, x: f64) -> f64 {
        self.piecewise_function.calculate_value_at_point(x)
    }

    pub fn calculate_rows_at_point(&self, x: f64) -> f64 {
        self.piecewise_function.calculate_rows_at_point(x)
    }

    pub fn get_self_join_size(&self) -> f64 {
        self.piecewise_function.get_self_join_size()
    }

    pub fn calculate_inverse(&self, y: f64) -> f64 {
        self.piecewise_function.calculate_inverse(y)
    }
}

pub type Pcf = PiecewiseConstantFunction;

pub fn alpha(pieces: &[Pcf]) -> Pcf {
    pointwise_function_mult(pieces)
}

pub fn alpha_refs(pieces: &[&Pcf]) -> Pcf {
    use super::function::pointwise_function_mult_refs;
    pointwise_function_mult_refs(pieces)
}

pub fn beta_left(rx: &Pcf, ry: &Pcf, sy: &Pcf) -> Pcf {
    calculate_non_joining_column_frequency(sy, ry, rx)
}

pub fn beta_right(rx: &Pcf, sx: &Pcf, sy: &Pcf) -> Pcf {
    calculate_non_joining_column_frequency(rx,sx,sy)
}

pub fn beta(rx: &Pcf, ry: &Pcf, sy: &Pcf, sz: &Pcf) -> (Pcf, Pcf) {
    (beta_left(rx,ry,sy), beta_right(ry,sy,sz))
}

fn project_child_via_parent(
    child: &PiecewiseConstantFunction,
    parent_from: &PiecewiseConstantFunction,
    parent_to: &PiecewiseConstantFunction,
) -> PiecewiseConstantFunction {
    if child.constants.is_empty()
        || parent_from.constants.is_empty()
        || parent_to.constants.is_empty()
    {
        return PiecewiseConstantFunction::empty();
    }

    let mut res_constants = Vec::new();
    let mut res_right_edges = Vec::new();
    let mut child_idx = 0;
    let mut child_const = child.constants[child_idx];
    let mut child_bound = child.right_interval_edges[child_idx];
    let mut x = 0.0;
    let mut cum = 0.0;

    for seg_idx in 0..parent_to.constants.len() {
        let seg_right = parent_to.right_interval_edges[seg_idx];
        let seg_const = parent_to.constants[seg_idx];
        let seg_cum_end = parent_to.cumulative_rows[seg_idx];

        while cum < seg_cum_end - 1e-9 {
            let target_rows = if child_idx < child.constants.len() {
                parent_from.calculate_rows_at_point(child_bound)
            } else {
                f64::INFINITY
            };

            let next_cum = seg_cum_end.min(target_rows);
            let next_cum = if next_cum.is_infinite() {
                seg_cum_end
            } else {
                next_cum
            };

            let width = if seg_const != 0.0 {
                (next_cum - cum) / seg_const
            } else {
                0.0
            };
            let next_x = seg_right.min(x + width);

            if next_x <= x + 1e-9 {
                res_constants.push(child_const);
                res_right_edges.push(seg_right);
                break;
            }

            res_constants.push(child_const);
            res_right_edges.push(next_x);
            x = next_x;
            cum = next_cum;

            if (cum - target_rows).abs() < 1e-6 {
                child_idx += 1;
                if child_idx < child.constants.len() {
                    child_const = child.constants[child_idx];
                    child_bound = child.right_interval_edges[child_idx];
                } else {
                    child_const = 0.0;
                    child_bound = f64::INFINITY;
                }
            }

            if (cum - seg_cum_end).abs() < 1e-6 {
                break;
            }
        }

        x = seg_right;
        cum = seg_cum_end;
    }

    let mut res_cumulative_rows = Vec::new();
    let mut cur_row = 0.0;
    let mut cur_left = 0.0;
    for i in 0..res_right_edges.len() {
        let cur_right = res_right_edges[i];
        cur_row += res_constants[i] * (cur_right - cur_left);
        res_cumulative_rows.push(cur_row);
        cur_left = cur_right;
    }

    PiecewiseConstantFunction {
        constants: res_constants,
        right_interval_edges: res_right_edges,
        cumulative_rows: res_cumulative_rows,
    }
}

fn maxreduce_edge(edge_a: &Pcf, edge_b: &Pcf) -> (Pcf, Pcf) {
    let max_a = edge_a.max_value();
    let max_b = edge_b.max_value();
    
    let mut new_a = edge_a.copy();
    let mut new_b = edge_b.copy();

    if max_a < max_b - 1e-9 {
        new_b = new_b.cap_constants(max_a);
    }
    if max_b < max_a - 1e-9 {
        new_a = new_a.cap_constants(max_b);
    }

    let total_a = new_a.get_num_rows();
    let total_b = new_b.get_num_rows();
    let aligned_total = total_a.min(total_b);
    
    let aligned_a = new_a.crop_to_total(aligned_total);
    let aligned_b = new_b.crop_to_total(aligned_total);
    
    (aligned_a, aligned_b)
}

pub fn gamma(paths: Vec<(&Pcf, &Pcf)>) -> (Pcf, Pcf) {
    if paths.is_empty() {
        return (Pcf::empty(), Pcf::empty());
    }

    let mut aligned_paths: Vec<(Pcf, Pcf)> = paths
        .iter()
        .map(|(a, b)| maxreduce_edge(a, b))
        .collect();

    if aligned_paths.len() == 1 {
        return (aligned_paths[0].0.clone(), aligned_paths[0].1.clone());
    }

    while aligned_paths.len() > 1 {
        let p1 = aligned_paths.remove(0);
        let p2 = aligned_paths.remove(0);

        // Plan A1: alpha(P1.A, min(P2.A, beta(P2.B->P1.A via P1.B)))
        let p2_b_to_p1_a = project_child_via_parent(&p2.1, &p1.1, &p1.0);
        let min_p2_a_or_beta = pointwise_function_min(&[p2.0.clone(), p2_b_to_p1_a]);
        let plan_a1 = alpha(&[p1.0.clone(), min_p2_a_or_beta]);

        // Plan A2: alpha(P2.A, min(P1.A, beta(P1.B->P2.A via P2.B)))
        let p1_b_to_p2_a = project_child_via_parent(&p1.1, &p2.1, &p2.0);
        let min_p1_a_or_beta = pointwise_function_min(&[p1.0.clone(), p1_b_to_p2_a]);
        let plan_a2 = alpha(&[p2.0.clone(), min_p1_a_or_beta]);

        // Plan B1: alpha(P1.B, min(P2.B, beta(P2.A->P1.B via P1.A)))
        let p2_a_to_p1_b = project_child_via_parent(&p2.0, &p1.0, &p1.1);
        let min_p2_b_or_beta = pointwise_function_min(&[p2.1.clone(), p2_a_to_p1_b]);
        let plan_b1 = alpha(&[p1.1.clone(), min_p2_b_or_beta]);

        // Plan B2: alpha(P2.B, min(P1.B, beta(P1.A->P2.B via P2.A)))
        let p1_a_to_p2_b = project_child_via_parent(&p1.0, &p2.0, &p2.1);
        let min_p1_b_or_beta = pointwise_function_min(&[p1.1.clone(), p1_a_to_p2_b]);
        let plan_b2 = alpha(&[p2.1.clone(), min_p1_b_or_beta]);

        let min_cds = [
            plan_a1.get_num_rows(),
            plan_a2.get_num_rows(),
            plan_b1.get_num_rows(),
            plan_b2.get_num_rows(),
        ]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

        let aligned_plan_a1 = plan_a1.crop_to_total(min_cds);
        let aligned_plan_a2 = plan_a2.crop_to_total(min_cds);
        let aligned_plan_b1 = plan_b1.crop_to_total(min_cds);
        let aligned_plan_b2 = plan_b2.crop_to_total(min_cds);

        let a1_self_join = aligned_plan_a1.get_self_join_size();
        let a2_self_join = aligned_plan_a2.get_self_join_size();
        let b1_self_join = aligned_plan_b1.get_self_join_size();
        let b2_self_join = aligned_plan_b2.get_self_join_size();

        let best_a = if a1_self_join <= a2_self_join {
            aligned_plan_a1
        } else {
            aligned_plan_a2
        };

        let best_b = if b1_self_join <= b2_self_join {
            aligned_plan_b1
        } else {
            aligned_plan_b2
        };

        aligned_paths.insert(0, (best_a, best_b));
    }

    (aligned_paths[0].0.clone(), aligned_paths[0].1.clone())
}
pub fn calculate_non_joining_column_frequency(
    rx: &PiecewiseConstantFunction,
    sx: &PiecewiseConstantFunction,
    sy: &PiecewiseConstantFunction,
) -> PiecewiseConstantFunction {
    let mut sy_to_x_right_x = Vec::new();
    let mut sy_to_x_right_y = Vec::new();
    let mut sx_to_y_slope = Vec::new();

    let mut idx0 = 0;
    let mut idx1 = 0;
    let mut finished = false;

    while !finished {
        let new_row = sx.cumulative_rows[idx0].min(sy.cumulative_rows[idx1]);

        sy_to_x_right_x.push(sx.calculate_inverse(new_row));
        sy_to_x_right_y.push(sy.calculate_inverse(new_row));

        let slope = if sy.constants[idx1] != 0.0 {
            sx.constants[idx0] / sy.constants[idx1]
        } else {
            0.0
        };
        sx_to_y_slope.push(slope);

        if new_row >= sx.cumulative_rows[idx0] {
            idx0 += 1;
        }
        if new_row >= sy.cumulative_rows[idx1] {
            idx1 += 1;
        }
        if idx0 >= sx.cumulative_rows.len() || idx1 >= sy.cumulative_rows.len() {
            finished = true;
        }
    }

    if sy_to_x_right_x.is_empty() {
        return PiecewiseConstantFunction::empty();
    }

    let mut final_constants = Vec::new();
    let mut final_right_interval_edges = Vec::new();

    idx0 = 0;
    idx1 = 0;
    finished = false;

    while !finished {
        let new_x_val = rx.right_interval_edges[idx0].min(sy_to_x_right_x[idx1]);

        let mut left_y = 0.0;
        let mut left_x = 0.0;
        if idx1 > 0 {
            left_y = sy_to_x_right_y[idx1 - 1];
            left_x = sy_to_x_right_x[idx1 - 1];
        }

        let right_edge = left_y + (new_x_val - left_x) * sx_to_y_slope[idx1];
        final_right_interval_edges.push(right_edge);
        final_constants.push(rx.constants[idx0]);

        if new_x_val >= rx.right_interval_edges[idx0] {
            idx0 += 1;
        }
        if new_x_val >= sy_to_x_right_x[idx1] {
            idx1 += 1;
        }
        if idx0 >= rx.right_interval_edges.len() || idx1 >= sy_to_x_right_x.len() {
            finished = true;
        }
    }

    if final_constants.is_empty() {
        return PiecewiseConstantFunction::empty();
    }

    let mut final_cumulative_rows = Vec::new();
    let mut cur_row = 0.0;
    let mut cur_left = 0.0;

    for i in 0..final_right_interval_edges.len() {
        let cur_right = final_right_interval_edges[i];
        cur_row += final_constants[i] * (cur_right - cur_left);
        final_cumulative_rows.push(cur_row);
        cur_left = cur_right;
    }

    PiecewiseConstantFunction {
        constants: final_constants,
        right_interval_edges: final_right_interval_edges,
        cumulative_rows: final_cumulative_rows,
    }
}
