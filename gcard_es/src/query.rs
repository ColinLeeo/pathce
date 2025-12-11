use crate::degreepiecewise::{gamma_core, Pcf};
use crate::error::{GCardError, GCardResult};
use crate::graph::DegreeSeqGraph;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Expr {
    Single {
        pcf: Rc<Pcf>,
        node: String,
    },
    Pair {
        left: Rc<Pcf>,
        left_name: String,
        right: Rc<Pcf>,
        right_name: String,
        node_map: HashMap<String, Rc<Pcf>>,
    },
}

impl Expr {
    pub fn new_single(node: impl Into<String>, pcf: Rc<Pcf>) -> Expr {
        Expr::Single {
            pcf,
            node: node.into(),
        }
    }

    pub fn new_pair(
        left_name: impl Into<String>,
        right_name: impl Into<String>,
        left: Rc<Pcf>,
        right: Rc<Pcf>,
    ) -> Expr {
        let left_name = left_name.into();
        let right_name = right_name.into();

        let mut map = HashMap::new();
        map.insert(left_name.clone(), left.clone());
        map.insert(right_name.clone(), right.clone());

        Expr::Pair {
            left,
            left_name,
            right,
            right_name,
            node_map: map,
        }
    }

    pub fn as_pcf(&self) -> &Pcf {
        match self {
            Expr::Single { pcf, .. } => pcf.as_ref(),
            Expr::Pair { left, .. } => left.as_ref(),
        }
    }

    pub fn left(&self) -> Rc<Pcf> {
        match self {
            Self::Single { pcf, .. } => Rc::clone(pcf),
            Self::Pair { left, .. } => Rc::clone(left),
        }
    }

    pub fn from_path(
        graph: &DegreeSeqGraph,
        edges: HashSet<&str>,
        target_node: &str,
    ) -> GCardResult<Self> {
        let degree_seq = graph
            .get_degree_seq_vec_by_edges(&edges, target_node)
            .ok_or_else(|| {
                GCardError::InvalidData(format!("Path not found: {:?}.{}", edges, target_node))
            })?;

        let degree_piecewise =
            crate::degreepiecewise::DegreePiecewise::from_degree_sequence_default(degree_seq)?;
        let pcf = Rc::new(degree_piecewise.get_piecewise_function().clone());

        Ok(Expr::Single {
            pcf,
            node: target_node.to_string(),
        })
    }

    pub fn right(&self) -> Rc<Pcf> {
        match self {
            Self::Single { pcf, .. } => Rc::clone(pcf),
            Self::Pair { right, .. } => Rc::clone(right),
        }
    }

    pub fn node_name(&self) -> &str {
        match self {
            Self::Single { node, .. } => node,
            Self::Pair { left_name, .. } => left_name,
        }
    }

    pub fn left_name(&self) -> &str {
        match self {
            Self::Single { node, .. } => node,
            Self::Pair { left_name, .. } => left_name,
        }
    }

    pub fn right_name(&self) -> &str {
        match self {
            Self::Single { .. } => panic!("cannot get name"),
            Self::Pair { right_name, .. } => right_name,
        }
    }

    pub fn get(&self, node_type: &str) -> Expr {
        match self {
            Self::Single { .. } => panic!("Cannot find node type"),
            Self::Pair { node_map, .. } => match node_map.get(node_type) {
                Some(pcf_rc) => Expr::Single {
                    pcf: Rc::clone(pcf_rc),
                    node: node_type.to_string(),
                },
                None => panic!("Cannot find node type"),
            },
        }
    }

    pub fn get_num(&self) -> f64 {
        match self {
            Self::Single { pcf, .. } => pcf.get_num_rows(),
            Self::Pair { left, .. } => left.get_num_rows(),
        }
    }

    pub fn sum(&self) -> f64 {
        self.get_num()
    }
}

#[macro_export]
macro_rules! expr {
    ($graph:expr, [$($edge:ident),*], $target:ident) => {
        {
            let edges: HashSet<&str> = [$(stringify!($edge)),*].iter().cloned().collect();
            Expr::from_path($graph, edges, stringify!($target))
        }
    };
    ($graph:expr, [$($edge:ident),*], $target:expr) => {
        {
            let edges: HashSet<&str> = [$(stringify!($edge)),*].iter().cloned().collect();
            Expr::from_path($graph, edges, $target)
        }
    };
}
pub fn alpha<'a, I>(exprs: I) -> Expr
where
    I: IntoIterator<Item = &'a Expr>,
{
    let exprs: Vec<&Expr> = exprs.into_iter().collect();

    let first_node = match exprs.first() {
        Some(e) => e.node_name().to_string(),
        None => panic!("alpha requires at least one Expr"),
    };

    let pcfs: Vec<&Pcf> = exprs.iter().map(|e| e.as_pcf()).collect();
    let result = crate::degreepiecewise::alpha_refs(&pcfs);

    Expr::new_single(first_node, Rc::new(result))
}

pub fn beta(rx: &Expr, sx: &Expr, sy: &Expr) -> Expr {
    let result = crate::degreepiecewise::calculate_non_joining_column_frequency(
        rx.as_pcf(),
        sx.as_pcf(),
        sy.as_pcf(),
    );

    Expr::new_single(sy.node_name().to_string(), Rc::new(result))
}

pub fn gamma(paris: Vec<(Expr, Expr)>) -> Expr {
    let item: Vec<(&Pcf, &Pcf)> = paris.iter().map(|t| (t.0.as_pcf(), t.1.as_pcf())).collect();
    let first_node = paris.first().unwrap();
    let left_name = first_node.0.node_name().to_string();
    let right_name = first_node.1.node_name().to_string();
    let (left_pcf, right_pcf) = gamma_core(item);
    Expr::new_pair(left_name, right_name, Rc::new(left_pcf), Rc::new(right_pcf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_graph() -> DegreeSeqGraph {
        DegreeSeqGraph::import_bincode(Path::new("/Users/colin/dev/pathce/gcard_es/graph.bincode"))
            .unwrap()
    }
    ///
    /// Post-hasTag-Tag
    // Forum-hasTag2-Tag
    // Person-hasInterest-Tag
    // Person-likes1-Comment
    // Forum-hasMember-Person
    // City-isPartOf-Country
    // Person-likes-Post
    // Comment-hasTag1-Tag
    // Person-knows-Person
    // Person-studyAt-University
    // Person-isLocatedIn-City
    // Person-workAt-Company
    // Person-hasMember-Forum-hasTag2-Tag
    // Person-knows-Person-workAt-Company
    // Tag-hasInterest-Person-likes1-Comment
    // Forum-hasMember-Person-hasInterest-Tag
    // Comment-likes1-Person-workAt-Company
    // Forum-hasTag2-Tag-hasTag-Post
    // Comment-hasTag1-Tag-hasTag-Post
    // Forum-hasMember-Person-isLocatedIn-City
    // University-studyAt-Person-likes-Post
    // Person-isLocatedIn-City-isPartOf-Country
    // Forum-hasMember-Person-workAt-Company
    // Post-likes-Person-isLocatedIn-City
    // Person-likes-Post-hasTag-Tag
    // Person-knows-Person-studyAt-University
    // Forum-hasMember-Person-studyAt-University
    // Tag-hasInterest-Person-workAt-Company
    // Post-likes-Person-likes1-Comment
    // Person-hasInterest-Tag-hasTag2-Forum
    // Post-likes-Person-workAt-Company
    // Forum-hasMember-Person-likes1-Comment
    // University-studyAt-Person-likes1-Comment
    // City-isLocatedIn-Person-likes1-Comment
    // City-isLocatedIn-Person-workAt-Company
    // Forum-hasMember-Person-likes-Post
    // Comment-hasTag1-Tag-hasTag2-Forum
    // Person-hasInterest-Tag-hasTag-Post
    // Person-hasInterest-Tag-hasTag1-Comment
    // University-studyAt-Person-hasInterest-Tag
    // University-studyAt-Person-workAt-Company
    // University-studyAt-Person-isLocatedIn-City
    // Person-likes1-Comment-hasTag1-Tag
    // Person-knows-Person-likes1-Comment
    // Person-knows-Person-hasInterest-Tag
    // Post-likes-Person-hasInterest-Tag
    // City-isLocatedIn-Person-hasInterest-Tag

    #[test]
    fn test_q1() -> GCardResult<()> {
        let graph = load_graph();
        let b1 = beta(
            &expr!(&graph, [isLocatedIn, isPartOf], Person)?,
            &expr!(&graph, [hasMember], Person)?,
            &expr!(&graph, [hasMember], Forum)?,
        );

        let b2 = beta(
            &expr!(&graph, [hasTag1, hasType], Comment)?,
            &expr!(&graph, [containerOf, replyOf], Comment)?,
            &expr!(&graph, [containerOf, replyOf], Forum)?,
        );

        let result = alpha(&[b1, b2]).get_num();
        println!("get row num is {}", result);
        Ok(())
    }

    #[test]
    fn test_q2() -> GCardResult<()> {
        let graph = load_graph();

        let g = gamma(vec![
            (
                expr!(&graph, [hasCreator, knows], Comment)?,
                expr!(&graph, [hasCreator, knows], Person)?,
            ),
            (
                expr!(&graph, [replyof, hasCreator], Comment)?,
                expr!(&graph, [replyof, hasCreator], Person)?,
            ),
        ]);

        let result = g.get_num();
        println!("q2 row num is {}", result);

        Ok(())
    }

    #[test]
    fn test_q3() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(&graph, [hasTag], Message)?;
        let e2 = expr!(&graph, [hasCreator], Message)?;
        let e3 = expr!(&graph, [likes], Message)?;
        let e4 = expr!(&graph, [replyof], Message)?;

        let result = alpha(&[e1, e2, e3, e4]).sum();

        println!("Q3 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q5() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(&graph, [hasTag], Message)?;
        let e2 = expr!(&graph, [replyof, hasTag], Message)?;

        let result = alpha(&[e1, e2]).sum();

        println!("Q5 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q6() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(&graph, [knows, knows], Person)?;
        let e2 = expr!(&graph, [hasInterest], Person)?;

        let result = alpha(&[e1, e2]).sum();

        println!("Q6 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q8() -> GCardResult<()> {
        let graph = load_graph();

        let p1_l = expr!(&graph, [replyof, hasTag1], Tag)?;
        let p1_r = expr!(&graph, [replyof, hasTag1], Comment)?;
        let p2_l = expr!(&graph, [hasTag1], Tag)?;
        let p2_r = expr!(&graph, [hasTag1], Comment)?;

        let g = gamma(vec![(p1_l, p1_r), (p2_l, p2_r)]);

        let g_comment = g.get("Comment");

        let e = expr!(&graph, [hasTag], Comment)?;

        let result = alpha(&[g_comment, e]).sum();

        println!("Q8 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q9() -> GCardResult<()> {
        let graph = load_graph();
        let p1_l = expr!(&graph, [knows, knows], Person)?; // [knows,knows].person
        let p1_r = expr!(&graph, [knows], Person)?; // [knows].person

        let g = gamma(vec![(p1_l, p1_r)]);

        let g_person = g.get("Person");

        let e = expr!(&graph, [hasInterest], Person)?;

        let result = alpha(&[g_person, e]).sum();

        println!("Q9 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p1() -> GCardResult<()> {
        let graph = load_graph();

        // 1 : ([hasTag, has_creator].tag , [hasTag, has_creator].person)
        let p1_l = expr!(&graph, [hasTag, has_creator], Tag)?; // tag
        let p1_r = expr!(&graph, [hasTag, has_creator], Person)?; // person

        // 2 : ([has_interest].tag , [has_interest].person)
        let p2_l = expr!(&graph, [has_interest], Tag)?; // tag
        let p2_r = expr!(&graph, [has_interest], Person)?; // person

        let g = gamma(vec![(p1_l, p1_r), (p2_l, p2_r)]);

        let result = g.sum();

        println!("Q8 result = {}", result);
        Ok(())
    }
    #[test]
    fn test_p2() -> GCardResult<()> {
        let graph = load_graph();

        //  ([has_creator, is_located_in].comment, [has_creator, is_located_in].city)
        let p1_l = expr!(&graph, [has_creator, is_located_in], Comment)?;
        let p1_r = expr!(&graph, [has_creator, is_located_in], City)?;

        //  ([likes, is_located_in].comment, [likes, is_located_in].city)
        let p2_l = expr!(&graph, [likes, is_located_in], Comment)?;
        let p2_r = expr!(&graph, [likes, is_located_in], City)?;

        // Gamma  Pair<Comment, City>
        let g = gamma(vec![
            (p1_l, p1_r),
            (p2_l, p2_r),
        ]);

        // DSL: Gamma(...).left.sum()
        let result = g.sum();

        println!("Q9 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p4() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(&graph, [has_creator, replayof], Post)?;          // [has_creator, replayof].post
        let e2 = expr!(&graph, [hasMember, container_of], Post)?;        // [hasMember,container_of].post

        let result = alpha(&[e1, e2]).sum();

        println!("p4 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p5() -> GCardResult<()> {
        let graph = load_graph();

        let c_like  = expr!(&graph, [hasCreator, likes], Comment)?; // [hasCreator, likes].comment
        let c_reply = expr!(&graph, [replyof], Comment)?;           // [replyof].comment

        // Gamma(
        //   ([hasCreator, likes].comment,[hasCreator, likes].comment),
        //   ([replyof].comment, [replyof].comment),
        //   ([hasCreator, likes].comment,[hasCreator, likes].comment)
        // )
        let g = gamma(vec![
            (c_like.clone(),  c_like.clone()),
            (c_reply.clone(), c_reply.clone()),
            (c_like.clone(),  c_like),
        ]);

        // ..sum()
        let result = g.sum();

        println!("p5 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p6() -> GCardResult<()> {
        let graph = load_graph();

        // t = Gamma(
        //   ([has_member, likes].forum, [has_member,likes].post),
        //   ([container_of].forum, [container_of].post)
        // )
        let t = gamma(vec![
            (
                expr!(&graph, [has_member, likes], Forum)?,
                expr!(&graph, [has_member, likes], Post)?,
            ),
            (
                expr!(&graph, [container_of], Forum)?,
                expr!(&graph, [container_of], Post)?,
            ),
        ]);

        // t.post / t.forum
        let t_post = t.get("Post");
        let t_forum = t.get("Forum");

        // Beta([likes].post, t.post, t.forum)
        let beta1 = beta(
            &expr!(&graph, [likes], Post)?,
            &t_post,
            &t_forum,
        );

        // Beta(t.post, [likes].post, [likes].person)
        let beta2 = beta(
            &t_post,
            &expr!(&graph, [likes], Post)?,
            &expr!(&graph, [likes], Person)?,
        );

        // Gamma(
        //   (Beta(...), Beta(...)),
        //   ([has_member].forum, [has_member].person)
        // )
        let g_inner = gamma(vec![
            (beta1, beta2),
            (
                expr!(&graph, [has_member], Forum)?,
                expr!(&graph, [has_member], Person)?,
            ),
        ]);

        // Gamma(...).person
        let g_person = g_inner.get("Person");

        // [knows].person
        let knows_person = expr!(&graph, [knows], Person)?;

        // Alpha( Gamma(...).person, [knows].person ).sum()
        let result = alpha(&[g_person, knows_person]).sum();

        println!("p6 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p7() -> GCardResult<()> {
        let graph = load_graph();

        // ---------- pair 1: Beta([hasMember].forum, [hasMember].forum, [hasMember].forum) ----------

        let hm_forum = expr!(&graph, [hasMember], Forum)?; // [hasMember].forum

        let b_hm1 = beta(&hm_forum, &hm_forum, &hm_forum);
        let b_hm2 = beta(&hm_forum, &hm_forum, &hm_forum);

        // ---------- pair 2: ([knows].person, [knows].person) ----------

        let knows_p1 = expr!(&graph, [knows], Person)?; // [knows].person
        let knows_p2 = expr!(&graph, [knows], Person)?;

        // Beta([has_creator, replyof].comment, [has_creator].comment, [has_creator].person)
        let b_c1 = beta(
            &expr!(&graph, [has_creator, replyof], Comment)?,
            &expr!(&graph, [has_creator], Comment)?,
            &expr!(&graph, [has_creator], Person)?,
        );

        // Beta([has_creator].comment, [has_creator,replyof].comment,[has_creator,replyof].person)
        let b_c2 = beta(
            &expr!(&graph, [has_creator], Comment)?,
            &expr!(&graph, [has_creator, replyof], Comment)?,
            &expr!(&graph, [has_creator, replyof], Person)?,
        );

        // ---------- Gamma( pair1, pair2, pair3 ).sum() ----------

        let g = gamma(vec![
            (b_hm1, b_hm2),
            (knows_p1, knows_p2),
            (b_c1, b_c2),
        ]);

        let result = g.sum();

        println!("p7 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p8() -> GCardResult<()> {
        let graph = load_graph();

        let tag_expr     = expr!(&graph, [hasTag], Tag)?;      // [hasTag].tag
        let tag_comment  = expr!(&graph, [hasTag], Comment)?;  // [hasTag].Comment

        let b_tag_1 = beta(&tag_expr, &tag_expr, &tag_comment);
        let b_tag_2 = beta(&tag_expr, &tag_expr, &tag_comment);


        let reply_comment_1 = expr!(&graph, [replyof], Comment)?; // [replyof].comment
        let reply_comment_2 = expr!(&graph, [replyof], Comment)?;


        // Beta([has_creator, knows].person, [has_creator].person, [has_creator].comment)
        let b_creator_1 = beta(
            &expr!(&graph, [has_creator, knows], Person)?,
            &expr!(&graph, [has_creator], Person)?,
            &expr!(&graph, [has_creator], Comment)?,
        );

        // Beta([has_creator].person, [has_creator,knows].person,[has_creator,knows].comment)
        let b_creator_2 = beta(
            &expr!(&graph, [has_creator], Person)?,
            &expr!(&graph, [has_creator, knows], Person)?,
            &expr!(&graph, [has_creator, knows], Comment)?,
        );

        // ---------- Gamma( pair1, pair2, pair3 ).sum() ----------

        let g = gamma(vec![
            (b_tag_1,           b_tag_2),
            (reply_comment_1,   reply_comment_2),
            (b_creator_1,       b_creator_2),
        ]);

        let result = g.sum();

        println!("p8 result = {}", result);
        Ok(())
    }



}
