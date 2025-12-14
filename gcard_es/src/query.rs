use crate::degreepiecewise::{gamma_core, Pcf};
use crate::error::{GCardError, GCardResult};
use crate::graph::DegreeSeqGraph;
use crate::AltKey;
use std::collections::HashMap;
use std::fmt;
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

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Single { pcf, node } => {
                writeln!(f, "Expr::Single")?;
                writeln!(f, "  node: {}", node)?;
                writeln!(f, "  pcf:")?;

                for line in format!("{}", pcf).lines() {
                    writeln!(f, "    {}", line)?;
                }

                Ok(())
            }

            Expr::Pair { node_map, .. } => {
                writeln!(f, "Expr::Pair")?;
                writeln!(f, "  node_map ({}):", node_map.len())?;

                let mut keys: Vec<_> = node_map.keys().collect();
                keys.sort();

                for k in keys {
                    let pcf = &node_map[k];
                    writeln!(f, "    {}:", k)?;
                    for line in format!("{}", pcf).lines() {
                        writeln!(f, "      {}", line)?;
                    }
                }

                Ok(())
            }
        }
    }
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
        path: &AltKey,
        target_node: &str,
    ) -> GCardResult<Self> {
        let degree_seq = graph
            .get_degree_seq_vec_by_path(path, target_node)
            .ok_or_else(|| {
                GCardError::InvalidData(format!("Path not found: {:?}.{}", path, target_node))
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
        let key = AltKey(vec![$(stringify!($edge).to_string()),*]);
            Expr::from_path($graph, &key, stringify!($target))
        }
    };
    ($graph:expr, [$($edge:ident),*], $target:expr) => {
        {
        let key = AltKey(vec![$(stringify!($edge).to_string()),*]);
            Expr::from_path($graph, &key, $target)
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
        DegreeSeqGraph::import_bincode(Path::new(
            "/Users/colin/dev/pathce/gcard_es/statistic.bincode",
        ))
        .unwrap()
    }

    fn print_path(graph: &DegreeSeqGraph) {
        let path = &graph.edge_set_to_endpoints;
        for (key, _) in path {
            println!("{:?}", key);
        }
    }

    #[test]
    fn test_q1() -> GCardResult<()> {
        let graph = load_graph();
        let result = alpha(&[
            beta(
                &expr!(&graph, [Person, Person_isLocatedIn_City, City, City_isPartOf_Country, Country], Person)?,
                &expr!(&graph, [Forum, Forum_hasMember_Person, Person], Person)?,
                &expr!(&graph, [Forum, Forum_hasMember_Person, Person], Forum)?,
            ),
            beta(
                &expr!(&graph, [Comment, Comment_hasTag_Tag, Tag, Tag_hasType_TagClass, TagClass], Comment)?,
                &expr!(&graph, [Forum, Forum_containerOf_Post, Post, Comment_replyOf_Post, Comment], Comment)?,
                &expr!(&graph, [Forum, Forum_containerOf_Post, Post, Comment_replyOf_Post, Comment], Forum)?,
            ),
        ])
        .get_num();
        println!("get row num is {}", result);
        Ok(())
    }

    #[test]
    fn test_q2() -> GCardResult<()> {
        let graph = load_graph();
        let g = gamma(vec![
            (
                expr!(&graph, [Comment, Comment_hasCreator_Person, Person, Person_knows_Person, Person], Comment)?,
                expr!(&graph, [Comment, Comment_hasCreator_Person, Person, Person_knows_Person, Person], Person)?,
            ),
            (
                expr!(&graph, [Comment, Comment_replyOf_Post, Post, Post_hasCreator_Person, Person], Comment)?,
                expr!(&graph, [Comment, Comment_replyOf_Post, Post, Post_hasCreator_Person, Person], Person)?,
            ),
        ]);
        let result = g.get_num();
        println!("q2 row num is {}", result);
        Ok(())
    }

    #[test]
    fn test_q4() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(&graph, [Post, Post_hasTag_Tag, Tag], Post)?;
        let e2 = expr!(&graph, [Post, Post_hasCreator_Person, Person], Post)?;
        let e3 = expr!(&graph, [Person, Person_likes_Post, Post], Post)?;
        let e4 = expr!(&graph, [Comment, Comment_replyOf_Post, Post], Post)?;

        let result = alpha(&[e1, e2, e3, e4]).sum();

        println!("Q4 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q5() -> GCardResult<()> {
        let graph = load_graph();
        print_path(&graph);

        let e1 = expr!(&graph, [Post, Post_hasTag_Tag, Tag], Post)?;
        let e2 = expr!(&graph, [Post, Comment_replyOf_Post, Comment, Comment_hasTag_Tag, Tag], Post)?;

        let result = alpha(&[e1, e2]).sum();

        println!("Q5 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q6() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(&graph, [Person, Person_knows_Person, Person, Person_knows_Person, Person], Person)?;
        let e2 = expr!(&graph, [Person, Person_hasInterest_Tag, Tag], Person)?;

        let result = alpha(&[e1, e2]).sum();

        println!("Q6 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q8() -> GCardResult<()> {
        let graph = load_graph();

        let p1_l = expr!(&graph, [replyOf1, hasTag2], Tag)?;
        let p1_r = expr!(&graph, [replyOf1, hasTag2], Comment)?;
        let p2_l = expr!(&graph, [hasTag], Tag)?;
        let p2_r = expr!(&graph, [hasTag], Comment)?;

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

        let g = gamma(vec![(p1_l.clone(), p1_l), (p1_r.clone(), p1_r)]);

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
        let p1_l = expr!(&graph, [hasTag, hasCreator1], Tag)?; // tag
        let p1_r = expr!(&graph, [hasTag, hasCreator1], Person)?; // person

        // 2 : ([has_interest].tag , [has_interest].person)
        let p2_l = expr!(&graph, [hasInterest], Tag)?; // tag
        let p2_r = expr!(&graph, [hasInterest], Person)?; // person

        let g = gamma(vec![(p1_l, p1_r), (p2_l, p2_r)]);

        let result = g.sum();

        println!("Q8 result = {}", result);
        Ok(())
    }
    #[test]
    fn test_p2() -> GCardResult<()> {
        let graph = load_graph();

        //  ([has_creator, is_located_in].comment, [has_creator, is_located_in].city)
        let p1_l = expr!(&graph, [hasCreator1, isLocatedIn2], Comment)?;
        let p1_r = expr!(&graph, [hasCreator1, isLocatedIn2], City)?;

        //  ([likes, is_located_in].comment, [likes, is_located_in].city)
        let p2_l = expr!(&graph, [likes1, isLocatedIn2], Comment)?;
        let p2_r = expr!(&graph, [likes1, isLocatedIn2], City)?;

        // Gamma  Pair<Comment, City>
        let g = gamma(vec![(p1_l, p1_r), (p2_l, p2_r)]);

        // DSL: Gamma(...).left.sum()
        let result = g.sum();

        println!("Q9 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p4() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(&graph, [hasCreator1, replyOf1], Post)?; // [has_creator, replayof].post
        let e2 = expr!(&graph, [hasMember, containerOf], Post)?; // [hasMember,container_of].post

        let result = alpha(&[e1, e2]).sum();

        println!("p4 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p5() -> GCardResult<()> {
        let graph = load_graph();
        print_path(&graph);
        let c_like = expr!(&graph, [Comment,Comment_hasCreator_Person, Person, Person_likes_Comment, Comment], Comment)?; // [hasCreator, likes].comment
        let c_reply = expr!(&graph, [Comment, Comment_replyOf_Comment, Comment], Comment)?; // [replyof].comment

        // Gamma(
        //   ([hasCreator, likes].comment,[hasCreator, likes].comment),
        //   ([replyof].comment, [replyof].comment),
        //   ([hasCreator, likes].comment,[hasCreator, likes].comment)
        // )
        let g = gamma(vec![
            (c_like.clone(), c_like.clone()),
            (c_reply.clone(), c_reply.clone()),
            (c_like.clone(), c_like),
        ]);

        // ..sum()
        let result = g.sum();

        println!("p5 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p6() -> GCardResult<()> {
        let graph = load_graph();
        print_path(&graph);
        // t=Gamma(
        //     ([hasMember, likes].post, [hasMember, likes].Forum),
        //     ([containerof].post, [containerof]. forum)
        // )
        let t = gamma(vec![
            (
                expr!(&graph, [Forum, Forum_hasMember_Person, Person, Person_likes_Post, Post], Post)?,
                expr!(&graph, [Forum, Forum_hasMember_Person, Person, Person_likes_Post, Post], Forum)?,
            ),
            (
                expr!(&graph, [Forum, Forum_containerOf_Post, Post], Post)?,
                expr!(&graph, [Forum, Forum_containerOf_Post, Post], Forum)?,
            ),
        ]);

        // Alpha(
        //     Gamma(
        //     beta([like].Post, t.Post, t.forum),
        //     beta(t.Post, [like].Post, [like].person)
        //     ),
        //     ([hasMember].forum, [hasMember].Person)
        //     )
        // )

        // t.post / t.forum
        let t_post = t.get("Post");
        let t_forum = t.get("Forum");

        // Beta([likes].post, t.post, t.forum)
        let beta1 = beta(&expr!(&graph, [Person, Person_likes_Post, Post], Post)?, &t_post, &t_forum);

        // Beta(t.post, [likes].post, [likes].person)
        let beta2 = beta(
            &t_post,
            &expr!(&graph, [Person, Person_likes_Post, Post], Post)?,
            &expr!(&graph, [Person, Person_likes_Post, Post], Person)?,
        );

        // Gamma(
        //   (Beta(...), Beta(...)),
        //   ([has_member].forum, [has_member].person)
        // )
        let g_inner = gamma(vec![
            (beta1, beta2),
            (
                expr!(&graph, [Forum, Forum_hasMember_Person, Person], Forum)?,
                expr!(&graph, [Forum, Forum_hasMember_Person, Person], Person)?,
            ),
        ]);

        // Gamma(...).person
        let g_person = g_inner.get("Person");

        // Alpha( Gamma(...).person, [knows].person ).sum()
        let result = g_person.sum();

        println!("p6 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p7() -> GCardResult<()> {
        let graph = load_graph();
        print_path(&graph);
        let t1 = gamma(vec![
            (
                beta(
                    &expr!(&graph, [Person, Comment_hasCreator_Person, Comment, Comment_replyOf_Comment, Comment], Comment)?,
                    &expr!(&graph, [Comment, Comment_hasCreator_Person, Person], Comment)?,
                    &expr!(&graph, [Comment, Comment_hasCreator_Person, Person], Person)?,
                ),
                beta(
                    &expr!(&graph, [Comment, Comment_hasCreator_Person, Person], Comment)?,
                    &expr!(&graph, [Person, Comment_hasCreator_Person, Comment, Comment_replyOf_Comment, Comment], Comment)?,
                    &expr!(&graph, [Person, Comment_hasCreator_Person, Comment, Comment_replyOf_Comment, Comment], Person)?,
                ),
            ),
            (
                expr!(&graph, [Person, Person_knows_Person, Person], Person)?,
                expr!(&graph, [Person, Person_knows_Person, Person], Person)?,
            ),
        ]);

        let g = gamma(vec![
            (
                expr!(&graph, [Forum, Forum_hasMember_Person, Person], Forum)?,
                expr!(&graph, [Forum, Forum_hasMember_Person, Person], Person)?,
            ),
            (
                beta(
                    &t1.get("Person"),
                    &expr!(&graph, [Forum, Forum_hasMember_Person, Person], Person)?,
                    &expr!(&graph, [Forum, Forum_hasMember_Person, Person], Forum)?,
                ),
                beta(
                    &expr!(&graph, [Forum, Forum_hasMember_Person, Person], Person)?,
                    &t1.get("Person"),
                    &t1.get("Person"),
                ),
            ),
        ]);

        let result = g.sum();

        println!("p7 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p8() -> GCardResult<()> {
        let graph = load_graph();

        let tag_expr = expr!(&graph, [hasTag], Tag)?; // [hasTag].tag
        let tag_comment = expr!(&graph, [hasTag], Comment)?; // [hasTag].Comment

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
            (b_tag_1, b_tag_2),
            (reply_comment_1, reply_comment_2),
            (b_creator_1, b_creator_2),
        ]);

        let result = g.sum();

        println!("p8 result = {}", result);
        Ok(())
    }
}
