use crate::degreepiecewise::{gamma_core, Pcf};
use crate::error::GCardResult;
use crate::{AltKey, DegreeSeqGraphCompressed};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
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
        graph: &DegreeSeqGraphCompressed,
        path: &AltKey,
        target_node: &str,
        f: f64,
    ) -> GCardResult<Self> {
        let degree_seq = graph.get_piece_func_by_path(path, target_node);
        let pcf = Rc::new(degree_seq.truncate_by_selectivity(f).clone());

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

    pub fn get_num(&self) -> u64 {
        match self {
            Self::Single { pcf, .. } => pcf.get_num_rows() as u64,
            Self::Pair { left, .. } => left.get_num_rows() as u64,
        }
    }

    pub fn sum(&self) -> u64 {
        self.get_num()
    }
}

#[macro_export]
macro_rules! expr {
    ($graph:expr, [$($edge:ident),*], $target:ident, $f:expr) => {
        {
        let key = AltKey(vec![$(stringify!($edge).to_string()),*]);
            Expr::from_path($graph, &key, stringify!($target), $f)
        }
    };
    ($graph:expr, [$($edge:ident),*], $target:expr, $f:expr) => {
        {
        let key = AltKey(vec![$(stringify!($edge).to_string()),*]);
            Expr::from_path($graph, &key, $target, $f)
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
    let tmp = Expr::new_single(sy.node_name().to_string(), Rc::new(result));
    let res = alpha(&[tmp, sy.clone()]);
    res
}

pub fn gamma(paris: Vec<(Expr, Expr)>) -> Expr {
    let item: Vec<(&Pcf, &Pcf)> = paris.iter().map(|t| (t.0.as_pcf(), t.1.as_pcf())).collect();
    let first_node = paris.first().unwrap();
    let left_name = first_node.0.node_name().to_string();
    let right_name = first_node.1.node_name().to_string();
    let (left_pcf, right_pcf) = gamma_core(item);
    Expr::new_pair(left_name, right_name, Rc::new(left_pcf), Rc::new(right_pcf))
}

/// 从GQL查询文件中解析GQL查询
///
/// 文件格式：
/// -- query_id
/// (gql pattern)
///
/// # Arguments
/// * `file_path` - GQL查询文件路径
///
/// # Returns
/// 返回GQL查询字符串的向量
pub fn parse_gql_queries<P: AsRef<Path>>(file_path: P) -> GCardResult<Vec<String>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;

    let mut queries = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // 查找注释行（以 -- 开头）
        if line.starts_with("--") {
            // 查找下一行非空行作为GQL查询
            i += 1;
            while i < lines.len() && lines[i].trim().is_empty() {
                i += 1;
            }

            if i < lines.len() {
                let gql = lines[i].trim().to_string();
                if !gql.is_empty() {
                    queries.push(gql);
                }
            }
        }
        i += 1;
    }

    Ok(queries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_pattern;
    use crate::wander_join_selectivity;
    use duckdb::Connection;
    use std::path::Path;

    fn load_graph() -> DegreeSeqGraphCompressed {
        DegreeSeqGraphCompressed::import_bincode(Path::new(
            "/Users/colin/dev/pathce/gcard_es/statistic_graph_sf0.003_fastcomp.bincode",
        ))
        .unwrap()
    }
    #[allow(dead_code)]
    fn print_path(graph: &DegreeSeqGraphCompressed) {
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
                &expr!(
                    &graph,
                    [
                        Person,
                        Person_isLocatedIn_City,
                        City,
                        City_isPartOf_Country,
                        Country
                    ],
                    Person, 1.0
                )?,
                &expr!(&graph, [Forum, Forum_hasMember_Person, Person], Person, 1.0)?,
                &expr!(&graph, [Forum, Forum_hasMember_Person, Person], Forum, 1.0)?,
            ),
            beta(
                &expr!(
                    &graph,
                    [
                        Comment,
                        Comment_hasTag_Tag,
                        Tag,
                        Tag_hasType_TagClass,
                        TagClass
                    ],
                    Comment,
                    1.0
                )?,
                &expr!(
                    &graph,
                    [
                        Forum,
                        Forum_containerOf_Post,
                        Post,
                        Comment_replyOf_Post,
                        Comment
                    ],
                    Comment,
                    1.0
                )?,
                &expr!(
                    &graph,
                    [
                        Forum,
                        Forum_containerOf_Post,
                        Post,
                        Comment_replyOf_Post,
                        Comment
                    ],
                    Forum,
                    1.0
                )?,
            ),
        ])
        .get_num();
        println!("Q1 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q2() -> GCardResult<()> {
        // (p1:Person {gender='female'})-[pkp:KNOWS]-(p2:Person)-[php:HAS_CREATOR]-(po:Post)-[crp:REPLY_OF]-(c:Comment)-[chp:HAS_CREATOR {creationDate>=1350093427787}]-(p1:Person {gender='female'})
        let gqls = parse_gql_queries("/Users/colin/dev/pathce/benchmark/q2/q2_predicates_100_gql.txt")?;
        for gql in gqls {
            let s = parse_pattern(&gql).unwrap();
            let expr1: Vec<String> = vec![
                "Comment",
                "Comment_replyOf_Post",
                "Post",
                "Post_hasCreator_Person",
                "Person",
            ].into_iter().map(|s| s.to_string()).collect();
            let expr1_var: Vec<String> = vec!["c", "crp", "po", "php", "p2"]
                .into_iter().map(|s| s.to_string()).collect();
            let expr2 :  Vec<String> = vec![
                "Comment", "Comment_hasCreator_Person", "Person",
            ].into_iter().map(|s| s.to_string()).collect();

            let expr2_var:  Vec<String>  = vec!["c", "chp", "p1"].into_iter().map(|s| s.to_string()).collect();
            let graph = load_graph();
            let conn = match Connection::open("/Users/colin/dev/pathce/graphs/ldbc/duckdb/ldbc_with_attrs_sf0.1.duckdb") {
                Ok(conn) => conn,
                Err(e) => {
                    std::process::exit(1);
                }
            };
            let f1 = wander_join_selectivity(&conn, &s, &expr1, &expr1_var, 20).unwrap();
            let f2 = wander_join_selectivity(&conn, &s, &expr2, &expr2_var, 200).unwrap();
            let res = alpha(&[
                expr!(
                    &graph,
                    [
                        Comment,
                        Comment_replyOf_Post,
                        Post,
                        Post_hasCreator_Person,
                        Person
                    ],
                    Comment, f1
                )?,
                expr!(
                    &graph,
                    [Comment, Comment_hasCreator_Person, Person],
                    Comment,
                    f2
                )?,
            ]);
            println!("Q2 result = {}", res.get_num());
        }

        Ok(())
    }

    #[test]
    fn test_q4() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(&graph, [Post, Post_hasTag_Tag, Tag], Post, 1.0)?;
        let e2 = expr!(&graph, [Post, Post_hasCreator_Person, Person], Post, 1.0)?;
        let e3 = expr!(&graph, [Person, Person_likes_Post, Post], Post, 1.0)?;
        let e4 = expr!(&graph, [Comment, Comment_replyOf_Post, Post], Post, 1.0)?;

        let result = alpha(&[e1, e2, e3, e4]).sum();

        println!("Q4 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q5() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(&graph, [Post, Post_hasTag_Tag, Tag], Post, 1.0)?;
        let e2 = expr!(
            &graph,
            [Post, Comment_replyOf_Post, Comment, Comment_hasTag_Tag, Tag],
            Post,
            1.0
        )?;

        let result = alpha(&[e1, e2]).sum();

        println!("Q5 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q6() -> GCardResult<()> {
        let graph = load_graph();

        let e1 = expr!(
            &graph,
            [
                Person,
                Person_knows_Person,
                Person,
                Person_knows_Person,
                Person
            ],
            Person,
            1.0
        )?;
        let e2 = expr!(&graph, [Person, Person_hasInterest_Tag, Tag], Person, 1.0)?;

        let result = alpha(&[e1, e2]).sum();

        println!("Q6 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_q8() -> GCardResult<()> {
        let graph = load_graph();
        let result = alpha(&[
            expr!(&graph, [Comment, Comment_hasTag_Tag, Tag], Comment, 1.0)?,
            expr!(
                &graph,
                [Comment, Comment_replyOf_Post, Post, Post_hasTag_Tag, Tag],
                Comment,
                1.0
            )?,
        ]);

        println!("Q8 result = {}", result.sum());
        Ok(())
    }

    #[test]
    fn test_q9() -> GCardResult<()> {
        let graph = load_graph();

        let result = alpha(&[
            expr!(&graph, [Person, Person_knows_Person, Person], Person, 1.0)?,
            expr!(
                &graph,
                [
                    Person,
                    Person_knows_Person,
                    Person,
                    Person_hasInterest_Tag,
                    Tag
                ],
                Person,
                1.0
            )?,
        ])
        .sum();

        println!("Q9 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p1() -> GCardResult<()> {
        let graph = load_graph();

        let result = expr!(
            &graph,
            [
                Person,
                Person_hasInterest_Tag,
                Tag,
                Comment_hasTag_Tag,
                Comment
            ],
            Person,
            1.0
        )?;

        println!("P1 result = {}", result.get_num());
        Ok(())
    }

    #[test]
    fn test_p2() -> GCardResult<()> {
        let graph = load_graph();

        // DSL: Gamma(...).left.sum()
        let result = alpha(&[
            expr!(&graph, [Person, Person_likes_Comment, Comment], Comment, 1.0)?,
            expr!(
                &graph,
                [
                    Comment,
                    Person_likes_Comment,
                    Person,
                    Person_isLocatedIn_City,
                    City
                ],
                Comment,
                1.0
            )?,
        ])
        .sum();

        println!("P2 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p4() -> GCardResult<()> {
        let graph = load_graph();
        let e1 = expr!(
            &graph,
            [
                Person,
                Comment_hasCreator_Person,
                Comment,
                Comment_replyOf_Post,
                Post
            ],
            Post,
            1.0
        )?; // [has_creator, replayof].post
        let e2 = expr!(
            &graph,
            [
                Person,
                Forum_hasMember_Person,
                Forum,
                Forum_containerOf_Post,
                Post
            ],
            Post,
            1.0
        )?; // [hasMember,container_of].post

        let result = alpha(&[e1, e2]).sum();

        println!("P4 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p5() -> GCardResult<()> {
        let graph = load_graph();
        // ..sum()
        let result = alpha(&[
            expr!(&graph, [Person, Person_likes_Comment, Comment], Comment, 1.0)?,
            expr!(&graph, [Comment, Comment_replyOf_Comment, Comment], Comment, 1.0)?,
            expr!(&graph, [Comment, Person_likes_Comment, Person], Comment, 1.0)?,
        ])
        .sum();

        println!("P5 result = {}", result);
        Ok(())
    }

    #[test]
    fn test_p6() -> GCardResult<()> {
        let graph = load_graph();

        println!(
            "P6 result = {}",
            alpha(&[
                expr!(
                    &graph,
                    [
                        Person,
                        Person_likes_Post,
                        Post,
                        Forum_containerOf_Post,
                        Forum
                    ],
                    Forum,
                    1.0
                )?,
                expr!(&graph, [Forum, Forum_hasMember_Person, Person], Forum, 1.0)?
            ])
            .get_num()
        );
        Ok(())
    }

    #[test]
    fn test_p7() -> GCardResult<()> {
        let graph = load_graph();
        println!(
            "P7 result = {}",
            alpha(&[
                expr!(
                    &graph,
                    [
                        Forum,
                        Forum_hasMember_Person,
                        Person,
                        Comment_hasCreator_Person,
                        Comment
                    ],
                    Comment,
                    1.0
                )?,
                expr!(
                    &graph,
                    [
                        Person,
                        Comment_hasCreator_Person,
                        Comment,
                        Comment_replyOf_Comment,
                        Comment
                    ],
                    Comment,
                    1.0
                )?
            ])
            .sum()
        );
        Ok(())
    }

    #[test]
    fn test_p8() -> GCardResult<()> {
        let graph = load_graph();
        let result = alpha(&[
            expr!(
                &graph,
                [Comment, Comment_hasCreator_Person, Person],
                Comment,
                1.0
            )?,
            expr!(
                &graph,
                [
                    Comment,
                    Comment_replyOf_Comment,
                    Comment,
                    Comment_hasCreator_Person,
                    Person
                ],
                Comment,
                1.0
            )?,
            expr!(&graph, [Comment, Comment_hasTag_Tag, Tag], Comment, 1.0)?,
        ])
        .sum();

        println!("P8 result = {}", result);
        Ok(())
    }
}
