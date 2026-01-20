// A lightweight parser for your GQL *path pattern* strings, e.g.
// (ci:City)<-[plc1:IS_LOCATED_IN]-(p1:Person {language='zh;en'})-[plc_like:LIKE]-(c:Comment)-[chp:HAS_CREATOR {explicitlydeleted=false}]-(p2:Person)<-[plc2:IS_LOCATED_IN {explicitlydeleted=false}]-(ci:City)
//
// It parses into var:label/type + predicates (from {...}) for BOTH nodes and edges,
// and also outputs the chain order (Node/Edge/Node/Edge/...).
//
// Notes:
// - Supports properties in {...} with ops: =, >=, <=, >, <
// - Supports values: 'string', true/false, integers
// - Merges repeated vars (e.g., ci appears twice) by AND-ing predicates (no dedup logic here beyond simple append)
// - This is NOT a full Cypher/GQL parser—it's intentionally limited to your pattern subset.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Eq,
    Ge,
    Le,
    Gt,
    Lt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Str(String),
    Int(i64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredTerm {
    pub key: String,
    pub op: Op,
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct NodePat {
    pub var: String,
    pub label: Option<String>,
    pub preds: Vec<PredTerm>,
}

#[derive(Debug, Clone)]
pub struct EdgePat {
    pub var: String,
    pub typ: Option<String>,
    pub preds: Vec<PredTerm>,
    pub dir: Dir,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    Left,      // <-[e]- (edge arrow points left, i.e., pattern has "<-")
    Right,     // -[e]-> (pattern has "->")
    Undirected // -[e]-  (no arrow)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Atom {
    Node(String),
    Edge(String),
}

#[derive(Debug)]
pub struct Pattern {
    pub nodes: HashMap<String, NodePat>,
    pub edges: HashMap<String, EdgePat>,
    pub chain: Vec<Atom>, // Node, Edge, Node, ...
}

#[derive(Debug)]
pub enum ParseError {
    UnexpectedEof,
    UnexpectedChar(char, usize),
    Expected(&'static str, usize),
    InvalidIdent(usize),
    InvalidNumber(usize),
    InvalidValue(usize),
    DuplicateVarConflict { var: String, what: &'static str },
    TrailingJunk(usize),
}

pub fn parse_pattern(input: &str) -> Result<Pattern, ParseError> {
    let mut p = Parser::new(input);
    p.skip_ws();

    let mut out = Pattern {
        nodes: HashMap::new(),
        edges: HashMap::new(),
        chain: Vec::new(),
    };

    // First must be a node
    let n = p.parse_node()?;
    merge_node(&mut out.nodes, n)?;
    out.chain.push(Atom::Node(p.last_node_var.clone()));

    // Then repeat: connector + edge + connector + node
    loop {
        p.skip_ws();
        if p.eof() {
            break;
        }

        // Left connector part: could be "<-" or "-" or "->" but for edges it is commonly "<-" or "-"
        let left_dir = p.parse_link_dir_left()?; // consumes "<-" or "-" (or "->" if someone writes weird)
        p.skip_ws();

        // Expect edge "[...]"
        let e = p.parse_edge(left_dir)?;
        merge_edge(&mut out.edges, e)?;
        out.chain.push(Atom::Edge(p.last_edge_var.clone()));

        p.skip_ws();

        // Right connector must include "-" and possibly "->" or just "-"
        let _right_dir = p.parse_link_dir_right()?; // consumes "-"/"->"
        p.skip_ws();

        // Next node
        let n = p.parse_node()?;
        merge_node(&mut out.nodes, n)?;
        out.chain.push(Atom::Node(p.last_node_var.clone()));
    }

    Ok(out)
}

fn merge_node(map: &mut HashMap<String, NodePat>, node: NodePat) -> Result<(), ParseError> {
    if let Some(prev) = map.get_mut(&node.var) {
        // label conflict?
        if prev.label.is_some() && node.label.is_some() && prev.label != node.label {
            return Err(ParseError::DuplicateVarConflict { var: node.var, what: "node label" });
        }
        if prev.label.is_none() {
            prev.label = node.label;
        }
        prev.preds.extend(node.preds);
        Ok(())
    } else {
        map.insert(node.var.clone(), node);
        Ok(())
    }
}

fn merge_edge(map: &mut HashMap<String, EdgePat>, edge: EdgePat) -> Result<(), ParseError> {
    if let Some(prev) = map.get_mut(&edge.var) {
        // type conflict?
        if prev.typ.is_some() && edge.typ.is_some() && prev.typ != edge.typ {
            return Err(ParseError::DuplicateVarConflict { var: edge.var, what: "edge type" });
        }
        if prev.typ.is_none() {
            prev.typ = edge.typ;
        }
        // direction conflict isn't necessarily illegal, but we treat as conflict (can relax if you want)
        if prev.dir != edge.dir {
            return Err(ParseError::DuplicateVarConflict { var: edge.var, what: "edge direction" });
        }
        prev.preds.extend(edge.preds);
        Ok(())
    } else {
        map.insert(edge.var.clone(), edge);
        Ok(())
    }
}

struct Parser<'a> {
    s: &'a str,
    i: usize,
    // convenience for chain
    last_node_var: String,
    last_edge_var: String,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self { s, i: 0, last_node_var: String::new(), last_edge_var: String::new() }
    }

    fn eof(&self) -> bool {
        self.i >= self.s.len()
    }

    fn pos(&self) -> usize {
        self.i
    }

    fn peek(&self) -> Option<char> {
        self.s[self.i..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        if self.eof() { return None; }
        let c = self.peek()?;
        self.i += c.len_utf8();
        Some(c)
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() { self.bump(); } else { break; }
        }
    }

    fn expect_char(&mut self, expected: char) -> Result<(), ParseError> {
        match self.bump() {
            Some(c) if c == expected => Ok(()),
            Some(c) => Err(ParseError::UnexpectedChar(c, self.pos())),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_ident(&mut self) -> Result<String, ParseError> {
        self.skip_ws();
        let start = self.pos();
        let mut out = String::new();

        let mut it = self.s[self.i..].chars().peekable();
        match it.peek().copied() {
            Some(c) if is_ident_start(c) => {}
            _ => return Err(ParseError::InvalidIdent(start)),
        }

        while let Some(c) = self.peek() {
            if is_ident_continue(c) {
                out.push(c);
                self.bump();
            } else {
                break;
            }
        }
        Ok(out)
    }

    fn parse_node(&mut self) -> Result<NodePat, ParseError> {
        self.skip_ws();
        self.expect_char('(')?;

        self.skip_ws();
        let var = self.parse_ident()?;

        self.skip_ws();
        let label = if self.peek() == Some(':') {
            self.bump();
            Some(self.parse_ident()?)
        } else {
            None
        };

        self.skip_ws();
        let preds = if self.peek() == Some('{') {
            self.parse_props()?
        } else {
            vec![]
        };

        self.skip_ws();
        self.expect_char(')')?;

        self.last_node_var = var.clone();
        Ok(NodePat { var, label, preds })
    }

    fn parse_edge(&mut self, left_dir: Dir) -> Result<EdgePat, ParseError> {
        self.skip_ws();
        self.expect_char('[')?;

        self.skip_ws();
        let var = self.parse_ident()?;

        self.skip_ws();
        let typ = if self.peek() == Some(':') {
            self.bump();
            Some(self.parse_ident()?)
        } else {
            None
        };

        self.skip_ws();
        let preds = if self.peek() == Some('{') {
            self.parse_props()?
        } else {
            vec![]
        };

        self.skip_ws();
        self.expect_char(']')?;

        // Determine final direction:
        // In your syntax, direction is indicated by the connector before/after the edge.
        // We pass in "left_dir" based on "<-" or "-" seen before "[...]".
        // We'll finalize direction in parse_link_dir_right() by adjusting if we see "->".
        // For simplicity: if left_dir is Left => edge dir is Left; else default Undirected here.
        let dir = left_dir;

        self.last_edge_var = var.clone();
        Ok(EdgePat { var, typ, preds, dir })
    }

    // Parses the link that comes BEFORE the edge, e.g. "<-" or "-"
    // Returns Dir::Left if "<-" is present, otherwise Undirected.
    fn parse_link_dir_left(&mut self) -> Result<Dir, ParseError> {
        self.skip_ws();
        // Expect either "<-" or "-"
        match self.peek() {
            Some('<') => {
                self.bump();
                match self.bump() {
                    Some('-') => Ok(Dir::Left),
                    Some(c) => Err(ParseError::UnexpectedChar(c, self.pos())),
                    None => Err(ParseError::UnexpectedEof),
                }
            }
            Some('-') => {
                self.bump();
                Ok(Dir::Undirected)
            }
            Some(c) => Err(ParseError::UnexpectedChar(c, self.pos())),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    // Parses the link that comes AFTER the edge, e.g. "-" or "->"
    // If it sees "->", you probably want edge dir = Right.
    // In this minimal version, we only consume and ignore. If you want exact dir, you can
    // store it and reconcile with left_dir inside parse_edge() or after edge parse.
    fn parse_link_dir_right(&mut self) -> Result<Dir, ParseError> {
        self.skip_ws();
        match self.peek() {
            Some('-') => {
                self.bump();
                if self.peek() == Some('>') {
                    self.bump();
                    Ok(Dir::Right)
                } else {
                    Ok(Dir::Undirected)
                }
            }
            Some(c) => Err(ParseError::UnexpectedChar(c, self.pos())),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_props(&mut self) -> Result<Vec<PredTerm>, ParseError> {
        self.skip_ws();
        self.expect_char('{')?;
        self.skip_ws();

        let mut preds = Vec::new();

        if self.peek() == Some('}') {
            self.bump();
            return Ok(preds);
        }

        loop {
            self.skip_ws();
            let key = self.parse_ident()?;
            self.skip_ws();
            let op = self.parse_op()?;
            self.skip_ws();
            let value = self.parse_value()?;
            preds.push(PredTerm { key, op, value });

            self.skip_ws();
            match self.peek() {
                Some(',') => { self.bump(); self.skip_ws(); }
                Some('}') => { self.bump(); break; }
                Some(c) => return Err(ParseError::UnexpectedChar(c, self.pos())),
                None => return Err(ParseError::UnexpectedEof),
            }
        }

        Ok(preds)
    }

    fn parse_op(&mut self) -> Result<Op, ParseError> {
        self.skip_ws();
        let start = self.pos();
        match self.peek() {
            Some('=') => { self.bump(); Ok(Op::Eq) }
            Some('>') => {
                self.bump();
                if self.peek() == Some('=') { self.bump(); Ok(Op::Ge) } else { Ok(Op::Gt) }
            }
            Some('<') => {
                self.bump();
                if self.peek() == Some('=') { self.bump(); Ok(Op::Le) } else { Ok(Op::Lt) }
            }
            Some(_) => Err(ParseError::Expected("operator (=,>=,<=,>,<)", start)),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        self.skip_ws();
        let start = self.pos();
        match self.peek() {
            Some('\'') => {
                let s = self.parse_string()?;
                Ok(Value::Str(s))
            }
            Some(c) if c.is_ascii_digit() || c == '-' => {
                let n = self.parse_int()?;
                Ok(Value::Int(n))
            }
            Some(c) if is_ident_start(c) => {
                let ident = self.parse_ident()?;
                match ident.as_str() {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    _ => Err(ParseError::InvalidValue(start)),
                }
            }
            Some(_) => Err(ParseError::InvalidValue(start)),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.skip_ws();
        self.expect_char('\'')?;
        let mut out = String::new();
        while let Some(c) = self.bump() {
            match c {
                '\'' => return Ok(out),
                '\\' => {
                    // simple escape support
                    if let Some(n) = self.bump() {
                        out.push(n);
                    } else {
                        return Err(ParseError::UnexpectedEof);
                    }
                }
                _ => out.push(c),
            }
        }
        Err(ParseError::UnexpectedEof)
    }

    fn parse_int(&mut self) -> Result<i64, ParseError> {
        self.skip_ws();
        let start = self.pos();
        let mut s = String::new();
        if self.peek() == Some('-') {
            s.push('-');
            self.bump();
        }
        let mut any = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                any = true;
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if !any {
            return Err(ParseError::InvalidNumber(start));
        }
        s.parse::<i64>().map_err(|_| ParseError::InvalidNumber(start))
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}
fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// --------------------------
// Example usage (in your tests)
// --------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_q8_pattern() {
        let s = "(ci:City)<-[plc1:IS_LOCATED_IN]-(p1:Person {language='zh;en'})-[plc_like:LIKE]-(c:Comment)-[chp:HAS_CREATOR {explicitlydeleted=false}]-(p2:Person)<-[plc2:IS_LOCATED_IN {explicitlydeleted=false}]-(ci:City)";
        let pat = parse_pattern(s).unwrap();

        // Quick sanity checks
        assert!(pat.nodes.contains_key("ci"));
        assert!(pat.nodes.contains_key("p1"));
        assert!(pat.edges.contains_key("chp"));
        assert_eq!(pat.nodes["p1"].label.as_deref(), Some("Person"));
        assert_eq!(pat.nodes["p1"].preds.len(), 1);
        assert_eq!(pat.edges["chp"].preds.len(), 1);

        // Chain order: Node, Edge, Node, Edge, ...
        assert!(matches!(pat.chain[0], Atom::Node(_)));
        assert!(matches!(pat.chain[1], Atom::Edge(_)));
    }

    #[test]
    fn parse_q8_variant_with_comment_predicate() {
        let s = "(ci:City)<-[plc1:IS_LOCATED_IN]-(p1:Person)-[plc_like:LIKE]-(c:Comment {creationdate>=1350093427787})-[chp:HAS_CREATOR]-(p2:Person)<-[plc2:IS_LOCATED_IN]-(ci:City)";
        let pat = parse_pattern(s).unwrap();

        // ---- basic existence checks ----
        assert!(pat.nodes.contains_key("ci"));
        assert!(pat.nodes.contains_key("p1"));
        assert!(pat.nodes.contains_key("c"));
        assert!(pat.nodes.contains_key("p2"));

        assert!(pat.edges.contains_key("plc1"));
        assert!(pat.edges.contains_key("plc_like"));
        assert!(pat.edges.contains_key("chp"));
        assert!(pat.edges.contains_key("plc2"));

        // ---- labels / types ----
        assert_eq!(pat.nodes["ci"].label.as_deref(), Some("City"));
        assert_eq!(pat.nodes["p1"].label.as_deref(), Some("Person"));
        assert_eq!(pat.nodes["c"].label.as_deref(), Some("Comment"));
        assert_eq!(pat.nodes["p2"].label.as_deref(), Some("Person"));

        assert_eq!(pat.edges["plc1"].typ.as_deref(), Some("IS_LOCATED_IN"));
        assert_eq!(pat.edges["plc_like"].typ.as_deref(), Some("LIKE"));
        assert_eq!(pat.edges["chp"].typ.as_deref(), Some("HAS_CREATOR"));
        assert_eq!(pat.edges["plc2"].typ.as_deref(), Some("IS_LOCATED_IN"));

        // ---- predicate extraction (the key point of this test) ----
        // Only (c:Comment {creationdate>=1350093427787}) has a predicate.
        assert_eq!(pat.nodes["c"].preds.len(), 1);

        let p = &pat.nodes["c"].preds[0];
        assert_eq!(p.key, "creationdate");
        assert_eq!(p.op, Op::Ge);
        assert_eq!(p.value, Value::Int(1350093427787));

        // Other nodes/edges should have no predicates in this query
        assert_eq!(pat.nodes["p1"].preds.len(), 0);
        assert_eq!(pat.nodes["p2"].preds.len(), 0);
        assert_eq!(pat.nodes["ci"].preds.len(), 0);

        assert_eq!(pat.edges["plc1"].preds.len(), 0);
        assert_eq!(pat.edges["plc_like"].preds.len(), 0);
        assert_eq!(pat.edges["chp"].preds.len(), 0);
        assert_eq!(pat.edges["plc2"].preds.len(), 0);

        // ---- chain order check ----
        // Expected chain:
        // N(ci) E(plc1) N(p1) E(plc_like) N(c) E(chp) N(p2) E(plc2) N(ci)
        assert_eq!(
            pat.chain,
            vec![
                Atom::Node("ci".to_string()),
                Atom::Edge("plc1".to_string()),
                Atom::Node("p1".to_string()),
                Atom::Edge("plc_like".to_string()),
                Atom::Node("c".to_string()),
                Atom::Edge("chp".to_string()),
                Atom::Node("p2".to_string()),
                Atom::Edge("plc2".to_string()),
                Atom::Node("ci".to_string()),
            ]
        );

        // Optional: print to inspect manually when running `cargo test -- --nocapture`
        println!("Parsed nodes: {:#?}", pat.nodes);
        println!("Parsed edges: {:#?}", pat.edges);
        println!("Chain: {:#?}", pat.chain);
    }

}
