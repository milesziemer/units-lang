use super::{
    token::{Token, TokenKind},
    tracker::Tracker,
};

pub struct Parser {
    tokens: Vec<Token>,
    curr: Option<Token>,
    index: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Parser {
        let curr = tokens.first().copied();
        Parser {
            tokens,
            curr,
            index: 0,
        }
    }

    fn advance(&mut self) -> Option<Token> {
        self.index += 1;
        if self.index < self.tokens.len() {
            self.curr = self.tokens.get(self.index).copied();
        } else {
            self.curr = None;
        }
        return self.curr;
    }

    pub fn parse(&mut self) -> Result<Node, ParseError> {
        let result = self.expr();
        match result {
            Node::Error(ParseError {
                start,
                end,
                kind,
                details,
            }) => Err(ParseError {
                start,
                end,
                kind,
                details,
            }),
            _ => Ok(result),
        }
    }

    fn expr(&mut self) -> Node {
        return self.add_sub_expr();
    }

    fn add_sub_expr(&mut self) -> Node {
        // println!("add_sub_expr");
        let mut left = self.mul_div_expr();
        while let Some(token) = self.curr {
            match token.kind {
                TokenKind::ADD | TokenKind::SUBTRACT => {
                    let op = token;
                    self.advance();
                    let right = self.mul_div_expr();
                    left = Node::BinaryOp {
                        left: Box::new(left),
                        right: Box::new(right),
                        op,
                    }
                }
                _ => break,
            }
        }
        return left;
    }

    fn mul_div_expr(&mut self) -> Node {
        // println!("mul_div_expr");
        let mut left = self.sign_expr();
        while let Some(token) = self.curr {
            match token.kind {
                TokenKind::MULTIPLY | TokenKind::DIVIDE => {
                    let op = token;
                    self.advance();
                    let right = self.sign_expr();
                    left = Node::BinaryOp {
                        left: Box::new(left),
                        right: Box::new(right),
                        op,
                    };
                }
                _ => break,
            }
        }
        // println!("{:?}", left);
        return left;
    }

    fn sign_expr(&mut self) -> Node {
        // println!("sign_expr");
        if let Some(token) = self.curr {
            let is_sign_expr = match token.kind {
                TokenKind::ADD | TokenKind::SUBTRACT => true,
                _ => false,
            };
            if is_sign_expr {
                // println!("{:?}", token);
                let op = token.clone();
                self.advance();
                let sign_expr = self.sign_expr();
                return Node::UnaryOp {
                    node: Box::new(sign_expr),
                    op,
                };
            }
        }
        let node = self.pow_expr();
        // println!("sign node: {:?}", node);
        return node;
        // return self.pow_expr();
    }

    fn pow_expr(&mut self) -> Node {
        // println!("pow_expr");
        let mut left = self.stmt();
        while let Some(token) = self.curr {
            match token.kind {
                TokenKind::POWER => {
                    let op = token;
                    self.advance();
                    let right = self.sign_expr();
                    left = Node::BinaryOp {
                        left: Box::new(left),
                        right: Box::new(right),
                        op,
                    };
                }
                _ => break,
            }
        }
        // println!("pow node: {:?}", left);
        return left;
    }

    fn stmt(&mut self) -> Node {
        // println!("stmt");
        if let Some(token) = self.curr {
            let (node, adv) = match token.kind {
                TokenKind::NUM => (Node::Number(token), true),
                TokenKind::LPAREN => {
                    self.advance();
                    let expr = self.expr();
                    if let Some(tok) = self.curr {
                        match tok.kind {
                            TokenKind::RPAREN => (expr, true),
                            _ => (
                                Node::Error(ParseError::from(
                                    Some(token),
                                    ParseErrorKind::InvalidSyntax,
                                    "Expected ')'".to_string(),
                                )),
                                false,
                            ),
                        }
                    } else {
                        (expr, false)
                    }
                }
                _ => (
                    Node::Error(ParseError::from(
                        Some(token),
                        ParseErrorKind::InvalidSyntax,
                        "Expected number or '('".to_string(),
                    )),
                    true,
                ),
            };
            if adv {
                self.advance();
            }
            // println!("stmt node: {:?}", node);
            return node;
        }
        Node::Error(ParseError::from(
            self.curr,
            ParseErrorKind::InvalidSyntax,
            "Expected number, '+', '-', or '('".to_string(),
        ))
    }
}

#[derive(Debug)]
pub enum ParseErrorKind {
    InvalidSyntax,
    IllegalChar,
    IllegalNumber,
}

#[derive(Debug)]
pub struct ParseError {
    pub start: Option<Tracker>,
    pub end: Option<Tracker>,
    pub kind: ParseErrorKind,
    pub details: String,
}

impl ParseError {
    fn from(token: Option<Token>, kind: ParseErrorKind, details: String) -> ParseError {
        let (start, end) = match token {
            Some(token) => (token.start, token.end),
            None => (None, None),
        };
        ParseError {
            start,
            end,
            kind,
            details,
        }
    }
}

#[derive(Debug)]
pub enum Node {
    BinaryOp {
        left: Box<Node>,
        right: Box<Node>,
        op: Token,
    },
    UnaryOp {
        node: Box<Node>,
        op: Token,
    },
    Number(Token),
    Error(ParseError),
}
