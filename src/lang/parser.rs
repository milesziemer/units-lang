use super::token::{Token, TokenKind};

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
        }
        return self.curr;
    }

    pub fn parse(&mut self) -> Result<Node, String> {
        let result = self.expr();
        Ok(result)
    }

    fn expr(&mut self) -> Node {
        return self.add_sub_expr();
    }

    fn add_sub_expr(&mut self) -> Node {
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
        return left;
    }
    fn sign_expr(&mut self) -> Node {
        if let Some(token) = self.curr {
            let is_sign_expr = match token.kind {
                TokenKind::ADD | TokenKind::SUBTRACT => true,
                _ => false,
            };
            if is_sign_expr {
                self.advance();
                let sign_expr = self.sign_expr();
                return Node::UnaryOp {
                    node: Box::new(sign_expr),
                    op: token,
                };
            }
        }
        return self.pow_expr();
    }

    fn pow_expr(&mut self) -> Node {
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
        return left;
    }

    fn stmt(&mut self) -> Node {
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
                                Node::Error("Invalid Syntax Expected ')'".to_string()),
                                false,
                            ),
                        }
                    } else {
                        (expr, false)
                    }
                }
                _ => (
                    Node::Error("Invalid Syntax, Expected number or '('".to_string()),
                    false,
                ),
            };
            if adv {
                self.advance();
            }
            return node;
        }
        return Node::Error("Invalid Syntax".to_string());
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
    Error(String),
}
