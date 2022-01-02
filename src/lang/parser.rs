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
        Ok(self.expr())
    }

    fn factor(&mut self) -> Node {
        let token = self.curr.unwrap();
        self.advance();
        Node::Number(token)
    }

    fn expr(&mut self) -> Node {
        let mut left = self.factor();
        while let Some(curr) = self.curr {
            match curr.kind {
                TokenKind::SUBTRACT | TokenKind::ADD => {
                    self.advance();
                    let right = self.factor();
                    left = Node::BinaryOp {
                        left: Box::new(left),
                        right: Box::new(right),
                        op: curr,
                    };
                }
                _ => break,
            }
        }
        return left;
    }
}

#[derive(Debug)]
pub enum Node {
    BinaryOp {
        left: Box<Node>,
        right: Box<Node>,
        op: Token,
    },
    Number(Token),
}
