use super::{
    token::{KeywordKind, Token, TokenKind},
    tracker::Tracker,
};

pub struct Parser {
    tokens: Vec<Token>,
    curr: Option<Token>,
    index: usize,
}

enum ExprType {
    Expr,
    AddSub,
    MulDiv,
    Sign,
    Pow,
    Stmt,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Parser {
        let curr = tokens.first().cloned();
        Parser {
            tokens,
            curr,
            index: 0,
        }
    }

    fn advance(&mut self) {
        self.index += 1;
        if self.index < self.tokens.len() {
            self.curr = self.tokens.get(self.index).cloned();
        } else {
            self.curr = None;
        }
    }

    pub fn parse(&mut self) -> Result<Node, ParseError> {
        let result = self.expr();
        match result {
            Node::Error(e) => Err(e),
            _ => Ok(result),
        }
    }

    fn get_node(&mut self, expr_type: &ExprType) -> Node {
        match *expr_type {
            ExprType::Expr => self.expr(),
            ExprType::AddSub => self.add_sub_expr(),
            ExprType::MulDiv => self.mul_div_expr(),
            ExprType::Pow => self.pow_expr(),
            ExprType::Sign => self.sign_expr(),
            ExprType::Stmt => self.stmt(),
        }
    }

    fn binary_op(
        &mut self,
        left_expr_type: ExprType,
        right_expr_type: ExprType,
        comp: &dyn Fn(Token) -> bool,
    ) -> Node {
        let mut left = self.get_node(&left_expr_type);
        while let Some(token) = self.curr.to_owned() {
            if !comp(token.clone()) {
                break;
            }
            let op = token;
            self.advance();
            let right = self.get_node(&right_expr_type);
            left = Node::BinaryOp {
                left: Box::new(left),
                right: Box::new(right),
                op,
            }
        }
        return left;
    }

    fn expr(&mut self) -> Node {
        if let Some(token) = self.curr.to_owned() {
            if token.kind == TokenKind::KEYWORD(KeywordKind::LET) {
                self.advance();
                if let Some(token) = self.curr.to_owned() {
                    if let TokenKind::IDENTIFIER(_) = &token.kind {
                        let id_token = token.clone();
                        self.advance();
                        if let Some(token) = self.curr.to_owned() {
                            if token.kind != TokenKind::EQUALS {
                                return Node::Error(ParseError::from(
                                    Some(token),
                                    ParseErrorKind::InvalidSyntax,
                                    "Expexted '='".to_string(),
                                ));
                            } else {
                                self.advance();
                                let node = Box::new(self.expr());
                                return Node::Assignment { id_token, node };
                            }
                        }
                    }
                }
            }
        }
        return self.add_sub_expr();
    }

    fn add_sub_expr(&mut self) -> Node {
        self.binary_op(ExprType::MulDiv, ExprType::MulDiv, &|tok: Token| {
            [TokenKind::ADD, TokenKind::SUBTRACT].contains(&tok.kind)
        })
    }

    fn mul_div_expr(&mut self) -> Node {
        self.binary_op(ExprType::Sign, ExprType::Sign, &|tok: Token| {
            [TokenKind::MULTIPLY, TokenKind::DIVIDE].contains(&tok.kind)
        })
    }

    fn sign_expr(&mut self) -> Node {
        if let Some(token) = self.curr.to_owned() {
            if [TokenKind::ADD, TokenKind::SUBTRACT].contains(&token.kind) {
                let op = token.clone();
                self.advance();
                let node = self.sign_expr();
                // println!("node after: {:?}", node);
                return Node::UnaryOp {
                    node: Box::new(node),
                    op,
                };
            }
        }
        return self.pow_expr();
    }

    fn pow_expr(&mut self) -> Node {
        self.binary_op(ExprType::Stmt, ExprType::Sign, &|tok: Token| {
            tok.kind == TokenKind::POWER
        })
    }

    fn stmt(&mut self) -> Node {
        let token = self.curr.clone();
        // println!("t: {:?}", token);
        if let Some(token) = token.clone() {
            if token.kind == TokenKind::NUM {
                self.advance();
                return Node::Number(token);
            }

            if token.kind == TokenKind::LPAREN {
                self.advance();
                let expr = self.expr();
                // println!("expr: {:?}", expr);
                if let Some(tok) = self.curr.clone() {
                    if tok.kind == TokenKind::RPAREN {
                        self.advance();
                        return expr;
                    } else {
                        return Node::Error(ParseError::from(
                            self.curr.clone(),
                            ParseErrorKind::InvalidSyntax,
                            "Expected ')'".to_string(),
                        ));
                    }
                }
            }
            if let TokenKind::IDENTIFIER(_) = &token.kind {
                self.advance();
                return Node::Access(token);
            }
        }
        Node::Error(ParseError::from(
            token,
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
    UnknownIdentifier,
}

#[derive(Debug)]
pub struct ParseError {
    pub start: Option<Tracker>,
    pub end: Option<Tracker>,
    pub kind: ParseErrorKind,
    pub details: String,
}

impl ParseError {
    pub fn from(token: Option<Token>, kind: ParseErrorKind, details: String) -> ParseError {
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
    Access(Token),
    Assignment {
        id_token: Token,
        node: Box<Node>,
    },
    Error(ParseError),
}
