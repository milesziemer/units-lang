use std::collections::HashMap;

use super::{
    parser::{Node, ParseError, ParseErrorKind},
    token::TokenKind,
};

#[derive(Debug)]
pub struct SymbolTable {
    pub table: HashMap<String, NumberType>,
}

impl SymbolTable {
    pub fn new() -> SymbolTable {
        SymbolTable {
            table: HashMap::new(),
        }
    }

    pub fn get(&self, identifier: String) -> Option<&NumberType> {
        match self.table.get(&identifier) {
            Some(nt) => Some(nt),
            None => None,
        }
    }

    pub fn set(&mut self, identifier: String, value: &NumberType) {
        self.table.insert(identifier, value.clone());
    }

    pub fn _delete(&mut self, identifier: String) {
        self.table.remove(&identifier);
    }
}

#[derive(Debug, Clone)]
pub struct NumberType {
    pub value: f64,
}

impl NumberType {
    fn new(value: f64) -> NumberType {
        NumberType { value }
    }

    fn add(&self, num: NumberType) -> NumberType {
        NumberType {
            value: self.value + num.value,
        }
    }

    fn subtract(&self, num: NumberType) -> NumberType {
        NumberType {
            value: self.value - num.value,
        }
    }

    fn multiply(&self, num: NumberType) -> NumberType {
        NumberType {
            value: self.value * num.value,
        }
    }

    fn divide(&self, num: NumberType) -> NumberType {
        NumberType {
            value: self.value / num.value,
        }
    }

    fn power(&self, num: NumberType) -> NumberType {
        NumberType {
            value: self.value.powf(num.value),
        }
    }

    fn negate(&self) -> NumberType {
        NumberType { value: -self.value }
    }
}

pub struct Interpreter<'a> {
    pub symbol_table: &'a mut SymbolTable,
}

impl Interpreter<'_> {
    pub fn visit(&mut self, n: Node) -> Result<NumberType, ParseError> {
        return match n {
            Node::BinaryOp { left, right, op } => {
                let left = self.visit(*left)?;
                let right = self.visit(*right)?;
                Ok(match op.kind {
                    TokenKind::SUBTRACT => left.subtract(right),
                    TokenKind::ADD => left.add(right),
                    TokenKind::MULTIPLY => left.multiply(right),
                    TokenKind::DIVIDE => left.divide(right),
                    TokenKind::POWER => left.power(right),
                    _ => left,
                })
            }
            Node::UnaryOp { node, op } => {
                let node = self.visit(*node)?;
                Ok(match op.kind {
                    TokenKind::SUBTRACT => node.negate(),
                    _ => node,
                })
            }
            Node::Number(token) => Ok(NumberType::new(token.value.unwrap())),
            Node::Access(token) => {
                if let TokenKind::IDENTIFIER(id) = token.kind.to_owned() {
                    let num = self.symbol_table.get(id.clone()).cloned();
                    match num {
                        Some(n) => Ok(n),
                        None => Err(ParseError::from(
                            Some(token),
                            ParseErrorKind::UnknownIdentifier,
                            format!("'{}' is not defined", id),
                        )),
                    }
                    // Ok(self.symbol_table.get(id).cloned().unwrap())
                } else {
                    Err(ParseError::from(
                        Some(token),
                        ParseErrorKind::InvalidSyntax,
                        "".to_string(),
                    ))
                }
            }
            Node::Assignment { node, id_token } => {
                if let TokenKind::IDENTIFIER(id) = id_token.kind {
                    let node = self.visit(*node)?;
                    self.symbol_table.set(id, &node);
                    Ok(node)
                } else {
                    Err(ParseError::from(
                        Some(id_token),
                        ParseErrorKind::InvalidSyntax,
                        "".to_string(),
                    ))
                }
            }
            Node::Error(e) => Err(e),
        };
    }
}
