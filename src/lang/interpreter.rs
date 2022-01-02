use super::{parser::Node, token::TokenKind};

#[derive(Debug)]
pub struct NumberType {
    value: f64,
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
}

pub struct Interpreter;

impl Interpreter {
    pub fn visit(&mut self, node: Node) -> NumberType {
        return match node {
            Node::BinaryOp { left, right, op } => {
                let left = self.visit(*left);
                let right = self.visit(*right);
                match op.kind {
                    TokenKind::SUBTRACT => left.subtract(right),
                    TokenKind::ADD => left.add(right),
                    _ => left,
                }
            }
            Node::Number(token) => NumberType::new(token.value.unwrap()),
        };
    }
}
