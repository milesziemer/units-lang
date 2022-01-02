use super::{parser::Node, token::TokenKind};

#[derive(Debug)]
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

    fn negate(&self) -> NumberType {
        NumberType { value: -self.value }
    }
}

pub struct Interpreter;

impl Interpreter {
    pub fn visit(&mut self, n: Node) -> NumberType {
        return match n {
            Node::BinaryOp { left, right, op } => {
                let left = self.visit(*left);
                let right = self.visit(*right);
                match op.kind {
                    TokenKind::SUBTRACT => left.subtract(right),
                    TokenKind::ADD => left.add(right),
                    TokenKind::MULTIPLY => left.multiply(right),
                    TokenKind::DIVIDE => left.divide(right),
                    _ => left,
                }
            }
            Node::UnaryOp { node, op } => {
                let node = self.visit(*node);
                match op.kind {
                    TokenKind::SUBTRACT => node.negate(),
                    _ => node,
                }
            }
            Node::Number(token) => NumberType::new(token.value.unwrap()),
            Node::Error(e) => {
                println!("{}", e);
                NumberType::new(0.0)
            }
        };
    }
}
