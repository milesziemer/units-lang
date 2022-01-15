mod lang;

use interpreter::{Interpreter, NumberType, SymbolTable};

use std::io::{stdin, stdout, Write};

fn main() {
    let mut symbol_table = SymbolTable::new();
    loop {
        let mut line = String::new();
        print!("units > ");
        let _ = stdout().flush();
        stdin()
            .read_line(&mut line)
            .expect("Did not enter a valid string");
        if let Some('\n') = line.chars().next_back() {
            line.pop();
        }
        if let Some('\r') = line.chars().next_back() {
            line.pop();
        }
        if line.trim().len() == 0 {
            continue;
        }
        let result = run(line, &mut symbol_table);
        match result {
            Ok(n) => println!("{:?}", n),
            Err(e) => println!("{:?}", e),
        }
    }
}

fn run(line: String, symbol_table: &mut SymbolTable) -> Result<NumberType, error::Error> {
    let mut lexer = lexer::Lexer::new(line.as_bytes());
    let tokens = lexer.get_tokens()?;
    let mut parser = parser::Parser::new(&tokens);
    let ast = parser.parse()?;
    let mut interpreter = Interpreter {
        symbols: symbol_table,
    };
    let result = interpreter.visit(ast)?;
    Ok(result)
}

mod error {
    use crate::Tracer;

    #[derive(Debug)]
    pub struct ErrorData {
        pub trace: Tracer,
        pub details: String,
    }

    #[derive(Debug)]
    pub enum Error {
        InvalidSyntax(ErrorData),
        IllegalChar(ErrorData),
        _IllegalNumber(ErrorData),
        UnknownIdentifier(ErrorData),
        Unknown,
    }
}

pub trait Advances<T> {
    fn advance(&mut self, curr: Option<T>) -> Option<T>;
}

#[derive(Clone, Copy, Debug)]
pub struct Location {
    pub index: usize,
    line: i32,
    column: i32,
}

impl Location {
    pub fn new(index: usize, line: i32, column: i32) -> Location {
        Location {
            index,
            line,
            column,
        }
    }
}

impl Advances<char> for Location {
    fn advance(&mut self, curr: Option<char>) -> Option<char> {
        self.index += 1;
        self.column += 1;
        if let Some('\n') = curr {
            self.line += 1;
            self.column += 1;
        }
        curr
    }
}

#[derive(Debug, Clone)]
pub struct Tracer {
    pub start: Location,
    pub end: Location,
}

pub trait Traceable {
    fn get_current_location(&mut self) -> Location;
}

mod units {

    #[derive(Debug, Clone, PartialEq)]
    pub enum ImperialLength {
        Inches,
        Feet,
        Yards,
        Miles,
    }

    impl ImperialLength {
        pub fn to(&self, other: ImperialLength) -> f64 {
            use ImperialLength::*;
            match (self.clone(), other) {
                (a, b) if a == b => 1.0,
                (Inches, Feet) => 1.0 / 12.0,
                (Feet, Yards) => 1.0 / 3.0,
                (Yards, Miles) => 1.0 / 1760.0,
                (Inches, Yards) => Inches.to(Feet) * Feet.to(Yards),
                (Inches, Miles) => Inches.to(Yards) * Yards.to(Miles),
                (Feet, Miles) => Feet.to(Yards) * Yards.to(Miles),
                (a, b) => 1.0 / b.to(a),
            }
        }
        pub fn to_metric(&self) -> f64 {
            self.to(ImperialLength::Feet) * 1.0 / 3.28084
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum MetricLength {
        Millimeters,
        Centimeters,
        Meters,
        Kilometers,
    }

    impl MetricLength {
        pub fn to(&self, other: MetricLength) -> f64 {
            use MetricLength::*;
            match (self.clone(), other) {
                (a, b) if a == b => 1.0,
                (Millimeters, Centimeters) => 0.1,
                (Centimeters, Meters) => 0.01,
                (Meters, Kilometers) => 0.001,
                (Millimeters, Meters) => Millimeters.to(Centimeters) * Centimeters.to(Meters),
                (Millimeters, Kilometers) => Millimeters.to(Meters) * Meters.to(Kilometers),
                (Centimeters, Kilometers) => Centimeters.to(Meters) * Meters.to(Kilometers),
                (a, b) => 1.0 / b.to(a),
            }
        }
        pub fn to_feet(&self) -> f64 {
            self.to(MetricLength::Meters) * 3.28084
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum LengthUnit {
        Imperial(ImperialLength),
        Metric(MetricLength),
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum Unit {
        Empty,
        Length(LengthUnit),
    }

    impl Unit {
        pub fn from(s: String) -> Option<Unit> {
            use ImperialLength::*;
            use LengthUnit::*;
            use MetricLength::*;
            use Unit::Length;
            match s.as_str() {
                "in" => Some(Length(Imperial(Inches))),
                "ft" => Some(Length(Imperial(Feet))),
                "yd" => Some(Length(Imperial(Yards))),
                "mi" => Some(Length(Imperial(Miles))),
                "mm" => Some(Length(Metric(Millimeters))),
                "cm" => Some(Length(Metric(Centimeters))),
                "m" => Some(Length(Metric(Meters))),
                "km" => Some(Length(Metric(Kilometers))),
                _ => None,
            }
        }
    }

    pub trait ToLength {
        fn to(&self, other: LengthUnit) -> f64;
    }

    impl ToLength for LengthUnit {
        fn to(&self, other: LengthUnit) -> f64 {
            use crate::units::{ImperialLength::*, LengthUnit::*, MetricLength::*};
            match (self.clone(), other) {
                (Metric(a), Metric(b)) => a.to(b),
                (Imperial(a), Imperial(b)) => a.to(b),
                (Metric(a), Imperial(b)) => a.to_feet() * Feet.to(b),
                (Imperial(a), Metric(b)) => a.to_metric() * Meters.to(b),
            }
        }
    }
}

mod token {
    use crate::units::Unit;
    use crate::{Advances, Traceable, Tracer};

    #[derive(Debug, Clone)]
    pub enum Token {
        Empty,
        Unknown(Tracer),
        Add(Tracer),
        Subtract(Tracer),
        Multiply(Tracer),
        Divide(Tracer),
        Power(Tracer),
        OpenParen(Tracer),
        CloseParen(Tracer),
        Equals(Tracer),
        Let(Tracer),
        Id(Tracer, String),
        Number(Tracer, f64),
        Unit(Tracer, Unit),
    }

    struct NumberValidator {
        dots: u8,
    }
    struct IdentifierValidator;

    struct UnitValidator;

    trait Validates {
        fn validate(&mut self, c: char) -> bool;
    }

    impl Validates for NumberValidator {
        fn validate(&mut self, c: char) -> bool {
            match c {
                c if c.is_numeric() => true,
                '.' if self.dots < 1 => {
                    self.dots += 1;
                    true
                }
                _ => false,
            }
        }
    }

    impl Validates for IdentifierValidator {
        fn validate(&mut self, c: char) -> bool {
            c.is_alphanumeric() || c == '_'
        }
    }

    impl Validates for UnitValidator {
        fn validate(&mut self, c: char) -> bool {
            c.is_alphanumeric()
        }
    }

    impl Token {
        pub fn get_trace(self) -> Option<Tracer> {
            match self {
                Token::Unknown(t)
                | Token::Add(t)
                | Token::Subtract(t)
                | Token::Multiply(t)
                | Token::Divide(t)
                | Token::Power(t)
                | Token::OpenParen(t)
                | Token::CloseParen(t)
                | Token::Equals(t)
                | Token::Let(t) => Some(t),
                Token::Id(t, _) | Token::Number(t, _) => Some(t),
                _ => None,
            }
        }

        pub fn fromchar(c: char, trc: &mut impl Traceable) -> Token {
            let t = Tracer {
                start: trc.get_current_location(),
                end: trc.get_current_location(),
            };
            match c {
                ' ' | '\t' => Token::Empty,
                '+' => Token::Add(t),
                '-' => Token::Subtract(t),
                '*' => Token::Multiply(t),
                '/' => Token::Divide(t),
                '^' => Token::Power(t),
                '(' => Token::OpenParen(t),
                ')' => Token::CloseParen(t),
                '=' => Token::Equals(t),
                _ => Token::Unknown(t),
            }
        }

        fn build(
            c: char,
            adv: &mut (impl Advances<char> + Traceable),
            vdr: &mut impl Validates,
        ) -> (String, Tracer) {
            let mut acc = c.to_string();
            let start = adv.get_current_location();
            while let Some(c) = adv.advance(None) {
                if !vdr.validate(c) {
                    break;
                }
                acc.push_str(&c.to_string());
            }
            let end = adv.get_current_location();
            let tracer = Tracer { start, end };
            (acc, tracer)
        }

        pub fn make_number(c: char, adv: &mut (impl Advances<char> + Traceable)) -> Token {
            let mut validator = NumberValidator { dots: 0 };
            let (num_str, tracer) = Token::build(c, adv, &mut validator);
            match num_str.parse::<f64>() {
                Ok(n) => Token::Number(tracer, n),
                Err(_) => Token::Unknown(tracer),
            }
        }

        pub fn make_identifier(c: char, adv: &mut (impl Advances<char> + Traceable)) -> Token {
            let mut validator = IdentifierValidator;
            let (identifier, tracer) = Token::build(c, adv, &mut validator);
            if let Some(unit) = Unit::from(identifier.clone()) {
                return Token::Unit(tracer, unit);
            }
            match identifier.as_str() {
                "let" => Token::Let(tracer),
                _ => Token::Id(tracer, identifier),
            }
        }

        pub fn make_unit(_: char, adv: &mut (impl Advances<char> + Traceable)) -> Token {
            let mut validator = UnitValidator;
            let start = adv.advance(None).unwrap_or(' ');
            let (unit, tracer) = Token::build(start, adv, &mut validator);
            if let Some(unit) = Unit::from(unit.trim().to_string().clone()) {
                return Token::Unit(tracer, unit);
            } else {
                return Token::Unknown(tracer);
            }
        }
    }
}

mod lexer {
    use crate::{
        error::{Error, Error::*, ErrorData},
        token::Token,
        Advances, Location, Traceable,
    };
    pub struct Lexer<'a> {
        text: &'a [u8],
        curr: Option<char>,
        location: Location,
    }

    impl Lexer<'_> {
        pub fn new(text: &[u8]) -> Lexer {
            Lexer {
                text,
                curr: Some(text[0] as char),
                location: Location::new(0, 0, 0),
            }
        }

        pub fn get_tokens(&mut self) -> Result<Vec<Token>, Error> {
            let mut tokens = Vec::new();
            while let Some(c) = self.curr {
                let token = if c.is_numeric() {
                    Token::make_number(c, self)
                } else if c.is_alphabetic() || c == '_' {
                    Token::make_identifier(c, self)
                } else if c == ':' {
                    Token::make_unit(c, self)
                } else {
                    self.advance(None);
                    Token::fromchar(c, self)
                };
                match token {
                    Token::Unknown(trace) => {
                        let token_slice = &self.text[trace.start.index..trace.end.index];
                        let token_string = match std::str::from_utf8(token_slice) {
                            Ok(s) => Some(s),
                            Err(_) => None,
                        };
                        return Err(IllegalChar(ErrorData {
                            trace,
                            details: format!(
                                "Unexpected '{}'",
                                token_string.unwrap_or(&c.to_string())
                            ),
                        }));
                    }
                    Token::Empty => (),
                    _ => tokens.push(token),
                }
            }
            Ok(tokens)
        }
    }

    impl Advances<char> for Lexer<'_> {
        fn advance(&mut self, _: Option<char>) -> Option<char> {
            self.location.advance(None);
            self.curr = match self.location.index < self.text.len() {
                true => Some(self.text[self.location.index] as char),
                _ => None,
            };
            self.curr
        }
    }

    impl Traceable for Lexer<'_> {
        fn get_current_location(&mut self) -> Location {
            self.location
        }
    }
}

mod parser {
    use crate::{
        error::{Error, Error::*, ErrorData},
        token::Token,
        Advances,
    };

    pub struct Parser<'a> {
        tokens: &'a Vec<Token>,
        curr: Option<&'a Token>,
        last: Option<&'a Token>,
        index: usize,
    }

    enum StatementType {
        _Expression,
        _Arithmetic,
        Rational,
        Exponential,
        Apply,
        _Unit,
    }

    #[derive(Debug)]
    pub enum Node {
        BinaryOp {
            left: Box<Node>,
            right: Box<Node>,
            op: Box<Token>,
        },
        UnaryOp {
            node: Box<Node>,
            op: Token,
        },
        Number {
            value: Token,
            unit: Option<Token>,
        },
        Access {
            value: Token,
            unit: Option<Token>,
        },
        Assignment {
            id: Token,
            unit: Option<Token>,
            node: Box<Node>,
        },
        Error(Error),
    }

    impl Parser<'_> {
        pub fn new<'a>(tokens: &'a Vec<Token>) -> Parser<'a> {
            Parser {
                tokens,
                curr: tokens.first(),
                last: tokens.last(),
                index: 0,
            }
        }

        pub fn parse(&mut self) -> Result<Node, Error> {
            Ok(self.expr())
        }

        fn get_node(&mut self, stmt_type: &StatementType) -> Node {
            match *stmt_type {
                StatementType::_Expression => self.expr(),
                StatementType::_Arithmetic => self.arith(),
                StatementType::Rational => self.rational(),
                StatementType::Exponential => self.exp(),
                StatementType::Apply => self.apply(),
                StatementType::_Unit => self.unit(),
            }
        }

        fn binary_op(&mut self, stmt_type: StatementType, comp: &dyn Fn(&Token) -> bool) -> Node {
            let mut left = self.get_node(&stmt_type);
            while let Some(token) = self.curr {
                if !comp(token) {
                    break;
                }
                let op = Box::new(token.clone());
                self.advance(None);
                let right = self.get_node(&stmt_type);
                left = Node::BinaryOp {
                    left: Box::new(left),
                    right: Box::new(right),
                    op,
                }
            }
            return left;
        }

        fn expr(&mut self) -> Node {
            if let Some(Token::Let(let_trace)) = self.curr.cloned() {
                // We have a 'let' token, check for identifier
                if let Some(Token::Id(id_trace, id)) = self.advance(None).cloned() {
                    // We have an identifier, check for type and then equals sign
                    let next = self.advance(None).cloned();
                    if let Some(Token::Unit(unit_trace, unit)) = next {
                        if let Some(Token::Equals(_)) = self.advance(None) {
                            self.advance(None);
                            let node = Box::new(self.expr());
                            return Node::Assignment {
                                id: Token::Id(id_trace, id),
                                unit: Some(Token::Unit(unit_trace, unit)),
                                node,
                            };
                        } else {
                            return Node::Error(InvalidSyntax(ErrorData {
                                trace: unit_trace,
                                details: format!("expected '='"),
                            }));
                        }
                    } else if let Some(Token::Equals(_)) = next {
                        // We have an equals sign, evaluate expression
                        self.advance(None);
                        let node = Box::new(self.expr());
                        return Node::Assignment {
                            id: Token::Id(id_trace, id),
                            unit: None,
                            node,
                        };
                    } else {
                        // 'let', 'ID' tokens with no equals sign, error
                        return Node::Error(InvalidSyntax(ErrorData {
                            trace: id_trace,
                            details: format!("expected '='"),
                        }));
                    }
                } else {
                    // 'let' token with no identifier, error
                    return Node::Error(InvalidSyntax(ErrorData {
                        trace: let_trace,
                        details: format!("expected identifier"),
                    }));
                }
            }
            // No assignment, evaluate
            return self.arith();
        }

        fn arith(&mut self) -> Node {
            self.binary_op(StatementType::Rational, &|tok: &Token| match *tok {
                Token::Add(_) | Token::Subtract(_) => true,
                _ => false,
            })
        }

        fn rational(&mut self) -> Node {
            self.binary_op(StatementType::Exponential, &|tok: &Token| match *tok {
                Token::Multiply(_) | Token::Divide(_) => true,
                _ => false,
            })
        }

        fn exp(&mut self) -> Node {
            self.binary_op(StatementType::Apply, &|tok: &Token| match *tok {
                Token::Power(_) => true,
                _ => false,
            })
        }

        fn apply(&mut self) -> Node {
            if let Some(token) = match self.curr {
                Some(Token::Add(_)) | Some(Token::Subtract(_)) => self.curr,
                _ => None,
            } {
                let op = token.clone();
                self.advance(None);
                let node = self.apply();
                return Node::UnaryOp {
                    node: Box::new(node),
                    op,
                };
            }
            return self.unit();
        }

        fn unit(&mut self) -> Node {
            if let Some(token) = self.curr {
                let token = token.clone();
                match token {
                    Token::Number(_, _) => {
                        self.advance(None);
                        // Check for a unit type identifier
                        let unit = self.curr.cloned();
                        if let Some(Token::Unit(_, _)) = unit {
                            self.advance(None);
                            return Node::Number { value: token, unit };
                        } else {
                            return Node::Number {
                                value: token,
                                unit: None,
                            };
                        }
                    }
                    Token::OpenParen(trace) => {
                        self.advance(None);
                        let expr = self.expr();
                        let token = self.curr.clone();
                        if let Some(Token::CloseParen(_)) = token {
                            self.advance(None);
                            return expr;
                        } else {
                            return Node::Error(InvalidSyntax(ErrorData {
                                trace,
                                details: format!("no matching ')' found"),
                            }));
                        }
                    }
                    Token::Id(_, _) => {
                        self.advance(None);
                        let unit = self.curr.cloned();
                        if let Some(Token::Unit(_, _)) = unit {
                            self.advance(None);
                            return Node::Access { value: token, unit };
                        } else {
                            return Node::Access {
                                value: token,
                                unit: None,
                            };
                        }
                    }
                    _ => (),
                }
            }
            if let Some(token) = self.last {
                if let Some(trace) = token.clone().get_trace() {
                    return Node::Error(InvalidSyntax(ErrorData {
                        trace,
                        details: format!("expected number, identifier, unit or '('"),
                    }));
                }
            }
            return Node::Error(Unknown);
        }
    }

    impl<'a> Advances<&'a Token> for Parser<'a> {
        fn advance(&mut self, _: Option<&'a Token>) -> Option<&'a Token> {
            self.index += 1;
            self.curr = match self.index < self.tokens.len() {
                true => Some(&self.tokens[self.index]),
                false => None,
            };
            self.curr
        }
    }
}

mod interpreter {

    use std::collections::HashMap;

    use crate::{
        error::{self, Error::*, ErrorData},
        parser::Node,
        token::Token,
        units::ToLength,
        units::Unit,
    };

    #[derive(Debug, Clone)]
    pub struct NumberType {
        pub value: f64,
        pub unit: Unit,
    }

    impl NumberType {
        fn new(value: f64, unit: Unit) -> NumberType {
            NumberType { value, unit }
        }

        fn convert(&mut self, other: Unit) {
            use crate::units::Unit::*;
            let factor = match (self.unit.clone(), other.clone()) {
                (Length(a), Length(b)) => a.to(b),
                _ => 1.0,
            };
            self.value = self.value * factor;
            self.unit = other;
        }

        fn add(&self, num: NumberType) -> NumberType {
            NumberType {
                value: self.value + num.value,
                unit: self.unit.clone(),
            }
        }

        fn subtract(&self, num: NumberType) -> NumberType {
            NumberType {
                value: self.value - num.value,
                unit: self.unit.clone(),
            }
        }

        fn multiply(&self, num: NumberType) -> NumberType {
            NumberType {
                value: self.value * num.value,
                unit: self.unit.clone(),
            }
        }

        fn divide(&self, num: NumberType) -> NumberType {
            NumberType {
                value: self.value / num.value,
                unit: self.unit.clone(),
            }
        }

        fn power(&self, num: NumberType) -> NumberType {
            NumberType {
                value: self.value.powf(num.value),
                unit: self.unit.clone(),
            }
        }

        fn negate(&self) -> NumberType {
            NumberType {
                value: -self.value,
                unit: self.unit.clone(),
            }
        }
    }

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

        pub fn get(&self, id: String) -> Option<&NumberType> {
            match self.table.get(&id) {
                Some(n) => Some(n),
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

    pub struct Interpreter<'a> {
        pub symbols: &'a mut SymbolTable,
    }

    impl Interpreter<'_> {
        pub fn visit(&mut self, n: Node) -> Result<NumberType, error::Error> {
            return match n {
                Node::BinaryOp { left, right, op } => {
                    let left = self.visit(*left)?;
                    let mut right = self.visit(*right)?;
                    right.convert(left.unit.clone());
                    Ok(match *op {
                        Token::Add(_) => left.add(right),
                        Token::Subtract(_) => left.subtract(right),
                        Token::Multiply(_) => left.multiply(right),
                        Token::Divide(_) => left.divide(right),
                        Token::Power(_) => left.power(right),
                        _ => left,
                    })
                }
                Node::UnaryOp { node, op } => {
                    let node = self.visit(*node)?;
                    Ok(match op {
                        Token::Subtract(_) => node.negate(),
                        _ => node,
                    })
                }
                Node::Number {
                    value: Token::Number(_, value),
                    unit,
                } => match unit {
                    Some(Token::Unit(_, unit)) => Ok(NumberType::new(value, unit)),
                    _ => Ok(NumberType::new(value, Unit::Empty)),
                },
                Node::Access {
                    value: Token::Id(trace, id),
                    unit,
                } => match self.symbols.get(id.to_string()).cloned() {
                    Some(mut n) => {
                        if let Some(Token::Unit(_, u)) = unit {
                            n.convert(u);
                        }
                        Ok(n.clone())
                    }
                    _ => Err(UnknownIdentifier(ErrorData {
                        trace,
                        details: format!("'{id}' is not defined"),
                    })),
                },
                Node::Assignment {
                    id: Token::Id(_, id),
                    unit,
                    node,
                } => {
                    let mut node = self.visit(*node)?;
                    if let Some(Token::Unit(_, u)) = unit {
                        node.convert(u);
                    }
                    self.symbols.set(id, &node);
                    Ok(node)
                }
                Node::Error(e) => Err(e),
                _ => Err(error::Error::Unknown),
            };
        }
    }
}
