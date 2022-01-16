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
            Ok(n) => println!("{}", n.to_string()),
            Err(e) => println!("{}", e.to_string()),
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
        UnknownUnit(ErrorData),
        Unknown,
    }

    impl Error {
        fn get_data(self) -> Option<ErrorData> {
            use Error::*;
            match self {
                IllegalChar(e) | InvalidSyntax(e) | _IllegalNumber(e) | UnknownIdentifier(e)
                | UnknownUnit(e) => Some(e),
                _ => None,
            }
        }

        pub fn to_string(self) -> String {
            use Error::*;
            let error_name = match self {
                IllegalChar(_) => format!("Illegal Character"),
                InvalidSyntax(_) => format!("Invalid Syntax"),
                _IllegalNumber(_) => format!("Illegal Number"),
                UnknownIdentifier(_) => format!("Unknown Identifier"),
                UnknownUnit(_) => format!("Unknown Unit"),
                _ => format!(""),
            };
            let details = if let Some(ErrorData { trace: _, details }) = self.get_data() {
                details
            } else {
                "".to_string()
            };
            format!("{error_name}: {details}")
        }
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

        pub fn to_string(self) -> String {
            use ImperialLength::*;
            use LengthUnit::*;
            use MetricLength::*;
            use Unit::Length;
            let unit_str = match self {
                Length(Imperial(Inches)) => "in",
                Length(Imperial(Feet)) => "ft",
                Length(Imperial(Yards)) => "yd",
                Length(Imperial(Miles)) => "mi",
                Length(Metric(Millimeters)) => "mm",
                Length(Metric(Centimeters)) => "cm",
                Length(Metric(Meters)) => "m",
                Length(Metric(Kilometers)) => "km",
                _ => "",
            };
            unit_str.to_string()
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

mod function {

    #[derive(Debug, Clone)]
    pub enum Function {
        Sin,
        Cos,
        Tan,
        Ln,
    }

    impl Function {
        pub fn from(s: String) -> Option<Function> {
            match s.as_str() {
                "sin" => Some(Function::Sin),
                "cos" => Some(Function::Cos),
                "tan" => Some(Function::Tan),
                "ln" => Some(Function::Ln),
                _ => None,
            }
        }

        pub fn to_string(self) -> String {
            "".to_string()
        }
    }
}

mod token {
    use crate::units::Unit;
    use crate::{function::Function, Advances, Traceable, Tracer};

    #[derive(Debug, Clone)]
    pub enum Token {
        Empty,
        Unknown(Tracer, String),
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
        Function(Tracer, Function),
    }

    struct NumberValidator {
        dots: u8,
    }
    struct IdentifierValidator;

    struct UnitValidator;

    struct FunctionValidator;

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

    impl Validates for FunctionValidator {
        fn validate(&mut self, c: char) -> bool {
            c.is_alphabetic()
        }
    }

    impl Token {
        pub fn to_string(self) -> String {
            match self {
                Token::Add(_) => "+".to_string(),
                Token::Subtract(_) => "-".to_string(),
                Token::Multiply(_) => "*".to_string(),
                Token::Divide(_) => "/".to_string(),
                Token::Power(_) => "^".to_string(),
                Token::OpenParen(_) => "(".to_string(),
                Token::CloseParen(_) => ")".to_string(),
                Token::Equals(_) => "=".to_string(),
                Token::Let(_) => "let".to_string(),
                Token::Unknown(_, value) => value,
                Token::Id(_, id) => id,
                Token::Number(_, num) => num.to_string(),
                Token::Unit(_, unit) => unit.to_string(),
                Token::Function(_, func) => func.to_string(),
                _ => "".to_string(),
            }
        }

        pub fn get_trace(self) -> Option<Tracer> {
            match self {
                Token::Add(t)
                | Token::Subtract(t)
                | Token::Multiply(t)
                | Token::Divide(t)
                | Token::Power(t)
                | Token::OpenParen(t)
                | Token::CloseParen(t)
                | Token::Equals(t)
                | Token::Let(t) => Some(t),
                Token::Unknown(t, _)
                | Token::Id(t, _)
                | Token::Number(t, _)
                | Token::Unit(t, _)
                | Token::Function(t, _) => Some(t),
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
                _ => Token::Unknown(t, c.to_string()),
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
                Err(_) => Token::Unknown(tracer, num_str),
            }
        }

        pub fn make_identifier(c: char, adv: &mut (impl Advances<char> + Traceable)) -> Token {
            let mut validator = IdentifierValidator;
            let (identifier, tracer) = Token::build(c, adv, &mut validator);
            if let Some(func) = Function::from(identifier.clone()) {
                return Token::Function(tracer, func);
            } else if let Some(unit) = Unit::from(identifier.clone()) {
                return Token::Unit(tracer, unit);
            }
            match identifier.as_str() {
                "let" => Token::Let(tracer),
                _ => Token::Id(tracer, identifier),
            }
        }

        pub fn make_unit(c: char, adv: &mut (impl Advances<char> + Traceable)) -> Token {
            let mut validator = UnitValidator;
            let start = if c.is_alphabetic() {
                c
            } else {
                adv.advance(None).unwrap_or(' ')
            };
            let (unit_str, tracer) = Token::build(start, adv, &mut validator);
            if let Some(unit) = Unit::from(unit_str.trim().to_string().clone()) {
                return Token::Unit(tracer, unit);
            } else {
                return Token::Unknown(tracer, unit_str);
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
                    Token::Unknown(trace, token_str) => {
                        return Err(IllegalChar(ErrorData {
                            trace,
                            details: format!("Unexpected '{token_str}'",),
                        }));
                    }
                    Token::Number(_, _) => {
                        tokens.push(token);
                        if let Some(c) = self.curr {
                            if c.is_alphabetic() {
                                let unit_token = Token::make_unit(c, self);
                                if let Token::Unit(_, _) = unit_token {
                                    tokens.push(unit_token);
                                } else {
                                    let trace = unit_token.clone().get_trace();
                                    let details = unit_token.to_string();
                                    return if let Some(trace) = trace {
                                        Err(UnknownUnit(ErrorData { trace, details }))
                                    } else {
                                        Err(Unknown)
                                    };
                                }
                            }
                        }
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
        Function {
            func: Token,
            arg: Box<Node>,
        },
        Conversion {
            node: Box<Node>,
            unit: Option<Token>,
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
            while let Some(token) = self.curr.cloned() {
                if !comp(&token) {
                    break;
                }
                let op = Box::new(token);
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
            let node = self.unit();
            let token = self.curr.cloned();
            if let Some(Token::Unit(_, _)) = token {
                return Node::Conversion {
                    node: Box::new(node),
                    unit: token,
                };
            }
            return node;
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
                        } else if let Some(Token::Id(trace, id)) = unit {
                            return Node::Error(UnknownUnit(ErrorData {
                                trace,
                                details: format!("'{id}'"),
                            }));
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
                    Token::Function(func_trace, func) => {
                        self.advance(None);
                        let open_token = self.curr.clone();
                        if let Some(Token::OpenParen(open_trace)) = open_token {
                            self.advance(None);
                            let expr = self.expr();
                            let close_token = self.curr.clone();
                            if let Some(Token::CloseParen(_)) = close_token {
                                self.advance(None);
                                return Node::Function {
                                    func: Token::Function(func_trace, func),
                                    arg: Box::new(expr),
                                };
                            } else {
                                return Node::Error(InvalidSyntax(ErrorData {
                                    trace: open_trace.clone(),
                                    details: format!("expected ')'"),
                                }));
                            }
                        } else {
                            return Node::Error(InvalidSyntax(ErrorData {
                                trace: func_trace,
                                details: format!("expected '('"),
                            }));
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
        function::Function,
        parser::Node,
        token::Token,
        units::{ToLength, Unit},
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

        pub fn to_string(self) -> String {
            let unit = self.unit.to_string();
            let value = self.value.to_string();
            format!("{value} {unit}")
        }

        fn convert(&mut self, other: Unit) {
            use crate::units::Unit::*;
            let factor = match (&self.unit, &other) {
                (Length(a), Length(b)) => a.to(b.clone()),
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

        fn apply(&mut self, func: Function) {
            self.value = match func {
                Function::Sin => self.value.sin(),
                Function::Cos => self.value.cos(),
                Function::Tan => self.value.tan(),
                Function::Ln => self.value.ln(),
            };
            self.unit = Unit::Empty
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
                    let mut left = self.visit(*left)?;
                    let mut right = self.visit(*right)?;
                    match (&left.unit, &right.unit) {
                        (Unit::Empty, Unit::Empty) => (),
                        (Unit::Empty, _) => left.convert(right.unit.clone()),
                        (_, _) => right.convert(left.unit.clone()),
                    }
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
                    let mut num = self.visit(*node)?;
                    if let Some(Token::Unit(_, u)) = unit {
                        num.convert(u);
                    }
                    self.symbols.set(id, &num);
                    Ok(num)
                }
                Node::Function {
                    func: Token::Function(_, func),
                    arg,
                } => {
                    let mut num = self.visit(*arg)?;
                    num.apply(func);
                    Ok(num)
                }
                Node::Conversion { node, unit } => {
                    let mut num = self.visit(*node)?;
                    if let Some(Token::Unit(_, u)) = unit {
                        num.convert(u);
                    }
                    Ok(num)
                }
                Node::Error(e) => Err(e),
                _ => Err(Unknown),
            };
        }
    }
}
