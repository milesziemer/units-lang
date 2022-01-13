mod lang;

use interpreter::{Interpreter, NumberType, SymbolTable};
// use lang::{
//     interpreter::{Interpreter, NumberType, SymbolTable},
//     lexer::Lexer,
//     parser::Parser,
// };

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

mod token {
    use crate::{Advances, Traceable, Tracer};

    #[derive(Debug, Clone)]
    pub struct TokenData(pub Tracer);

    #[derive(Debug, Clone)]
    pub struct ValueToken<T>(pub TokenData, pub T);

    #[derive(Debug, Clone)]
    pub enum Token {
        Empty,
        Unknown(TokenData),
        Add(TokenData),
        Subtract(TokenData),
        Multiply(TokenData),
        Divide(TokenData),
        Power(TokenData),
        OpenParen(TokenData),
        CloseParen(TokenData),
        Equals(TokenData),
        Let(TokenData),
        Identifier(ValueToken<String>),
        Number(ValueToken<f64>),
    }

    struct NumberValidator {
        dots: u8,
    }
    struct IdentifierValidator;
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

    impl Token {
        pub fn get_trace(self) -> Option<Tracer> {
            match self {
                Token::Unknown(TokenData(t))
                | Token::Add(TokenData(t))
                | Token::Subtract(TokenData(t))
                | Token::Multiply(TokenData(t))
                | Token::Divide(TokenData(t))
                | Token::Power(TokenData(t))
                | Token::OpenParen(TokenData(t))
                | Token::CloseParen(TokenData(t))
                | Token::Equals(TokenData(t))
                | Token::Let(TokenData(t)) => Some(t),
                Token::Identifier(ValueToken(TokenData(t), _))
                | Token::Number(ValueToken(TokenData(t), _)) => Some(t),
                _ => None,
            }
        }

        pub fn fromchar(c: char, trc: &mut impl Traceable) -> Token {
            let data = TokenData(Tracer {
                start: trc.get_current_location(),
                end: trc.get_current_location(),
            });
            match c {
                ' ' | '\t' => Token::Empty,
                '+' => Token::Add(data),
                '-' => Token::Subtract(data),
                '*' => Token::Multiply(data),
                '/' => Token::Divide(data),
                '^' => Token::Power(data),
                '(' => Token::OpenParen(data),
                ')' => Token::CloseParen(data),
                '=' => Token::Equals(data),
                _ => Token::Unknown(data),
            }
        }

        fn build(
            c: char,
            adv: &mut (impl Advances<char> + Traceable),
            vdr: &mut impl Validates,
        ) -> (String, TokenData) {
            let mut acc = c.to_string();
            let start = adv.get_current_location();
            while let Some(c) = adv.advance(None) {
                if !vdr.validate(c) {
                    break;
                }
                acc.push_str(&c.to_string());
            }
            let end = adv.get_current_location();
            let token_data = TokenData(Tracer { start, end });
            (acc, token_data)
        }

        pub fn make_number(c: char, adv: &mut (impl Advances<char> + Traceable)) -> Token {
            let mut validator = NumberValidator { dots: 0 };
            let (num_str, token_data) = Token::build(c, adv, &mut validator);
            match num_str.parse::<f64>() {
                Ok(n) => Token::Number(ValueToken(token_data, n)),
                Err(_) => Token::Unknown(token_data),
            }
        }

        pub fn make_identifier(c: char, adv: &mut (impl Advances<char> + Traceable)) -> Token {
            let mut validator = IdentifierValidator;
            let (identifier, token_data) = Token::build(c, adv, &mut validator);
            match identifier.as_str() {
                "let" => Token::Let(token_data),
                _ => Token::Identifier(ValueToken(token_data, identifier)),
            }
        }
    }
}

mod lexer {
    use crate::{
        error::{Error, ErrorData},
        token::{Token, TokenData},
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
                } else {
                    self.advance(None);
                    Token::fromchar(c, self)
                };
                match token {
                    Token::Unknown(TokenData(trace)) => {
                        return Err(Error::IllegalChar(ErrorData {
                            trace,
                            details: format!("Unexpected '{}'", &c),
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
        error::{self, ErrorData},
        token::{Token, TokenData, ValueToken},
        Advances,
    };

    pub struct Parser<'a> {
        tokens: &'a Vec<Token>,
        curr: Option<&'a Token>,
        last: Option<&'a Token>,
        index: usize,
    }

    enum StatementType {
        Expression,
        Arithmetic,
        Rational,
        Exponential,
        Apply,
        Unit,
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
        Number(Token),
        Access(Token),
        Assignment {
            id: Token,
            node: Box<Node>,
        },
        Error(error::Error),
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

        pub fn parse(&mut self) -> Result<Node, error::Error> {
            Ok(self.expr())
        }

        fn get_node(&mut self, stmt_type: &StatementType) -> Node {
            match *stmt_type {
                StatementType::Expression => self.expr(),
                StatementType::Arithmetic => self.arith(),
                StatementType::Rational => self.rational(),
                StatementType::Exponential => self.exp(),
                StatementType::Apply => self.apply(),
                StatementType::Unit => self.unit(),
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
            if let Some(Token::Let(TokenData(trace))) = self.curr {
                // We have a 'let' token, check for identifier
                if let Some(Token::Identifier(value)) = self.advance(None) {
                    // We have an identifier, check for equals sign
                    if let Some(Token::Equals(TokenData(_))) = self.advance(None) {
                        // We have an equals sign, evaluate expression
                        self.advance(None);
                        let node = Box::new(self.expr());
                        return Node::Assignment {
                            id: Token::Identifier(value.clone()),
                            node,
                        };
                    } else {
                        let ValueToken(TokenData(value), _) = value.clone();
                        // 'let', 'ID' tokens with no equals sign, error
                        return Node::Error(error::Error::InvalidSyntax(ErrorData {
                            trace: value,
                            details: format!("expected '='"),
                        }));
                    }
                } else {
                    // 'let' token with no identifier, error
                    return Node::Error(error::Error::InvalidSyntax(ErrorData {
                        trace: trace.clone(),
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
                match token.clone() {
                    Token::Number(_) => {
                        self.advance(None);
                        return Node::Number(token.clone());
                    }
                    Token::OpenParen(TokenData(trace)) => {
                        self.advance(None);
                        let expr = self.expr();
                        let token = self.curr.clone();
                        if let Some(Token::CloseParen(_)) = token {
                            self.advance(None);
                            return expr;
                        } else {
                            return Node::Error(error::Error::InvalidSyntax(ErrorData {
                                trace,
                                details: format!("no matching ')' found"),
                            }));
                        }
                    }
                    Token::Identifier(_) => {
                        self.advance(None);
                        return Node::Access(token.clone());
                    }
                    _ => (),
                }
            }
            if let Some(token) = self.last {
                if let Some(trace) = token.clone().get_trace() {
                    return Node::Error(error::Error::InvalidSyntax(ErrorData {
                        trace,
                        details: format!("expected number, identifier or '('"),
                    }));
                }
            }
            return Node::Error(error::Error::Unknown);
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

    use std::collections::HashMap;

    use crate::{
        error::{self, ErrorData},
        parser::Node,
        token::{Token, TokenData, ValueToken},
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
                    let right = self.visit(*right)?;
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
                Node::Number(Token::Number(ValueToken(_, value))) => Ok(NumberType::new(value)),
                Node::Access(Token::Identifier(ValueToken(TokenData(trace), id))) => {
                    let num = self.symbols.get(id.clone());
                    match num {
                        Some(n) => Ok(n.clone()),
                        None => Err(error::Error::UnknownIdentifier(ErrorData {
                            trace,
                            details: format!("'{}' is not defined", id),
                        })),
                    }
                }
                Node::Assignment {
                    id: Token::Identifier(ValueToken(_, id)),
                    node,
                } => {
                    let node = self.visit(*node)?;
                    self.symbols.set(id, &node);
                    Ok(node)
                }
                Node::Error(e) => Err(e),
                _ => Err(error::Error::Unknown),
            };
        }
    }
}
