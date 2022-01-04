mod lang;

use lang::{
    interpreter::{Interpreter, NumberType, SymbolTable},
    lexer::Lexer,
    parser::{ParseError, Parser},
};

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
        let _result = run(line, &mut symbol_table);
        // match result {
        //     Ok(num) => println!("{:?}", num.value),
        //     Err(e) => println!("{:?}", e),
        // }
    }
}

fn run(line: String, _symbol_table: &mut SymbolTable) -> Result<(), ()> {
    let mut lexer = lexer::Lexer::new(line.as_bytes());
    let tokens = lexer.get_tokens()?;
    for token in tokens.iter() {
        println!("{:?}", token);
    }
    Ok(())
    // let mut lexer = Lexer::new(line.as_bytes());
    // let tokens = lexer.get_tokens()?;
    // let mut parser = Parser::new(tokens);
    // let ast = parser.parse()?;
    // let mut interpreter = Interpreter { symbol_table };
    // interpreter.visit(ast)
}

pub enum _Error<T> {
    InvalidSyntax(T),
    IllegalChar(T),
    IllegalNumber(T),
    UnknownIdentifier(T),
}

pub trait Advances {
    type Item;
    fn advance(&mut self, curr: Self::Item) -> Self::Item;
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

impl Advances for Location {
    type Item = Option<char>;
    fn advance(&mut self, curr: Self::Item) -> Self::Item {
        self.index += 1;
        self.column += 1;
        if let Some('\n') = curr {
            self.line += 1;
            self.column += 1;
        }
        curr
    }
}

#[derive(Debug)]
pub struct Tracer {
    pub start: Location,
    pub end: Location,
}

pub trait Traceable {
    fn get_current_location(&mut self) -> Location;
}

mod token {
    use crate::{Advances, Traceable, Tracer};

    #[derive(Debug)]
    pub struct TokenData(Tracer);

    #[derive(Debug)]
    pub struct ValueToken<T>(TokenData, T);

    #[derive(Debug)]
    pub enum Token {
        Unknown,
        Empty,
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
                _ => Token::Unknown,
            }
        }

        fn build(
            c: char,
            adv: &mut (impl Advances<Item = Option<char>> + Traceable),
            validator: &mut impl Validates,
        ) -> (String, TokenData) {
            let mut acc = c.to_string();
            let start = adv.get_current_location();
            while let Some(c) = adv.advance(None) {
                if !validator.validate(c) {
                    break;
                }
                acc.push_str(&c.to_string());
            }
            let end = adv.get_current_location();
            let token_data = TokenData(Tracer { start, end });
            (acc, token_data)
        }

        pub fn make_number(
            c: char,
            adv: &mut (impl Advances<Item = Option<char>> + Traceable),
        ) -> Token {
            let mut validator = NumberValidator { dots: 0 };
            let (num_str, token_data) = Token::build(c, adv, &mut validator);
            match num_str.parse::<f64>() {
                Ok(n) => Token::Number(ValueToken(token_data, n)),
                Err(_) => Token::Unknown,
            }
        }

        pub fn make_identifier(
            c: char,
            adv: &mut (impl Advances<Item = Option<char>> + Traceable),
        ) -> Token {
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
    use crate::{token::Token, Advances, Location, Traceable};
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

        pub fn get_tokens(&mut self) -> Result<Vec<Token>, ()> {
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
                    Token::Empty => (),
                    Token::Unknown => println!("Error"),
                    _ => tokens.push(token),
                }
            }
            Ok(tokens)
        }
    }

    impl Advances for Lexer<'_> {
        type Item = Option<char>;
        fn advance(&mut self, _: Self::Item) -> Self::Item {
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

pub struct _ParseError {
    pub trace: Tracer,
    pub details: String,
}
