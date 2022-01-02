use super::token::{Token, TokenKind};

// const ADD: char = '+';
// const SUBTRACT: char = '+';
// const MULTIPLY: char = '+';
// const DIVIDE: char = '+';
pub struct Lexer<'a> {
    text: &'a [u8],
    curr: Option<char>,
    pos: usize,
}

impl Lexer<'_> {
    pub fn new<'a>(text: &'a [u8]) -> Lexer<'a> {
        Lexer {
            text,
            curr: Some(text[0] as char),
            pos: 0,
        }
    }

    fn advance(&mut self) {
        self.pos += 1;
        self.curr = if self.pos < self.text.len() {
            Some(self.text[self.pos] as char)
        } else {
            None
        }
    }

    pub fn get_tokens(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        while let Some(curr) = self.curr {
            let (token, adv) = match curr {
                c if c.is_numeric() => (Some(self.make_number()), false),
                c => (Token::from(&c.to_string()), true),
            };
            if let Some(token) = token {
                tokens.push(token);
            }
            if adv {
                self.advance();
            }
        }
        return tokens;
    }

    fn make_number(&mut self) -> Token {
        let mut num_str = "".to_owned();
        let mut dots = 0;

        while let Some(curr) = self.curr {
            match curr {
                c if c.is_numeric() => num_str.push_str(&curr.to_string()),
                '.' if dots == 1 => break,
                '.' => {
                    dots += 1;
                    num_str.push_str(".");
                }
                _ => break,
            }
            self.advance();
        }

        Token {
            kind: TokenKind::NUM,
            value: match num_str.parse::<f64>() {
                Ok(n) => Some(n),
                Err(_) => None,
            },
        }
    }
}
