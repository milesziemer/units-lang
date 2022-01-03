use super::{
    parser::{ParseError, ParseErrorKind},
    token::{KeywordKind, Token, TokenKind},
    tracker::Tracker,
};

// const ADD: char = '+';
// const SUBTRACT: char = '+';
// const MULTIPLY: char = '+';
// const DIVIDE: char = '+';
pub struct Lexer<'a> {
    text: &'a [u8],
    curr: Option<char>,
    tracker: Tracker,
}

impl Lexer<'_> {
    pub fn new<'a>(text: &'a [u8]) -> Lexer<'a> {
        Lexer {
            text,
            curr: Some(text[0] as char),
            tracker: Tracker::new(0, 0, 0),
        }
    }

    fn advance(&mut self) {
        self.tracker.advance(None);
        self.curr = if self.tracker.index < self.text.len() {
            Some(self.text[self.tracker.index] as char)
        } else {
            None
        }
    }

    pub fn get_tokens(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();
        while let Some(curr) = self.curr {
            let (token, adv) = match curr {
                c if c.is_numeric() => (self.make_number(), false),
                c if c.is_alphabetic() || c == '_' => (self.make_identifier(), false),
                c => (Token::from(&c.to_string(), Some(self.tracker), None), true),
            };
            if adv {
                self.advance();
            }
            if let Ok(Some(token)) = token {
                tokens.push(token);
            } else if let Err(e) = token {
                return Err(e);
            }
        }
        return Ok(tokens);
    }

    fn make_identifier(&mut self) -> Result<Option<Token>, ParseError> {
        let mut identifier = "".to_owned();
        let start = self.tracker.clone();

        while let Some(curr) = self.curr {
            if !curr.is_alphanumeric() && curr != '_' {
                break;
            }
            identifier += &curr.to_string();
            self.advance();
        }

        let kind = if let Some(kw_kind) = KeywordKind::from(&identifier) {
            TokenKind::KEYWORD(kw_kind)
        } else {
            TokenKind::IDENTIFIER(identifier)
        };

        Ok(Some(Token {
            kind,
            start: Some(start),
            end: Some(self.tracker),
            value: None,
        }))
    }

    fn make_number(&mut self) -> Result<Option<Token>, ParseError> {
        let mut num_str = "".to_owned();
        let mut dots = 0;
        let start = self.tracker.clone();

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

        match num_str.parse::<f64>() {
            Ok(n) => Ok(Some(Token {
                kind: TokenKind::NUM,
                value: Some(n),
                start: Some(start),
                end: Some(self.tracker),
            })),
            Err(_) => Err(ParseError {
                kind: ParseErrorKind::IllegalNumber,
                details: num_str,
                start: Some(start),
                end: Some(self.tracker),
            }),
        }
    }
}
