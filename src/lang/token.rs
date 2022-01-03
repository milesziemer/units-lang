use super::parser::{ParseError, ParseErrorKind};
use super::tracker::Tracker;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeywordKind {
    LET,
}

impl KeywordKind {
    pub fn from(s: &str) -> Option<KeywordKind> {
        match s {
            "let" => Some(KeywordKind::LET),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    NUM,
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    POWER,
    LPAREN,
    RPAREN,
    EQUALS,
    IDENTIFIER(String),
    KEYWORD(KeywordKind),
}

#[derive(Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub value: Option<f64>,
    pub start: Option<Tracker>,
    pub end: Option<Tracker>,
}

impl Token {
    pub fn from(
        s: &str,
        start: Option<Tracker>,
        end: Option<Tracker>,
    ) -> Result<Option<Token>, ParseError> {
        if s == " " || s == "\t" {
            return Ok(None);
        }
        let parsed_token = match s {
            "+" => Some((TokenKind::ADD, None)),
            "-" => Some((TokenKind::SUBTRACT, None)),
            "*" => Some((TokenKind::MULTIPLY, None)),
            "/" => Some((TokenKind::DIVIDE, None)),
            "^" => Some((TokenKind::POWER, None)),
            "(" => Some((TokenKind::LPAREN, None)),
            ")" => Some((TokenKind::RPAREN, None)),
            "=" => Some((TokenKind::EQUALS, None)),
            _ => None,
        };
        if let Some((kind, value)) = parsed_token {
            Ok(Some(Token {
                kind,
                value,
                start,
                end,
            }))
        } else {
            Err(ParseError {
                start,
                end,
                kind: ParseErrorKind::IllegalChar,
                details: format!("'{}'", s),
            })
        }
    }
}

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Token")
            .field("kind", &self.kind)
            .field("value", &self.value)
            .finish()
    }
}
