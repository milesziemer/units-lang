#[derive(Debug, Clone, Copy)]
pub enum TokenKind {
    NUM,
    ADD,
    SUBTRACT,
}

#[derive(Clone, Copy)]
pub struct Token {
    pub kind: TokenKind,
    pub value: Option<f64>,
}

impl Token {
    pub fn from(s: &str) -> Option<Token> {
        match s {
            "+" => Some(Token {
                kind: TokenKind::ADD,
                value: None,
            }),
            "-" => Some(Token {
                kind: TokenKind::SUBTRACT,
                value: None,
            }),
            _ => None,
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
