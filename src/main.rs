mod lang;

use lang::{
    interpreter::{Interpreter, NumberType},
    lexer::Lexer,
    parser::{ParseError, Parser},
};

use std::io::{stdin, stdout, Write};

fn main() {
    loop {
        let mut line = String::new();
        print!("poop > ");
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
        let result = run(line);
        match result {
            Ok(num) => println!("{:?}", num.value),
            Err(e) => println!("{:?}", e),
        }
    }
}

fn run(line: String) -> Result<NumberType, ParseError> {
    let mut lexer = Lexer::new(line.as_bytes());
    let tokens = lexer.get_tokens()?;
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;
    let mut interpreter = Interpreter;
    interpreter.visit(ast)
}
