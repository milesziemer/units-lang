mod lang;

use lang::{interpreter::Interpreter, lexer::Lexer, parser::Parser};

use std::io::{stdin, stdout, Write};

fn main() {
    // let mut chars = Vec::new();
    // chars.push('1');
    // chars.push('.');
    // chars.push('1');
    // let mut dots = 0;
    // for curr in chars.iter() {
    //     match &curr {
    //         c if c.is_numeric() => println!("numeric"),
    //         '.' if dots == 1 => {
    //             println!("dot count is 1 already");
    //             break;
    //         }
    //         '.' => {
    //             println!("incr dots");
    //             dots += 1;
    //         }
    //         _ => break,
    //     }
    // }
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
        let _result = run(line);
    }
}

fn run(line: String) -> String {
    let mut lexer = Lexer::new(line.as_bytes());
    let tokens = lexer.get_tokens();
    let mut parser = Parser::new(tokens);
    let ast = parser.parse();
    if let Ok(res) = ast {
        let mut interpreter = Interpreter;
        let result = interpreter.visit(res);
        println!("{:?}", result.value);
    } else if let Err(res) = ast {
        println!("{:?}", res);
    }
    return line;
}
