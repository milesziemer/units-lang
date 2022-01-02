#[derive(Clone, Copy, Debug)]
pub struct Tracker {
    pub index: usize,
    line: i32,
    column: i32,
}

impl Tracker {
    pub fn new(index: usize, line: i32, column: i32) -> Tracker {
        Tracker {
            index,
            line,
            column,
        }
    }

    pub fn advance(&mut self, curr: Option<char>) -> Tracker {
        self.index += 1;
        self.column += 1;

        if let Some(curr) = curr {
            if curr == '\n' {
                self.line += 1;
                self.column = 0;
            }
        }

        return *self;
    }
}
