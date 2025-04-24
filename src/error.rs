use thiserror::Error;

#[derive(Error, Debug)]
pub enum LumoraError {
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("Undefined function: {0}")]
    UndefinedFunction(String),

    #[error("Undefined struct: {0}")]
    UndefinedStruct(String),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
}
