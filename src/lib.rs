pub mod ast;
pub mod compiler;
pub mod error;
pub mod parser;
pub mod types;

pub fn compile(code: &str) -> Result<Vec<u8>, error::LumoraError> {
    let ast = parser::parse(code)?;
    compiler::Compiler::new(ast).compile()
}
