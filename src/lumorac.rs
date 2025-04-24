use lumora::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: lumorac <input.lum> <output.wasm>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let source = std::fs::read_to_string(input_path)?;

    let ast = parser::parse(&source)?;

    let wasm_bytes = compiler::Compiler::new(ast).compile()?;

    std::fs::write(output_path, wasm_bytes)?;

    println!("Compiled {} -> {}", input_path, output_path);
    Ok(())
}
