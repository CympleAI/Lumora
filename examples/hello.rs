use std::env;
use std::fs;
use std::path::Path;

use lumora::*;
use wasmtime::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: lumorac <input.lum> <output.wasm>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let source = fs::read_to_string(input_path)?;

    let ast = parser::parse(&source)?;

    let wasm_bytes = compiler::Compiler::new(ast).compile()?;

    fs::write(output_path, wasm_bytes)?;

    let engine = Engine::default();
    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);

    // 加载你的模块
    let module = Module::from_file(&engine, output_path)?;
    let instance = linker.instantiate(&mut store, &module)?;

    // 调用函数
    let func = instance.get_typed_func::<(i32, i32), i32>(&mut store, "init")?;
    let result = func.call(&mut store, (1, 2))?;
    println!("Result: {}", result);

    println!("Compiled {} -> {}", input_path, output_path);
    Ok(())
}
