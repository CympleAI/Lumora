use lumora::*;
use wasmtime::*;

const CODE: &str = r#"
(mod main
  (fn fib (n int) -> int
    (if (<= n 1)
        n
        (+ (fib (- n 1)) (fib (- n 2)))
    )
  )
  (export fib)
)
"#;

fn main() {
    let wasm_bytes = compile(CODE).unwrap();
    // std::fs::write("test.wasm", &wasm_bytes);

    let engine = Engine::default();
    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);

    let module = Module::from_binary(&engine, &wasm_bytes).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let func = instance
        .get_typed_func::<i32, i32>(&mut store, "fib")
        .unwrap();
    let result = func.call(&mut store, 10).unwrap();
    println!("Fib result: {}", result);
}
