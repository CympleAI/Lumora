use lumora::*;
use wasmtime::*;

const CODE: &str = r#"
(mod main
  (fn add (x int) (y int) -> int
    (+ x y)
  )
  (export add)
)
"#;

fn main() {
    let wasm_bytes = compile(CODE).unwrap();

    let engine = Engine::default();
    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);

    let module = Module::from_binary(&engine, &wasm_bytes).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let func = instance
        .get_typed_func::<(i32, i32), i32>(&mut store, "add")
        .unwrap();
    let result = func.call(&mut store, (1, 2)).unwrap();
    println!("Add result: {}", result);
}
