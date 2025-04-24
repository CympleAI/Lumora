# Lumora
Lumora is a statically typed, S-expression-based programming language designed for AI-driven development and WebAssembly (WASM) runtimes. Lumora aims to minimize error rates in code generation by large language models (LLMs) through a minimalist, regularized syntax while delivering fast compilation, high performance, and cross-platform compatibility. It is particularly suited for AI inference, data processing, and edge computing.

## Features
- AI-Friendly: Regularized S-expression syntax reduces LLM generation errors, with type inference and bracket repair for robustness.
- WASM-Optimized: Compilation in <1 second, compact WASM modules (<100KB), and near-Rust performance (90-95%).
- Static Typing: Supports basic types (int, float, string, tensor), composite types (array, stream, struct), and function types.
- AI-Specific Support: Built-in tensor operations (tensor.matmul), inference pipelines (model.infer), and stream processing (stream.map).
- Cross-Platform: Runs on browsers, servers, and edge devices, leveraging WASM's sandboxed security.

## Example
Here's a Lumora program example `fibonacci.lum`:
```lisp
(mod main
  (fn fib (n int) -> int
    (if (<= n 1)
        n
        (+ (fib (- n 1)) (fib (- n 2)))
    )
  )
  (export fib)
)
```

1. Compile `cargo run fibonacci.lum fibonacci.wasm`
2. Try run `wasmtime fibonacci.wasm --invoke fib 10`

or `cargo run --example fibonacci`

## License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.
