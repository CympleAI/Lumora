use std::collections::HashMap;
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, Instruction, MemorySection, MemoryType, Module as WasmModule, TypeSection,
    ValType,
};

use crate::ast::*;
use crate::error::LumoraError;
use crate::types::*;

pub struct Compiler {
    module: Module,
    env: TypeEnvironment,
    wasm_module: WasmModule,
    type_map: HashMap<Type, u32>, // Maps Lumora types to WebAssembly types
    struct_offsets: HashMap<String, Vec<(String, u32)>>, // Maps struct fields to their offsets
    struct_sizes: HashMap<String, u32>, // Maps struct names to their sizes in bytes
    function_indices: HashMap<String, u32>, // Maps function names to their indices
}

impl Compiler {
    pub fn new(module: Module) -> Self {
        let env = TypeEnvironment::new();

        Self {
            module,
            env,
            wasm_module: WasmModule::new(),
            type_map: HashMap::new(),
            struct_offsets: HashMap::new(),
            struct_sizes: HashMap::new(),
            function_indices: HashMap::new(),
        }
    }

    pub fn compile(mut self) -> Result<Vec<u8>, LumoraError> {
        // Initialize environment
        self.initialize_environment()?;

        // Calculate struct layouts
        self.calculate_struct_layouts()?;

        // Compile types: SectionType 1
        self.compile_types()?;

        // Compile imports: SectionImport 2
        self.compile_imports()?;

        // Compile function signatures: SectionFunction 3
        self.compile_function_signatures()?;

        // Compile imports: SectionMemory 5
        self.compile_memories()?;

        // Compile exports: SectionExport 7
        self.compile_exports()?;

        // Compile function bodies: Section Code 10
        self.compile_function_bodies()?;

        // Finalize the module
        Ok(self.wasm_module.finish())
    }

    fn initialize_environment(&mut self) -> Result<(), LumoraError> {
        // First pass: register all struct types
        for struct_def in &self.module.structs {
            self.env.add_struct(struct_def.clone());
        }

        // Second pass: register all function types
        for fn_def in &self.module.functions {
            let param_types = fn_def.params.iter().map(|p| p.typ.clone()).collect();
            self.env
                .add_function(fn_def.name.clone(), param_types, fn_def.return_type.clone());

            // Store the function index
            let index = self.function_indices.len() as u32;
            self.function_indices.insert(fn_def.name.clone(), index);
        }

        // Third pass: typecheck the whole module
        check_module(&self.module)?;

        Ok(())
    }

    fn calculate_struct_layouts(&mut self) -> Result<(), LumoraError> {
        for struct_def in &self.module.structs {
            let mut field_offsets = Vec::new();
            let mut offset = 0;

            for field in &struct_def.fields {
                field_offsets.push((field.name.clone(), offset));
                offset += self.get_type_size(&field.typ)?;
            }

            self.struct_offsets
                .insert(struct_def.name.clone(), field_offsets);
            self.struct_sizes.insert(struct_def.name.clone(), offset);
        }

        Ok(())
    }

    fn get_type_size(&self, typ: &Type) -> Result<u32, LumoraError> {
        match typ {
            Type::Int => Ok(4),
            Type::Float => Ok(4),
            Type::String => Ok(8),          // Pointer + length
            Type::Tensor => Ok(12),         // Pointer + dimensions
            Type::Array(_) => Ok(8),        // Pointer + length
            Type::Stream(_) => Ok(8),       // Pointer + state
            Type::Function { .. } => Ok(4), // Function index
            Type::Struct(name) => self
                .struct_sizes
                .get(name)
                .cloned()
                .ok_or_else(|| LumoraError::UndefinedStruct(name.clone())),
        }
    }

    fn compile_types(&mut self) -> Result<(), LumoraError> {
        let mut type_section = TypeSection::new();

        // Add function types
        for fn_def in &self.module.functions {
            let params = fn_def
                .params
                .iter()
                .map(|p| self.lumora_type_to_wasm_type(&p.typ))
                .collect::<Result<Vec<_>, _>>()?;

            let returns = vec![self.lumora_type_to_wasm_type(&fn_def.return_type)?];

            type_section.ty().function(params, returns);
        }

        self.wasm_module.section(&type_section);

        Ok(())
    }

    fn lumora_type_to_wasm_type(&self, typ: &Type) -> Result<ValType, LumoraError> {
        match typ {
            Type::Int => Ok(ValType::I32),
            Type::Float => Ok(ValType::F32),
            Type::String => Ok(ValType::I32), // Pointer to string data
            Type::Tensor => Ok(ValType::I32), // Pointer to tensor data
            Type::Array(_) => Ok(ValType::I32), // Pointer to array data
            Type::Stream(_) => Ok(ValType::I32), // Pointer to stream data
            Type::Function { .. } => Ok(ValType::I32), // Function index
            Type::Struct(_) => Ok(ValType::I32), // Pointer to struct data
        }
    }

    fn compile_imports(&mut self) -> Result<(), LumoraError> {
        let mut import_section = ImportSection::new();
        let mut type_section = TypeSection::new();

        // Add imports
        for import in self.module.imports.iter() {
            // Create a function type for the import
            let (params_types, return_types) = match &import.typ {
                Type::Function {
                    params,
                    return_type,
                } => {
                    let params = params
                        .iter()
                        .map(|p| self.lumora_type_to_wasm_type(p))
                        .collect::<Result<Vec<_>, _>>()?;
                    let returns = vec![self.lumora_type_to_wasm_type(return_type)?];

                    (params, returns)
                }
                _ => {
                    return Err(LumoraError::Compilation(format!(
                        "Import {} must be a function type",
                        import.name
                    )))
                }
            };

            let type_index = self.function_indices.len() as u32;
            self.function_indices
                .insert(import.name.clone(), type_index);
            type_section.ty().function(params_types, return_types);

            import_section.import(
                &import.module,
                &import.name,
                EntityType::Function(type_index),
            );
        }

        // self.wasm_module.section(&type_section);
        self.wasm_module.section(&import_section);

        Ok(())
    }

    fn compile_function_signatures(&mut self) -> Result<(), LumoraError> {
        let mut function_section = FunctionSection::new();

        // Add function types
        for fn_def in &self.module.functions {
            // The type index is the same as the function index
            function_section.function(self.function_indices[&fn_def.name]);
        }

        self.wasm_module.section(&function_section);

        Ok(())
    }

    fn compile_memories(&mut self) -> Result<(), LumoraError> {
        let mut memories = MemorySection::new();

        memories.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });

        self.wasm_module.section(&memories);

        Ok(())
    }

    fn compile_exports(&mut self) -> Result<(), LumoraError> {
        let mut exports = ExportSection::new();
        for export in &self.module.exports {
            if let Some(&index) = self.function_indices.get(&export.name) {
                exports.export(&export.name, ExportKind::Func, index);
            } else {
                return Err(LumoraError::UndefinedFunction(export.name.clone()));
            }
            // ext. memory, global
        }

        self.wasm_module.section(&exports);

        Ok(())
    }

    fn compile_function_bodies(&mut self) -> Result<(), LumoraError> {
        let mut code_section = CodeSection::new();

        for fn_def in &self.module.functions {
            let mut env = self.env.create_child_scope(false);

            // Add parameters to the environment and local indices
            for (i, param) in fn_def.params.iter().enumerate() {
                env.add_variable(param.name.clone(), param.typ.clone());
                env.add_local(param.name.clone(), i as u32);
            }

            // Compile the function body
            let mut body = Function::new(vec![]);
            let mut instructions = vec![];
            self.compile_block(&mut env, &fn_def.body, &mut instructions)?;

            let mut no_return = true;
            for instruction in instructions {
                if matches!(instruction, Instruction::Return) {
                    no_return = false;
                }
                body.instruction(&instruction);
            }

            // Ensure the function returns a value
            if no_return {
                body.instruction(&Instruction::Return);
            }

            // End the body
            body.instruction(&Instruction::End);

            code_section.function(&body);
        }

        self.wasm_module.section(&code_section);

        Ok(())
    }

    fn compile_block(
        &self,
        env: &mut TypeEnvironment,
        block: &Block,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Type>, LumoraError> {
        let mut return_types = vec![];

        let mut next = block.statements.len();
        for stmt in &block.statements {
            if let Some(return_type) = self.compile_stmt(env, stmt, instructions, next != 1)? {
                return_types.push(return_type);
            }
            next -= 1;
        }

        // TODO check return_types is same

        Ok(return_types.pop())
    }

    fn compile_stmt(
        &self,
        env: &mut TypeEnvironment,
        stmt: &Stmt,
        instructions: &mut Vec<Instruction>,
        drop: bool,
    ) -> Result<Option<Type>, LumoraError> {
        match stmt {
            Stmt::Let { name, typ, value } => {
                // Compile the value expression
                self.compile_expr(env, value, instructions)?;

                // Add the variable to the environment
                let var_type = if let Some(t) = typ {
                    t.clone()
                } else {
                    // Infer the type from the value
                    check_expr(env, value)?
                };

                env.add_variable(name.clone(), var_type);

                // Add the variable to locals
                let local_idx = env.next_local();
                env.add_local(name.clone(), local_idx);

                // Store the value in the local
                instructions.push(Instruction::LocalSet(local_idx));

                Ok(None)
            }
            Stmt::Set { name, value } => {
                // Compile the value expression
                self.compile_expr(env, value, instructions)?;

                // Find the local index
                let local_idx = env.get_local(&name)?;

                // Store the value in the local
                instructions.push(Instruction::LocalSet(local_idx));

                Ok(None)
            }
            Stmt::Return { value } => {
                // Compile the value expression
                let ty = self.compile_expr(env, value, instructions)?;

                // Return the value
                instructions.push(Instruction::Return);

                // mark return type
                Ok(Some(ty))
            }
            Stmt::If {
                condition,
                then_block,
                else_block,
            } => {
                // Compile the condition
                self.compile_expr(env, condition, instructions)?;

                // Compile the then block
                let mut then_env = env.create_child_scope(true);

                // If-else block
                let mut then_instructions = vec![];
                let mut else_instructions = vec![];
                let mut block_ty =
                    self.compile_block(&mut then_env, then_block, &mut then_instructions)?;

                if let Some(else_block) = else_block {
                    // Compile the else block
                    let mut else_env = env.create_child_scope(true);
                    if let Some(ty) =
                        self.compile_block(&mut else_env, else_block, &mut else_instructions)?
                    {
                        // TODO check return is same
                        block_ty = Some(ty);
                    }
                }

                // check block result
                let wasm_block_ty = if let Some(ty) = &block_ty {
                    let wasm_ty = self.lumora_type_to_wasm_type(ty)?;
                    BlockType::Result(wasm_ty)
                } else {
                    BlockType::Empty
                };

                instructions.push(Instruction::If(wasm_block_ty));

                // insert instructions
                for i in then_instructions {
                    instructions.push(i);
                }
                if !else_instructions.is_empty() {
                    instructions.push(Instruction::Else);
                    for i in else_instructions {
                        instructions.push(i);
                    }
                }

                instructions.push(Instruction::End);

                Ok(block_ty)
            }
            Stmt::Loop { body } => {
                // Compile the loop body
                let mut block_instructions = vec![];
                let mut loop_env = env.create_child_scope(true);
                let block_ty = self.compile_block(&mut loop_env, body, &mut block_instructions)?;

                let wasm_block_ty = if let Some(ty) = &block_ty {
                    let wasm_ty = self.lumora_type_to_wasm_type(ty)?;
                    BlockType::Result(wasm_ty)
                } else {
                    BlockType::Empty
                };

                instructions.push(Instruction::Block(wasm_block_ty));
                instructions.push(Instruction::Loop(wasm_block_ty));

                for i in block_instructions {
                    instructions.push(i);
                }

                // Continue the loop
                instructions.push(Instruction::Br(0));
                instructions.push(Instruction::End);
                instructions.push(Instruction::End);

                Ok(block_ty)
            }
            Stmt::For {
                var,
                start,
                end,
                body,
            } => {
                // Compile the range
                self.compile_expr(env, start, instructions)?;

                // Create the index variable
                let local_idx = env.next_local();
                env.add_local(var.clone(), local_idx);

                // Store the start value in the local
                instructions.push(Instruction::LocalSet(local_idx));

                // Firstly compile the loop body for body return
                let mut body_instructions = vec![];
                let mut for_env = env.create_child_scope(true);
                for_env.add_variable(var.clone(), Type::Int);
                let block_ty = self.compile_block(&mut for_env, body, &mut body_instructions)?;

                let wasm_block_ty = if let Some(ty) = &block_ty {
                    let wasm_ty = self.lumora_type_to_wasm_type(ty)?;
                    BlockType::Result(wasm_ty)
                } else {
                    BlockType::Empty
                };

                // Loop block
                instructions.push(Instruction::Block(wasm_block_ty));
                instructions.push(Instruction::Loop(wasm_block_ty));

                // Check if index < end
                instructions.push(Instruction::LocalGet(local_idx));
                self.compile_expr(env, end, instructions)?;
                instructions.push(Instruction::I32LtS);

                // If index >= end, break out of the loop
                instructions.push(Instruction::I32Eqz);
                instructions.push(Instruction::BrIf(1));

                // Compile the loop body
                for i in body_instructions {
                    instructions.push(i);
                }

                // Increment the index
                instructions.push(Instruction::LocalGet(local_idx));
                instructions.push(Instruction::I32Const(1));
                instructions.push(Instruction::I32Add);
                instructions.push(Instruction::LocalSet(local_idx));

                // Continue the loop
                instructions.push(Instruction::Br(0));
                instructions.push(Instruction::End);
                instructions.push(Instruction::End);

                Ok(block_ty)
            }
            Stmt::Break => {
                // Break out of the innermost loop
                instructions.push(Instruction::Br(1));

                Ok(None)
            }
            Stmt::Expr(expr) => {
                // Compile the expression
                let ty = self.compile_expr(env, expr, instructions)?;

                // Discard the value
                if drop {
                    instructions.push(Instruction::Drop);
                    Ok(None)
                } else {
                    Ok(Some(ty))
                }
            }
        }
    }

    fn compile_expr(
        &self,
        env: &TypeEnvironment,
        expr: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Type, LumoraError> {
        let ty = match expr {
            Expr::Identifier(name) => {
                // Get the local index
                let local_idx = env.get_local(name)?;

                // Load the value from the local
                instructions.push(Instruction::LocalGet(local_idx));

                // TODO
                Type::Int
            }
            Expr::Integer(value) => {
                instructions.push(Instruction::I32Const(*value as i32));

                Type::Int
            }
            Expr::Float(value) => {
                instructions.push(Instruction::F32Const(*value as f32));

                Type::Float
            }
            Expr::String(value) => {
                // In a real implementation, we would allocate memory for the string
                // and store the string data. For simplicity, we'll just push a dummy pointer.
                instructions.push(Instruction::I32Const(0));

                Type::String
            }
            Expr::Array(elements) => {
                // In a real implementation, we would allocate memory for the array
                // and store each element. For simplicity, we'll just push a dummy pointer.
                instructions.push(Instruction::I32Const(0));

                // TODO
                Type::Array(Box::new(Type::Int))
            }
            Expr::Tensor(elements) => {
                // In a real implementation, we would allocate memory for the tensor
                // and store the tensor data. For simplicity, we'll just push a dummy pointer.
                instructions.push(Instruction::I32Const(0));

                Type::Tensor
            }
            Expr::Operation { op, args } => self.compile_operation(env, op, args, instructions)?,
            Expr::StructLiteral { name, fields } => {
                // In a real implementation, we would allocate memory for the struct
                // and store each field. For simplicity, we'll just push a dummy pointer.
                instructions.push(Instruction::I32Const(0));

                Type::Struct(name.to_owned())
            }
            Expr::FieldAccess { expr, field } => {
                // In a real implementation, we would load the struct pointer,
                // compute the field offset, and load the field value.
                // For simplicity, we'll just push a dummy value.
                instructions.push(Instruction::I32Const(0));

                // TODO
                Type::Int
            }
        };

        Ok(ty)
    }

    fn compile_operation(
        &self,
        env: &TypeEnvironment,
        op: &str,
        args: &[Expr],
        instructions: &mut Vec<Instruction>,
    ) -> Result<Type, LumoraError> {
        let ty = match op {
            // Arithmetic operations
            "+" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32Add),
                    Type::Float => instructions.push(Instruction::F32Add),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator + to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            "-" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32Sub),
                    Type::Float => instructions.push(Instruction::F32Sub),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator - to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            "*" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32Mul),
                    Type::Float => instructions.push(Instruction::F32Mul),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator * to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            "/" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32DivS),
                    Type::Float => instructions.push(Instruction::F32Div),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator / to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            "%" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32RemS),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator % to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            "<=" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32LeS),
                    // Type::Float => instructions.push(Instruction::F32LeS),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator <= to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            ">=" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32GeS),
                    // Type::Float => instructions.push(Instruction::F32GeS),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator >= to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            "<" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32LtS),
                    // Type::Float => instructions.push(Instruction::F32LtS),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator < to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            ">" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32GtS),
                    // Type::Float => instructions.push(Instruction::F32GtS),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator > to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            "=" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32Eq),
                    Type::Float => instructions.push(Instruction::F32Eq),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator > to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            "!=" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => instructions.push(Instruction::I32Ne),
                    Type::Float => instructions.push(Instruction::F32Ne),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator > to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            "and" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type1 = check_expr(env, &args[0])?;
                let arg_type2 = check_expr(env, &args[2])?;
                match (&arg_type1, &arg_type2) {
                    (&Type::Int, &Type::Int) => instructions.push(Instruction::I32And),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator and to type {:?} {:?}",
                            arg_type1, arg_type2
                        )))
                    }
                }

                arg_type1
            }
            "or" => {
                self.compile_expr(env, &args[0], instructions)?;
                self.compile_expr(env, &args[1], instructions)?;

                let arg_type1 = check_expr(env, &args[0])?;
                let arg_type2 = check_expr(env, &args[2])?;
                match (&arg_type1, &arg_type2) {
                    (&Type::Int, &Type::Int) => instructions.push(Instruction::I32Or),
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator or to type {:?} {:?}",
                            arg_type1, arg_type2
                        )))
                    }
                }

                arg_type1
            }
            "not" => {
                self.compile_expr(env, &args[0], instructions)?;

                let arg_type = check_expr(env, &args[0])?;
                match arg_type {
                    Type::Int => {
                        instructions.push(Instruction::I32Const(-1));
                        instructions.push(Instruction::I32Xor);
                    }
                    _ => {
                        return Err(LumoraError::Compilation(format!(
                            "Cannot apply operator not to type {:?}",
                            arg_type
                        )))
                    }
                }

                arg_type
            }
            _ => {
                // println!("func_op: {}", op);
                if let Some(&func_idx) = self.function_indices.get(op) {
                    // println!("func_idx: {}", func_idx);
                    for arg in args {
                        self.compile_expr(env, arg, instructions)?;
                    }
                    instructions.push(Instruction::Call(func_idx));
                } else {
                    return Err(LumoraError::UndefinedFunction(op.to_string()));
                }

                // TODO get function return_type
                Type::Int
            }
        };

        Ok(ty)
    }
}
