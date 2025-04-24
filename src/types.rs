use crate::ast::*;
use crate::error::LumoraError;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TypeEnvironment {
    structs: HashMap<String, StructDef>,
    variables: HashMap<String, Type>,
    functions: HashMap<String, FunctionType>,
    local_indices: HashMap<String, u32>,
}

#[derive(Debug, Clone)]
pub struct FunctionType {
    pub params: Vec<Type>,
    pub return_type: Type,
}

impl TypeEnvironment {
    pub fn new() -> Self {
        Self {
            structs: HashMap::new(),
            variables: HashMap::new(),
            functions: HashMap::new(),
            local_indices: HashMap::new(),
        }
    }

    pub fn add_struct(&mut self, struct_def: StructDef) {
        self.structs.insert(struct_def.name.clone(), struct_def);
    }

    pub fn add_variable(&mut self, name: String, typ: Type) {
        self.variables.insert(name, typ);
    }

    pub fn get_variable_type(&self, name: &str) -> Result<&Type, LumoraError> {
        self.variables
            .get(name)
            .ok_or_else(|| LumoraError::UndefinedVariable(name.to_string()))
    }

    pub fn add_function(&mut self, name: String, params: Vec<Type>, return_type: Type) {
        self.functions.insert(
            name,
            FunctionType {
                params,
                return_type,
            },
        );
    }

    pub fn get_function_type(&self, name: &str) -> Result<&FunctionType, LumoraError> {
        self.functions
            .get(name)
            .ok_or_else(|| LumoraError::UndefinedFunction(name.to_string()))
    }

    pub fn get_struct(&self, name: &str) -> Result<&StructDef, LumoraError> {
        self.structs
            .get(name)
            .ok_or_else(|| LumoraError::UndefinedStruct(name.to_string()))
    }

    pub fn next_local(&self) -> u32 {
        self.local_indices.len() as u32
    }

    pub fn add_local(&mut self, name: String, index: u32) {
        self.local_indices.insert(name, index);
    }

    pub fn get_local(&self, name: &str) -> Result<u32, LumoraError> {
        self.local_indices
            .get(name)
            .map(|v| *v)
            .ok_or_else(|| LumoraError::UndefinedVariable(name.to_string()))
    }

    pub fn create_child_scope(&self, keep: bool) -> Self {
        // Clone the environment for a new scope
        let local_indices = if keep {
            self.local_indices.clone()
        } else {
            HashMap::new()
        };
        Self {
            structs: self.structs.clone(),
            variables: self.variables.clone(),
            functions: self.functions.clone(),
            local_indices,
        }
    }
}

pub fn check_module(module: &Module) -> Result<(), LumoraError> {
    let mut env = TypeEnvironment::new();

    // First pass: register all struct types
    for struct_def in &module.structs {
        env.add_struct(struct_def.clone());
    }

    // Second pass: register all function types
    for fn_def in &module.functions {
        let param_types = fn_def.params.iter().map(|p| p.typ.clone()).collect();
        env.add_function(fn_def.name.clone(), param_types, fn_def.return_type.clone());
    }

    // Third pass: typecheck function bodies
    for fn_def in &module.functions {
        check_function(&env, fn_def)?;
    }

    // Fourth pass: check exports
    for export in &module.exports {
        if !env.functions.contains_key(&export.name) {
            return Err(LumoraError::UndefinedFunction(export.name.clone()));
        }
    }

    Ok(())
}

pub fn check_function(env: &TypeEnvironment, fn_def: &FnDef) -> Result<(), LumoraError> {
    let mut function_env = env.create_child_scope(false);

    // Add parameters to the environment
    for param in &fn_def.params {
        function_env.add_variable(param.name.clone(), param.typ.clone());
    }

    // Check the function body
    let result_type = check_block(&mut function_env, &fn_def.body)?;

    // Check that the result type matches the declared return type
    if result_type != fn_def.return_type {
        return Err(LumoraError::Type(format!(
            "Function {} declares return type {:?} but body returns {:?}",
            fn_def.name, fn_def.return_type, result_type
        )));
    }

    Ok(())
}

pub fn check_block(env: &mut TypeEnvironment, block: &Block) -> Result<Type, LumoraError> {
    let mut result_type = Type::Int; // Default result type

    for stmt in &block.statements {
        match stmt {
            Stmt::Return { value } => {
                result_type = check_expr(env, value)?;
                return Ok(result_type);
            }
            _ => {
                check_stmt(env, stmt)?;
            }
        }
    }

    Ok(result_type)
}

pub fn check_stmt(env: &mut TypeEnvironment, stmt: &Stmt) -> Result<(), LumoraError> {
    match stmt {
        Stmt::Let { name, typ, value } => {
            let value_type = check_expr(env, value)?;

            if let Some(declared_type) = typ {
                if *declared_type != value_type {
                    return Err(LumoraError::Type(format!(
                        "Variable {} is declared with type {:?} but initialized with value of type {:?}",
                        name, declared_type, value_type
                    )));
                }
                env.add_variable(name.clone(), declared_type.clone());
            } else {
                env.add_variable(name.clone(), value_type);
            }
        }
        Stmt::Set { name, value } => {
            let variable_type = env.get_variable_type(name)?.clone();
            let value_type = check_expr(env, value)?;

            if variable_type != value_type {
                return Err(LumoraError::Type(format!(
                    "Cannot assign value of type {:?} to variable {} of type {:?}",
                    value_type, name, variable_type
                )));
            }
        }
        Stmt::Return { value } => {
            check_expr(env, value)?;
        }
        Stmt::If {
            condition,
            then_block,
            else_block,
        } => {
            let condition_type = check_expr(env, condition)?;

            // Condition should be a boolean (represented as int in our simple type system)
            if condition_type != Type::Int {
                return Err(LumoraError::Type(format!(
                    "If condition must be of type int (boolean), got {:?}",
                    condition_type
                )));
            }

            let mut then_env = env.create_child_scope(true);
            check_block(&mut then_env, then_block)?;

            if let Some(else_block) = else_block {
                let mut else_env = env.create_child_scope(true);
                check_block(&mut else_env, else_block)?;
            }
        }
        Stmt::Loop { body } => {
            let mut loop_env = env.create_child_scope(true);
            check_block(&mut loop_env, body)?;
        }
        Stmt::For {
            var,
            start,
            end,
            body,
        } => {
            let start_type = check_expr(env, start)?;
            let end_type = check_expr(env, end)?;

            if start_type != Type::Int || end_type != Type::Int {
                return Err(LumoraError::Type(
                    "For loop range must be of type int".to_string(),
                ));
            }

            let mut for_env = env.create_child_scope(true);
            for_env.add_variable(var.clone(), Type::Int);
            check_block(&mut for_env, body)?;
        }
        Stmt::Break => {
            // Nothing to type check for break
        }
        Stmt::Expr(expr) => {
            check_expr(env, expr)?;
        }
    }

    Ok(())
}

pub fn check_expr(env: &TypeEnvironment, expr: &Expr) -> Result<Type, LumoraError> {
    match expr {
        Expr::Identifier(name) => env.get_variable_type(name).cloned(),
        Expr::Integer(_) => Ok(Type::Int),
        Expr::Float(_) => Ok(Type::Float),
        Expr::String(_) => Ok(Type::String),
        Expr::Array(elements) => {
            if elements.is_empty() {
                return Err(LumoraError::Type(
                    "Cannot infer type of empty array".to_string(),
                ));
            }

            let first_type = check_expr(env, &elements[0])?;

            for (i, elem) in elements.iter().enumerate().skip(1) {
                let elem_type = check_expr(env, elem)?;
                if elem_type != first_type {
                    return Err(LumoraError::Type(format!(
                        "Array elements must have the same type. Element 0 has type {:?}, element {} has type {:?}",
                        first_type, i, elem_type
                    )));
                }
            }

            Ok(Type::Array(Box::new(first_type)))
        }
        Expr::Tensor(elements) => {
            for (i, elem) in elements.iter().enumerate() {
                let elem_type = check_expr(env, elem)?;
                if elem_type != Type::Float {
                    return Err(LumoraError::Type(format!(
                        "Tensor elements must be of type float. Element {} has type {:?}",
                        i, elem_type
                    )));
                }
            }

            Ok(Type::Tensor)
        }
        Expr::Operation { op, args } => check_operation(env, op, args),
        Expr::StructLiteral { name, fields } => {
            let struct_def = env.get_struct(name)?;

            // Check that all required fields are provided
            let mut provided_fields = std::collections::HashSet::new();

            for field_init in fields {
                let field_name = &field_init.name;

                // Check for duplicate fields
                if provided_fields.contains(field_name) {
                    return Err(LumoraError::Type(format!(
                        "Duplicate field {} in struct literal",
                        field_name
                    )));
                }
                provided_fields.insert(field_name.clone());

                // Find the field in the struct definition
                let field_def = struct_def
                    .fields
                    .iter()
                    .find(|f| f.name == *field_name)
                    .ok_or_else(|| {
                        LumoraError::Type(format!(
                            "Field {} not found in struct {}",
                            field_name, name
                        ))
                    })?;

                // Check that the field value has the correct type
                let value_type = check_expr(env, &field_init.value)?;
                if value_type != field_def.typ {
                    return Err(LumoraError::Type(format!(
                        "Field {} of struct {} expects type {:?}, got {:?}",
                        field_name, name, field_def.typ, value_type
                    )));
                }
            }

            // Check that all required fields are provided
            for field_def in &struct_def.fields {
                if !provided_fields.contains(&field_def.name) {
                    return Err(LumoraError::Type(format!(
                        "Missing field {} in struct literal for {}",
                        field_def.name, name
                    )));
                }
            }

            Ok(Type::Struct(name.clone()))
        }
        Expr::FieldAccess { expr, field } => {
            let expr_type = check_expr(env, expr)?;

            match expr_type {
                Type::Struct(struct_name) => {
                    let struct_def = env.get_struct(&struct_name)?;

                    let field_def = struct_def
                        .fields
                        .iter()
                        .find(|f| f.name == *field)
                        .ok_or_else(|| {
                            LumoraError::Type(format!(
                                "Field {} not found in struct {}",
                                field, struct_name
                            ))
                        })?;

                    Ok(field_def.typ.clone())
                }
                _ => Err(LumoraError::Type(format!(
                    "Cannot access field {} on non-struct type {:?}",
                    field, expr_type
                ))),
            }
        }
    }
}

pub fn check_operation(
    env: &TypeEnvironment,
    op: &str,
    args: &[Expr],
) -> Result<Type, LumoraError> {
    match op {
        // Arithmetic operations
        "+" | "-" | "*" | "/" | "%" => {
            if args.len() != 2 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires exactly 2 arguments",
                    op
                )));
            }

            let left_type = check_expr(env, &args[0])?;
            let right_type = check_expr(env, &args[1])?;

            let err = format!(
                "Cannot apply operator {} to types {:?} and {:?}",
                op, left_type, right_type
            );

            match (left_type, right_type) {
                (Type::Int, Type::Int) => Ok(Type::Int),
                (Type::Float, Type::Float) => Ok(Type::Float),
                (Type::Tensor, Type::Tensor) => Ok(Type::Tensor),
                _ => Err(LumoraError::Type(err)),
            }
        }
        // Comparison operations
        "<" | ">" | "<=" | ">=" | "=" | "!=" => {
            if args.len() != 2 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires exactly 2 arguments",
                    op
                )));
            }

            let left_type = check_expr(env, &args[0])?;
            let right_type = check_expr(env, &args[1])?;

            // Check that the types are comparable
            if left_type != right_type {
                return Err(LumoraError::Type(format!(
                    "Cannot compare different types: {:?} and {:?}",
                    left_type, right_type
                )));
            }

            // All comparisons return int (boolean)
            Ok(Type::Int)
        }
        // Logical operations
        "and" | "or" => {
            if args.len() != 2 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires exactly 2 arguments",
                    op
                )));
            }

            let left_type = check_expr(env, &args[0])?;
            let right_type = check_expr(env, &args[1])?;

            if left_type != Type::Int || right_type != Type::Int {
                return Err(LumoraError::Type(format!(
                    "Logical operator {} requires int (boolean) arguments, got {:?} and {:?}",
                    op, left_type, right_type
                )));
            }

            Ok(Type::Int)
        }
        "not" => {
            if args.len() != 1 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires exactly 1 argument",
                    op
                )));
            }

            let arg_type = check_expr(env, &args[0])?;

            if arg_type != Type::Int {
                return Err(LumoraError::Type(format!(
                    "Logical operator {} requires int (boolean) argument, got {:?}",
                    op, arg_type
                )));
            }

            Ok(Type::Int)
        }
        // Array operations
        "array.get" => {
            if args.len() != 2 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires exactly 2 arguments",
                    op
                )));
            }

            let array_type = check_expr(env, &args[0])?;
            let index_type = check_expr(env, &args[1])?;

            match array_type {
                Type::Array(element_type) => {
                    if index_type != Type::Int {
                        return Err(LumoraError::Type(format!(
                            "Array index must be int, got {:?}",
                            index_type
                        )));
                    }

                    Ok(*element_type)
                }
                _ => Err(LumoraError::Type(format!(
                    "Cannot index non-array type {:?}",
                    array_type
                ))),
            }
        }
        "array.set" => {
            if args.len() != 3 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires exactly 3 arguments",
                    op
                )));
            }

            let array_type = check_expr(env, &args[0])?;
            let index_type = check_expr(env, &args[1])?;
            let value_type = check_expr(env, &args[2])?;

            match array_type {
                Type::Array(element_type) => {
                    if index_type != Type::Int {
                        return Err(LumoraError::Type(format!(
                            "Array index must be int, got {:?}",
                            index_type
                        )));
                    }

                    if value_type != *element_type {
                        return Err(LumoraError::Type(format!(
                            "Cannot set array element of type {:?} to value of type {:?}",
                            element_type, value_type
                        )));
                    }

                    Ok(Type::Array(element_type))
                }
                _ => Err(LumoraError::Type(format!(
                    "Cannot index non-array type {:?}",
                    array_type
                ))),
            }
        }
        "array.new" => {
            if args.len() != 2 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires exactly 2 arguments",
                    op
                )));
            }

            let size_type = check_expr(env, &args[0])?;
            let element_type = check_expr(env, &args[1])?;

            if size_type != Type::Int {
                return Err(LumoraError::Type(format!(
                    "Array size must be int, got {:?}",
                    size_type
                )));
            }

            Ok(Type::Array(Box::new(element_type)))
        }
        "array.length" => {
            if args.len() != 1 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires exactly 1 argument",
                    op
                )));
            }

            let array_type = check_expr(env, &args[0])?;

            match array_type {
                Type::Array(_) => Ok(Type::Int),
                _ => Err(LumoraError::Type(format!(
                    "Cannot get length of non-array type {:?}",
                    array_type
                ))),
            }
        }
        // String operations
        "string.get" | "string.set" | "string.length" | "string.concat" => {
            // Simplified implementation - just check that the first argument is a string
            let first_type = check_expr(env, &args[0])?;

            if first_type != Type::String {
                return Err(LumoraError::Type(format!(
                    "String operator {} requires string as first argument, got {:?}",
                    op, first_type
                )));
            }

            match op {
                "string.get" => Ok(Type::String),
                "string.set" => Ok(Type::String),
                "string.length" => Ok(Type::Int),
                "string.concat" => Ok(Type::String),
                _ => unreachable!(),
            }
        }
        // Tensor operations
        "tensor.add" | "tensor.sub" | "tensor.matmul" => {
            if args.len() != 2 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires exactly 2 arguments",
                    op
                )));
            }

            let left_type = check_expr(env, &args[0])?;
            let right_type = check_expr(env, &args[1])?;

            if left_type != Type::Tensor || right_type != Type::Tensor {
                return Err(LumoraError::Type(format!(
                    "Tensor operator {} requires tensor arguments, got {:?} and {:?}",
                    op, left_type, right_type
                )));
            }

            Ok(Type::Tensor)
        }
        "tensor.const" => {
            // This creates a tensor from a list of floats
            for arg in args {
                let arg_type = check_expr(env, arg)?;
                if arg_type != Type::Float {
                    return Err(LumoraError::Type(format!(
                        "tensor.const requires float arguments, got {:?}",
                        arg_type
                    )));
                }
            }

            Ok(Type::Tensor)
        }
        "tensor.reshape" => {
            if args.len() < 2 {
                return Err(LumoraError::Type(format!(
                    "Operator {} requires at least 2 arguments",
                    op
                )));
            }

            let tensor_type = check_expr(env, &args[0])?;

            if tensor_type != Type::Tensor {
                return Err(LumoraError::Type(format!(
                    "First argument to tensor.reshape must be a tensor, got {:?}",
                    tensor_type
                )));
            }

            // Check that all dimensions are integers
            for (i, dim) in args.iter().enumerate().skip(1) {
                let dim_type = check_expr(env, dim)?;
                if dim_type != Type::Int {
                    return Err(LumoraError::Type(format!(
                        "Tensor dimension must be int, argument {} has type {:?}",
                        i, dim_type
                    )));
                }
            }

            Ok(Type::Tensor)
        }
        // Function calls
        _ => {
            // Check if this is a user-defined function
            if let Ok(fn_type) = env.get_function_type(op) {
                if args.len() != fn_type.params.len() {
                    return Err(LumoraError::Type(format!(
                        "Function {} expects {} arguments, got {}",
                        op,
                        fn_type.params.len(),
                        args.len()
                    )));
                }

                // Check that all arguments have the correct types
                for (i, (arg, expected_type)) in args.iter().zip(fn_type.params.iter()).enumerate()
                {
                    let arg_type = check_expr(env, arg)?;
                    if arg_type != *expected_type {
                        return Err(LumoraError::Type(format!(
                            "Function {} argument {} expects type {:?}, got {:?}",
                            op, i, expected_type, arg_type
                        )));
                    }
                }

                Ok(fn_type.return_type.clone())
            } else {
                Err(LumoraError::UndefinedFunction(op.to_string()))
            }
        }
    }
}
