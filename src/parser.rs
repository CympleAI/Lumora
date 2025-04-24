use pest::{iterators::Pair, Parser};
use pest_derive::Parser;

use crate::ast::*;
use crate::error::LumoraError;

#[derive(Parser)]
#[grammar = "lumora.pest"]
pub struct LumoraParser;

pub fn parse(input: &str) -> Result<Module, LumoraError> {
    let pairs =
        LumoraParser::parse(Rule::program, input).map_err(|e| LumoraError::Parse(e.to_string()))?;

    // Get the first (and only) pair from the top rule which should be 'program'
    let program_pair = pairs.peek().unwrap();

    // The first (and only) inner pair of 'program' should be 'module'
    let module_pair = program_pair.into_inner().next().unwrap();

    parse_module(module_pair)
}

fn parse_module(pair: Pair<Rule>) -> Result<Module, LumoraError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();

    let mut imports = Vec::new();
    let mut structs = Vec::new();
    let mut functions = Vec::new();
    let mut exports = Vec::new();

    for item in inner {
        match item.as_rule() {
            Rule::import => imports.push(parse_import(item)?),
            Rule::struct_def => structs.push(parse_struct_def(item)?),
            Rule::fn_def => functions.push(parse_fn_def(item)?),
            Rule::export => exports.push(parse_export(item)?),
            _ => {
                return Err(LumoraError::Parse(format!(
                    "Unexpected rule in module: {:?}",
                    item.as_rule()
                )))
            }
        }
    }

    Ok(Module {
        name,
        imports,
        structs,
        functions,
        exports,
    })
}

fn parse_import(pair: Pair<Rule>) -> Result<Import, LumoraError> {
    let mut inner = pair.into_inner();
    let module = inner.next().unwrap().as_str().to_string();
    let name = inner.next().unwrap().as_str().to_string();
    let typ = parse_type(inner.next().unwrap())?;

    Ok(Import { module, name, typ })
}

fn parse_struct_def(pair: Pair<Rule>) -> Result<StructDef, LumoraError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();

    let mut fields = Vec::new();
    for field_pair in inner {
        fields.push(parse_field(field_pair)?);
    }

    Ok(StructDef { name, fields })
}

fn parse_field(pair: Pair<Rule>) -> Result<Field, LumoraError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let typ = parse_type(inner.next().unwrap())?;

    Ok(Field { name, typ })
}

fn parse_fn_def(pair: Pair<Rule>) -> Result<FnDef, LumoraError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();

    let mut params = Vec::new();
    while let Some(next) = inner.peek() {
        if next.as_rule() == Rule::param {
            params.push(parse_param(inner.next().unwrap())?);
        } else {
            break;
        }
    }

    let return_type = parse_type(inner.next().unwrap())?;
    let body = parse_block(inner.next().unwrap())?;

    Ok(FnDef {
        name,
        params,
        return_type,
        body,
    })
}

fn parse_param(pair: Pair<Rule>) -> Result<Param, LumoraError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let typ = parse_type(inner.next().unwrap())?;

    Ok(Param { name, typ })
}

fn parse_export(pair: Pair<Rule>) -> Result<Export, LumoraError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();

    Ok(Export { name })
}

fn parse_block(pair: Pair<Rule>) -> Result<Block, LumoraError> {
    let mut statements = Vec::new();
    for stmt_pair in pair.into_inner() {
        statements.push(parse_stmt(stmt_pair)?);
    }

    Ok(Block { statements })
}

fn parse_stmt(pair: Pair<Rule>) -> Result<Stmt, LumoraError> {
    if pair.as_rule() == Rule::stmt {
        let inner_pair = pair.into_inner().next().unwrap();
        return parse_stmt(inner_pair);
    }

    match pair.as_rule() {
        Rule::let_stmt => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();

            // Type annotation is optional
            let next_pair = inner.next().unwrap();
            let (typ, value) = if next_pair.as_rule() == Rule::ty {
                let typ = Some(parse_type(next_pair)?);
                let value = parse_expr(inner.next().unwrap())?;
                (typ, value)
            } else {
                let value = parse_expr(next_pair)?;
                (None, value)
            };

            Ok(Stmt::Let { name, typ, value })
        }
        Rule::set_stmt => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();
            let value = parse_expr(inner.next().unwrap())?;

            Ok(Stmt::Set { name, value })
        }
        Rule::return_stmt => {
            let mut inner = pair.into_inner();
            let value = parse_expr(inner.next().unwrap())?;

            Ok(Stmt::Return { value })
        }
        Rule::if_stmt => {
            let mut inner = pair.into_inner();
            let condition = parse_expr(inner.next().unwrap())?;

            let then_block = parse_block(inner.next().unwrap())?;
            let else_block = if let Some(else_pair) = inner.next() {
                Some(parse_block(else_pair)?)
            } else {
                None
            };

            Ok(Stmt::If {
                condition,
                then_block,
                else_block,
            })
        }
        Rule::loop_stmt => {
            let mut inner = pair.into_inner();
            let body = parse_block(inner.next().unwrap())?;

            Ok(Stmt::Loop { body })
        }
        Rule::for_stmt => {
            let mut inner = pair.into_inner();
            let var = inner.next().unwrap().as_str().to_string();
            let start = parse_expr(inner.next().unwrap())?;
            let end = parse_expr(inner.next().unwrap())?;
            let body = parse_block(inner.next().unwrap())?;

            Ok(Stmt::For {
                var,
                start,
                end,
                body,
            })
        }
        Rule::break_stmt => Ok(Stmt::Break),
        Rule::expr => {
            let expr = parse_expr(pair)?;
            Ok(Stmt::Expr(expr))
        }
        _ => Err(LumoraError::Parse(format!(
            "Unexpected rule in statement: {:?}",
            pair.as_rule()
        ))),
    }
}

fn parse_expr(pair: Pair<Rule>) -> Result<Expr, LumoraError> {
    match pair.as_rule() {
        Rule::expr => {
            // Unwrap the inner expression
            let inner = pair.into_inner().next().unwrap();
            parse_expr(inner)
        }
        Rule::identifier => Ok(Expr::Identifier(pair.as_str().to_string())),
        Rule::integer => {
            let value = pair
                .as_str()
                .parse::<i64>()
                .map_err(|e| LumoraError::Parse(e.to_string()))?;
            Ok(Expr::Integer(value))
        }
        Rule::float => {
            let value = pair
                .as_str()
                .parse::<f64>()
                .map_err(|e| LumoraError::Parse(e.to_string()))?;
            Ok(Expr::Float(value))
        }
        Rule::string => {
            // Remove the quotes from the string
            let value = pair.as_str();
            let value = value[1..value.len() - 1].to_string();
            Ok(Expr::String(value))
        }
        Rule::array => {
            let mut elements = Vec::new();
            for item in pair.into_inner() {
                elements.push(parse_expr(item)?);
            }
            Ok(Expr::Array(elements))
        }
        Rule::tensor => {
            let mut elements = Vec::new();
            for item in pair.into_inner() {
                match item.as_rule() {
                    Rule::float => {
                        let value = item
                            .as_str()
                            .parse::<f64>()
                            .map_err(|e| LumoraError::Parse(e.to_string()))?;
                        elements.push(Expr::Float(value));
                    }
                    _ => elements.push(parse_expr(item)?),
                }
            }
            Ok(Expr::Tensor(elements))
        }
        Rule::operation => {
            let mut inner = pair.into_inner();
            let op = inner.next().unwrap().as_str().to_string();

            let mut args = Vec::new();
            for arg_pair in inner {
                args.push(parse_expr(arg_pair)?);
            }

            Ok(Expr::Operation { op, args })
        }
        Rule::struct_literal => {
            let mut inner = pair.into_inner();
            let name = inner.next().unwrap().as_str().to_string();

            let mut fields = Vec::new();
            for field_pair in inner {
                fields.push(parse_field_init(field_pair)?);
            }

            Ok(Expr::StructLiteral { name, fields })
        }
        Rule::field_access => {
            let mut inner = pair.into_inner();
            let expr = parse_expr(inner.next().unwrap())?;
            let field = inner.next().unwrap().as_str().to_string();

            Ok(Expr::FieldAccess {
                expr: Box::new(expr),
                field,
            })
        }
        _ => Err(LumoraError::Parse(format!(
            "Unexpected rule in expression: {:?}",
            pair.as_rule()
        ))),
    }
}

fn parse_field_init(pair: Pair<Rule>) -> Result<FieldInit, LumoraError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let value = parse_expr(inner.next().unwrap())?;

    Ok(FieldInit { name, value })
}

fn parse_type(pair: Pair<Rule>) -> Result<Type, LumoraError> {
    match pair.as_rule() {
        Rule::ty => {
            let inner = pair.into_inner().next().unwrap();
            parse_type(inner)
        }
        Rule::type_base => match pair.as_str() {
            "int" => Ok(Type::Int),
            "float" => Ok(Type::Float),
            "string" => Ok(Type::String),
            "tensor" => Ok(Type::Tensor),
            _ => Err(LumoraError::Parse(format!(
                "Unknown base type: {}",
                pair.as_str()
            ))),
        },
        Rule::type_array => {
            let mut inner = pair.into_inner();
            let element_type = parse_type(inner.next().unwrap())?;

            Ok(Type::Array(Box::new(element_type)))
        }
        Rule::type_stream => {
            let mut inner = pair.into_inner();
            let element_type = parse_type(inner.next().unwrap())?;

            Ok(Type::Stream(Box::new(element_type)))
        }
        Rule::type_function => {
            let mut inner = pair.into_inner();

            let mut param_types = Vec::new();
            let mut next = inner.next();

            // Parse parameter types until we reach the return type
            while next.is_some() && next.as_ref().unwrap().as_rule() != Rule::ty {
                param_types.push(parse_type(next.unwrap())?);
                next = inner.next();
            }

            // The last type is the return type
            let return_type = parse_type(next.unwrap())?;

            Ok(Type::Function {
                params: param_types,
                return_type: Box::new(return_type),
            })
        }
        Rule::identifier => {
            // This is a user-defined type (struct name)
            Ok(Type::Struct(pair.as_str().to_string()))
        }
        _ => Err(LumoraError::Parse(format!(
            "Unexpected rule in type: {:?}",
            pair.as_rule()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::*;

    #[test]
    fn test_simple_function() {
        let code = r#"
            (mod mymod
                (fn add (a int) (b int) -> int
                  (return (+ a b))
                )
            )
        "#;

        let module = parse(code.trim()).expect("Failed to parse");
        assert_eq!(module.name, "mymod");
        assert_eq!(module.functions.len(), 1);

        let func = &module.functions[0];
        assert_eq!(func.name, "add");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.return_type, Type::Int);

        match &func.body.statements[0] {
            Stmt::Return {
                value: Expr::Operation { op, args },
            } => {
                assert_eq!(op.as_str(), "+");
                assert_eq!(args.len(), 2);
                match &args[0] {
                    Expr::Identifier(s) => assert_eq!(s, "a"),
                    _ => panic!("Expected identifier"),
                }
            }
            _ => panic!("Expected return add expression"),
        }
    }

    #[test]
    fn test_struct_definition() {
        let code = r#"
            (mod mymod
                (struct Vec2 (x float) (y float))
            )
        "#;

        let module = parse(code).expect("Failed to parse");
        assert_eq!(module.structs.len(), 1);
        let s = &module.structs[0];
        assert_eq!(s.name, "Vec2");
        assert_eq!(s.fields.len(), 2);
        assert_eq!(s.fields[0].name, "x");
        assert_eq!(s.fields[0].typ, Type::Float);
    }

    #[test]
    fn test_let_binding() {
        let code = r#"
            (mod main
                (fn init -> int
                    (let x int 42)
                    (return x)
                )
            )
        "#;

        let module = parse(code).expect("Failed to parse");
        let func = &module.functions[0];
        // println!("test {:?}", func.body);
        // Let("(let x int 42)", None, Ident("x"))
        match &func.body.statements[0] {
            Stmt::Let {
                name,
                typ: Some(Type::Int),
                value: Expr::Integer(42),
            } => {
                assert_eq!(name, "x");
            }
            _ => panic!("Expected let binding"),
        }
    }
}
