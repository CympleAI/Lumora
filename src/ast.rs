#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub imports: Vec<Import>,
    pub structs: Vec<StructDef>,
    pub functions: Vec<FnDef>,
    pub exports: Vec<Export>,
}

#[derive(Debug, Clone)]
pub struct Import {
    pub module: String,
    pub name: String,
    pub typ: Type,
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub typ: Type,
}

#[derive(Debug, Clone)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub typ: Type,
}

#[derive(Debug, Clone)]
pub struct Export {
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        typ: Option<Type>,
        value: Expr,
    },
    Set {
        name: String,
        value: Expr,
    },
    Return {
        value: Expr,
    },
    If {
        condition: Expr,
        then_block: Block,
        else_block: Option<Block>,
    },
    Loop {
        body: Block,
    },
    For {
        var: String,
        start: Expr,
        end: Expr,
        body: Block,
    },
    Break,
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Identifier(String),
    Integer(i64),
    Float(f64),
    String(String),
    Array(Vec<Expr>),
    Tensor(Vec<Expr>),
    Operation {
        op: String,
        args: Vec<Expr>,
    },
    StructLiteral {
        name: String,
        fields: Vec<FieldInit>,
    },
    FieldAccess {
        expr: Box<Expr>,
        field: String,
    },
}

#[derive(Debug, Clone)]
pub struct FieldInit {
    pub name: String,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    String,
    Tensor,
    Array(Box<Type>),
    Stream(Box<Type>),
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
    Struct(String),
}
