#include "parser.h"
#include <functional>
#include <unordered_map>
#include <sstream>

map<NodeType, string> nodeTypeMap = 
{
    { NODE_PROG,        "Program" },
    { NODE_FUNC_DECL,   "FunctionDecl" },
    { NODE_VAR_DECL,    "VarDecl" },
    { NODE_BLOCK,       "Block" },
    { NODE_EXPR,        "Expr" },
    { NODE_ASSIGN,      "Assign" },
    { NODE_IF,          "If" },
    { NODE_FOR,         "For" },
    { NODE_RET,         "Return" },
    { NODE_LIT,         "Literal" },
    { NODE_UNOP,        "UnaryOp" },
    { NODE_BINOP,       "BinaryOp" },
    { NODE_CALL,        "Call" },
    { NODE_IDX,         "Index" },
    { NODE_IDENT,       "Identifier" }
};

map<Type, string> typeMap = 
{
    { TYPE_INT,     "int" },
    { TYPE_FLOAT,   "float" },
    { TYPE_CGAVEC,  "cgavec" },
    { TYPE_VOID,    "void" },
    { TYPE_UNKNOWN, "unknown" }
};

map<SymbolKind, string> symbolKindMap = 
{
    { SYM_VAR,      "var" },
    { SYM_PARAM,    "param" },
    { SYM_FUNC,     "func" },
    { SYM_GLOBAL,   "global" }
};

/**
 * GAParser::PrintTokens - Print all tokens in the parser.
 */

void GAParser::PrintTokens() 
{
    printf("\n==============\n");
    printf("Tokens:\n");
    printf("==============\n");
    
    for (const Token &tok : tokens)
        printf("Token: Type=%d, Text='%s', Line=%u, Column=%u\n",
            tok.type, tok.text.c_str(), tok.line, tok.column);
}

/**
 * GAParser::PrintAST - Print an AST.
 */

void GAParser::PrintAST() 
{
    printf("\n==============\n");
    printf("AST:\n");
    printf("==============\n");

    vector<pair<ASTNode, uint32_t>> stack;
    stack.push_back({root, 0});

    while (!stack.empty())
    {
        ASTNode cur     = stack.back().first;
        uint32_t depth  = stack.back().second;

        stack.pop_back();

        printf("%*sNode: type=%s, txt='%s', l=%u, c=%u\n", 
            depth * 2, "", nodeTypeMap[cur.nodeType].c_str(), cur.text.c_str(), cur.l, cur.c);

        for (uint32_t i = cur.children.size(); i-- > 0;)
            stack.push_back({ cur.children[i], depth + 1 });
    }
}

/**
 * GAParser::PrintSymbolTable - Print the symbol table.
 */

void GAParser::PrintSymbolTable()
{
    printf("\n==============\n");
    printf("Symbols:\n");
    printf("==============\n");

    for (const auto &entry : symbolTable)
    {
        const Symbol &sym = entry.second;

        printf("Symbol: Name='%s', Kind=%s, Type=%s, ScopeLevel=%u\n",
            sym.name.c_str(), symbolKindMap[sym.kind].c_str(), typeMap[sym.type].c_str(), sym.scopeLevel);
    }
}

/**
 * GAParser::GenerateAST - Generate the AST from the parsed tokens.
 */

void GAParser::GenerateAST()
{
    root = ASTNode();    

    while (current < tokens.size())
    {
        if (tokens[current].type == TOKEN_KW_INT || 
            tokens[current].type == TOKEN_KW_FLOAT ||
            tokens[current].type == TOKEN_KW_CGAVEC)
        {
            ParseVarDecl(root);
            continue;
        }

        if (tokens[current].type == TOKEN_EOF)
            break;

        if (tokens[current].type == TOKEN_OP_EQ)
        {
            ParseAssignment(root);
        }
    }
}

/**
 * IsTypeToken - Check if a token represents a type and output the type.
 * 
 * @param t         [in]    Token to check.
 * @param outType   [out]   Output type if token is a type.
 */

static bool IsTypeToken(const Token &t, Type &outType)
{
    if (t.type == TOKEN_IDENTIFIER)
    {
        if (t.text == "int") 
        {
            outType = TYPE_INT; 
            return true; 
        }

        if (t.text == "float") 
        {
            outType = TYPE_FLOAT; 
            return true; 
        }

        if (t.text == "cgavec") 
        {
            outType = TYPE_CGAVEC; 
            return true; 
        }

        return false;
    }

    if (t.type == TOKEN_KW_CGAVEC)
    {
        outType = TYPE_CGAVEC;
        return true;
    }

    return false;
}

/**
 * IsBinaryOp - Check if a token type is a binary operator.
 * 
 * @param t     [in]    Token type to check.
 * 
 * @return      true if binary operator, false otherwise.
 */

static inline bool IsBinaryOp(TokenType t)
{
    return t == TOKEN_OP_PLUS || t == TOKEN_OP_MINUS || t == TOKEN_OP_STAR || t == TOKEN_OP_SLASH;
}

/**
 * IsAddSub - Check if a token type is addition or subtraction.
 * 
 * @param t     [in]    Token type to check.
 * 
 * @return      true if addition or subtraction, false otherwise.
 */

static inline bool IsAddSub(TokenType t)
{
    return t == TOKEN_OP_PLUS || t == TOKEN_OP_MINUS;
}

/**
 * IsMulDiv - Check if a token type is multiplication or division.
 * 
 * @param t     [in]    Token type to check.
 * 
 * @return      true if multiplication or division, false otherwise.
 */

static inline bool IsMulDiv(TokenType t)
{
    return t == TOKEN_OP_STAR || t == TOKEN_OP_SLASH;
}
  
/**
 * IsIdentOrLiteral - Check if a token type is an identifier or literal.
 * 
 * @param t     [in]    Token type to check.
 * 
 * @return      true if identifier or literal, false otherwise.
 */

static inline bool IsIdentOrLiteral(TokenType t)
{
    return t == TOKEN_IDENTIFIER || t == TOKEN_INT_LITERAL || t == TOKEN_FLOAT_LITERAL;
}

/**
 * GAParser::ParseAssignment - Parse an assignment statement.
 * 
 * @param parent    [in/out]    Parent AST node to attach the assignment to.
 */

void GAParser::ParseAssignment(ASTNode &parent)
{
    if (tokens[current + 1].type != TOKEN_OP_ASSIGN)
    {
        assert(!"Expected '=' in assignment");
        exit(1);
    }

    ASTNode assign(NODE_ASSIGN, tokens[current + 1]);
    ASTNode lhs(NODE_IDENT, tokens[current]);
    assign.children.push_back(lhs);
    current += 2;

    if (current + 2 < tokens.size() &&
        IsIdentOrLiteral(tokens[current].type) &&
        tokens[current + 1].type == TOKEN_SEMICOLON)
    {
        NodeType type = tokens[current].type == TOKEN_IDENTIFIER ? NODE_IDENT : NODE_LIT;

        ASTNode rhs(type, tokens[current]);
        assign.children.push_back(rhs);
        current += 2;
    }
    else
    {
        ParseExpr(assign);
    }


    parent.children.push_back(assign);
}

/**
 * GAParser::ParseFuncDecl -Parse a function declaration.
 * 
 * @param parent    [in/out]    Parent AST node to attach the function declaration to.
 */

void GAParser::ParseFuncDecl(ASTNode &parent)
{

}

/**
 * GAParser::ParseVarDecl - Parse a variable declaration.
 * 
 * @param parent    [in/out]    Parent AST node to attach the variable declaration to.
 */

void GAParser::ParseVarDecl(ASTNode &parent)
{
    TokenType type = tokens[current].type;

    if (type == TOKEN_KW_INT || type == TOKEN_KW_FLOAT || type == TOKEN_KW_CGAVEC) 
    {
        if (current + 1 < tokens.size() && tokens[current + 1].type == TOKEN_IDENTIFIER)
        {
            ASTNode varDecl(NODE_VAR_DECL, type, tokens[current + 1].text);
            parent.children.push_back(varDecl);

            if (current + 2 < tokens.size())
            {
                if (tokens[current + 2].type == TOKEN_SEMICOLON)
                {
                    current = current + 3;
                    return;
                }

                if (tokens[current + 2].type == TOKEN_OP_ASSIGN)
                {
                    current = current + 1;
                    ParseAssignment(parent);
                    return;
                }
            }
        }
        else
        {
            assert(!"Expected identifier after type in variable declaration");
            return;
        }
    }
}

/**
 * GAParser::ParseBlock - Parse a block of code.
 * 
 * @param parent    [in/out]    Parent AST node to attach the block to.
 */

void GAParser::ParseBlock(ASTNode &parent)
{
    
}

/**
 * GAParser::ParseExpression - Parse an expression.
 * 
 * @param parent    [in/out]    Parent AST node to attach the expression to.
 */

void GAParser::ParseExpr(ASTNode &parent)
{
    if (tokens[current].type == TOKEN_LPAREN)
        current++;

    if (IsIdentOrLiteral(tokens[current].type))
    {
        if (current + 1 <= tokens.size() && tokens[current + 1].type == TOKEN_SEMICOLON)
        {
            ASTNode ident(tokens[current].type == TOKEN_IDENTIFIER ? NODE_IDENT : NODE_LIT, 
                tokens[current]);
            
            current += 2;
            parent.children.push_back(ident);
            return;
        }

        if (IsAddSub(tokens[current + 1].type))
        {
            ASTNode binop(NODE_BINOP, tokens[current + 1]);
            ParseTerm(binop);
            binop.text = tokens[current].text;
            current++;
            ParseExpr(binop);
            parent.children.push_back(binop);
            return;
        }
    }
    else
    {
        assert(!"Unexpected non-literal/identifier encountered parsing expression");
        exit(1);
    }
}

/**
 * GAParser::ParseTerm - Parse a term in an expression.
 * 
 * @param parent    [in/out]    Parent AST node to attach the term to.
 */

void GAParser::ParseTerm(ASTNode &parent)
{
    if (tokens[current].type == TOKEN_LPAREN)
        current++;

    if (IsIdentOrLiteral(tokens[current].type))
    { 
        NodeType type = tokens[current].type == TOKEN_IDENTIFIER ? NODE_IDENT : NODE_LIT;
        ASTNode lhs(type, tokens[current]);

        if (IsMulDiv(tokens[current + 1].type))
        {
            ASTNode binop(NODE_BINOP, tokens[current + 1]);
            binop.children.push_back(lhs);

            binop.text  = tokens[current + 1].text;
            current     = current + 2;
            
            ParseTerm(binop);
            parent.children.push_back(binop);    
            return;
        }
        else
        {
            parent.children.push_back(lhs);
            current++;
            return;
        }
        
        current++;
    }
}

/**
 * GAParser::ParseForLoop - Parse a for loop.
 * 
 * @param parent    [in/out]    Parent AST node to attach the for loop to.
 */

void GAParser::ParseForLoop(ASTNode &parent)
{
    
}

/**
 * GAParser::ParseParams - Parse a param list.
 * 
 * @param parent    [in/out]    Parent AST node to attach the parameter list to.
 */

void GAParser::ParseParams(ASTNode &parent)
{
    
}

/**
 * GAParser::ParseCallExpr - Parse a function call.
 * 
 * @param parent    [in/out]    Parent AST node to attach the call to.
 */

void GAParser::ParseCallExpr(ASTNode &parent)
{
   
}

/**
 * GAParser::BuildSymbolTable - Build the symbol table from the AST.
 */

void GAParser::BuildSymbolTable()
{
    vector<pair<ASTNode, uint32_t>> stack;

    stack.push_back({root, 0});

    while (!stack.empty())
    {
        ASTNode cur     = stack.back().first;
        uint32_t depth  = stack.back().second;

        stack.pop_back();

        for (uint32_t i = cur.children.size(); i-- > 0;)
            stack.push_back({ cur.children[i], depth + 1 });

        if (cur.nodeType == NODE_VAR_DECL)
        {
            Symbol sym;
            sym.name        = cur.text;
            sym.kind        = SYM_VAR;
            sym.type        = cur.type;
            sym.scopeLevel  = depth;

            symbolTable[sym.name] = sym;
        }
    }
}


/**
 * IsNumber - Check if a type is numeric (int or float).
 * 
 * @param t     [in]    Type to check.
 * 
 * @return      true if numeric, false otherwise.
 */

static bool IsNumeric(Type t)
{
    return t == TYPE_INT || t == TYPE_FLOAT;
}

/**
 * PromoteNumberic - Promote two numeric types to a common type.
 */

static Type PromoteNumeric(Type a, Type b)
{
    return TYPE_UNKNOWN;
}

/**
 * GAParser::EmitASM - Emit assembly code from the AST.
 * 
 * @param filename  [in]    Output filename.
 */

void GAParser::EmitASM(string &filename)
{
    FILE *pFile = fopen(filename.c_str(), "w");
    
    if (!pFile)
    {
        printf("Error: Unable to open output file '%s' for writing\n", filename.c_str());
        return;
    }

    fprintf(pFile, "section .data\n");
    fprintf(pFile, "fmt: db \"%%d\", 10, 0\n");

    for (const auto &entry : symbolTable)
    {
        const Symbol &sym = entry.second;

        if (sym.kind == SYM_VAR && sym.type == TYPE_INT)
        {
            fprintf(pFile, "%s: dq 0\n", sym.name.c_str());
        }
    }

    fprintf(pFile, "section .text\n");
    fprintf(pFile, "global main\n");
    fprintf(pFile, "extern printf\n");
    
    fprintf(pFile, "main:\n");

    vector<pair<ASTNode, uint32_t>> stack;
    stack.push_back({root, 0});

    while (!stack.empty())
    {
        ASTNode cur     = stack.back().first;
        uint32_t depth  = stack.back().second;

        stack.pop_back();

        for (uint32_t i = cur.children.size(); i-- > 0;)
            stack.push_back({ cur.children[i], depth + 1 });

        if (cur.nodeType == NODE_ASSIGN)
        {
            const ASTNode &lhs = cur.children[0];
            const ASTNode &rhs = cur.children[1];

            if (lhs.nodeType == NODE_IDENT && rhs.nodeType == NODE_LIT)
            {
                fprintf(pFile, "    mov qword [%s], %s\n", lhs.text.c_str(), rhs.text.c_str());
            }
            else if (rhs.nodeType == NODE_BINOP)
            {
                const ASTNode &binopLHS = rhs.children[0];
                const ASTNode &binopRHS = rhs.children[1];

                if (rhs.text == "+")
                {
                    fprintf(pFile, "    mov rax, [%s]\n", binopLHS.text.c_str());
                    fprintf(pFile, "    add rax, [%s]\n", binopRHS.text.c_str());
                    fprintf(pFile, "    mov [%s], rax\n", lhs.text.c_str());
                }
                else if (rhs.text == "-")
                {
                    fprintf(pFile, "    mov rax, %s\n", binopLHS.text.c_str());
                    fprintf(pFile, "    sub rax, %s\n", binopRHS.text.c_str());
                    fprintf(pFile, "    mov qword [%s], rax\n", lhs.text.c_str());
                }
                else if (rhs.text == "*")
                {
                    fprintf(pFile, "    mov rax, %s\n", binopLHS.text.c_str());
                    fprintf(pFile, "    imul rax, %s\n", binopRHS.text.c_str());
                    fprintf(pFile, "    mov qword [%s], rax\n", lhs.text.c_str());
                }
                else if (rhs.text == "/")
                {
                    fprintf(pFile, "    mov rax, %s\n", binopLHS.text.c_str());
                    fprintf(pFile, "    cqo\n");
                    fprintf(pFile, "    idiv qword %s\n", binopRHS.text.c_str());
                    fprintf(pFile, "    mov qword [%s], rax\n", lhs.text.c_str());
                }
            }
                
        }
    }

    fprintf(pFile, "    sub rsp, 8\n");
    fprintf(pFile, "    mov rdi, fmt\n");
    fprintf(pFile, "    mov rsi, [c]\n");
    fprintf(pFile, "    xor rax, rax\n");
    fprintf(pFile, "    call printf\n");
    fprintf(pFile, "    add rsp, 8\n");
    fprintf(pFile, "    mov eax, 0\n");
    fprintf(pFile, "    ret\n");

    fclose(pFile);
}