#include "parser.h"

static bool IsTypeToken(const Token &t, Types &outType)
{
    if (t.type == TOKEN_IDENTIFIER)
    {
        if (t.text == "int") { outType = TYPE_INT; return true; }
        if (t.text == "float32") { outType = TYPE_FLOAT32; return true; }
        if (t.text == "float64") { outType = TYPE_FLOAT64; return true; }
        if (t.text == "void") { outType = TYPE_VOID; return true; }
        if (t.text == "cgavec") { outType = TYPE_CGAVEC; return true; }

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
 * GAParser::PrintTokens - Print all tokens in the parser.
 */

void GAParser::PrintTokens() 
{
    for (const Token &tok : tokens)
        printf("Token: Type=%d, Text='%s', Line=%u, Column=%u\n",
            tok.type, tok.text.c_str(), tok.line, tok.column);
}

/**
 * GAParser::ResolveNames - Resolve names in the AST and link to symbol table.
 */

void GAParser::ResolveNames()
{
    vector<map<string,int>> scopeStack;
    scopeStack.push_back(map<string,int>());

    for (size_t i = 0; i < symbolTable.size(); ++i)
    {
        if (symbolTable[i].scopeLevel == 0)
            scopeStack[0][symbolTable[i].name] = (int)i;
    }

    struct PFrame { vector<int> path; bool entered; };

    vector<PFrame> pstk;
    pstk.push_back(PFrame());
    pstk.back().path.clear();
    pstk.back().entered = false;

    while (!pstk.empty())
    {
        PFrame pf = pstk.back();
        pstk.pop_back();

        ASTNode &n = GetNodeByPath(pf.path);

        if (!pf.entered)
        {
            if (n.type == NODE_FUNC_DECL)
            {
                scopeStack.push_back(map<string,int>());

                for (size_t ci = 0; ci < n.children.size(); ++ci)
                {
                    const ASTNode &c = n.children[ci];

                    if (c.type == NODE_VAR_DECL)
                    {
                        int foundId = -1;

                        for (size_t si = 0; si < symbolTable.size(); ++si)
                        {
                            if (symbolTable[si].name == c.text && symbolTable[si].scopeLevel == 1)
                            {
                                foundId = (int)si;
                                break;
                            }
                        }

                        if (foundId >= 0)
                            scopeStack.back()[c.text] = foundId;
                    }
                }
            }

            if (n.type == NODE_BLOCK)
                scopeStack.push_back(map<string,int>());

            if (n.type == NODE_VAR_DECL)
            {
                int foundId = -1;

                for (size_t si = 0; si < symbolTable.size(); ++si)
                {
                    if (symbolTable[si].name == n.text && symbolTable[si].scopeLevel >= 1)
                    {
                        foundId = (int)si;
                        break;
                    }
                }

                if (foundId >= 0)
                    scopeStack.back()[n.text] = foundId;
            }

            if (n.type == NODE_EXPR || n.type == NODE_IDX || n.type == NODE_CALL)
            {
                if (!n.text.empty())
                {
                    int resolved = -1;

                    for (int si = (int)scopeStack.size() - 1; si >= 0; --si)
                    {
                        map<string,int> &m = scopeStack[si];
                        map<string,int>::iterator it = m.find(n.text);
                        
                        if (it != m.end())
                        {
                            resolved = it->second;
                            break;
                        }
                    }

                    if (resolved >= 0)
                        n.symbolId = resolved;
                    else
                        PushDiagnostic(n.line, n.column, string("unresolved identifier '") + n.text + "'", true);
                }
            }

            pstk.push_back(PFrame());
            pstk.back().path = pf.path;
            pstk.back().entered = true;

            for (int i = (int)n.children.size() - 1; i >= 0; --i)
            {
                PFrame child;
                child.path = pf.path;
                child.path.push_back(i);
                child.entered = false;
                pstk.push_back(child);
            }
        }
        else
        {
            if (n.type == NODE_BLOCK || n.type == NODE_FUNC_DECL)
            {
                if (!scopeStack.empty())
                    scopeStack.pop_back();
            }
        }
    }
}

/**
 * GAParser::GetNodeByPath - Get a reference to an AST node by its path.
 */

ASTNode &GAParser::GetNodeByPath(const vector<int> &path)
{
    ASTNode *cur = &root;
    for (size_t i = 0; i < path.size(); ++i)
    {
        int idx = path[i];
        if (idx < 0 || idx >= (int)cur->children.size())
            return *cur;

        cur = &cur->children[idx];
    }

    return *cur;
}

/**
 * GAParser::PushDiagnostic - Push a diagnostic message.
 */

void GAParser::PushDiagnostic(uint32_t line, uint32_t column, const string &msg, bool isError)
{
    Diagnostic d;

    d.line      = line;
    d.column    = column;
    d.message   = msg;
    d.isError   = isError;
    diagnostics.push_back(std::move(d));
}

/**
 * GAParser::PrintDiagnostics - Print all diagnostics.
 */

void GAParser::PrintDiagnostics()
{
    for (const Diagnostic &d : diagnostics)
    {
        printf("%s at %u:%u: %s\n", d.isError ? "error" : "warning", d.line, d.column, d.message.c_str());
    }
}

/**
 * GAParser::HasErrors - Check if there are any error diagnostics.
 */

bool GAParser::HasErrors()
{
    for (const Diagnostic &d : diagnostics)
        if (d.isError) return true;

    return false;
}

/**
 * GAParser::PrintAST - Print an AST.
 */

void GAParser::PrintAST() 
{
    struct StackItem
    {
        ASTNode *node;
        int depth;
        bool isLast;
    };

    vector<StackItem> stack;

    printf("%s\n", "PROG");

    for (size_t i = 0; i < root.children.size(); ++i)
    {
        bool last = (i + 1 == root.children.size());
        stack.push_back({ &root.children[i], 0, last });
    }

    while (!stack.empty())
    {
        StackItem item = stack.back();
        stack.pop_back();

        for (int d = 0; d < item.depth; ++d)
        {
            if (d + 1 == item.depth)
                printf("%s", item.isLast ? "`--" : "|--");
            else
                printf("|   ");
        }

        if (item.depth == 0)
            printf("%s", item.isLast ? "`--" : "|--");

        const char *label = "";
        
        switch (item.node->type)
        {
            case NODE_FUNC_DECL: 
                
                label = "FUNC";
                break;
            
            case NODE_VAR_DECL:
                label = "VAR"; 
                break;
            
            case NODE_BLOCK:
            
                label = "BLOCK"; 
                break;
            
            case NODE_EXPR: 
            
                label = "EXPR";
                break;
            
            case NODE_IF: 
                
                label = "IF"; 
                break;
            
            case NODE_FOR: 
                
                label = "FOR"; 
                break;
            
            case NODE_RET:
            
                label = "RET"; 
                break;
            
            case NODE_LIT:
                
                label = "LIT"; 
                break;
            
            case NODE_UNOP:
            
                label = "UNOP"; 
                break;

            case NODE_BINOP: 
                
                label = "BINOP"; 
                break;
            
            case NODE_CALL:
            
                label = "CALL"; 
                break;

            case NODE_IDX: 
                
                label = "IDX"; 
                break;

            case NODE_PROG: 
            
                label = "PROG"; 
                break;

            default: 

                label = "NODE"; 
                break;
        }

        if (item.node->text.empty())
            printf("%s\n", label);
        else
            printf("%s(%s)\n", label, item.node->text.c_str());

        for (int i = (int)item.node->children.size() - 1; i >= 0; --i)
        {
            bool childLast = (i == (int)item.node->children.size() - 1);
            stack.push_back({ &item.node->children[i], item.depth + 1, childLast });
        }
    }
}

/**
 * GAParser::PrintSymbolTable - Print the symbol table.
 */

void GAParser::PrintSymbolTable()
{
    printf("Symbol Table:\n");

    for (size_t i = 0; i < symbolTable.size(); ++i)
    {
        const Symbol &sym   = symbolTable[i];
        const char *kindStr = "VAR";

        if (sym.kind == SYM_PARAM)
            kindStr = "PARAM";
        
        if (sym.kind == SYM_FUNC)
            kindStr = "FUNC";

        if (sym.kind == SYM_GLOBAL)
            kindStr = "GLOBAL";

        const char *typeStr = "UNKNOWN";

        switch (sym.type)
        {
            case TYPE_INT:

                typeStr = "INT";
                break;

            case TYPE_FLOAT32:

                typeStr = "FLOAT32"; 
                break;

            case TYPE_FLOAT64:

                typeStr = "FLOAT64"; 
                break;

            case TYPE_VOID:

                typeStr = "VOID"; 
                break;

            case TYPE_CGAVEC:

                typeStr = "CGAVEC"; 
                break;

            default:

                typeStr = "UNKNOWN"; 
                break;
        }

        printf("  %zu: %s (%s) : %s [scope=%u] at %u:%u\n", i, sym.name.c_str(), 
            kindStr, typeStr, sym.scopeLevel, sym.declLine, sym.declCol);
    }
}

/**
 * GAParser::GenerateAST - Generate the AST from the parsed tokens.
 */

void GAParser::GenerateAST()
{
    while (current < tokens.size())
    {
        if (tokens[current].type == TOKEN_EOF)
            break;

        Token t = tokens[current];

        if (t.type == TOKEN_KW_FOR)
        {
            ParseForLoop(root);
            continue;
        }

        if (t.type == TOKEN_KW_CONST || t.type == TOKEN_KW_LET)
        {
            ParseVarDecl(root);
            continue;
        }

        if (t.type == TOKEN_LBRACE)
        {
            ParseBlock(root);
            continue;
        }

        Types tmpType;
        bool isTypeTok = IsTypeToken(t, tmpType);

        if (t.type == TOKEN_IDENTIFIER || isTypeTok)
        {
            bool isFunc = false;
            bool isTypedDecl = false;

            if (current + 1 < tokens.size())
            {
                if (tokens[current + 1].type == TOKEN_LPAREN)
                    isFunc = true;
                else if (tokens[current + 1].type == TOKEN_IDENTIFIER)
                {
                    if (current + 2 < tokens.size() && tokens[current + 2].type == TOKEN_LPAREN)
                        isFunc = true;
                    else
                        isTypedDecl = true;
                }
            }

            if (isTypedDecl)
            {
                ParseVarDecl(root);
                continue;
            }

            if (isFunc)
            {
                ParseFuncDecl(root);
                continue;
            }

            ParseExpression(root);
            continue;
        }

        ++current;
    }
}

/**
 * GAParser::ParseFuncDecl -Parse a function declaration.
 * 
 * @param parent    [in/out]    Parent AST node to attach the function declaration to.
 */

void GAParser::ParseFuncDecl(ASTNode &parent)
{
    if (current >= tokens.size())
        return;

    Types retType = TYPE_UNKNOWN;
    bool hasRetType = false;

    if (current + 2 < tokens.size())
    {
        Types tmp;

        if (IsTypeToken(tokens[current], tmp) && tokens[current + 1].type == TOKEN_IDENTIFIER && 
            tokens[current + 2].type == TOKEN_LPAREN)
        {
            hasRetType = true;
            retType = tmp;
        }
    }

    if (hasRetType || (current + 2 < tokens.size() &&
        tokens[current].type == TOKEN_IDENTIFIER && 
        tokens[current + 1].type == TOKEN_IDENTIFIER && 
        tokens[current + 2].type == TOKEN_LPAREN))
    {
        Token name = tokens[current + (hasRetType ? 1 : 1)];

        ASTNode fn(NODE_FUNC_DECL, name.line, name.column);
        fn.text = name.text;
        
        if (hasRetType)
            fn.declaredType = retType;

        current += (hasRetType ? 2 : 2);
        
        if (tokens[current].type == TOKEN_LPAREN)
            ++current;
        
        ParseParams(fn);
        
        if (current < tokens.size() && tokens[current].type == TOKEN_RPAREN)
            ++current;
        
        if (current < tokens.size() && tokens[current].type == TOKEN_LBRACE)
            ParseBlock(fn);
        else if (current < tokens.size() && tokens[current].type == TOKEN_SEMICOLON)
            ++current;
        
        parent.children.push_back(std::move(fn));
        
        return;
    }

    if (tokens[current].type == TOKEN_IDENTIFIER && 
        current + 1 < tokens.size() && 
        tokens[current + 1].type == TOKEN_LPAREN)
    {
        Token name = tokens[current];
        ASTNode fn(NODE_FUNC_DECL, name.line, name.column);
        fn.text = name.text;
        ++current;

        if (tokens[current].type == TOKEN_LPAREN)
            ++current;

        ParseParams(fn);

        if (current < tokens.size() && tokens[current].type == TOKEN_RPAREN)
            ++current;

        if (current < tokens.size() && tokens[current].type == TOKEN_LBRACE)
            ParseBlock(fn);
        else if (current < tokens.size() && tokens[current].type == TOKEN_SEMICOLON)
            ++current;

        parent.children.push_back(std::move(fn));

        return;
    }
}

/**
 * GAParser::ParseVarDecl - Parse a variable declaration.
 * 
 * @param parent    [in/out]    Parent AST node to attach the variable declaration to.
 */

void GAParser::ParseVarDecl(ASTNode &parent)
{
    if (current >= tokens.size())
        return;

    Types leadingType = TYPE_UNKNOWN;

    if (tokens[current].type == TOKEN_KW_CONST || tokens[current].type == TOKEN_KW_LET)
    {
        ++current;
    }
        else if ((tokens[current].type == TOKEN_IDENTIFIER || tokens[current].type == TOKEN_KW_CGAVEC) &&
                 current + 1 < tokens.size() &&
                 tokens[current + 1].type == TOKEN_IDENTIFIER)
    {
        string tkn = tokens[current].text;

        if (tkn == "int")
            leadingType = TYPE_INT;
        else if (tkn == "float32")
            leadingType = TYPE_FLOAT32;
        else if (tkn == "float64")
            leadingType = TYPE_FLOAT64;
        else if (tkn == "void")
            leadingType = TYPE_VOID;
        else if (tkn == "cgavec")
            leadingType = TYPE_CGAVEC;

        ++current;
    }

        if (leadingType == TYPE_UNKNOWN && current + 1 < tokens.size() &&
            (tokens[current].type == TOKEN_IDENTIFIER || tokens[current].type == TOKEN_KW_CGAVEC) &&
            tokens[current + 1].type == TOKEN_IDENTIFIER)
        {
            string tkn2 = tokens[current].text;
            
            if (tkn2 == "int") 
                leadingType = TYPE_INT;
            else if (tkn2 == "float32") 
                leadingType = TYPE_FLOAT32;
            else if (tkn2 == "float64") 
                leadingType = TYPE_FLOAT64;
            else if (tkn2 == "void")
                leadingType = TYPE_VOID;
            else if (tkn2 == "cgavec") 
                leadingType = TYPE_CGAVEC;

            ++current;
        }

    if (current >= tokens.size())
        return;

    Token name = tokens[current];
    ASTNode vd(NODE_VAR_DECL, name.line, name.column);

    if (name.type == TOKEN_IDENTIFIER)
    {
        vd.text = name.text;

        if (leadingType != TYPE_UNKNOWN)
            vd.declaredType = leadingType;

        ++current;
    }

    if (current < tokens.size() && 
        tokens[current].type == TOKEN_LBRACK)
    {
        ++current;

        if (current < tokens.size())
        {
            Token idx = tokens[current];
            if (idx.type == TOKEN_INT_LITERAL || idx.type == TOKEN_IDENTIFIER)
            {
                ASTNode lit(NODE_LIT, idx.line, idx.column);
                lit.text = idx.text;
                vd.children.push_back(std::move(lit));
                ++current;
            }
        }

        if (current < tokens.size() && tokens[current].type == TOKEN_RBRACK)
            ++current;
    }

    if (current < tokens.size() && tokens[current].type == TOKEN_OP_ASSIGN)
    {
        ++current;
        ParseExpression(vd);
    }

    if (current < tokens.size() && tokens[current].type == TOKEN_SEMICOLON)
        ++current;

    parent.children.push_back(std::move(vd));
}

/**
 * GAParser::ParseBlock - Parse a block of code.
 * 
 * @param parent    [in/out]    Parent AST node to attach the block to.
 */

void GAParser::ParseBlock(ASTNode &parent)
{
    if (current >= tokens.size())
        return;

    Token lb = tokens[current];
    ASTNode block(NODE_BLOCK, lb.line, lb.column);

    if (tokens[current].type == TOKEN_LBRACE)
        ++current;

    while (current < tokens.size() && 
        tokens[current].type != TOKEN_RBRACE && 
        tokens[current].type != TOKEN_EOF)
    {
        Token t = tokens[current];
        Types tmp;
        bool isTypeTok = IsTypeToken(t, tmp);

        if (t.type == TOKEN_KW_CONST || t.type == TOKEN_KW_LET || isTypeTok)
        {
            ParseVarDecl(block);
            continue;
        }

        if (t.type == TOKEN_KW_FOR)
        {
            ParseForLoop(block);
            continue;
        }

        if (t.type == TOKEN_IDENTIFIER)
        {
            ParseExpression(block);
            continue;
        }

        ++current;
    }

    if (current < tokens.size() && tokens[current].type == TOKEN_RBRACE)
        ++current;

    parent.children.push_back(std::move(block));
}

/**
 * GAParser::ParseBinaryOp - Parse a binary op.
 * 
 * @param parent    [in/out]    Parent AST node to attach the binary operation to.
 */

void GAParser::ParseBinaryOp(ASTNode &parent)
{
    ParseExpression(parent);

    if (current < tokens.size())
    {
        TokenType tt = tokens[current].type;
        
        if (tt == TOKEN_OP_PLUS || 
            tt == TOKEN_OP_MINUS || 
            tt == TOKEN_OP_STAR || 
            tt == TOKEN_OP_SLASH)
        {
            Token op = tokens[current];
            ++current;
            ASTNode bin(NODE_BINOP, op.line, op.column);
            bin.text = op.text;
            
            if (!parent.children.empty())
            {
                ASTNode lhs = std::move(parent.children.back());
                parent.children.pop_back();
                bin.children.push_back(std::move(lhs));
            }
            
            ParseExpression(parent);
            
            if (!parent.children.empty())
            {
                ASTNode rhs = std::move(parent.children.back());
                parent.children.pop_back();
                bin.children.push_back(std::move(rhs));
            }
            
            parent.children.push_back(std::move(bin));
        }
    }
}

/**
 * GAParser::ParseExpression - Parse an expression.
 * 
 * @param parent    [in/out]    Parent AST node to attach the expression to.
 */

void GAParser::ParseExpression(ASTNode &parent)
{
    if (current >= tokens.size())
        return;

    Token t = tokens[current];

    if (t.type == TOKEN_INT_LITERAL ||
        t.type == TOKEN_FLOAT_LITERAL ||
        t.type == TOKEN_STRING_LITERAL)
    {
        ASTNode lit(NODE_LIT, t.line, t.column);
        lit.text = t.text;
        ++current;
        parent.children.push_back(std::move(lit));

        if (current < tokens.size() && tokens[current].type == TOKEN_SEMICOLON)
            ++current;
            
        return;
    }

    if (t.type == TOKEN_IDENTIFIER)
    {
        ASTNode id(NODE_EXPR, t.line, t.column);
        id.text = t.text;
        ++current;

        if (current < tokens.size() &&
            tokens[current].type == TOKEN_LPAREN)
        {
            ++current;
            ASTNode call(NODE_CALL, t.line, t.column);
            call.text = id.text;
            
            while (current < tokens.size() && 
                   tokens[current].type != TOKEN_RPAREN &&
                   tokens[current].type != TOKEN_EOF)
            {
                ParseExpression(call);
                if (current < tokens.size() && tokens[current].type == TOKEN_COMMA)
                    ++current;
            }

            if (current < tokens.size() && tokens[current].type == TOKEN_RPAREN)
                ++current;
            
            parent.children.push_back(std::move(call));
            
            if (current < tokens.size() && tokens[current].type == TOKEN_SEMICOLON)
                ++current;
            
                return;
        }

        if (current < tokens.size() && tokens[current].type == TOKEN_LBRACK)
        {
            ++current;
            ASTNode idx(NODE_IDX, t.line, t.column);
            idx.text = id.text;

            ParseExpression(idx);
            
            if (current < tokens.size() && tokens[current].type == TOKEN_RBRACK)
                ++current;
            
            parent.children.push_back(std::move(idx));
            
            if (current < tokens.size() && tokens[current].type == TOKEN_OP_ASSIGN)
            {
                ++current;
                ParseExpression(parent);
            }
            
            if (current < tokens.size() && tokens[current].type == TOKEN_SEMICOLON)
                ++current;
            
                return;
        }

        parent.children.push_back(std::move(id));
        
        if (current < tokens.size() && tokens[current].type == TOKEN_SEMICOLON)
            ++current;
        
        return;
    }

    ++current;
}

/**
 * GAParser::ParseForLoop - Parse a for loop.
 * 
 * @param parent    [in/out]    Parent AST node to attach the for loop to.
 */

void GAParser::ParseForLoop(ASTNode &parent)
{
    if (current >= tokens.size())
        return;

    Token f = tokens[current];
    ASTNode fl(NODE_FOR, f.line, f.column);
    ++current;

    if (current < tokens.size() && tokens[current].type == TOKEN_LPAREN)
        ++current;

    if (current + 1 < tokens.size() && 
        tokens[current].type == TOKEN_IDENTIFIER && 
        tokens[current + 1].type == TOKEN_IDENTIFIER)
    {
        Token typeTok = tokens[current];
        Token nameTok = tokens[current + 1];
        ASTNode vd(NODE_VAR_DECL, nameTok.line, nameTok.column);

        if (typeTok.text == "int")
            vd.declaredType = TYPE_INT;
        else if (typeTok.text == "float32")
            vd.declaredType = TYPE_FLOAT32;
        else if (typeTok.text == "float64")
            vd.declaredType = TYPE_FLOAT64;
        else if (typeTok.text == "void")
            vd.declaredType = TYPE_VOID;
        else if (typeTok.text == "cgavec")
            vd.declaredType = TYPE_CGAVEC;

        vd.text = nameTok.text;
        current += 2;
        
        if (current < tokens.size() && tokens[current].type == TOKEN_OP_ASSIGN)
        {
            ++current;
            ParseExpression(vd);
        }

        fl.children.push_back(std::move(vd));
    }
    else
    {
        ParseExpression(fl);
    }

    ParseExpression(fl);

    if (current < tokens.size() && tokens[current].type == TOKEN_SEMICOLON)
        ++current;
    
    ParseExpression(fl);

    if (current < tokens.size() && tokens[current].type == TOKEN_RPAREN)
        ++current;

    if (current < tokens.size() && tokens[current].type == TOKEN_LBRACE)
        ParseBlock(fl);

    parent.children.push_back(std::move(fl));
}

/**
 * GAParser::ParseParams - Parse a param list.
 * 
 * @param parent    [in/out]    Parent AST node to attach the parameter list to.
 */

void GAParser::ParseParams(ASTNode &parent)
{
    while (current < tokens.size() && 
           tokens[current].type != TOKEN_RPAREN &&
           tokens[current].type != TOKEN_EOF)
    {
        Token t = tokens[current];
    
        if (t.type == TOKEN_IDENTIFIER)
        {
            Types leadingType = TYPE_UNKNOWN;

            if (current + 1 < tokens.size() && tokens[current + 1].type == TOKEN_IDENTIFIER)
            {
                string tt = tokens[current].text;
                
                if (tt == "int")
                    leadingType = TYPE_INT;
                else if (tt == "float32")
                    leadingType = TYPE_FLOAT32;
                else if (tt == "float64")
                    leadingType = TYPE_FLOAT64;
                else if (tt == "void")
                    leadingType = TYPE_VOID;
                else if (tt == "cgavec")
                    leadingType = TYPE_CGAVEC;
                
                ++current;
            }

            Token name = tokens[current];
            ASTNode p(NODE_VAR_DECL, name.line, name.column);
            p.text = name.text;
            
            if (leadingType != TYPE_UNKNOWN)
                p.declaredType = leadingType;

            parent.children.push_back(std::move(p));
            ++current;
            
            continue;
        }

        if (t.type == TOKEN_COMMA)
        {
            ++current;
            continue;
        }

        ++current;
    }
}

/**
 * GAParser::ParseCallExpr - Parse a function call.
 * 
 * @param parent    [in/out]    Parent AST node to attach the call to.
 */

void GAParser::ParseCallExpr(ASTNode &parent)
{
    if (current >= tokens.size())
        return;

    Token id = tokens[current];
    ASTNode call(NODE_CALL, id.line, id.column);
    call.text = id.text;
    ++current;

    if (current < tokens.size() && tokens[current].type == TOKEN_LPAREN)
        ++current;
    
    while (current < tokens.size() && 
           tokens[current].type != TOKEN_RPAREN &&
           tokens[current].type != TOKEN_EOF)
    {
        ParseExpression(call);
    
        if (current < tokens.size() && tokens[current].type == TOKEN_COMMA)
            ++current;
    }

    if (current < tokens.size() && tokens[current].type == TOKEN_RPAREN)
        ++current;

    if (current < tokens.size() && tokens[current].type == TOKEN_SEMICOLON)
        ++current;
    
    parent.children.push_back(std::move(call));
}

/**
 * GAParser::BuildSymbolTable - Build the symbol table from the AST.
 */

void GAParser::BuildSymbolTable()
{
    symbolTable.clear();

    Symbol ext1;
    ext1.name           = "printf";
    ext1.kind           = SYM_FUNC;
    ext1.scopeLevel     = 0;
    ext1.declaredType   = TYPE_INT;
    ext1.type           = TYPE_INT;
    ext1.declLine       = 0;
    ext1.declCol        = 0;
    ext1.paramCount     = -1;

    symbolTable.push_back(ext1);

    Symbol ext2;
    ext2.name           = "cgarand";
    ext2.kind           = SYM_FUNC;
    ext2.scopeLevel     = 0;
    ext2.declaredType   = TYPE_CGAVEC;
    ext2.type           = TYPE_CGAVEC;
    ext2.declLine       = 0;
    ext2.declCol        = 0;
    ext2.paramCount     = 0;

    symbolTable.push_back(ext2);

    uint32_t nextId = 1;
    vector<vector<int>> stackPaths;
    stackPaths.push_back(vector<int>());

    while (!stackPaths.empty())
    {
        vector<int> path = stackPaths.back();
        stackPaths.pop_back();

        ASTNode &n = GetNodeByPath(path);
        n.nodeId = nextId++;

        for (int i = (int)n.children.size() - 1; i >= 0; --i)
        {
            vector<int> childPath = path;
            childPath.push_back(i);
            stackPaths.push_back(childPath);
        }
    }

    for (size_t i = 0; i < root.children.size(); ++i)
    {
        const ASTNode &n = root.children[i];

        if (n.type == NODE_FUNC_DECL)
        {
            Symbol f;
            f.name = n.text;
            f.kind = SYM_FUNC;
            f.declaredType = n.declaredType;

            if (f.declaredType != TYPE_UNKNOWN)
                f.type = f.declaredType;

            f.scopeLevel = 0;
            f.declLine = n.line;
            f.declCol = n.column;

            int pc = 0;
            for (const ASTNode &c : n.children)
            {
                if (c.type == NODE_VAR_DECL)
                    ++pc;
            }

            f.paramCount = pc;
            f.astNodeIdx = n.nodeId;

            for (const ASTNode &c : n.children)
            {
                if (c.type == NODE_VAR_DECL)
                {
                    f.paramTypes.push_back(c.declaredType);
                }
            }
            symbolTable.push_back(f);
        }

        if (n.type == NODE_VAR_DECL)
        {
            Symbol g;
            g.name           = n.text;
            g.kind           = SYM_GLOBAL;
            g.declaredType   = n.declaredType;

            if (g.declaredType != TYPE_UNKNOWN)
                g.type = g.declaredType;
            
            g.scopeLevel     = 0;
            g.declLine       = n.line;
            g.declCol        = n.column;
            g.astNodeIdx     = n.nodeId;

            symbolTable.push_back(g);
        }

        if ((n.type == NODE_IDX || n.type == NODE_EXPR) && !n.text.empty())
        {
            bool found = false;
            for (size_t si = 0; si < symbolTable.size(); ++si)
            {
                if (symbolTable[si].name == n.text)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                Symbol g2;
                g2.name           = n.text;
                g2.kind           = SYM_GLOBAL;
                g2.declaredType   = n.declaredType;

                if (g2.declaredType != TYPE_UNKNOWN)
                    g2.type = g2.declaredType;
                
                g2.scopeLevel     = 0;
                g2.declLine       = n.line;
                g2.declCol        = n.column;
                g2.astNodeIdx     = n.nodeId;

                symbolTable.push_back(g2);
            }
        }
    }

    for (size_t fi = 0; fi < root.children.size(); ++fi)
    {
        const ASTNode &fn = root.children[fi];
        if (fn.type != NODE_FUNC_DECL)
            continue;

        for (size_t ci = 0; ci < fn.children.size(); ++ci)
        {
            const ASTNode &c = fn.children[ci];
            if (c.type == NODE_VAR_DECL)
            {
                Symbol p;
                p.name           = c.text;
                p.kind           = SYM_PARAM;
                p.declaredType   = c.declaredType;

                if (p.declaredType != TYPE_UNKNOWN)
                    p.type = p.declaredType;

                p.scopeLevel     = 1;
                p.declLine       = c.line;
                p.declCol        = c.column;
                p.astNodeIdx     = c.nodeId;

                symbolTable.push_back(p);
                continue;
            }

            vector<vector<int>> stack;
            vector<unsigned> stackLevel;
            vector<int> startPath;
            startPath.push_back((int)fi);
            startPath.push_back((int)ci);
            stack.push_back(startPath);
            stackLevel.push_back(2);

            while (!stack.empty())
            {
                vector<int> path = stack.back();
                stack.pop_back();
                unsigned level = stackLevel.back();
                stackLevel.pop_back();

                ASTNode &node = GetNodeByPath(path);

                if (node.type == NODE_VAR_DECL)
                {
                    Symbol s;
                    s.name          = node.text;
                    s.kind          = (level == 0) ? SYM_GLOBAL : SYM_VAR;
                    s.declaredType  = node.declaredType;

                    if (s.declaredType != TYPE_UNKNOWN)
                        s.type = s.declaredType;
                    
                    s.scopeLevel    = level;
                    s.declLine      = node.line;
                    s.declCol       = node.column;
                    s.astNodeIdx    = node.nodeId;

                    symbolTable.push_back(s);

                    continue;
                }

                unsigned nextLevel = level;
                if (node.type == NODE_BLOCK && nextLevel < 2u)
                    nextLevel = 2u;

                for (int i = (int)node.children.size() - 1; i >= 0; --i)
                {
                    vector<int> childPath = path;
                    childPath.push_back(i);
                    stack.push_back(childPath);
                    stackLevel.push_back(nextLevel);
                }
            }
        }
    }

    vector<vector<int>> todo;
    todo.push_back(vector<int>());

    while (!todo.empty())
    {
        vector<int> path = todo.back();
        todo.pop_back();

        ASTNode &n = GetNodeByPath(path);

        if ((n.type == NODE_IDX || n.type == NODE_EXPR) && !n.text.empty())
        {
            bool found = false;
            for (size_t si = 0; si < symbolTable.size(); ++si)
            {
                if (symbolTable[si].name == n.text)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                Symbol g3;
                g3.name          = n.text;
                g3.kind          = SYM_GLOBAL;
                g3.declaredType  = n.declaredType;

                if (g3.declaredType != TYPE_UNKNOWN)
                    g3.type = g3.declaredType;

                g3.scopeLevel    = 0;
                g3.declLine      = n.line;
                g3.declCol       = n.column;
                g3.astNodeIdx    = n.nodeId;

                symbolTable.push_back(g3);
            }
        }

        for (int i = (int)n.children.size() - 1; i >= 0; --i)
        {
            vector<int> childPath = path;
            childPath.push_back(i);
            todo.push_back(childPath);
        }
    }
}
/**
 * TypeToString - Convert a type enum to a string.
 * 
 * @param t     [in]    Type to convert.
 * 
 * @return      String representation of the type.
 */

static const char *TypeToString(Types t)
{
    switch (t)
    {
        case TYPE_INT: 
        
            return "INT";
        
        case TYPE_FLOAT32: 
        
            return "FLOAT32";
        
        case TYPE_FLOAT64:
        
            return "FLOAT64";
        
        case TYPE_VOID:
        
            return "VOID";
        
        case TYPE_CGAVEC:
        
            return "CGAVEC";
        
        default: 
        
            return "UNKNOWN";
    }
}

/**
 * IsNumber - Check if a type is numeric (int or float).
 * 
 * @param t     [in]    Type to check.
 * 
 * @return      true if numeric, false otherwise.
 */

static bool IsNumeric(Types t)
{
    return t == TYPE_INT || t == TYPE_FLOAT32 || t == TYPE_FLOAT64;
}

/**
 * PromoteNumberic - Promote two numeric types to a common type.
 */

static Types PromoteNumeric(Types a, Types b)
{
    if (a == TYPE_FLOAT64 || b == TYPE_FLOAT64) 
        return TYPE_FLOAT64;
    
    if (a == TYPE_FLOAT32 || b == TYPE_FLOAT32) 
        return TYPE_FLOAT32;
    
    if (a == TYPE_INT && b == TYPE_INT) 
        return TYPE_INT;

    return TYPE_UNKNOWN;
}


/**
 * GAParser::InferTypes - Traverse the AST in post-order and infer
 * types for symbols used in exprssions.
 */

void GAParser::InferTypes()
{
    const int maxIter = 8;

    for (int iter = 0; iter < maxIter; ++iter)
    {
        bool changed = false;

        struct Frame { vector<int> path; bool entered; };
        vector<Frame> stack;
        stack.push_back(Frame());
        stack.back().path.clear();
        stack.back().entered = false;

        while (!stack.empty())
        {
            Frame fr = stack.back();
            stack.pop_back();

            ASTNode &node = GetNodeByPath(fr.path);

            if (!fr.entered)
            {
                fr.entered = true;
                stack.push_back(fr);

                for (int i = (int)node.children.size() - 1; i >= 0; --i)
                {
                    Frame child;
                    child.path = fr.path;
                    child.path.push_back(i);
                    child.entered = false;
                    stack.push_back(child);
                }

                continue;
            }

            switch (node.type)
            {
                case NODE_LIT:
                {
                    if (!node.text.empty() && node.text.find('.') != string::npos)
                        node.inferredType = TYPE_FLOAT32;
                    else
                        node.inferredType = TYPE_INT;
                    break;
                }

                case NODE_EXPR:
                {
                    if (node.symbolId >= 0 && node.symbolId < (int)symbolTable.size())
                        node.inferredType = symbolTable[node.symbolId].type;
                    break;
                }

                case NODE_IDX:
                {
                    if (node.symbolId >= 0 && node.symbolId < (int)symbolTable.size())
                    {
                        Types container = symbolTable[node.symbolId].type;
                        if (container == TYPE_CGAVEC)
                            node.inferredType = TYPE_CGAVEC;
                        else
                            node.inferredType = TYPE_UNKNOWN;
                    }
                    break;
                }

                case NODE_CALL:
                {
                    if (node.symbolId >= 0 && node.symbolId < (int)symbolTable.size())
                    {
                        Symbol &sym = symbolTable[node.symbolId];
                        node.inferredType = sym.type;

                        int provided = 0;
                        for (const ASTNode &c : node.children)
                            ++provided;

                        if (sym.paramCount >= 0 && sym.paramCount != 0 && sym.paramCount != provided)
                        {
                            PushDiagnostic(node.line, node.column,
                                           string("call to '") + sym.name + "' expected " +
                                           to_string(sym.paramCount) + " args but got " + to_string(provided),
                                           true);
                        }
                    }
                    break;
                }

                case NODE_BINOP:
                {
                    if (node.children.size() >= 2)
                    {
                        Types L = node.children[0].inferredType;
                        Types R = node.children[1].inferredType;

                        if (IsNumeric(L) && IsNumeric(R))
                            node.inferredType = PromoteNumeric(L, R);
                        else if (L == TYPE_CGAVEC && R == TYPE_CGAVEC)
                            node.inferredType = TYPE_CGAVEC;
                        else
                            node.inferredType = TYPE_UNKNOWN;
                    }
                    break;
                }

                case NODE_VAR_DECL:
                {
                    bool hasInit = false;
                    Types init = TYPE_UNKNOWN;

                    auto isNumericLiteralString = [&](const string &s) -> bool
                    {
                        if (s.empty())
                            return false;
                        size_t i = 0;
                        if (s[0] == '+' || s[0] == '-')
                            i = 1;
                        bool seenDigit = false;
                        for (; i < s.size(); ++i)
                        {
                            if (isdigit((unsigned char)s[i]))
                            {
                                seenDigit = true;
                                continue;
                            }
                            if (s[i] == '.')
                                continue;
                            return false;
                        }
                        return seenDigit;
                    };

                    if (!node.children.empty())
                    {
                        if (node.children.size() >= 2)
                        {
                            hasInit = true;
                            init = node.children.back().inferredType;
                        }
                        else
                        {
                            const ASTNode &c0 = node.children[0];
                            if (c0.type == NODE_LIT && !isNumericLiteralString(c0.text))
                            {
                                hasInit = false;
                            }
                            else
                            {
                                hasInit = true;
                                init = c0.inferredType;
                            }
                        }
                    }

                    if (hasInit)
                    {
                        if (node.declaredType != TYPE_UNKNOWN)
                        {
                            if (!(IsNumeric(init) && IsNumeric(node.declaredType)) && init != node.declaredType)
                            {
                                PushDiagnostic(node.line, node.column,
                                               string("initializer type '") + TypeToString(init) +
                                               "' does not match declared type '" + TypeToString(node.declaredType) + "' for '" +
                                               node.text + "'",
                                               true);
                            }

                            for (Symbol &s : symbolTable)
                            {
                                if (s.astNodeIdx == node.nodeId)
                                {
                                    if (s.type != node.declaredType)
                                    {
                                        s.type = node.declaredType;
                                        changed = true;
                                    }
                                    break;
                                }
                            }
                        }
                        else
                        {
                            for (Symbol &s : symbolTable)
                            {
                                if (s.astNodeIdx == node.nodeId)
                                {
                                    if (s.type != init)
                                    {
                                        s.type = init;
                                        changed = true;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    break;
                }

                case NODE_RET:
                {
                    if (!node.children.empty())
                    {
                        Types rt            = node.children[0].inferredType;
                        node.inferredType   = rt;

                        vector<int> path;

                        struct SFrame 
                        { 
                            vector<int> p;
                        };

                        vector<SFrame> sstack;
                        sstack.push_back(SFrame());
                        bool foundFunc = false;
                        uint32_t funcNodeId = 0;

                        while (!sstack.empty() && !foundFunc)
                        {
                            SFrame cur = sstack.back();
                            sstack.pop_back();
                            ASTNode &ncur = GetNodeByPath(cur.p);
                            
                            if (ncur.nodeId == node.nodeId)
                            {
                                vector<int> up = cur.p;
                                while (!up.empty())
                                {
                                    ASTNode &maybe = GetNodeByPath(up);
                                    if (maybe.type == NODE_FUNC_DECL)
                                    {
                                        funcNodeId = maybe.nodeId;
                                        foundFunc = true;
                                        break;
                                    }
                                    up.pop_back();
                                }
                                break;
                            }

                            for (int i = (int)ncur.children.size() - 1; i >= 0; --i)
                            {
                                SFrame nf;
                                nf.p = cur.p;
                                nf.p.push_back(i);
                                sstack.push_back(nf);
                            }
                        }

                        if (foundFunc)
                        {
                            for (Symbol &s : symbolTable)
                            {
                                if (s.kind == SYM_FUNC && s.astNodeIdx == funcNodeId)
                                {
                                    if (s.declaredType == TYPE_UNKNOWN)
                                    {
                                        if (s.type != rt)
                                        {
                                            s.type = rt;
                                            changed = true;
                                        }
                                    }
                                    else
                                    {
                                        if (!(IsNumeric(rt) && IsNumeric(s.declaredType)) && rt != s.declaredType)
                                        {
                                            PushDiagnostic(node.line, node.column, string("return type '") +
                                                           TypeToString(rt) + "' does not match function declared type '" +
                                                           TypeToString(s.declaredType) + "'",
                                                           true);
                                        }
                                    }

                                    break;
                                }
                            }
                        }
                    }
                    break;
                }

                default:
                    break;
            }
        }

        if (!changed)
            break;
    }
}