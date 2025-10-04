#include "parser.h"

/**
 * Print all tokens in the parser.
 */

void GAParser::PrintTokens() 
{
    for (const Token &tok : tokens)
        printf("Token: Type=%d, Text='%s', Line=%u, Column=%u\n", tok.type, tok.text.c_str(), tok.line, tok.column);
}

/**
 * Print an AST.
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
 * Print the symbol table.
 */

void GAParser::PrintSymbolTable()
{
    printf("Symbol Table:\n");
    
    for (const auto &entry : symbolTable)
    {
        const string &name  = entry.first;
        Types type          = entry.second;
        const char *typeStr = "UNKNOWN";

        switch (type)
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

        printf("  %s : %s\n", name.c_str(), typeStr);
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

        if (t.type == TOKEN_IDENTIFIER)
        {
            bool isFunc = false;

            if (current + 2 < tokens.size())
            {
                if (tokens[current + 1].type == TOKEN_IDENTIFIER &&
                    tokens[current + 2].type == TOKEN_LPAREN)
                    isFunc = true;

                if (tokens[current + 1].type == TOKEN_LPAREN)
                    isFunc = true;
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

    if (current + 2 < tokens.size() &&
        tokens[current].type == TOKEN_IDENTIFIER && 
        tokens[current + 1].type == TOKEN_IDENTIFIER && 
        tokens[current + 2].type == TOKEN_LPAREN)
    {
        Token ret = tokens[current];
        Token name = tokens[current + 1];

        ASTNode fn(NODE_FUNC_DECL, name.line, name.column);
        fn.text = name.text;
        current += 2;
        
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

    Token kw = tokens[current];
    ++current;

    if (current < tokens.size() && tokens[current].type == TOKEN_IDENTIFIER)
        ++current;

    if (current >= tokens.size())
        return;

    Token name = tokens[current];
    ASTNode vd(NODE_VAR_DECL, name.line, name.column);

    if (name.type == TOKEN_IDENTIFIER)
    {
        vd.text = name.text;
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
        if (t.type == TOKEN_KW_CONST || t.type == TOKEN_KW_LET)
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
            if (current + 1 < tokens.size() && 
                tokens[current + 1].type == TOKEN_IDENTIFIER)
                ++current;

            Token name = tokens[current];
            ASTNode p(NODE_VAR_DECL, name.line, name.column);
            p.text = name.text;
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