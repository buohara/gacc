#include "parser.h"
#include <functional>
#include <unordered_map>
#include <sstream>

/**
 * GAParser::LowerSSA - Emit a simple textual SSA-like IR to stdout.
 * This is a minimal lowering used when MLIR is not available. It prints
 * function signatures, allocs for arrays and cgavec, simple for-loops,
 * calls and assignments in a readable SSA-like form.
 */

void GAParser::LowerSSA()
{
    
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
    
}


/**
 * GAParser::PrintAST - Print an AST.
 */

void GAParser::PrintAST() 
{
    
}

/**
 * GAParser::PrintSymbolTable - Print the symbol table.
 */

void GAParser::PrintSymbolTable()
{
    
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
 * GAParser::InferTypes - Traverse the AST in post-order and infer
 * types for symbols used in exprssions.
 */

void GAParser::InferTypes()
{
   
}