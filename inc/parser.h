#pragma once

#include "common.h"
#include "lexer.h"

using namespace std;

enum NodeType
{
    NODE_PROG,
    NODE_FUNC_DECL,
    NODE_VAR_DECL,
    NODE_BLOCK,
    NODE_EXPR,
    NODE_ASSIGN,
    NODE_IF,
    NODE_FOR,
    NODE_RET,
    NODE_LIT,
    NODE_UNOP,
    NODE_BINOP,
    NODE_CALL,
    NODE_IDX,
    NODE_IDENT
};

enum Type
{
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_CGAVEC,
    TYPE_VOID,
    TYPE_UNKNOWN
};

enum SymbolKind
{
    SYM_VAR,
    SYM_PARAM,
    SYM_FUNC,
    SYM_GLOBAL
};

struct Symbol 
{
    string name;
    SymbolKind kind;
    Type type;
    uint32_t scopeLevel;
};

struct ASTNode
{
    NodeType nodeType;
    uint32_t l;
    uint32_t c;
    uint32_t nodeId = 0;
    string text;
    Type type;
    
    vector<ASTNode> children;

    ASTNode() : nodeType(NODE_PROG), l(0), c(0) {}

    ASTNode(NodeType t, Token &token) : 
        nodeType(t),
        l(token.line), 
        c(token.column), 
        text(token.text)
    {
    }

    ASTNode(NodeType t, TokenType typeIn, string &txt) : 
        nodeType(t), 
        l(0), 
        c(0), 
        text(txt) 
    {
        switch (typeIn)
        {
            case TOKEN_KW_INT:

                type = TYPE_INT;
                break;

            case TOKEN_KW_FLOAT:

                type = TYPE_FLOAT;
                break;

            case TOKEN_KW_CGAVEC:

                type = TYPE_CGAVEC;
                break;

            default:

                type = TYPE_UNKNOWN;
                break;
        }
    }
};

struct GAParser
{
    vector<Token> tokens;
    map<string, Symbol> symbolTable;

    size_t current;
    ASTNode root;

    GAParser() : current(0) {}
    
    void PrintTokens();
    void PrintAST();
    void PrintSymbolTable();

    void GenerateAST();
    void BuildSymbolTable();

    void ParseFuncDecl(ASTNode &parent);
    void ParseVarDecl(ASTNode &parent);
    void ParseBlock(ASTNode &parent);
    void ParseExpr(ASTNode &parent);
    void ParseTerm(ASTNode &parent);
    void ParseForLoop(ASTNode &parent);
    void ParseParams(ASTNode &parent);
    void ParseCallExpr(ASTNode &parent);
    void ParseAssignment(ASTNode &parent);

    void EmitASM(string &filename);
};