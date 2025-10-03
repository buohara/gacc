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
    NODE_IF,
    NODE_FOR,
    NODE_RET,
    NODE_LIT,
    NODE_UNOP,
    NODE_BINOP,
    NODE_CALL,
    NODE_IDX
};

enum Types
{
    TYPE_INT,
    TYPE_FLOAT32,
    TYPE_FLOAT64,
    TYPE_VOID,
    TYPE_CGAVEC,
    TYPE_UNKNOWN
};

struct ASTNode
{
    NodeType type;
    uint32_t line;
    uint32_t column;

    string text;
    TokenType tokenType;

    vector<ASTNode> children;

    ASTNode() : type(NODE_PROG), line(0), column(0), tokenType(TOKEN_UNKNOWN) {}
    ASTNode(NodeType t, uint32_t l, uint32_t c) : type(t), line(l), column(c), tokenType(TOKEN_UNKNOWN) {}
};

struct GAParser
{
    vector<Token> tokens;
    vector<pair<string, Types>> symbolTable;

    size_t current;

    ASTNode root;

    GAParser() : current(0) {}
    
    void PrintTokens();
    void PrintAST();
    void PrintSymbolTable();

    void GenerateAST();

    void ParseFuncDecl(ASTNode &parent);
    void ParseVarDecl(ASTNode &parent);
    void ParseBlock(ASTNode &parent);
    void ParseBinaryOp(ASTNode &parent);
    void ParseExpression(ASTNode &parent);
    void ParseForLoop(ASTNode &parent);
    void ParseParams(ASTNode &parent);
    void ParseCallExpr(ASTNode &parent);
};