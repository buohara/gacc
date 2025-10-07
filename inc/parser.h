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
    Types type            = TYPE_UNKNOWN;
    Types declaredType    = TYPE_UNKNOWN;
    uint32_t astNodeIdx   = 0;
    uint32_t scopeLevel   = 0;
    uint32_t declLine     = 0;
    uint32_t declCol      = 0;
    int32_t paramCount    = 0;

    vector<Types> paramTypes;

    bool isAddressable    = false;
    bool isConst          = false;
    bool declaredOnly     = false;
    // For array declarations, 0 means unknown/dynamic, otherwise static size.
    uint32_t arraySize    = 0;

    string loweredType;
};

struct ASTNode
{
    NodeType type;
    uint32_t line;
    uint32_t column;
    uint32_t nodeId = 0;

    string text;
    TokenType tokenType;

    vector<ASTNode> children;

    int32_t symbolId    = -1;
    Types declaredType  = TYPE_UNKNOWN;
    Types inferredType  = TYPE_UNKNOWN;

    ASTNode() : type(NODE_PROG), line(0), column(0), tokenType(TOKEN_UNKNOWN) {}
    ASTNode(NodeType t, uint32_t l, uint32_t c) : type(t), line(l), column(c), tokenType(TOKEN_UNKNOWN) {}
};

struct Diagnostic
    {
        uint32_t line;
        uint32_t column;
        string message;
        bool isError;
    };

struct GAParser
{
    vector<Token> tokens;
    vector<Symbol> symbolTable;
    vector<Diagnostic> diagnostics;

    size_t current;
    ASTNode root;

    GAParser() : current(0) {}
    
    void PrintTokens();
    void PrintAST();
    void PrintSymbolTable();

    void GenerateAST();
    void BuildSymbolTable();
    void ResolveNames();
    ASTNode &GetNodeByPath(const std::vector<int> &path);

    void ParseFuncDecl(ASTNode &parent);
    void ParseVarDecl(ASTNode &parent);
    void ParseBlock(ASTNode &parent);
    void ParseBinaryOp(ASTNode &parent);
    void ParseExpression(ASTNode &parent);
    void ParseForLoop(ASTNode &parent);
    void ParseParams(ASTNode &parent);
    void ParseCallExpr(ASTNode &parent);
    void InferTypes();
    void PushDiagnostic(uint32_t line, uint32_t column, const string &msg, bool isError=true);
    void PrintDiagnostics();
    bool HasErrors();
};