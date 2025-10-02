#pragma once

#include "common.h"
#include "lexer.h"

struct GAParser
{
    vector<Token> tokens;
    size_t current;

    GAParser() : current(0) {}

    void PrintTokens();

    // Token peek() 
    // {
    //     if (current < tokens.size()) 
    //     {
    //         return tokens[current];
    //     }
    //     return Token{TOKEN_EOF, "", 0, 0};
    // }

    // Token advance() 
    // {
    //     if (current < tokens.size()) 
    //     {
    //         return tokens[current++];
    //     }
    //     return Token{TOKEN_EOF, "", 0, 0};
    // }

    // bool match(TokenType type) 
    // {
    //     if (peek().type == type) 
    //     {
    //         advance();
    //         return true;
    //     }
    //     return false;
    // }
};