#include "parser.h"


/**
 * Print all tokens in the parser.
 */

void GAParser::PrintTokens() 
{
    for (const auto &tok : tokens) 
        printf("Token: Type=%d, Text='%s', Line=%u, Column=%u\n", tok.type, tok.text.c_str(), tok.line, tok.column);
}