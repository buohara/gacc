#include "lexer.h"

using namespace std;

map<string, TokenType> tokenMap =
{
	{"fn",      TOKEN_KW_FN},
	{"let",     TOKEN_KW_LET},
	{"const",   TOKEN_KW_CONST},
	{"type",    TOKEN_KW_TYPE},
	{"if",      TOKEN_KW_IF},
	{"else",    TOKEN_KW_ELSE},
	{"for",     TOKEN_KW_FOR},
	{"while",   TOKEN_KW_WHILE},
	{"return",  TOKEN_KW_RETURN},
	{"import",  TOKEN_KW_IMPORT},
	{"export",  TOKEN_KW_EXPORT},
	{"extern",  TOKEN_KW_EXTERN},
	{"mv",      TOKEN_KW_MV},
	{"cgavec",  TOKEN_KW_CGAVEC},
	{"grade",   TOKEN_KW_GRADE},
	{"struct",  TOKEN_KW_STRUCT},
	{"match",   TOKEN_KW_MATCH},
	{"using",   TOKEN_KW_USING},
	{"inline",  TOKEN_KW_INLINE},
	{"break",   TOKEN_KW_BREAK},
	{"continue",TOKEN_KW_CONTINUE},
};

/**
 * GetTokens
 *
 * Get tokens from input file.
 *
 * @param file      Input source file.
 * @param tokens    List of tokens to populate.
 */
void GetTokens(const string &file, vector<Token> &tokens)
{
    FILE *f;
    int64_t size;
    char *buffer;
    int64_t i;
    int line;
    int column;

    f = fopen(file.c_str(), "rb");

    if (f == NULL)
        return;

    if (fseek(f, 0, SEEK_END) != 0)
    {
        fclose(f);
        return;
    }

    size = ftell(f);

    if (size < 0)
    {
        fclose(f);
        return;
    }

    if (fseek(f, 0, SEEK_SET) != 0)
    {
        fclose(f);
        return;
    }

    buffer = (char *)malloc((size_t)size + 1);

    if (buffer == NULL)
    {
        fclose(f);
        return;
    }

    if (fread(buffer, 1, (size_t)size, f) != (size_t)size)
    {
        free(buffer);
        fclose(f);
        return;
    }

    buffer[size] = '\0';
    fclose(f);

    i       = 0;
    line    = 1;
    column  = 1;

    while (i < size)
    {
        char c;
        Token tok;

        c = buffer[i];

        if (c == '\n')
        {
            i = i + 1;
            line = line + 1;
            column = 1;
            continue;
        }

        if (c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f')
        {
            i       = i + 1;
            column  = column + 1;

            continue;
        }

        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_')
        {
            int64_t start;
            start = i;
            
            while (i < size)
            {
                char nc;
                nc = buffer[i];

                if ((nc >= 'A' && nc <= 'Z') || 
                    (nc >= 'a' && nc <= 'z') || 
                    (nc >= '0' && nc <= '9') || 
                    nc == '_')
                {
                    i = i + 1;
                    continue;
                }

                break;
            }

            int64_t len;
            string lex;
            
            len = i - start;
            lex.assign(buffer + start, (size_t)len);
            tok.line    = (uint32_t)line;
            tok.column  = (uint32_t)column;
            tok.text    = lex;
            map<string, TokenType>::iterator it = tokenMap.find(lex);
            
            if (it != tokenMap.end())
            {
                tok.type = it->second;
            }
            else
            {
                tok.type = TOKEN_IDENTIFIER;
            }

            tokens.push_back(tok);
            column = column + (int)len;

            continue;
        }

        if (c >= '0' && c <= '9')
        {
            int64_t start;
            bool isFloat;
            start   = i;
            isFloat = false;
            
            while (i < size)
            {
                char nc;
                nc = buffer[i];
                
                if (nc >= '0' && nc <= '9')
                {
                    i = i + 1;
                    continue;
                }
                
                if (nc == '.' && !isFloat)
                {
                    isFloat = true;
                    i = i + 1;
                    continue;
                }

                if ((nc == 'e' || nc == 'E') && i + 1 < size)
                {
                    int64_t j;
                    j = i + 1;
                    
                    if (buffer[j] == '+' || buffer[j] == '-')
                    {
                        j = j + 1;
                    }

                    if (j < size && buffer[j] >= '0' && buffer[j] <= '9')
                    {
                        i = i + 1;
                        while (i < size && 
                               ((buffer[i] >= '0' && buffer[i] <= '9') || 
                                 buffer[i] == '+' || buffer[i] == '-'))
                        {
                            i = i + 1;
                        }

                        isFloat = true;
                        continue;
                    }
                }

                break;
            }

            int64_t len;
            len = i - start;
            string lex;
            
            lex.assign(buffer + start, (size_t)len);
            tok.line    = (uint32_t)line;
            tok.column  = (uint32_t)column;
            tok.text    = lex;

            if (isFloat)
                tok.type = TOKEN_FLOAT_LITERAL;
            else
                tok.type = TOKEN_INT_LITERAL;
            
            tokens.push_back(tok);
            column = column + (int)len;
            continue;
        }

        if (c == '"')
        {
            int64_t start;
            
            start   = i;
            i       = i + 1;
            
            while (i < size)
            {
                char nc;
                nc = buffer[i];

                if (nc == '\\' && i + 1 < size)
                {
                    i = i + 2;
                    continue;
                }

                if (nc == '"')
                {
                    i = i + 1;
                    break;
                }

                if (nc == '\n')
                {
                    line    = line + 1;
                    column  = 1;
                }

                i = i + 1;
            }

            int64_t len;
            len = i - start;
            string lex;
            
            lex.assign(buffer + start, (size_t)len);

            tok.line    = (uint32_t)line;
            tok.column  = (uint32_t)column;
            tok.text    = lex;
            tok.type    = TOKEN_STRING_LITERAL;
            tokens.push_back(tok);
            column      = column + (int)len;
            
            continue;
        }

        if (c == '\'')
        {
            int64_t start;
            start = i;
            i = i + 1;

            if (i < size && buffer[i] == '\\' && i + 1 < size)
                i = i + 2;
            else if (i < size)
                i = i + 1;

            if (i < size && buffer[i] == '\'')
                i = i + 1;
            
            int64_t len;
            len = i - start;
            string lex;
            
            lex.assign(buffer + start, (size_t)len);

            tok.line    = (uint32_t)line;
            tok.column  = (uint32_t)column;
            tok.text    = lex;
            tok.type    = TOKEN_CHAR_LITERAL;
            
            tokens.push_back(tok);
            column = column + (int)len;
            
            continue;
        }

        if (c == '/')
        {
            if (i + 1 < size && buffer[i + 1] == '/')
            {
                int64_t start;
                start   = i;
                i       = i + 2;
                
                while (i < size && buffer[i] != '\n')
                    i = i + 1;

                int64_t len;
                len = i - start;

                string lex;
                lex.assign(buffer + start, (size_t)len);
                
                tok.line    = (uint32_t)line;
                tok.column  = (uint32_t)column;
                tok.text    = lex;
                tok.type    = TOKEN_COMMENT;
                
                tokens.push_back(tok);
                column = column + (int)len;
                
                continue;
            }
            else if (i + 1 < size && buffer[i + 1] == '*')
            {
                int64_t start;
                start   = i;
                i       = i + 2;

                while (i + 1 < size)
                {
                    if (buffer[i] == '*' && buffer[i + 1] == '/')
                    {
                        i = i + 2;
                        break;
                    }

                    if (buffer[i] == '\n')
                    {
                        line = line + 1;
                        column = 1;
                    }

                    i = i + 1;
                }

                int64_t len;
                len = i - start;
                string lex;
                lex.assign(buffer + start, (size_t)len);

                tok.line    = (uint32_t)line;
                tok.column  = (uint32_t)column;
                tok.text    = lex;
                tok.type    = TOKEN_COMMENT;
                
                tokens.push_back(tok);
                column = column + (int)len;
                
                continue;
            }
        }

        int64_t remaining;
        remaining = size - i;

        if (remaining >= 2)
        {
            string two;
            two.assign(buffer + i, 2);

            if (two == "->")
            {
                tok.type = TOKEN_ARROW;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "==")
            {
                tok.type = TOKEN_OP_EQ;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "!=")
            {
                tok.type = TOKEN_OP_NE;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "<=")
            {
                tok.type = TOKEN_OP_LE;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == ">=")
            {
                tok.type = TOKEN_OP_GE;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "&&")
            {
                tok.type = TOKEN_OP_AND;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "||")
            {
                tok.type = TOKEN_OP_OR;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "<<")
            {
                tok.type = TOKEN_OP_SHL;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == ">>")
            {
                tok.type = TOKEN_OP_SHR;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "+=")
            {
                tok.type = TOKEN_OP_PLUS_EQ;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "-=")
            {
                tok.type = TOKEN_OP_MINUS_EQ;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "*=")
            {
                tok.type = TOKEN_OP_STAR_EQ;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }

            if (two == "/=")
            {
                tok.type = TOKEN_OP_SLASH_EQ;
                tok.text = two;
                tok.line = (uint32_t)line;
                tok.column = (uint32_t)column;
                tokens.push_back(tok);
                i = i + 2;
                column = column + 2;
                continue;
            }
        }

        string one;
        one.assign(buffer + i, 1);

        tok.text    = one;
        tok.line    = (uint32_t)line;
        tok.column  = (uint32_t)column;

        if (one == "+")
        {
            tok.type = TOKEN_OP_PLUS;
        }
        else if (one == "-")
        {
            tok.type = TOKEN_OP_MINUS;
        }
        else if (one == "*")
        {
            tok.type = TOKEN_OP_STAR;
        }
        else if (one == "/")
        {
            tok.type = TOKEN_OP_SLASH;
        }
        else if (one == "%")
        {
            tok.type = TOKEN_OP_PERCENT;
        }
        else if (one == "=")
        {
            tok.type = TOKEN_OP_ASSIGN;
        }
        else if (one == "<")
        {
            tok.type = TOKEN_OP_LT;
        }
        else if (one == ">")
        {
            tok.type = TOKEN_OP_GT;
        }
        else if (one == "(")
        {
            tok.type = TOKEN_LPAREN;
        }
        else if (one == ")")
        {
            tok.type = TOKEN_RPAREN;
        }
        else if (one == "{")
        {
            tok.type = TOKEN_LBRACE;
        }
        else if (one == "}")
        {
            tok.type = TOKEN_RBRACE;
        }
        else if (one == "[")
        {
            tok.type = TOKEN_LBRACK;
        }
        else if (one == "]")
        {
            tok.type = TOKEN_RBRACK;
        }
        else if (one == ",")
        {
            tok.type = TOKEN_COMMA;
        }
        else if (one == ";")
        {
            tok.type = TOKEN_SEMICOLON;
        }
        else if (one == ":")
        {
            tok.type = TOKEN_COLON;
        }
        else if (one == ".")
        {
            tok.type = TOKEN_DOT;
        }
        else if (one == "@")
        {
            tok.type = TOKEN_AT;
        }
        else if (one == "#")
        {
            tok.type = TOKEN_HASH;
        }
        else
        {
            tok.type = TOKEN_UNKNOWN;
        }

        tokens.push_back(tok);
        
        i       = i + 1;
        column  = column + 1;
    }

    free(buffer);
    Token eofTok;

    eofTok.type     = TOKEN_EOF;
    eofTok.text     = "";
    eofTok.line     = (uint32_t)line;
    eofTok.column   = (uint32_t)column;
    
    tokens.push_back(eofTok);

}