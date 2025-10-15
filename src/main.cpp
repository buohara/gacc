#include "common.h"
#include "lexer.h"
#include "parser.h"

struct CLArgs 
{
    string inputFilename;
    string outputFilename;
    bool printTokens;
    bool printAST;
    bool dumpSymTab;
    bool dumpSSA;

    CLArgs() : printTokens(false), printAST(false), dumpSymTab(false), dumpSSA(false) {}
};

/**
 * ParseCommandLine - Parse command line arguments.
 * 
 * @param args      [in]        Vector of command line arguments.
 * @param clArgs    [in/out]    CLArgs struct to populate.
 */

void ParseCommandLine(vector<string> &args, CLArgs &clArgs)
{
    for (uint32_t i = 0; i < args.size(); i++) 
    {
        if (args[i] == "-h" || args[i] == "--help") 
        {
            printf("Help message\n");
            continue;
        }

        if (args[i] == "-o" || args[i] == "--output") 
        {
            if (i + 1 < args.size()) 
            {
                clArgs.outputFilename = args[i + 1];
                i++;
            } 
            else 
            {
                printf("Error: Missing output filename after %s\n", args[i].c_str());
            }

            continue;
        }

        if (args[i] == "-tok")
        {
            clArgs.printTokens = true;
            continue;
        }

        if (args[i] == "-ast")
        {
            clArgs.printAST = true;
            continue;
        }

        if (args[i] == "-sym")
        {
            clArgs.dumpSymTab = true;
            continue;
        }

        clArgs.inputFilename = args[i];
    }
}

/**
 * AssembleAndLink - Assemble and link the generated assembly code.
 */

void AssembleAndLink()
{
    int ret = system("nasm -f elf64 testout.s -o testout.o");
    ret     = system("gcc testout.o -no-pie -o out");
}

/**
 * main - Compiler entry.
 * 
 * @param argc  [in] Argument count.
 * @param argv  [in] Argument list.
 * 
 * @return Zero if compilation succeeded.
 */

int main(int argc, char **argv) 
{   
    vector<string> args(argv, argv + argc);
    CLArgs clArgs;

    ParseCommandLine(args, clArgs);

    GAParser parser;
    GetTokens(clArgs.inputFilename, parser.tokens);
    parser.GenerateAST();
    parser.BuildSymbolTable();

    if (clArgs.printTokens)
        parser.PrintTokens();

    if (clArgs.printAST)
        parser.PrintAST();

    if (clArgs.dumpSymTab)
        parser.PrintSymbolTable();

    string outFile = "testout.s";
    parser.EmitASM(outFile);
    AssembleAndLink();

    return 0;
}