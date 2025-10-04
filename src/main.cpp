#include "common.h"
#include "lexer.h"
#include "parser.h"
#include "gadialect.h"

struct CLArgs 
{
    string inputFilename;
    string outputFilename;
    bool printTokens;
    bool printAST;

    CLArgs() : printTokens(false), printAST(false) {}
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

        clArgs.inputFilename = args[i];
    }
}

/**
 * InitMLIRContext - Initialize the MLIR context with necessary dialects.
 * 
 * @param context   [in/out]    MLIR context to initialize.
 */

void InitMLIRContext(MLIRContext &context)
{
    DialectRegistry registry;

    registry.insert<ga::GADialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();

    context.appendDialectRegistry(registry);
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

    MLIRContext context;
    InitMLIRContext(context);

    GAParser parser;
    GetTokens(clArgs.inputFilename, parser.tokens);
    parser.GenerateAST();

    if (clArgs.printTokens)
        parser.PrintTokens();

    if (clArgs.printAST)
        parser.PrintAST();

    parser.PrintSymbolTable();

    return 0;
}