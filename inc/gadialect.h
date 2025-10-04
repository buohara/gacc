#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace llvm;

namespace ga 
{
    class GADialect : public Dialect 
    {
        public:
        explicit GADialect(MLIRContext *ctx);

        static StringRef getDialectNamespace() { return "ga"; }
    };
}