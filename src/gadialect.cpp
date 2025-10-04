#include "gadialect.h"
#include "Ops.h.inc"
#include "Types.h.inc"

using namespace mlir;

ga::GADialect::GADialect(mlir::MLIRContext *ctx) : mlir::Dialect(getDialectNamespace(), ctx, mlir::TypeID::get<ga::GADialect>()) 
{
    addOperations<>();
    addTypes<>();
}