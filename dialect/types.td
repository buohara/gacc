include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypes.td"

def CGADialect : Dialect
{
  let name = "cga";
  let cppNamespace = "::cga";
}

def GA_CGAVecType : DialectType<CGADialect, CPred<"true">, "vec"> {
  let summary = "CGA Multivector";
  let description = [{
    Conformal Geometric Algebra Multivector
  }];
}