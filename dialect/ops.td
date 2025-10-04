include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "dialect/types.td"

def GA_AddOp : Op<CGADialect, "Add"> {
  let summary = "Addition of CGA vectors";
  let arguments = (ins GA_CGAVecType:$lhs, GA_CGAVecType:$rhs);
  let results   = (outs GA_CGAVecType:$res);
}
