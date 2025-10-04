LLVM_CONFIG 	?= /usr/local/bin/llvm-config
CXX 			?= clang++
CXXFLAGS 		?= -g -O2 -std=c++17 -fPIC

LLVM_CXXFLAGS 	:= $(shell $(LLVM_CONFIG) --cxxflags 2>/dev/null)
LLVM_LDFLAGS  	:= $(shell $(LLVM_CONFIG) --ldflags 2>/dev/null)
MLIR_INC      	:= /usr/local/include
MLIR_LIBDIR   	:= /usr/local/lib

MLIR_TBLGEN    	?= /usr/local/bin/mlir-tblgen
TD_DIR         	:= dialect
GEN_INC_DIR    	:= inc

GEN_OPS_HDR    	:= $(GEN_INC_DIR)/Ops.h.inc
GEN_TYPES_HDR  	:= $(GEN_INC_DIR)/Types.h.inc

LLVM_LIBDIR 	:= $(shell $(LLVM_CONFIG) --libdir 2>/dev/null)
MLIR_LIBS 		:= $(wildcard $(MLIR_LIBDIR)/lib*.a) $(wildcard $(MLIR_LIBDIR)/lib*.so)
LLVM_LIBS 		:= $(wildcard $(LLVM_LIBDIR)/lib*.a) $(wildcard $(LLVM_LIBDIR_FROM_CONFIG)/lib*.so)
ALL_LIB_FILES 	:= $(MLIR_LIBS) $(LLVM_LIBS)

SRCS 			:= src/main.cpp src/lexer.cpp src/parser.cpp src/gadialect.cpp
TARGET 			:= ga-opt
INCLUDES 		:= -I$(MLIR_INC) -Iinc -I$(GEN_INC_DIR)
LIBDIRS 		:= -L$(MLIR_LIBDIR)
LD_FLAGS 		:= $(LLVM_LDFLAGS) $(LIBDIRS) -lpthread -ldl -lrt -lm

.PHONY: all clean tblgen

all: tblgen $(TARGET)

tblgen: $(GEN_OPS_HDR) $(GEN_TYPES_HDR)

$(GEN_INC_DIR)/Ops.h.inc: $(TD_DIR)/ops.td | $(GEN_INC_DIR)
	$(MLIR_TBLGEN) --gen-op-decls -I $(TD_DIR) -I $(MLIR_INC) $(TD_DIR)/ops.td -o $@

$(GEN_INC_DIR)/Types.h.inc: $(TD_DIR)/types.td | $(GEN_INC_DIR)
	$(MLIR_TBLGEN) --gen-typedef-decls -I $(TD_DIR) -I $(MLIR_INC) $(TD_DIR)/types.td -o $@

$(GEN_INC_DIR):
	mkdir -p $(GEN_INC_DIR)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) $(INCLUDES) -o $@ $^ $(LD_FLAGS) $(MLIR_LIBS) $(ALL_LIB_FILES) -Wl,-rpath,$(MLIR_LIBDIR)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	-rm -f $(TARGET) *.o

