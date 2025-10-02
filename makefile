LLVM_CONFIG 		?= /usr/bin/llvm-config
CXX 				?= clang++
CXXFLAGS 			?= -g -O2 -std=c++17 -fPIC

LLVM_CXXFLAGS 		:= $(shell $(LLVM_CONFIG) --cxxflags 2>/dev/null)
LLVM_LDFLAGS  		:= $(shell $(LLVM_CONFIG) --ldflags 2>/dev/null)
LLVM_LIBS     		:= $(shell $(LLVM_CONFIG) --libs all 2>/dev/null)
LLVM_SYSTEM_LIBS 	:= $(shell $(LLVM_CONFIG) --system-libs 2>/dev/null)
MLIR_PREFIX    		:= /usr
MLIR_INC      		:= /usr/include
MLIR_LIBDIR   		:= /usr/lib

MLIR_LIBS := -lmlirIR -lmlirParser -lmlirPass -lmlirTransforms -lmlirConversion \
            -lmlirTargetLLVMIR -lmlirInitAllDialects -lmlirInitAllPasses -lmlirSupport

SRCS 		:= src/main.cpp src/lexer.cpp src/parser.cpp
TARGET 		:= ga-opt
INCLUDES 	:= -I$(MLIR_INC) -Iinc
LIBDIRS 	:= -L$(MLIR_LIBDIR)
LD_FLAGS 	:= $(LLVM_LDFLAGS) $(LIBDIRS) -lpthread -ldl -lrt -lm

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Building $(TARGET) with $(CXX)"
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) $(INCLUDES) -o $@ $^ $(LD_FLAGS) -Wl,-rpath,$(MLIR_LIBDIR)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(LLVM_CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	-rm -f $(TARGET) *.o

