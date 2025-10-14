CXX 			?= clang++
#CXXFLAGS 		?= -g -O2 -std=c++17 -fPIC
CXXFLAGS 		+= -g -O0 -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline -fvar-tracking-assignments -DDEBUG

SRCS 			:= src/main.cpp src/lexer.cpp src/parser.cpp
TARGET 			:= ga-opt
INCLUDES 		:= -Iinc
LD_FLAGS 		:= -lpthread -ldl -lrt -lm

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
#$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LD_FLAGS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LD_FLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	-rm -f $(TARGET) *.o

