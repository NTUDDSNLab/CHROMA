## snap2partition: Partition a SNAP .txt into multiple subgraphs using METIS

CXX       ?= g++
CXXFLAGS  ?= -O3 -std=c++17 -I../../lib/io -I../../CHROMA

# Adjust these if METIS is installed elsewhere
INCLUDES_METIS ?= -I$(HOME)/local/include
LIBS_METIS     ?= -L$(HOME)/local/lib

RPATH_FLAGS := -Wl,-rpath,$(HOME)/local/lib

TARGET := snap2partition
SRCS   := snap2partition.cpp ../../lib/io/graph.cpp ../../lib/io/mmio.cpp
OBJS   := $(notdir $(SRCS:.cpp=.o))

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES_METIS) $^ -o $@ $(LIBS_METIS) -lmetis -lGKlib $(RPATH_FLAGS)

# Compile sources into local objects (avoid writing outside this dir)
snap2partition.o: snap2partition.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES_METIS) -c $< -o $@

graph.o: ../../lib/io/graph.cpp ../../lib/io/graph.h ../../lib/io/mmio.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

mmio.o: ../../lib/io/mmio.cpp ../../lib/io/mmio.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
