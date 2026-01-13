# ---- User Configurable ----------------------------------
ARCH      ?= sm_86
PARTITIONER_FLAG := -DHAVE_METIS
PARTITIONER_FLAG += -DHAVE_KAHIP # Open this if you want to use KaHIP
CXXFLAGS  := -O3 -std=c++17 -I../lib/io
DC_FLAGS  := -dc -arch=$(ARCH) $(PARTITIONER_FLAG) -Xcompiler -fopenmp $(CXXFLAGS) 
# Add METIS and OpenMP linking parameters

# METIS (Keep original ~/local path)
INCLUDES_METIS := -I$(HOME)/local/include
LIBS_METIS := -L$(HOME)/local/lib

# KaHIP (Modified to External directory)
INCLUDES_KAHIP := -I../External/KaHIP/deploy
LIBS_KAHIP := -L../External/KaHIP/deploy

# Write rpath, let executable carry library search path (avoid LD_LIBRARY_PATH)
RPATH_FLAGS := -Xlinker -rpath -Xlinker $(HOME)/local/lib \
               -Xlinker -rpath -Xlinker $(abspath ../External/KaHIP/deploy) \
               -Xlinker --enable-new-dtags

# Unify LDFLAGS, remove redundant comment lines
LDFLAGS := -O3 -arch=$(ARCH) -diag-suppress=177 -Xcompiler -fopenmp \
           $(LIBS_METIS) $(LIBS_KAHIP) -lmetis -lGKlib -lkahip $(RPATH_FLAGS)

TARGET    := CHROMA_RGP

# ---- Source File List -----------------------------------
SRCS := CHROMA_RGP.cu
CPP_SRCS := ../lib/io/graph.cpp ../lib/io/mmio.cpp
OBJS := $(SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

# ---- Target ----------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "== Linking $(TARGET) (ARCH=$(ARCH)) =="
	nvcc $(LDFLAGS) $(INCLUDES_METIS) $(INCLUDES_KAHIP) $^ -o $@
	@rm -f $(OBJS)

%.o: %.cu
	@echo "== Compiling $< (ARCH=$(ARCH)) =="
	nvcc $(DC_FLAGS) $(INCLUDES_METIS) $(INCLUDES_KAHIP) -c $< -o $@

%.o: %.cpp
	@echo "== Compiling $< =="
	g++ $(CXXFLAGS) -c $< -o $@


clean:
	@echo "== Clean =="
	@rm -f $(TARGET) $(OBJS)

.PHONY: all clean

# ---- Tests (CPU-only) --------------------------------
# Build a small unit test for partitioners without CUDA/METIS
test_partitioners:
	@echo "== Building tests/partitioner_test =="
	g++ -O3 -std=c++17 -I. tests/partitioner_test.cpp -o tests/partitioner_test
	@echo "== Running tests/partitioner_test =="
	./tests/partitioner_test
