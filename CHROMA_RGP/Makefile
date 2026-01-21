# ---- User Configurable ----------------------------------
PARTITIONER_FLAG := 
ARCH      ?= sm_86
PRE_MODEL ?= 0 		# 0: not use predict model, 1: use predict model

# Partitioners Selection (0 or 1)
USE_METIS   ?= 1
USE_KAHIP   ?= 0
USE_MTKAHIP ?= 0
# ---- Variable setup based on selection --------------------
LDFLAGS_LIBS :=
INCLUDES     := -I../lib/io

# Common Libs (TBB, NUMA, etc needed for mt-KaHIP but also harmless if linked usually)
COMMON_LIBS := -ltbb -ltbbmalloc -ltbbmalloc_proxy -latomic -ldl -lnuma

# METIS
ifeq ($(USE_METIS), 1)
	PARTITIONER_FLAG += -DHAVE_METIS
	INCLUDES += -I$(HOME)/local/include
	RPATH_FLAGS += -Xlinker -rpath -Xlinker $(HOME)/local/lib
	LDFLAGS_LIBS += -L$(HOME)/local/lib -lmetis -lGKlib
endif

# KaHIP (Sequential)
ifeq ($(USE_KAHIP), 1)
	PARTITIONER_FLAG += -DHAVE_KAHIP
	INCLUDES += -I../External/KaHIP/deploy
	RPATH_FLAGS += -Xlinker -rpath -Xlinker $(abspath ../External/KaHIP/deploy)
	# Use static lib to avoid conflicts if possible, or dynamic
	LDFLAGS_LIBS += ../External/KaHIP/deploy/libkahip.a
endif

# mt-KaHIP (Parallel)
ifeq ($(USE_MTKAHIP), 1)
	PARTITIONER_FLAG += -DHAVE_MTKAHIP
	INCLUDES += -I../External/mt-KaHIP/deploy
	RPATH_FLAGS += -Xlinker -rpath -Xlinker $(abspath ../External/mt-KaHIP/deploy)
	LDFLAGS_LIBS += ../External/mt-KaHIP/deploy/libmtkahip.a $(COMMON_LIBS)
endif


RPATH_FLAGS += -Xlinker --enable-new-dtags

CXXFLAGS  := -O3 -std=c++17 $(INCLUDES)
DC_FLAGS  := -dc -arch=$(ARCH) $(PARTITIONER_FLAG) -Xcompiler -fopenmp $(CXXFLAGS) 

# Unify LDFLAGS
LDFLAGS := -O3 -arch=$(ARCH) -diag-suppress=177 -Xcompiler -fopenmp \
           $(LDFLAGS_LIBS) $(RPATH_FLAGS)

TARGET    := CHROMA_RGP

# ---- Source File List -----------------------------------
SRCS := CHROMA_RGP.cu
CPP_SRCS := ../lib/io/graph.cpp ../lib/io/mmio.cpp

ifeq ($(PRE_MODEL), 1)
	CPP_SRCS += ../model/model.cpp
	DC_FLAGS += -DPRED_MODEL
	CXXFLAGS += -DPRED_MODEL
endif
OBJS := $(SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

# ---- Target ----------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "== Linking $(TARGET) (ARCH=$(ARCH)) =="
	nvcc $(LDFLAGS) $^ -o $@
	@rm -f $(OBJS)

%.o: %.cu
	@echo "== Compiling $< (ARCH=$(ARCH)) =="
	nvcc $(DC_FLAGS) -c $< -o $@

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
