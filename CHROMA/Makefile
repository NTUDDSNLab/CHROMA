# ================== Usage ==================
# $ make                   Generate executable CHROMA
# $ make PRE_MODEL=1       Enable predict model
# $ make clean             Remove *.o and CHROMA
# =================================================

# ---- User Configurable Parameters ----------------------------------
ARCH      ?= sm_86
PRE_MODEL ?= 0 		# 0: not use predict model, 1: use predict model
PROF      ?= 0       # 0: not use profiler, 1: use profiler
CXXFLAGS  := -O3 -std=c++17 -I../lib/io
DC_FLAGS  := -dc -arch=$(ARCH) -rdc=true $(CXXFLAGS) # -lineinfo -Xptxas=-v
LINKFLAGS := -O3 -arch=$(ARCH) -rdc=true -diag-suppress=177
TARGET    := CHROMA


# ---- Source File List --------------------------------------
SRCS := globals.cu chroma_utils.cu ECLGC.cu PA.cu CHROMA.cu
CPP_SRCS := ../lib/io/graph.cpp ../lib/io/mmio.cpp

ifeq ($(PRE_MODEL), 1)
	CPP_SRCS += ../model/model.cpp
	DC_FLAGS += -DPRED_MODEL
	CXXFLAGS += -DPRED_MODEL
endif

ifeq ($(PROF), 1)
	DC_FLAGS += -DPROFILE
	CXXFLAGS += -DPROFILE
endif

OBJS := $(SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

# ---- Default Target ----------------------------------------
all: $(TARGET)

# ---- Executable ----------------------------------------
$(TARGET): $(OBJS)
	@echo "== Linking $(TARGET) =="
	nvcc $(LINKFLAGS) $^ -o $@
	@rm -f $(OBJS)

# ---- Individual Objects ----------------------------------------
%.o: %.cu
	@echo "== Compiling $< =="
	nvcc $(DC_FLAGS) $< -o $@

%.o: %.cpp
	@echo "== Compiling $< =="
	g++ $(CXXFLAGS) -c $< -o $@

# ---- Clean -------------------------------------------
clean:
	rm -f  $(TARGET)

.PHONY: all clean
