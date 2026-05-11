# ================== Usage ==================
# $ make                   Generate executable CHROMA
# $ make PRE_MODEL=1       Enable predict model (v3: 9 features + RF + floor + shift -0.5)
# $ make PRE_MODEL=1 DYNAMIC_THETA=1   Enable predict model + on-device θ controller
# $ make DEBUG=1           Enable debug mode (-lineinfo -O0)
# $ make clean             Remove *.o and CHROMA
# =================================================
#
# The deployed predictor (v3) is in model/model.cpp + model/scaler.h.
# Reference v1/v2 artefacts live in model/model_v{1,2}.cpp + scaler_v{1,2}.h
# but require a separate build to use (see T22 commit history).

# ---- User Configurable Parameters ----------------------------------
ARCH      ?= sm_86
PRE_MODEL ?= 0 		# 0: not use predict model, 1: use predict model
# 0: off (zero overhead); 1: enable on-device θ controller (DEFAULT post-T14)
DYNAMIC_THETA ?= 1
PROF      ?= 0       # 0: not use profiler, 1: use profiler
DEBUG     ?= 0       # 0: release mode, 1: debug mode

ifeq ($(DEBUG), 1)
	OPT_FLAGS := -O0
	DC_DEBUG  := -lineinfo
else
	OPT_FLAGS := -O3
	DC_DEBUG  :=
endif

CXXFLAGS  := $(OPT_FLAGS) -std=c++17 -I../lib/io
DC_FLAGS  := -dc -arch=$(ARCH) -rdc=true $(CXXFLAGS) $(DC_DEBUG) # -lineinfo -Xptxas=-v
LINKFLAGS := $(OPT_FLAGS) -arch=$(ARCH) -rdc=true -diag-suppress=177
TARGET    := CHROMA


# ---- Source File List --------------------------------------
SRCS := globals.cu chroma_utils.cu color_reduction.cu ECLGC.cu PA.cu PA_split.cu BB.cu BB_split.cu CHROMA.cu
CPP_SRCS := ../lib/io/graph.cpp ../lib/io/mmio.cpp ../lib/io/graph_features.cpp

ifeq ($(PRE_MODEL), 1)
	CPP_SRCS += ../model/model.cpp
	DC_FLAGS += -DPRED_MODEL -I../model
	CXXFLAGS += -DPRED_MODEL -I../model
	ifeq ($(DYNAMIC_THETA), 1)
		DC_FLAGS += -DDYNAMIC_THETA
		CXXFLAGS += -DDYNAMIC_THETA
	endif
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
