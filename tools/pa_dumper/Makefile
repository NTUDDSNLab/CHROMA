# tools/pa_dumper/Makefile
#
# Builds a single binary `pa_dumper` that supports CHROMA, JP-Series, and
# CPU JP-SL^M PA implementations behind one CLI. Intended for measuring
# priority orderings across frameworks under a unified dump format.
#
#   $ make ARCH=sm_86      # default ARCH=sm_86; override e.g. sm_89, sm_90

ARCH      ?= sm_86
PRE_MODEL ?= 1            # link the m2cgen predictor (model/model.cpp)
                          # so --predict works at runtime

CHROMA_DIR := ../../CHROMA
LIBIO_DIR  := ../../lib/io
MODEL_DIR  := ../../model

CXXFLAGS   := -O3 -std=c++17
DC_FLAGS   := -dc -arch=$(ARCH) -rdc=true -O3 -std=c++17 \
              -I. -I$(CHROMA_DIR) -I$(LIBIO_DIR)
LINKFLAGS  := -O3 -arch=$(ARCH) -rdc=true -diag-suppress=177

ifeq ($(PRE_MODEL), 1)
  DC_FLAGS += -DPRED_MODEL
endif

# Local sources (in this directory)
LOCAL_CU   := pa_dumper.cu jp_pa.cu
LOCAL_CPP  := cpu_sdl.cpp

# CHROMA sources we reuse (compiled into our object dir)
CHROMA_CU  := globals.cu chroma_utils.cu color_reduction.cu \
              ECLGC.cu PA.cu PA_split.cu BB.cu BB_split.cu

# CSR I/O helpers (currently only ECLgraph.h is needed; mmio/graph kept
# for future format support)
LIBIO_CPP  := graph.cpp mmio.cpp

LOCAL_CU_OBJ   := $(LOCAL_CU:.cu=.o)
LOCAL_CPP_OBJ  := $(LOCAL_CPP:.cpp=.o)
CHROMA_OBJ     := $(addprefix chroma_,$(CHROMA_CU:.cu=.o))
LIBIO_OBJ      := $(addprefix libio_,$(LIBIO_CPP:.cpp=.o))

ifeq ($(PRE_MODEL), 1)
  MODEL_OBJ := model_model.o
else
  MODEL_OBJ :=
endif

ALL_OBJ := $(LOCAL_CU_OBJ) $(LOCAL_CPP_OBJ) $(CHROMA_OBJ) $(LIBIO_OBJ) $(MODEL_OBJ)

all: pa_dumper

pa_dumper: $(ALL_OBJ)
	@echo "== Linking pa_dumper =="
	nvcc $(LINKFLAGS) $^ -o $@
	@rm -f $(ALL_OBJ)

# Local .cu files
%.o: %.cu
	@echo "== Compiling $< =="
	nvcc $(DC_FLAGS) $< -o $@

# Local .cpp files (build with nvcc since it's all linked together w/ rdc)
%.o: %.cpp
	@echo "== Compiling $< =="
	nvcc $(DC_FLAGS) -x c++ $< -o $@

# CHROMA .cu files: prefix-renamed so we don't clash with potential local
# files of the same stem
chroma_%.o: $(CHROMA_DIR)/%.cu
	@echo "== Compiling (CHROMA) $< =="
	nvcc $(DC_FLAGS) $< -o $@

# lib/io .cpp files
libio_%.o: $(LIBIO_DIR)/%.cpp
	@echo "== Compiling (lib/io) $< =="
	nvcc $(DC_FLAGS) -x c++ $< -o $@

# m2cgen predictor (only when PRE_MODEL=1)
model_model.o: $(MODEL_DIR)/model.cpp
	@echo "== Compiling (model) $< =="
	nvcc $(DC_FLAGS) -x c++ $< -o $@

clean:
	rm -f pa_dumper $(ALL_OBJ)

.PHONY: all clean
