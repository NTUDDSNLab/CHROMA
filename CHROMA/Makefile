# ================== 使用方法 ==================
# $ make                   生成執行檔 CHROMA
# $ make PRE_MODEL=1       啟用 predict model
# $ make clean             刪除 *.o 及 CHROMA
# =================================================

# ---- 使用者可調參數 ----------------------------------
ARCH      ?= sm_86
PRE_MODEL ?= 0 		# 0: not use predict model, 1: use predict model
PROF      ?= 0       # 0: not use profiler, 1: use profiler
CXXFLAGS  := -O3 -std=c++17 -I../lib/io
DC_FLAGS  := -dc -arch=$(ARCH) -rdc=true $(CXXFLAGS) # -lineinfo -Xptxas=-v
LINKFLAGS := -O3 -arch=$(ARCH) -rdc=true -diag-suppress=177
TARGET    := CHROMA


# ---- 原始檔清單 --------------------------------------
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

# ---- 預設目標 ----------------------------------------
all: $(TARGET)

# ---- 可執行檔 ----------------------------------------
$(TARGET): $(OBJS)
	@echo "== Linking $(TARGET) =="
	nvcc $(LINKFLAGS) $^ -o $@
	@rm -f $(OBJS)

# ---- 個別物件 ----------------------------------------
%.o: %.cu
	@echo "== Compiling $< =="
	nvcc $(DC_FLAGS) $< -o $@

%.o: %.cpp
	@echo "== Compiling $< =="
	g++ $(CXXFLAGS) -c $< -o $@

# ---- 清理 -------------------------------------------
clean:
	rm -f  $(TARGET)

.PHONY: all clean
