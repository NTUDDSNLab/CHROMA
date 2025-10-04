# ================== 使用方法 ==================
# $ make         生成執行檔 CHROMA
# $ make clean   刪除 *.o 及 CHROMA
# =================================================

# ---- 使用者可調參數 ----------------------------------
ARCH      ?= sm_86
CXXFLAGS  := -O3 -std=c++17 -I../lib/io
DC_FLAGS  := -dc -arch=$(ARCH) -rdc=true $(CXXFLAGS) # -lineinfo -Xptxas=-v
LINKFLAGS := -O3 -arch=$(ARCH) -rdc=true -diag-suppress=177
TARGET    := CHROMA

# ---- 原始檔清單 --------------------------------------
SRCS := globals.cu chroma_utils.cu ECLGC.cu PA.cu CHROMA.cu
CPP_SRCS := ../lib/io/graph.cpp ../lib/io/mmio.cpp
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
