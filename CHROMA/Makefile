# ================== 使用方法 ==================
# $ make         生成執行檔 CHROMA
# $ make clean   刪除 *.o 及 CHROMA
# =================================================

# ---- 使用者可調參數 ----------------------------------
ARCH      ?= sm_89
CXXFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -rdc=true
DC_FLAGS  := -dc $(CXXFLAGS)
LINKFLAGS := -O3 -arch=$(ARCH) -rdc=true -diag-suppress=177
TARGET    := CHROMA

# ---- 原始檔清單 --------------------------------------
SRCS := globals.cu chroma_utils.cu ECLGC.cu PA.cu CHROMA.cu
OBJS := $(SRCS:.cu=.o)

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

# ---- 清理 -------------------------------------------
clean:
	rm -f  $(TARGET)

.PHONY: all clean
