# ---- 使用者可調 ----------------------------------
ARCH      ?= sm_89
CXXFLAGS  := -O3 -std=c++17 -arch=$(ARCH)
DC_FLAGS  := -dc $(CXXFLAGS)
# 加入 METIS 與 OpenMP 的連結參數
LINKFLAGS := -O3 -arch=$(ARCH) -diag-suppress=177 -Xcompiler -fopenmp -lmetis
TARGET    := CHROMA

# ---- 原始檔清單 -----------------------------------
SRCS := CHROMA_RGP.cu
OBJS := $(SRCS:.cu=.o)

# ---- 目標 ----------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "== Linking $(TARGET) (ARCH=$(ARCH)) =="
	nvcc $(LINKFLAGS) $^ -o $@

%.o: %.cu
	@echo "== Compiling $< (ARCH=$(ARCH)) =="
	nvcc $(DC_FLAGS) -c $< -o $@

clean:
	@echo "== Clean =="
	@rm -f $(TARGET) $(OBJS)

.PHONY: all clean
