# ---- 使用者可調 ----------------------------------
ARCH      ?= sm_89
PARTITIONER_FLAG := -DHAVE_METIS
PARTITIONER_FLAG += -DHAVE_KAHIP # Open this if you want to use KaHIP
CXXFLAGS  := -O3 -std=c++17 -arch=$(ARCH) $(PARTITIONER_FLAG) -Xcompiler -fopenmp
DC_FLAGS  := -dc $(CXXFLAGS) 
# 加入 METIS 與 OpenMP 的連結參數

# METIS (保持原有的 ~/local 路徑)
INCLUDES_METIS := -I$(HOME)/local/include
LIBS_METIS := -L$(HOME)/local/lib

# KaHIP (修改為 External 目錄)
INCLUDES_KAHIP := -I../External/KaHIP/deploy
LIBS_KAHIP := -L../External/KaHIP/deploy

# 寫入 rpath，讓執行檔自帶庫的搜尋路徑（免 LD_LIBRARY_PATH）
RPATH_FLAGS := -Xlinker -rpath -Xlinker $(HOME)/local/lib \
               -Xlinker -rpath -Xlinker $(abspath ../External/KaHIP/deploy) \
               -Xlinker --enable-new-dtags

# 統一使用 LDFLAGS，移除冗餘的註釋行
LDFLAGS := -O3 -arch=$(ARCH) -diag-suppress=177 -Xcompiler -fopenmp \
           $(LIBS_METIS) $(LIBS_KAHIP) -lmetis -lGKlib -lkahip $(RPATH_FLAGS)

TARGET    := CHROMA_RGP

# ---- 原始檔清單 -----------------------------------
SRCS := CHROMA_RGP.cu
OBJS := $(SRCS:.cu=.o)

# ---- 目標 ----------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "== Linking $(TARGET) (ARCH=$(ARCH)) =="
	nvcc $(LDFLAGS) $(INCLUDES_METIS) $(INCLUDES_KAHIP) $^ -o $@

%.o: %.cu
	@echo "== Compiling $< (ARCH=$(ARCH)) =="
	nvcc $(DC_FLAGS) $(INCLUDES_METIS) $(INCLUDES_KAHIP) -c $< -o $@

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
