# ---- 使用者可調 ----------------------------------
ARCH      ?= sm_89
CXXFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -DHAVE_METIS -Xcompiler -fopenmp
DC_FLAGS  := -dc $(CXXFLAGS) 
# 加入 METIS 與 OpenMP 的連結參數

INCLUDES_METIS := -I$(HOME)/local/include
LIBS_METIS := -L$(HOME)/local/lib

# 寫入 rpath，讓執行檔自帶 ~/local/lib 的搜尋路徑（免 LD_LIBRARY_PATH）
RPATH_FLAGS := -Xlinker -rpath -Xlinker $(HOME)/local/lib -Xlinker --enable-new-dtags

LDFLAGS     := -O3 -arch=$(ARCH) -diag-suppress=177 -Xcompiler -fopenmp \
               $(LIBDIRS) -lmetis -lGKlib $(RPATH_FLAGS)
# LINKFLAGS := -O3 -arch=$(ARCH) -diag-suppress=177 -Xcompiler -fopenmp $(LIBS_METIS) -lmetis -lGKlib
TARGET    := CHROMA_RGP

# ---- 原始檔清單 -----------------------------------
SRCS := CHROMA_RGP.cu
OBJS := $(SRCS:.cu=.o)

# ---- 目標 ----------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "== Linking $(TARGET) (ARCH=$(ARCH)) =="
	nvcc $(LDFLAGS) $^ -o $@

%.o: %.cu
	@echo "== Compiling $< (ARCH=$(ARCH)) =="
	nvcc $(DC_FLAGS) -c $< -o $@

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
