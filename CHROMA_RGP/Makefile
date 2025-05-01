# ---- 使用者可調 ----------------------------------
ARCH      ?= sm_89
CXXFLAGS  := -O3 -std=c++17 -arch=$(ARCH)
DC_FLAGS  := -dc $(CXXFLAGS)
LINKFLAGS := -O3 -arch=$(ARCH) -diag-suppress=177
TARGET    := CHROMA

# ---- 原始檔清單 -----------------------------------
SRCS := globals.cu chroma_utils.cu ECLGC.cu PA.cu CHROMA.cu
OBJS := $(SRCS:.cu=.o)

# ---- 目標 ----------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "== Linking $(TARGET) (ARCH=$(ARCH)) =="
	nvcc $(LINKFLAGS) $^ -o $@
	@rm -f $(OBJS)

%.o: %.cu
	@echo "== Compiling $< (ARCH=$(ARCH)) =="
	nvcc $(DC_FLAGS) -c $< -o $@

clean:
	@echo "== Clean =="
	@rm -f $(TARGET) $(OBJS)

.PHONY: all clean
