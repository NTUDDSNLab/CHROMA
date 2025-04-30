ARCH      ?= sm_89
CXXFLAGS  := -O3 -std=c++17 -arch=$(ARCH)
DC_FLAGS  := -dc $(CXXFLAGS)
LINKFLAGS := -O3 -arch=$(ARCH) -diag-suppress=177
TARGET    := JP-Series

SRCS := globals.cu chroma_utils.cu ECLGC.cu PA.cu JP-Series.cu
OBJS := $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "== Linking $(TARGET) =="
	nvcc $(LINKFLAGS) $^ -o $@
	@rm -f $(OBJS)

%.o: %.cu
	@echo "== Compiling $< =="
	nvcc $(DC_FLAGS) $< -o $@

clean:
	rm -f  $(TARGET)

.PHONY: all clean
