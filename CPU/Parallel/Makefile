# Makefile for CHROMA CPU Parallel algorithms
CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall
INCLUDES = -I.
OMPFLAGS = -fopenmp

# Kokkos paths (adjust these paths according to your system)
KOKKOS_ROOT ?= /usr/local
KOKKOS_KERNELS_ROOT ?= /usr/local
KOKKOS_INCLUDES = -I$(KOKKOS_ROOT)/include -I$(KOKKOS_KERNELS_ROOT)/include
KOKKOS_LIBS = -L$(KOKKOS_ROOT)/lib -lkokkos -lkokkosalgorithms -lkokkoskernels

# Source files
BASIC_SOURCES =
OMP_SOURCES = ADG.cpp SL.cpp ecl.cpp SLL.cpp
KOKKOS_SOURCES = greedy_kokkos.cpp

# Targets with cpu_ prefix
BASIC_TARGETS = $(BASIC_SOURCES:.cpp=)
OMP_TARGETS = $(OMP_SOURCES:.cpp=)
KOKKOS_TARGETS = $(KOKKOS_SOURCES:.cpp=)
BASIC_TARGETS_CPU = $(addprefix cpu_,$(BASIC_TARGETS))
OMP_TARGETS_CPU = $(addprefix cpu_,$(OMP_TARGETS))
KOKKOS_TARGETS_CPU = $(addprefix cpu_,$(KOKKOS_TARGETS))
ALL_TARGETS = $(BASIC_TARGETS_CPU) $(OMP_TARGETS_CPU) $(KOKKOS_TARGETS_CPU)

# Default target
all: $(OMP_TARGETS_CPU)
	@echo "Note: To build Kokkos targets, use 'make kokkos' (requires Kokkos installation)"

# Basic targets (no special dependencies) - currently empty
$(BASIC_TARGETS_CPU): cpu_%: %.cpp ECLgraph.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# OpenMP targets
$(OMP_TARGETS_CPU): cpu_%: %.cpp ECLgraph.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) -o $@ $<

# Kokkos targets (separate target due to external dependencies)
kokkos: $(KOKKOS_TARGETS_CPU)

$(KOKKOS_TARGETS_CPU): cpu_%: %.cpp ECLgraph.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) $(KOKKOS_INCLUDES) -o $@ $< $(KOKKOS_LIBS)

# Clean target
clean:
	rm -f $(ALL_TARGETS)

# Phony targets
.PHONY: all kokkos clean help

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build all OpenMP executables"
	@echo "  kokkos           - Build Kokkos-dependent executables (requires Kokkos)"
	@echo "  cpu_SLL          - Build cpu_SLL executable" 
	@echo "  cpu_ecl          - Build cpu_ecl executable"
	@echo "  cpu_ADG          - Build cpu_ADG executable (OpenMP)"
	@echo "  cpu_SL           - Build cpu_SL executable (OpenMP)"
	@echo "  cpu_greedy_kokkos - Build cpu_greedy_kokkos executable (requires Kokkos)"
	@echo "  clean            - Remove all executables"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Note: Kokkos targets require KOKKOS_ROOT and KOKKOS_KERNELS_ROOT"
	@echo "      environment variables or modify the paths in this Makefile"