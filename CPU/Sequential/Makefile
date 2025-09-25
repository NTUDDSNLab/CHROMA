# Makefile for CHROMA CPU Sequential algorithms
CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall
INCLUDES = -I.

# Source files
SEQUENTIAL_SOURCES = Dstura.cpp greedy.cpp SDL.cpp

# Targets with cpu_ prefix
SEQUENTIAL_TARGETS = $(SEQUENTIAL_SOURCES:.cpp=)
SEQUENTIAL_TARGETS_CPU = $(addprefix cpu_,$(SEQUENTIAL_TARGETS))
ALL_TARGETS = $(SEQUENTIAL_TARGETS_CPU)

# Default target
all: $(SEQUENTIAL_TARGETS_CPU)
	@echo "All sequential CPU executables built successfully"

# Sequential targets (no special dependencies)
$(SEQUENTIAL_TARGETS_CPU): cpu_%: %.cpp ECLgraph.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

# Clean target
clean:
	rm -f $(ALL_TARGETS)

# Phony targets
.PHONY: all clean help

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build all sequential CPU executables"
	@echo "  cpu_Dstura  - Build cpu_Dstura executable"
	@echo "  cpu_greedy  - Build cpu_greedy executable"
	@echo "  cpu_SDL     - Build cpu_SDL executable"
	@echo "  clean       - Remove all executables"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Sequential algorithms - no OpenMP or external dependencies required"
