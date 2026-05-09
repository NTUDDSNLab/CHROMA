# tools/feature_extractor/Makefile
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall
LIBIO     = ../../lib/io

extract: extract.cpp $(LIBIO)/graph_features.cpp $(LIBIO)/graph_features.h $(LIBIO)/ECLgraph.h
	$(CXX) $(CXXFLAGS) -I$(LIBIO) extract.cpp $(LIBIO)/graph_features.cpp -o extract

clean:
	rm -f extract

.PHONY: clean
