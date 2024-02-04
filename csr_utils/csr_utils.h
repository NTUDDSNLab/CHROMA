#ifndef CSR_UTILS_H
#define CSR_UTILS_H
#include <iostream>
#include <fstream>

struct edge{
	int vertex1, vertex2;
};

struct csr_graph {
    int v_count;
    int *A;
    int *IA;
    int *color;
};

void read_graph(csr_graph *graph, char *filename,int mode);
// mode 0 for DAG
// mode 1 for Undirected Graph

void print_array(int* arr, int size);

#endif // CSR_UTILS_H