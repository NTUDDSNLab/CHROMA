#include "csr_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <math.h>
#include <sys/time.h>

using namespace std;


void read_graph(csr_graph *graph, char *file_name,int mode){
    edge *edge_list;
    ifstream file(file_name);
    string line;
    int edge_id=0, e_count;
    if(file.is_open()) {
    while (getline(file, line)) {
        if (line[0] == 'p') {
            istringstream iss(line);
            string part;
            vector<string> parts;
            while (iss >> part) {
                parts.push_back(part);
            }
            graph->v_count=stoi(parts[2]);
            e_count=stoi(parts[3]);
            graph->IA=new int[graph->v_count + 1];
            graph->A=new int[e_count*2];
            graph->color=new int[graph->v_count];
            edge_list=new edge[e_count];
            fill(graph->IA, graph->IA + graph->v_count + 1, 0);
            // cout<<graph->v_count;
        }
        else if (line[0] == 'e') {
            istringstream iss(line);
            string part;
            vector<string> parts;
            while (iss >> part) {
                parts.push_back(part);
            }
            if (mode==0)
            {
                edge_list[edge_id].vertex1=stoi(parts[1])-1;
				edge_list[edge_id].vertex2=stoi(parts[2])-1;
                graph->IA[stoi(parts[1])]++;
            }
            else{
                edge_list[edge_id].vertex1=stoi(parts[1])-1;
				edge_list[edge_id].vertex2=stoi(parts[2])-1;
                graph->IA[stoi(parts[1])]++;
                graph->IA[stoi(parts[2])]++;
            }
            edge_id++;
        }
    }
    file.close();
    int *vertex_p = new int[graph->v_count];
    for (int i = 0; i < graph->v_count; i++) {
        graph->IA[i + 1] += graph->IA[i];
        vertex_p[i] = 0;
    }

    for (int edge_id = 0; edge_id < e_count; edge_id++) {
        if (mode==0)
        {
            graph->A[graph->IA[edge_list[edge_id].vertex1] + (vertex_p[edge_list[edge_id].vertex1]++)] = edge_list[edge_id].vertex2;
        }
        else{
            graph->A[graph->IA[edge_list[edge_id].vertex1] + (vertex_p[edge_list[edge_id].vertex1]++)] = edge_list[edge_id].vertex2;
            graph->A[graph->IA[edge_list[edge_id].vertex2] + (vertex_p[edge_list[edge_id].vertex2]++)] = edge_list[edge_id].vertex1;
        }
    }
    delete[] edge_list;
    delete[] vertex_p;
    } else {
        cout << "can't open file" << endl;
    }
}

void print_array(int* arr, int size){
    for(int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

}