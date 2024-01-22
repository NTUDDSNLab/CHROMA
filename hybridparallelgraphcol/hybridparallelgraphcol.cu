#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include<cuda.h>
#include<curand_kernel.h>
#include <math.h>
#define NO_COLOR 0
#define MIN_COLOR -1
#define MAX_COLOR 1
#define BLOCK_SIZE 1000
using namespace std;
struct new_csr_graph{
	int v_count,*A, *IA, *color;
};
struct edge{
	int vertex1, vertex2;
};
int  read_graph(new_csr_graph *graph, char *file_name);
int printArray(int* arr, int size);
void graph_color(struct new_csr_graph *graph,string file_name);
void printArrayC(int* arr, int size);

__global__
void init_kernel(int *d_color, float *d_node_val, int *d_IA, int v_count){
    __shared__ int shared_IA[BLOCK_SIZE];  
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_id = threadIdx.x;
    
    if (thread_id < BLOCK_SIZE && vertex_id < v_count) {
        shared_IA[thread_id] = d_IA[vertex_id];
    }
    __syncthreads();
    if (vertex_id < v_count) {
        d_node_val[vertex_id] = shared_IA[thread_id + 1] - shared_IA[thread_id];
        d_color[vertex_id] = NO_COLOR;
    }
}
__global__
void random_generate(float *d_node_val, curandState* state, unsigned long seed, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
		curand_init ( seed, vertex_id, 0, &state[vertex_id] );
		d_node_val[vertex_id]=curand_uniform(state+vertex_id);
        // d_node_val[vertex_id]=vertex_id;
	}
}
__global__
void  minmax_kernel(int *d_A, int *d_IA, int *d_color, float *d_node_val, char *d_color_code, char *d_cont, char *d_change, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count && d_color[vertex_id]==NO_COLOR){
		int total=d_IA[vertex_id+1];
		float curr_node_val=d_node_val[vertex_id];
		float edge_node_val;
		char is_min=1, is_max=1;
		for(int i=d_IA[vertex_id];i<total;i++){
			if(d_color[d_A[i]]!=NO_COLOR){
				//if this adjacent vertex is already colored then continue
				continue;
			}
			edge_node_val=d_node_val[d_A[i]];
			if(edge_node_val<=curr_node_val){
				is_min=0;
			}
			if(edge_node_val>=curr_node_val){
				is_max=0;
			}
		}
		if(is_min){
			d_color_code[vertex_id]=MIN_COLOR;
			*d_change=1;
		}
		else if(is_max){
			d_color_code[vertex_id]=MAX_COLOR;
			*d_change=1;
		}
		else{
			d_color_code[vertex_id]=NO_COLOR;
			*d_cont=1;
		}
	}
}
__global__
void color_kernel(int *d_color, char *d_color_code, int curr_color, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count && d_color[vertex_id]==NO_COLOR){
		if(d_color_code[vertex_id]==MIN_COLOR){
			d_color[vertex_id]=curr_color;
		}
		else if(d_color_code[vertex_id]==MAX_COLOR){
			d_color[vertex_id]=curr_color+1;
		}
	}
}
int validate_coloring(struct new_csr_graph *input_graph){
	for(int i=0;i<input_graph->v_count;i++){
		for(int j=input_graph->IA[i];j<input_graph->IA[i+1];j++){
			if(input_graph->color[i]==input_graph->color[input_graph->A[j]]){
                cout<<input_graph->color[i]<<" "<<i<<" "<<input_graph->A[j];
				return 0;
			}
		}
	}
	return 1;
}

int main(int argc,char* argv[]){
	new_csr_graph graph;
	clock_t start, end;
	double cpu_time_used;
	start = clock();
    if (argc>0)
    {
	    read_graph(&graph,argv[1]);
        // cout<<argv[1];
    }
    
    end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout << "\n cpu time taken: " << cpu_time_used <<endl;  
    start = clock();
	graph_color(&graph,argv[1]);
    end = clock();
    double gpu_time_used;
    gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    cout << "\n gpu time taken: " << gpu_time_used <<endl;  
    // printArray(graph.IA,graph.v_count+1);
    // printArray(graph.A,graph.IA[graph.v_count]);
    if(validate_coloring(&graph)==0){
		printf("\nInvalid coloring!");
	}else{
        // cout<<"Right";
    }
	return 0;
}
int read_graph(new_csr_graph *graph, char *file_name){
    edge *edge_list;
    ifstream file(file_name);
    string line;
    int phase=1, edge_id=0, e_count, i;
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
                edge_list[edge_id].vertex1=stoi(parts[1])-1;
				edge_list[edge_id].vertex2=stoi(parts[2])-1;
				graph->IA[stoi(parts[1])]++;
				graph->IA[stoi(parts[2])]++;
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
        graph->A[graph->IA[edge_list[edge_id].vertex1] + (vertex_p[edge_list[edge_id].vertex1]++)] = edge_list[edge_id].vertex2;
        graph->A[graph->IA[edge_list[edge_id].vertex2] + (vertex_p[edge_list[edge_id].vertex2]++)] = edge_list[edge_id].vertex1;
    }
    // printArrayC(graph->IA,graph->v_count);
    // printArray(graph->A,graph->v_count);
    delete[] edge_list;
    delete[] vertex_p;
    } else {
        cout << "can't open file" << endl;
    }
    return 0;
}
int printArray(int* arr, int size) {
    int zero=0;
    for(int i = 0; i < size; i++) {
        // cout << arr[i] << " ";
        if (arr[i]==0)
        {
            zero++;
        }
    }
    // cout << endl;
    return zero;
}
void printArrayC(int* arr, int size) {
    for(int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}
void graph_color(struct new_csr_graph *graph,string file_name){
    int cur_color = NO_COLOR + 1;
    char cont = 1, change;
    int *d_A, *d_IA, *d_color;
    char *d_cont, *d_change, *d_color_code;
    int *color_code;
    float *d_node_val;
    cudaError_t err;
    color_code = new int[graph->v_count];
    cudaMallocManaged(&d_A, graph->IA[graph->v_count] * sizeof(int));
    cudaMallocManaged(&d_IA, (graph->v_count + 1) * sizeof(int));
    cudaMallocManaged(&d_color, graph->v_count * sizeof(int));
    cudaMallocManaged(&d_cont, sizeof(char));
    cudaMallocManaged(&d_change, sizeof(char));
    cudaMallocManaged(&d_color_code, graph->v_count * sizeof(char));
    cudaMallocManaged(&d_node_val, graph->v_count * sizeof(float));
    cudaMemcpy(d_A, graph->A, graph->IA[graph->v_count] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IA, graph->IA, (graph->v_count + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(graph->color, d_color, graph->v_count * sizeof(int), cudaMemcpyDeviceToHost);
	init_kernel<<<ceil(graph->v_count/256.0),256>>>(d_color, d_node_val, d_IA, graph->v_count);
	int rand_ver=0;
    int iteration=0;
    bool already_change=0;
    float pre_arrayData,cur_arrayData;
	while(cont){
        iteration++;
		cont=0;
		change=0;
	    cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
        pre_arrayData=cur_arrayData;
        cur_arrayData = printArray(graph->color,graph->v_count);
        // cout<<arrayData/graph->v_count<<endl;
        // csvFile << iteration << "," << arrayData << "\n";
		cudaMemcpy(d_cont,&cont,sizeof(char),cudaMemcpyHostToDevice);
		cudaMemcpy(d_change,&change,sizeof(char),cudaMemcpyHostToDevice);
		minmax_kernel<<<ceil(graph->v_count/256.0),256>>>(d_A, d_IA, d_color, d_node_val, d_color_code, d_cont, d_change, graph->v_count);
		color_kernel<<<ceil(graph->v_count/256.0),256>>>(d_color, d_color_code, cur_color, graph->v_count);
		cudaMemcpy(&cont,d_cont,sizeof(char),cudaMemcpyDeviceToHost);
		cudaMemcpy(&change,d_change,sizeof(char),cudaMemcpyDeviceToHost);

        if((pre_arrayData-cur_arrayData)<0.05&&already_change==0){
            change=1;
        }

		if(cont && !change){
            already_change=1;
            // cout<<"stuck in here"<<endl;
			curandState* d_states;
			cudaMalloc((void **)&d_states, graph->v_count * sizeof(curandState));
			random_generate<<<ceil(graph->v_count/256.0),256>>>(d_node_val, d_states, time(NULL)+rand_ver++, graph->v_count);
			cudaFree(d_states);
		}
		else{
			cur_color+=2;
		}
	}    
    cudaFree(d_A);
	cudaFree(d_IA);
	cudaFree(d_cont);
	cudaFree(d_node_val);
    cout<<"iteration:"<<iteration<<endl;
    cout<<"cur_color: "<<cur_color<<endl;
	cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
    // printArrayC(graph->color,graph->v_count);
    // csvFile << iteration << "," << arrayData << "\n";
    cudaFree(d_color);

}