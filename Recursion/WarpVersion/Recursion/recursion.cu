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
#include <sys/time.h>

#define NO_COLOR 0
#define MIN_COLOR -1
#define MAX_COLOR 1
using namespace std;


static const int WS = 32;  // warp size and bits per int
static const unsigned int Warp = 0xffffffff;
static const int threads_per_block = 1024;  
void graph_color(struct new_csr_graph *graph,string file_name);

struct new_csr_graph{
	int v_count,*A, *IA, *color;
};
struct edge{
	int vertex1, vertex2;
};
int read_graph(new_csr_graph *graph, char *file_name);

__global__
void init_kernel(int *d_color,char *d_color_code, int *pre_color, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
		pre_color[vertex_id]=NO_COLOR;
		d_color[vertex_id]=NO_COLOR;
        d_color_code[vertex_id]=0;
	}
}


__global__
void  color_kernel(int *d_A, int *d_IA, int *d_color,int *pre_color,char *d_color_code, float *d_node_val, int *curr_color, int v_count, char *d_cont){
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = vertex_id / WS; 
    int lane_id = threadIdx.x % WS;
    if(warp_id < v_count && d_color_code[warp_id] == 0){
        int start = d_IA[warp_id];
        int end = d_IA[warp_id + 1];
        bool colored = true;
        for(int i = start + lane_id; i < end; i += 32){ 
            if(i < end){
                if(pre_color[d_A[i]] == pre_color[warp_id]){
                    d_color[d_A[i]]=*curr_color;
                    *d_cont = 1;
                    colored = false;
                }
            }
        }
        unsigned int mask = __ballot_sync(Warp, colored);
        if(mask != 0xffffffff){
            d_color_code[warp_id]=colored;
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
double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

int main(int argc,char* argv[]){
	new_csr_graph graph;
    if (argc>0)
    {
        read_graph(&graph,argv[1]);
    }
	graph_color(&graph,argv[1]);
    
	if(validate_coloring(&graph)==0){
		printf("\nInvalid coloring!");
	}else{
        cout<<"Right";
    }

    return 0;

}
int read_graph(new_csr_graph *graph, char *file_name){
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
                edge_list[edge_id].vertex1=stoi(parts[1])-1;
				edge_list[edge_id].vertex2=stoi(parts[2])-1;
				graph->IA[stoi(parts[1])]++;
				// graph->IA[stoi(parts[2])]++;
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
        // graph->A[graph->IA[edge_list[edge_id].vertex2] + (vertex_p[edge_list[edge_id].vertex2]++)] = edge_list[edge_id].vertex1;
    }
    delete[] edge_list;
    delete[] vertex_p;
    } else {
        cout << "can't open file" << endl;
    }
    return 0;
}
void printArray(int* arr, int size) {
    for(int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}
void printArrayF(float* arr, int size) {
    for(int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void graph_color(struct new_csr_graph *graph,string file_name){
    int cur_color = NO_COLOR + 1;
    char cont = 1;
    int *d_A, *d_IA, *d_color,*pre_color;
    char *d_cont, *d_color_code;
    int *d_cur_color;
    float *d_node_val;
    cudaMallocManaged(&d_A, graph->IA[graph->v_count] * sizeof(int));
    cudaMallocManaged(&d_IA, (graph->v_count + 1) * sizeof(int));
    cudaMallocManaged(&d_color, graph->v_count * sizeof(int));
    cudaMallocManaged(&pre_color, graph->v_count * sizeof(int));
    cudaMallocManaged(&d_cont, sizeof(char));
    cudaMallocManaged(&d_cur_color, sizeof(int));
    cudaMallocManaged(&d_color_code, graph->v_count * sizeof(char));
    cudaMallocManaged(&d_node_val, graph->v_count * sizeof(float));
    cudaMemcpy(d_A, graph->A, graph->IA[graph->v_count] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IA, graph->IA, (graph->v_count + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    int warps_per_block = threads_per_block / WS;
    int blocks_per_grid = (graph->v_count + warps_per_block - 1) / warps_per_block;
	
    init_kernel<<<ceil(graph->v_count/256.0),256>>>(d_color,d_color_code, pre_color, graph->v_count);

    
    int iteration=0;
    double start, end;
    start = rtclock();

	while(cont){
        iteration++;
		cont=0;
		cudaMemcpy(d_cont,&cont,sizeof(char),cudaMemcpyHostToDevice);
		cudaMemcpy(d_cur_color,&cur_color,sizeof(char),cudaMemcpyHostToDevice);
		color_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_IA, d_color,pre_color,d_color_code, d_node_val,d_cur_color,graph->v_count,d_cont);
		cudaMemcpy(&cont,d_cont,sizeof(char),cudaMemcpyDeviceToHost);
        if(cont==0){
            break;
        }
        cudaMemcpy(pre_color, d_color, graph->v_count * sizeof(int), cudaMemcpyDeviceToDevice);
		cur_color++;
	}
    end = rtclock();
    cout<<"GPU used: "<<1000.0f * (end - start)<<" ms"<<endl;
  	cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
	cudaFree(d_IA);
	cudaFree(d_cont);
	cudaFree(d_node_val);
    
    cout<<"iteration:"<<iteration<<endl;
    cout<<"cur_color: "<<cur_color<<endl;

    cudaFree(d_color);
}