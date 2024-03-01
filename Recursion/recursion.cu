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
#include "csr_utils.h"

#define NO_COLOR 0
#define MIN_COLOR -1
#define MAX_COLOR 1
using namespace std;

static const int MAX_COLORS = 128;
void graph_color(struct csr_graph *graph,string file_name);


int getDigitCount(int number) {
    if (number == 0) return 1;
    int count = 0;
    while (number != 0) {
        count++;
        number /= 10;
    }
    return count;
}
__global__
void init_kernel(int *d_color,char *d_active_node,int *d_color_mask ,int v_count,int *d_node_val,int *d_IA,int count_num){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
		d_color[vertex_id]=0;
        d_active_node[vertex_id]=0;
        d_color_mask[vertex_id] = 0xFFFFFFFF; 
        d_node_val[vertex_id] = vertex_id+(d_IA[vertex_id + 1]-d_IA[vertex_id])*powf(10.0f, count_num);
	}
}
__global__
void reset_kernal(int *d_color_mask ,int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
        d_color_mask[vertex_id] = 0xFFFFFFFF; 
	}
}

__device__
void insertionSort(int *d_A, int *d_node_val, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = d_A[i];
        int key_node_val = d_node_val[key];
        int j = i - 1;

        while (j >= left && d_node_val[d_A[j]] < key_node_val) {
            d_A[j + 1] = d_A[j];
            j = j - 1;
        }
        d_A[j + 1] = key;
    }
}
__device__
void print_bits(int num) {
    char bit_rep[33];
    bit_rep[32] = '\0'; 
    for (int bit = 31; bit >= 0; bit--) {
        bit_rep[31 - bit] = (num & (1 << bit)) ? '1' : '0';
    }
    printf("%s\n", bit_rep);
}
__global__
void sort_kernel(int *d_color, char *d_active_node, int v_count, int *d_node_val, int *d_A, int *d_IA) {
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id < v_count) {
        int start = d_IA[vertex_id];
        int end = d_IA[vertex_id + 1] - 1; 
        insertionSort(d_A, d_node_val, start, end);
    }
}

__global__
void  broadcast_kernel(int *d_A, int *d_IA, int *d_color,int *d_color_mask,char *d_color_code, int v_count){
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = vertex_id / WS; 
    int lane_id = threadIdx.x % WS;
    if(warp_id < v_count){
        if (lane_id == 0) {
            d_color_mask[warp_id] = 0xFFFFFFFF; 
        }
        __syncwarp();
        int start = d_IA[warp_id];
        int end = d_IA[warp_id + 1];
        for(int i = start + lane_id; i < end; i += 32){ 
            int neighbor_id = d_A[i]; 
            int neighbor_color = d_color[neighbor_id]; 
            int mask = ~(1 << neighbor_color); 
            atomicAnd(&d_color_mask[warp_id], mask);
        }
    }
}

__global__
void  color_kernel(int *d_A, int *d_IA, int *d_color,int *d_color_mask,char *d_color_code, int *d_node_val, int v_count,char *d_cont){
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = vertex_id / WS; 
    int lane_id = threadIdx.x % WS;
    bool conflict = false;
    bool short_cut = false;
    int available_color=-1;
    if(warp_id < v_count){
        int start = d_IA[warp_id];
        int end = d_IA[warp_id + 1];
        int color = d_color[warp_id]; 
        for(int i = start + lane_id; i < end; i += 32){ 
            int neighbor_id = d_A[i]; 
            if(d_color[neighbor_id] == color){
                *d_cont = 1;
                if(d_node_val[warp_id]<d_node_val[neighbor_id]){
                conflict = true;
                break; 
                }
            }
        }
        short_cut = __any_sync(0xffffffff, conflict);
        for(int i = start + lane_id; i < end; i += 32){ 
            int neighbor_id = d_A[i]; 
            if(d_color[neighbor_id] == color){
                if(d_node_val[warp_id]>d_node_val[neighbor_id]){
                    if(short_cut){
                    available_color = __ffs(d_color_mask[neighbor_id])-1;
                    }
                    else{
                    available_color = 32-(__ffs(__brev(d_color_mask[neighbor_id])));
                    }
                    d_color[neighbor_id] = available_color;
                }
            }
        }
    }
}

int main(int argc,char* argv[]){
	csr_graph graph;
    if (argc>0)
    {
        read_graph(&graph,argv[1],1);
    }
	graph_color(&graph,argv[1]);
    print_array(graph.color,graph.v_count);

	if(validate_coloring(&graph)==0){
		printf("\nInvalid coloring!");
	}else{
        cout<<"Right";
    }

    return 0;

}

void graph_color(struct csr_graph *graph,string file_name){
    char cont = 1;

    int *d_A, *d_IA, *d_color,*d_color_mask;
    char *d_cont, *d_active_node, *d_color_code;

    int *d_node_val;
    cudaMallocManaged(&d_A, graph->IA[graph->v_count] * sizeof(int));
    cudaMallocManaged(&d_IA, (graph->v_count + 1) * sizeof(int));
    cudaMallocManaged(&d_color, graph->v_count * sizeof(int));
    cudaMallocManaged(&d_color_mask, graph->v_count * sizeof(int));
    cudaMallocManaged(&d_cont, sizeof(char));
    cudaMallocManaged(&d_active_node, graph->v_count * sizeof(char));
    cudaMallocManaged(&d_node_val, graph->v_count * sizeof(int));
    cudaMemcpy(d_A, graph->A, graph->IA[graph->v_count] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IA, graph->IA, (graph->v_count + 1) * sizeof(int), cudaMemcpyHostToDevice);
    int count_num=getDigitCount(graph->v_count);
    int threads_per_block=1024;
    int warps_per_block = threads_per_block / WS;
    int blocks_per_grid = (graph->v_count + warps_per_block - 1) / warps_per_block;

    int iteration=0;
    double start, end;

    init_kernel<<<ceil(graph->v_count/256.0),256>>>(d_color,d_active_node,d_color_mask ,graph->v_count,d_node_val,d_IA,count_num);
    cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
    print_array(graph->color,graph->v_count);

    sort_kernel<<<ceil(graph->v_count/256.0),256>>>(d_color,d_active_node ,graph->v_count,d_node_val,d_A,d_IA);
    start = rtclock();
	while(cont){
        iteration++;
		cont=0;
		cudaMemcpy(d_cont,&cont,sizeof(char),cudaMemcpyHostToDevice);
        // reset_kernal<<<ceil(graph->v_count/256.0),256>>>(d_color_mask,graph->v_count);
        broadcast_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_IA, d_color,d_color_mask,d_color_code,graph->v_count);
        color_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_IA, d_color,d_color_mask,d_color_code, d_node_val,graph->v_count,d_cont);
        cudaMemcpy(&cont,d_cont,sizeof(char),cudaMemcpyDeviceToHost);
        // cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
        // print_array(graph->color,graph->v_count);
    }
    end = rtclock();
    cout<<"GPU used: "<<1000.0f * (end - start)<<" ms"<<endl;
  	cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_A);
	cudaFree(d_IA);
	cudaFree(d_cont);
	cudaFree(d_node_val);
    cout<<"iteration:"<<iteration<<endl;
    cudaFree(d_color);
}