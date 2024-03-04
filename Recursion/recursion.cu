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
#include "csr_utils.cuh"
#include <cuda_runtime.h>
#define MAX_COLOR 1
using namespace std;

static const int MAX_COLORS = 512;
static const int MAX_COLORS_NUM = MAX_COLORS/WS;
int graph_color(struct csr_graph *graph,string file_name);
std::vector<float> timings;


int getDigitCount(int number) {
    if (number == 0) return 1;
    int count = 0;
    while (number != 0) {
        count++;
        number /= 10;
    }
    return count;
}
// __global__ void init_kernel(int *d_color, int v_count, int *d_node_val, int *d_IA, int count_num) {
// 	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
// 	if(vertex_id<v_count){
// 		d_color[vertex_id]=0;
//         d_node_val[vertex_id] = vertex_id+(d_IA[vertex_id + 1]-d_IA[vertex_id])*powf(10.0f, count_num);
// 	}
// }
__device__ int pow10(int exponent) {
    int result = 1;
    for (int i = 0; i < exponent; i++) {
        result *= 10;
    }
    return result;
}

__global__ void init_kernel(int *d_color, int *v_count, int *d_node_val,char *d_active_node, int *d_IA, int count_num, int rand_num) {
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_id < *v_count) {
        unsigned int seed = vertex_id;
        unsigned int randVal = (seed * 1103515245 + 12345) & 0x7fffffff;
        d_color[vertex_id] = randVal % (rand_num); // Ensure the random value is within [0, count_num]
        int power_of_ten = pow10(count_num);
        d_node_val[vertex_id] = ((d_IA[vertex_id + 1] - d_IA[vertex_id]) * power_of_ten) + vertex_id;
        d_active_node[vertex_id]=true;
        // d_node_val[vertex_id] =  ((d_IA[vertex_id + 1] - d_IA[vertex_id]) * powf(10.0f, count_num))+vertex_id;
        // printf("[%d-%d]",vertex_id, d_node_val[vertex_id]);
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

// __global__
// void  broadcast_kernel(int *d_A, int *d_IA, int *d_color,int *d_color_mask,char *d_color_code, int v_count){
//     int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
//     int warp_id = vertex_id / WS; 
//     int lane_id = threadIdx.x % WS;
//     if(warp_id < v_count){
//         if (lane_id == 0) {
//             d_color_mask[warp_id] = 0xFFFFFFFF; 
//         }
//         __syncwarp();
//         int start = d_IA[warp_id];
//         int end = d_IA[warp_id + 1];
//         for(int i = start + lane_id; i < end; i += 32){ 
//             int neighbor_id = d_A[i]; 
//             int neighbor_color = d_color[neighbor_id]; 
//             int mask = ~(1 << neighbor_color); 
//             atomicAnd(&d_color_mask[warp_id], mask);
//         }
//     }
// }

__global__
void broadcast_kernel(int *d_A, int *d_IA, int *d_color, int *d_color_mask, char *d_active_node, int *v_count) {
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = vertex_id / WS; 
    int lane_id = threadIdx.x % WS;
    bool active = false;
    if (warp_id < *v_count && d_active_node[warp_id]==true) {
        int color = d_color[warp_id]; 
        if (lane_id == 0) {
            for (int i = 0; i < MAX_COLORS_NUM; ++i) {
                d_color_mask[warp_id * MAX_COLORS_NUM + i] = 0xFFFFFFFF;
            }
        }
        __syncwarp();
        int start = d_IA[warp_id];
        int end = d_IA[warp_id + 1];
        for (int i = start + lane_id; i < end; i += WS) { 
            int neighbor_id = d_A[i]; 
            int neighbor_color = d_color[neighbor_id];
            if(d_color[neighbor_id] == color){
                active = true;
            }
            int index = neighbor_color / WS;
            int bit_position = neighbor_color % WS;
            int mask = ~(1 << bit_position);
            atomicAnd(&d_color_mask[warp_id * MAX_COLORS_NUM + index], mask);
        }
        d_active_node[warp_id] = __any_sync(0xffffffff, active);
    }
}

__global__
void  dual_color_kernel(int *d_A, int *d_IA, int *d_color,int *d_color_mask,char *d_active_node, int *d_node_val, int *v_count,char *d_cont){
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = vertex_id / WS; 
    int lane_id = threadIdx.x % WS;
    bool conflict = false;
    bool dual_color = false;
    // bool active = false;
    if(warp_id < *v_count && d_active_node[warp_id]==true){
        int start = d_IA[warp_id];
        int end = d_IA[warp_id + 1];
        int color = d_color[warp_id]; 
        for(int i = start + lane_id; i < end; i += 32){ 
            int neighbor_id = d_A[i]; 
            if(d_color[neighbor_id] == color){
                *d_cont = 1;
                // active = true;
                if(d_node_val[warp_id]<d_node_val[neighbor_id]){
                conflict = true;
                break; 
                }
            }
        }
        dual_color = __any_sync(0xffffffff, conflict);
        // d_active_node[warp_id] = __any_sync(0xffffffff, active);
        for(int i = start + lane_id; i < end; i += 32){ 
            int neighbor_id = d_A[i]; 
            if(d_color[neighbor_id] == color){
                if(d_node_val[warp_id]>d_node_val[neighbor_id]){
                for(int mask_index = 0; mask_index < MAX_COLORS_NUM; ++mask_index) {
                    int mask_value = d_color_mask[neighbor_id * MAX_COLORS_NUM + mask_index];
                    int available_color = -1;
                    int best_color = __ffs(mask_value)-1;
                    int inverse_best_color = 32-(__ffs(__brev(mask_value)));
                    if(dual_color){
                    available_color = best_color;
                    }
                    else{
                    available_color = inverse_best_color;
                    }
                    if(best_color != -1) { 
                        if(d_color[neighbor_id]!= inverse_best_color+ mask_index * 32){
                            d_color[neighbor_id] = available_color + mask_index * 32; 
                        }
                        break; 
                    }
                }
                }
            }
        }
    }
}


__global__
void  color_kernel(int *d_A, int *d_IA, int *d_color,int *d_color_mask,char *d_active_node, int *d_node_val, int *v_count,char *d_cont){
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = vertex_id / WS; 
    int lane_id = threadIdx.x % WS;
    if(warp_id < *v_count){
        int start = d_IA[warp_id];
        int end = d_IA[warp_id + 1];
        int color = d_color[warp_id]; 
        for(int i = start + lane_id; i < end; i += 32){ 
            int neighbor_id = d_A[i]; 
            if(d_color[neighbor_id] == color){
                *d_cont = 1;
                if(d_node_val[warp_id]>d_node_val[neighbor_id]){
                for(int mask_index = 0; mask_index < MAX_COLORS_NUM; ++mask_index) {
                    int mask_value = d_color_mask[warp_id * MAX_COLORS_NUM + mask_index];
                    int available_color = -1;
                    available_color = __ffs(mask_value)-1;
                    if(available_color != -1) { 
                        d_color[warp_id] = available_color + mask_index * 32; 
                        break; 
                    }
                }
                }
            }
        }
    }
}

// int main(int argc,char* argv[]){
// 	csr_graph graph;
//     if (argc>0)
//     {
//         read_graph(&graph,argv[1],1);
//     }
// 	graph_color(&graph,argv[1]);
//     print_array(graph.color,graph.v_count);

// 	if(validate_coloring(&graph)==0){
// 		printf("\nInvalid coloring!");
// 	}else{
//         cout<<"Right";
//     }

//     return 0;

// }
int main(int argc, char** argv) {
    if (argc <= 1) {
        std::cerr << "Usage: " << argv[0] << " <graph_file>" << std::endl;
        return 1;
    }

    std::vector<int> color_num;
    csr_graph graph;
    read_graph(&graph, argv[1], 1); 

    for (int i = 0; i < 1; ++i) {
        color_num.push_back(graph_color(&graph, argv[1]));
        // if (i == 0) {
        //     print_array(graph.color, graph.v_count);
        // }
    }
    float averageTime = 0;
    float averageColor = 0;
    
    for (int c : color_num) {
        averageColor += c;
    }
    averageColor /= color_num.size();

    for (float t : timings) {
        averageTime += t;
    }

    averageTime /= timings.size();

    std::cout << "Average Color usage: " << averageColor << "  " << std::endl;
    std::cout << "Average GPU usage: " << averageTime << " ms" << std::endl;

    return 0;
}

int graph_color(struct csr_graph *graph,string file_name){
    char cont = 1;

    int *d_A, *d_IA, *d_color,*d_color_mask;
    char *d_cont, *d_active_node;

    int *d_node_val;
    size_t size = graph->v_count * MAX_COLORS_NUM * sizeof(int);
    cudaMallocManaged(&d_color_mask, size);

    cudaMallocManaged(&d_A, graph->IA[graph->v_count] * sizeof(int));
    cudaMallocManaged(&d_IA, (graph->v_count + 1) * sizeof(int));
    cudaMallocManaged(&d_color, graph->v_count * sizeof(int));
    cudaMallocManaged(&d_cont, sizeof(char));
    cudaMallocManaged(&d_active_node, graph->v_count * sizeof(char));
    cudaMallocManaged(&d_node_val, graph->v_count * sizeof(int));
    cudaMemcpy(d_A, graph->A, graph->IA[graph->v_count] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IA, graph->IA, (graph->v_count + 1) * sizeof(int), cudaMemcpyHostToDevice);
    int count_num=getDigitCount(graph->v_count);
    // cout<<"count_num"<<count_num<<endl;
    int threads_per_block=1024;
    int warps_per_block = threads_per_block / WS;
    int blocks_per_grid = (graph->v_count + warps_per_block - 1) / warps_per_block;

    int iteration=0;

    int hv_count=graph->v_count;
    int *dv_count;
    cudaMallocManaged(&dv_count, sizeof(int));

    cudaMemcpy(dv_count,&hv_count,sizeof(int),cudaMemcpyHostToDevice);


    int rand_num=(graph->edge/graph->v_count)*count_num;
    // cout<<"rand_num:"<<rand_num<<endl;
    GPUTimer timer;
    
    timer.start();
    init_kernel<<<ceil(graph->v_count/1024.0),1024>>>(d_color ,dv_count,d_node_val,d_active_node,d_IA,count_num,rand_num);

    int* h_node_val = (int*)malloc(hv_count * sizeof(int)); // 分配主机内存
    // cudaMemcpy(h_node_val, d_node_val, hv_count * sizeof(int), cudaMemcpyDeviceToHost);
    // print_array(h_node_val,graph->v_count);

	while(cont){
        iteration++;
		cont=0;
		cudaMemcpy(d_cont,&cont,sizeof(char),cudaMemcpyHostToDevice);
        broadcast_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_IA, d_color,d_color_mask,d_active_node,dv_count);
        dual_color_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_IA, d_color,d_color_mask,d_active_node, d_node_val,dv_count,d_cont);
        // dual_color_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_IA, d_color,d_color_mask,d_color_code, d_node_val,dv_count,d_cont);
        cudaMemcpy(&cont,d_cont,sizeof(char),cudaMemcpyDeviceToHost);

        // cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
        // print_array(graph->color,graph->v_count);
    }
    const float runtime = timer.stop();

    timings.push_back(runtime*1000);
  	cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
    std::set<int> uniqueColors;
    for(int i = 0; i < graph->v_count; ++i) {
        uniqueColors.insert(graph->color[i]);
    }
    int uniqueColorCount = uniqueColors.size();
    // std::cout << "Number of unique colors: " << uniqueColorCount << std::endl;
    cudaFree(d_A);
	cudaFree(d_IA);
	cudaFree(d_cont);
	cudaFree(d_node_val);
    cout<<"iteration:"<<iteration<<endl;
    cudaFree(d_color);
    return uniqueColorCount;
}