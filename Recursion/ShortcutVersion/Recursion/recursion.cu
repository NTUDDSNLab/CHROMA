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
void graph_color(struct csr_graph *graph,string file_name);


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
void  color_kernel(int *d_A, int *d_IA, int *d_color,int *pre_color,char *d_color_code, float *d_node_val, int *curr_color, int v_count, char *d_cont, char *d_shortcut){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
    int colored=1;
	if(vertex_id<v_count && d_color_code[vertex_id]==0){
		int total=d_IA[vertex_id+1];
        bool incoming=false;
		for(int i=d_IA[vertex_id];i<total;i++){
			if(d_color[d_A[i]]==d_color[vertex_id]){
                if(vertex_id>d_A[i]){
                    *d_shortcut=1;
                    incoming=true;
                }
                else{
                if(incoming){
				        d_color[d_A[i]]=*curr_color+1;
                    }
                    else{
				        d_color[d_A[i]]=*curr_color;
                    }
                }
                *d_cont=1;
                colored=0;

			}
		}
        d_color_code[vertex_id]=colored;
	}
}



int validate_coloring(struct csr_graph *input_graph){
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
	csr_graph graph;
    if (argc>0)
    {
        read_graph(&graph,argv[1],1);
    }
    // print_array(graph.IA,graph.v_count+1);    
    // print_array(graph.A,graph.IA[graph.v_count-1]);    
	graph_color(&graph,argv[1]);
    
	if(validate_coloring(&graph)==0){
		printf("\nInvalid coloring!");
	}else{
        cout<<"Right";
    }

    return 0;

}

void graph_color(struct csr_graph *graph,string file_name){
    int cur_color = NO_COLOR + 1;
    char cont = 1;
    char shortcut;

    int *d_A, *d_IA, *d_color,*pre_color;
    char *d_cont, *d_color_code,*d_shortcut;
    int *d_cur_color;
    float *d_node_val;
    cudaMallocManaged(&d_A, graph->IA[graph->v_count] * sizeof(int));
    cudaMallocManaged(&d_IA, (graph->v_count + 1) * sizeof(int));
    cudaMallocManaged(&d_color, graph->v_count * sizeof(int));
    cudaMallocManaged(&pre_color, graph->v_count * sizeof(int));
    cudaMallocManaged(&d_cont, sizeof(char));
    cudaMallocManaged(&d_shortcut, sizeof(char));
    cudaMallocManaged(&d_cur_color, sizeof(int));
    cudaMallocManaged(&d_color_code, graph->v_count * sizeof(char));
    cudaMallocManaged(&d_node_val, graph->v_count * sizeof(float));
    cudaMemcpy(d_A, graph->A, graph->IA[graph->v_count] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IA, graph->IA, (graph->v_count + 1) * sizeof(int), cudaMemcpyHostToDevice);
	init_kernel<<<ceil(graph->v_count/256.0),256>>>(d_color,d_color_code, pre_color, graph->v_count);

    
    int iteration=0;
    double start, end;
    start = rtclock();

	while(cont){
        iteration++;
		cont=0;
        shortcut=0;
		cudaMemcpy(d_cont,&cont,sizeof(char),cudaMemcpyHostToDevice);
		cudaMemcpy(d_cur_color,&cur_color,sizeof(char),cudaMemcpyHostToDevice);
		color_kernel<<<ceil(graph->v_count/256.0),256>>>(d_A, d_IA, d_color,pre_color,d_color_code, d_node_val,d_cur_color,graph->v_count,d_cont,d_shortcut);
		cudaMemcpy(&cont,d_cont,sizeof(char),cudaMemcpyDeviceToHost);
		cudaMemcpy(&shortcut,d_shortcut,sizeof(char),cudaMemcpyDeviceToHost);
        if(cont==0){
            break;
        }
        // cout<<shortcut<<endl;
        cudaMemcpy(pre_color, d_color, graph->v_count * sizeof(int), cudaMemcpyDeviceToDevice);
        if(shortcut==1){
		cur_color+=2;
        }else{
		cur_color++;
        }
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