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

#include <unordered_set>
#define NO_COLOR 0
#define MIN_COLOR -1
#define MAX_COLOR 200
#define BLOCK_SIZE 1000
using namespace std;
// void printArray(int* arr, int size);
void graph_color(struct new_csr_graph *graph,string file_name);

struct new_csr_graph{
	int v_count,*A, *IA, *color;
};
struct edge{
	int vertex1, vertex2;
};
int read_graph(new_csr_graph *graph, char *file_name);

double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

__global__
void init_kernel(int *d_color,bool *d_color_code, float *d_node_val, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
		d_node_val[vertex_id]=vertex_id;
		d_color[vertex_id]=NO_COLOR;
        d_color_code[vertex_id]=true;
	}
}

__global__
void  color_kernel(int *d_A, int *d_IA, int *d_color,bool *d_color_code, float *d_node_val, int curr_color, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count && d_color_code[vertex_id]){
		int total=__ldg(&d_IA[vertex_id+1]);
		for(int i=__ldg(&d_IA[vertex_id]);i<total;i++){
			if(d_color[d_A[i]]==d_color[vertex_id]){
				d_color[d_A[i]]=curr_color;
			}
		}
	}
}

__global__
void  check_kernel(int *d_A, int *d_IA, int *d_color,bool *d_color_code, float *d_node_val, int curr_color, char *d_cont, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
    int colored=false;
	if(vertex_id<v_count && d_color_code[vertex_id]){
		int total=__ldg(&d_IA[vertex_id+1]);
		for(int i=__ldg(&d_IA[vertex_id]);i<total;i++){
			if(d_color[d_A[i]]==d_color[vertex_id]){
    			*d_cont=1;
                colored=true;
			}
		}
        d_color_code[vertex_id]=colored;
	}
}

__global__
void greedy_kernel(int *d_A, int *d_IA, int *d_color, bool *d_color_code, float *d_node_val, int curr_color, int v_count){
    int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;
    int min_neighbor_color=INT_MAX;
    if(vertex_id < v_count && d_color_code[vertex_id]){
        int total = __ldg(&d_IA[vertex_id + 1]);
        bool used_colors[MAX_COLOR] = { false };

        for(int i = __ldg(&d_IA[vertex_id]); i < total; i++){
            int neighbor_color = d_color[d_A[i]];
            if(neighbor_color < MAX_COLOR){
                if(neighbor_color<min_neighbor_color){
                    min_neighbor_color=neighbor_color;
                }
                used_colors[neighbor_color] = true;
                
            }
        }

        int smallestMissingColor = min_neighbor_color;
        while(smallestMissingColor < curr_color && used_colors[smallestMissingColor]){
            smallestMissingColor++;
        }

        d_color[vertex_id] = smallestMissingColor;
    }
}
__global__
void  greedy_check_kernel(int *d_A, int *d_IA, int *d_color,bool *d_color_code, float *d_node_val, int curr_color, char *d_cont, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
    int colored=false;
	if(vertex_id<v_count){
		int total=__ldg(&d_IA[vertex_id+1]);
		for(int i=__ldg(&d_IA[vertex_id]);i<total;i++){
			if(d_color[d_A[i]]==d_color[vertex_id]){
    			*d_cont=1;
                colored=true;
			}
		}
        d_color_code[vertex_id]=colored;
	}
}

int validate_coloring(struct new_csr_graph *input_graph){
	for(int i=0;i<input_graph->v_count;i++){
		for(int j=input_graph->IA[i];j<input_graph->IA[i+1];j++){
			if(input_graph->color[i]==input_graph->color[input_graph->A[j]]){
                // cout<<input_graph->color[i]<<" "<<i<<" "<<input_graph->A[j];
				return 0;
			}
		}
	}
	return 1;
}

int main(int argc,char* argv[]){
	new_csr_graph graph;
	double start, end;
	double cpu_time_used;
	start = rtclock();
    if (argc>0)
    {
	    read_graph(&graph,argv[1]);
    }

    end = rtclock();
	cpu_time_used = (1000.0f) * (end -start);
    cout << "\n cpu time taken: " << cpu_time_used <<endl;  
    start = rtclock();
	graph_color(&graph,argv[1]);
    end = rtclock();
    double gpu_time_used;
    gpu_time_used = (1000.0f) * (end -start);
    cout << "\n gpu time taken: " << gpu_time_used <<endl;  
    
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
int unColoring(bool* arr, int size) {
    int zero=0;
    for(int i = 0; i < size; i++) {
        // cout << arr[i] << " ";
        if (arr[i])
        {
            zero++;
        }
    }
    // cout << endl;
    return zero;
}
int count_color(int* arr, int size) {
    std::unordered_set<int> unique_colors;

    for (int i = 0; i < size; ++i) {
        unique_colors.insert(arr[i]);
    }

    return unique_colors.size();
}
void graph_color(struct new_csr_graph *graph,string file_name){
    ofstream csvFile(file_name+".csv"); 
    int cur_color = NO_COLOR;
    char cont = 1;
    int *d_A, *d_IA, *d_color;
    char *d_cont;
    bool *d_color_code;
    float *d_node_val;
    bool *active_node;
    active_node = new bool[graph->v_count];
    cudaMallocManaged(&d_A, graph->IA[graph->v_count] * sizeof(int));
    cudaMallocManaged(&d_IA, (graph->v_count + 1) * sizeof(int));
    cudaMallocManaged(&d_color, graph->v_count * sizeof(int));
    cudaMallocManaged(&d_cont, sizeof(char));
    cudaMallocManaged(&d_color_code, graph->v_count * sizeof(bool));
    cudaMallocManaged(&d_node_val, graph->v_count * sizeof(float));
    cudaMemcpy(d_A, graph->A, graph->IA[graph->v_count] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IA, graph->IA, (graph->v_count + 1) * sizeof(int), cudaMemcpyHostToDevice);
	init_kernel<<<ceil(graph->v_count/256.0),256>>>(d_color,d_color_code, d_node_val, graph->v_count);
	cudaMemcpy(active_node, d_color_code, graph->v_count * sizeof(bool), cudaMemcpyDeviceToHost);
    // printArrayF(d_node_val_cpu,graph->v_count);
    int iteration=0;
    float unColoringNum=graph->v_count;
    float threadhold=0.3;

	while(unColoringNum/graph->v_count>threadhold){
        iteration++;
		// cont=0;
		// cudaMemcpy(d_cont,&cont,sizeof(char),cudaMemcpyHostToDevice);
		color_kernel<<<ceil(graph->v_count/256.0),256>>>(d_A, d_IA, d_color,d_color_code, d_node_val,cur_color,graph->v_count);
		check_kernel<<<ceil(graph->v_count/256.0),256>>>(d_A, d_IA, d_color,d_color_code, d_node_val,cur_color, d_cont,graph->v_count);
		// cudaMemcpy(&cont,d_cont,sizeof(char),cudaMemcpyDeviceToHost);

        // cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(active_node, d_color_code, graph->v_count * sizeof(bool), cudaMemcpyDeviceToHost);
        unColoringNum = unColoring(active_node,graph->v_count);
        // printArray(graph->color,graph->v_count);
        // cout<<unColoringNum/graph->v_count<<endl;
        cur_color++;
	}

    // cout<<"switch mode:"<<" "<<iteration<<endl;
	while(cont){
        iteration++;
        // cout<<iteration<<endl;
		cont=0;
		cudaMemcpy(d_cont,&cont,sizeof(char),cudaMemcpyHostToDevice);
		greedy_kernel<<<ceil(graph->v_count/256.0),256>>>(d_A, d_IA, d_color,d_color_code, d_node_val,cur_color,graph->v_count);
		greedy_check_kernel<<<ceil(graph->v_count/256.0),256>>>(d_A, d_IA, d_color,d_color_code, d_node_val,cur_color, d_cont,graph->v_count);
		cudaMemcpy(&cont,d_cont,sizeof(char),cudaMemcpyDeviceToHost);
		cur_color++;
	}
    cout<<"finish:"<<" "<<iteration<<endl;

  	cudaMemcpy(graph->color,d_color,graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
    int color_num=count_color(graph->color,graph->v_count);
    
    cudaFree(d_A);
	cudaFree(d_IA);
	cudaFree(d_cont);
	cudaFree(d_node_val);
    cout<<"iteration:"<<iteration<<endl;
    cout<<"cur_color: "<<color_num<<endl;

    cudaFree(d_color);
}
