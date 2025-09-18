# CHROMA: Coloring with High-Quality Resilient Optimized Multi-GPU Allocation
## Getting started Instructions
+  🖥️ Hardware
    - `CPU`: Intel Core i9-14900 (32 cores) @ 5.5GHz
    - `Memory`: 128 GB
    - `GPU`: NVIDIA GeForce RTX 4090  (each with 24 GB VRAM)
    - `GPU Arch`: `sm_90` (Ada Lovelace, compute capability 8.9)

+ 🧰 OS & Compiler
    - `OS`: Ubuntu 22.04.4 LTS (x86_64)
    - `Kernel`: 6.8.0-52-generic
    - `CUDA`: 12.6
    - `nvcc`: 12.6
    - `NVIDIA Driver`: 560.35.03
    - `Open-MP`(For RGP)
    - `METIS`(For RGP)
+ Important Files/Directories
```
Repository Organization:
    |-- CHROMA
        |-- Source files of CHROMA
    |-- CHROMA_RGP
        |-- Distributed version of CHROMA with resilient graph partition (RGP) 
    |-- JP_Series
        |-- Other ordering heuristic implmented in CUDA on GPUs
    |-- Dataset
        |-- Experiment datasets
```

## State-of-the-art Design Sources
### MIS
1. JP-SL (in /JP-Series)
2. JP-SLL (in /JP-Series)
3. JP-ADG (in /JP-Series)
4. JP-LF([ECL-GC](https://userweb.cs.txstate.edu/~burtscher/research/ECL-GC/))
### SGR
1. [Csrcolor](https://github.com/chenxuhao/csrcolor)
2. [data_pq](https://github.com/chenxuhao/csrcolor)
3. [data_wlc](https://github.com/chenxuhao/csrcolor)
4. [kokkos VB-BIT](https://github.com/kokkos/kokkos-kernels)


## How to Run?
### 1. CHROMA
#### 1) Compile
```
cd ./CHROMA
make [ARCH=sm_xx(default sm_89)]
```
#### 2) Run
```
./CHROMA
```
#### 3) Choose dataset
```
Enter input filename: [input your dataset path]
```
#### 4) Choose RGC optimization (if no input 0)
```
Enter RGC (default θ=10): [input your θ]
```
#### 5) Enable SDC
```
Enable SDC optimization? (y/n): 
```
#### 6) check the result
You will get result like:
```
input: ../Datasets/facebook.egr
nodes: 4039
edges: 176468
avg degree: 43.69
runtime:    1.349632 ms
result verification passed
colors used: 75
```
### 2. CHROMA(RGP)
#### 1) Compile
```
cd ./CHROMA_RGP
make [ARCH=sm_xx(default sm_89)]
```
#### 2) Run
```
./CHROMA
```
#### 3) Choose dataset
```
Enter input filename: [input your dataset path]
```
#### 4) Choose RGC optimization (if no input 0)
```
Enter RGC (default θ=10): [input your θ]
```
#### 5) Choose Partition number
```
nPart: [input your Partition number]
```
#### 6) check the result
You will get result like:
```
Enter input filename: ../Datasets/facebook.egr
Enter RGC (default θ=10): 0
Using RGC θ = 0
nPart: 2
Devide to = 2graphs
input: ../Datasets/facebook.egr
nodes: 4039
edges: 176468
avg degree: 43.69
Boundary node count = 184
Partition 0 => subgraph has 2040 nodes, 51650 edges
Partition 1 => subgraph has 2183 nodes, 126886 edges
Boundary graph has 184 nodes, 2068 edges
h_nodes 184 
[GPU 0] final iteration = 28, final theta = 2147483647
INIT FINISH
RUN LARGE FINISH
RUN SMALL FINISH
runtime:    7.244800 ms
throughput: 0.557503 Mnodes/s
throughput: 24.357884 Medges/s
result verification passed
colors used: 73

```
### 3. Other Ordering Heursitic in JP Algorithm
#### 1) Compile
```
cd ./JP-Series
make [ARCH=sm_xx(default sm_89)]
```
#### 2) Run
```
./JP-Series
```
#### 3) Choose dataset
```
Enter input filename: [input your dataset path]
```
#### 4) Select Algorithm
```
Select algorithm:
1) JP-SL
2) JP-ADG
3) JP-SLL
Enter your choice (1/2/3): [input your algorithm]
```
#### 5) Check the result
You will get result like:
```
Using JP-SL
input: ../Datasets/facebook.egr
nodes: 4039
edges: 176468
avg degree: 43.69
runtime:    12.476577 ms
result verification passed
colors used: 74
```
## APPENDIX
### You can get datasets [here](https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/index.html)
