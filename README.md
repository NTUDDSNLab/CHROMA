# CHROMA: Coloring with High-Quality Resilient Optimized Multi-GPU Allocation
## Getting started Instructions.
+ Clone this project git clone git@github.com:NTUDDSNLab/CHROMA.git
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
+ Important Files/Directories
```
Repository Organization:
    |-- CHROMA
        |-- CHROMA.cu        
        |-- chroma_utils.cuh  
        |-- ECLgraph.h  
        |-- globals.cuh  
        |-- PA.cu
        |-- chroma_utils.cu  
        |-- ECLGC.cu          
        |-- globals.cu  
        |-- Makefile
    |-- Dataset
        |-- Email-Enron.col.egr  
        |-- school1.egr               
        |-- soc-Slashdot0902.col.egr  
        |-- wiki-Vote.col.egr
        |-- facebook.egr         
        |-- soc-Epinions1.col.egr     
        |-- Stanford.egr              
        |-- youtube.egr
        |-- le450_25d.egr        
        |-- soc-Slashdot0811.col.egr  
        |-- twitter_combined.col.egr
```


## Environment Setup
## 2. Environment Setup

### 1) Compile
```
cd ./CHROMA
make
```
### 2) Run
```
./CHROMA
```
### 3) Choose dataset
```
Enter input filename: [input your dataset path]
```
### 4) Choose RGC optimization (if no input 0)
```
Enter RGC (default $\theta$=10): [input your $\theta$]
```
### 5) Enable SDC
```
Enable SDC optimization? (y/n): 
```
### 6) check the result
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