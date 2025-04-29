# CHROMA: Coloring with High-Quality Resilient Optimized Multi-GPU Allocation
## Getting started Instructions.
+ Clone this project git clone git@github.com:NTUDDSNLab/CHROMA.git
+ Hardware:
+ OS & Compler:
+ Important Files/Directories
    + data/: contains all datasets that we want to compare with.
Repository Organization:
    |-- CHROMA :our main code(Color Allocate cite [ECL-GC](https://userweb.cs.txstate.edu/~burtscher/research/ECL-GC/))
        |-- CHROMA.cu        
        |-- chroma_utils.cuh  
        |-- ECLgraph.h  
        |-- globals.cuh  
        |-- PA.cu
        |-- chroma_utils.cu  
        |-- ECLGC.cu          
        |-- globals.cu  
        |-- Makefile
    |-- dataset :Some datasets(all datasets can download [here](https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/index.html))
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