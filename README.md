# GraphColoring
以下是項目結構：

    |-- Recursion
        |-- recursion.cu
    |-- hybridparallelgraphcol
        |-- hybridparallelgraphcol.cu
    |-- recustion_topo
        |-- recustion_topo.cu
    |-- dataset
        |-- complete_graph_2000.txt
        |-- complete_graph_4.txt
        |-- facebook.txt
        |-- jean.col.txt
        |-- le450_25d.col.txt
        |-- miles750.col.txt
        |-- queen11_11.col.txt
        |-- queen9_9.col.txt
## hybridparallelgraphcol
+ from:https://github.com/melchizedekdas/gpu-graph-coloring/tree/master

+ iteration的著色比例<5%時會由degree mode 切換成 random mode
### Compile command

```
nvcc hybridparallelgraphcol.cu -o hybridparallelgraphcol
```
### execute command
```
./hybridparallelgraphcol ../dataset_path
```
## Recursion
### Compile command

```
nvcc recursion.cu -o recursion
```
### execute command
```
./recursion ../dataset_path
```
## Fake Feluca ​
### Compile command

```
nvcc feluca.cu -o feluca
```
### execute command
```
./feluca ../dataset_path
```
## recustion_topo
+ recustion算法+topo算法
+ 當為著色節點<40%時會由recustion切換到topo，以此來避免長尾效應
### Compile command

```
nvcc recustion_topo.cu -o recustion_topo
```
### execute command
```
./recustion_topo ../dataset_path
```
## Comepare
### Facebook
#### Time(ms)
#### Our
|hybridparallelgraphcol|  Recursion | Fake Feluca   | Recustion+topo  |
|  :----:  |  :----:  | :----:  | :----:  |
|  45.4  |  43.5  | 61.5  | 52.23  |
|  43.4  |  42.7  | 59.9  | 49.8  |

#### [csrcolor](https://github.com/chenxuhao/csrcolor/tree/master)
|DataSet|serial|  topo | GM   | csrcolor  |
|  :----:  |  :----:  |  :----:  | :----:  | :----:  |
|  facebook  |  0.231075  |  3.701735  | 85.5281  | 9.082413  |
|  asia_osm  |  109  |  49.621177  | -  | 42.564845  |

#### color
#### Our(後面trace了一下發現csrcolor的時間單單只有執行時間，沒有把數據遷移算進去，所以差距很大)

|hybridparallelgraphcol|  Recursion | Fake Feluca   | Recustion+topo  |
|  :----:  |  :----:  | :----:  | :----:  |
|  186.344  |  156.7  | 95  | 87.85  |
#### [csrcolor](https://github.com/chenxuhao/csrcolor/tree/master)
|DataSet|serial|  topo | GM   | csrcolor  |
|  :----:  |  :----:  |  :----:  | :----:  | :----:  |
|  facebook |  86.00  |  86.9  | 87.8  | 208.00  |
|  asia_osm  |  5  |  5  | -  | 32  |
