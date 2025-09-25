
// A C++ program to implement greedy algorithm for graph coloring
#include <iostream>
#include <list>
#include "ECLgraph.h"
#include <vector>
#include <algorithm>
#include <list>
#include <utility>
#include <cstdio>

#include <chrono>  // 引入 <chrono> 標頭檔案
#include <set>
using namespace std;
 
// A class that represents an undirected graph
class Graph
{
    int V;    // No. of vertices
    list<int> *adj;    // A dynamic array of adjacency lists
public:
    // Constructor and destructor
    Graph(int V)   { this->V = V; adj = new list<int>[V]; }
    ~Graph()       { delete [] adj; }
 
    // function to add an edge to graph
    void addEdge(int v, int w);
 
    // Prints greedy coloring of the vertices
    void greedyColoring();
};
 
void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w);
    adj[w].push_back(v);  // Note: the graph is undirected
}
 void Graph::greedyColoring()
{
    std::cout << "Number of vertices (V): " <<  V << std::endl;
    // int result[V];  // Array to store result (color assignment)
    // int saturation[V]; // Array to store the saturation degree of each vertex
    int* result = new int[V];  // Array to store result (color assignment)
    int* saturation = new int[V]; // Array to store the saturation degree of each vertex
    
    // Initialize all vertices as unassigned and saturation as 0
    for (int u = 0; u < V; u++) {
        result[u] = -1;  // No color is assigned
        saturation[u] = 0;  // Initial saturation is 0
    }
    // Initialize available colors array
    bool available[V];
    for (int cr = 0; cr < V; cr++)
        available[cr] = false;

    // Array to store the degree of each vertex
    std::vector<int> degree(V);
    for (int u = 0; u < V; u++)
        degree[u] = adj[u].size(); // Degree of each vertex

    // Start with the vertex of maximum degree
    int u = std::max_element(degree.begin(), degree.end()) - degree.begin();
    result[u] = 0; // Assign the first color

    // Update saturation degrees of adjacent vertices
    for (int neighbor : adj[u])
        saturation[neighbor]++;

    // Repeat for remaining V-1 vertices
    for (int i = 1; i < V; i++)
    {
        // Select the next vertex to color
        int max_sat = -1, max_deg = -1, next_vertex = -1;
        for (int v = 0; v < V; v++)
        {
            if (result[v] == -1)  // If v is not colored
            {
                if (saturation[v] > max_sat || 
                    (saturation[v] == max_sat && degree[v] > max_deg))
                {
                    max_sat = saturation[v];
                    max_deg = degree[v];
                    next_vertex = v;
                }
            }
        }

        u = next_vertex;

        // Mark the colors of adjacent vertices as unavailable
        for (int neighbor : adj[u])
            if (result[neighbor] != -1)
                available[result[neighbor]] = true;

        // Find the first available color
        int cr;
        for (cr = 0; cr < V; cr++)
            if (!available[cr])
                break;

        result[u] = cr; // Assign the found color

        // Reset the available array back to false
        for (int neighbor : adj[u])
            if (result[neighbor] != -1)
                available[result[neighbor]] = false;

        // Update saturation degrees of adjacent vertices
        for (int neighbor : adj[u])
            if (result[neighbor] == -1)
            {
                std::set<int> unique_colors; // Track unique colors of adjacent nodes
                for (int adj_of_neighbor : adj[neighbor])
                    if (result[adj_of_neighbor] != -1)
                        unique_colors.insert(result[adj_of_neighbor]);

                saturation[neighbor] = unique_colors.size();
            }
    }

    // 檢查顏色衝突
    bool conflict_found = false;
    for (int u = 0; u < V; u++)
    {
        for (int neighbor : adj[u])
        {
            if (result[u] == result[neighbor] && u != neighbor)
            {
                printf("Conflict detected between vertex %d and vertex %d\n", u, neighbor);
                conflict_found = true;
            }
        }
    }

    if (!conflict_found)
    {
        printf("No conflicts found.\n");
    }

    int cols = -1;

    // Calculate the number of colors used
    for (int u = 0; u < V; u++)
        cols = std::max(cols, result[u]);

    cols++;
    printf("colors used: %d\n", cols);
}

// Assigns colors (starting from 0) to all vertices and prints
// the assignment of colors

// void Graph::greedyColoring()
// {
//     int result[V];

//     // Initialize all vertices as unassigned
//     for (int u = 0; u < V; u++)
//         result[u] = -1;  // no color is assigned to u

//     // A temporary array to store the available colors. True
//     // value of available[cr] would mean that the color cr is
//     // assigned to one of its adjacent vertices
//     bool available[V];
//     for (int cr = 0; cr < V; cr++)
//         available[cr] = false;

//     // Array to store the degree of each vertex
//     std::vector<std::pair<int, int>> vertex_degree(V);
//     for (int u = 0; u < V; u++)
//         vertex_degree[u] = {adj[u].size(), u}; // pair (degree, vertex)

//     // Sort vertices in descending order of degree
//     std::sort(vertex_degree.rbegin(), vertex_degree.rend());

//     // Assign colors to all vertices
//     for (int i = 0; i < V; i++)
//     {
//         int u = vertex_degree[i].second; // Get the vertex with the current highest degree

//         // Process all adjacent vertices and flag their colors as unavailable
//         for (int neighbor : adj[u])
//             if (result[neighbor] != -1)
//                 available[result[neighbor]] = true;

//         // Find the first available color
//         int cr;
//         for (cr = 0; cr < V; cr++)
//             if (available[cr] == false)
//                 break;

//         result[u] = cr; // Assign the found color

//         // Reset the values back to false for the next iteration
//         for (int neighbor : adj[u])
//             if (result[neighbor] != -1)
//                 available[result[neighbor]] = false;
//     }

//     int cols = -1;

//     // Calculate the number of colors used
//     for (int u = 0; u < V; u++)
//         cols = std::max(cols, result[u]);

//     cols++;
//     printf("colors used: %d\n", cols);
// }   
 
// Driver program to test above function
int main(int argc, char** argv) 
{
    ECLgraph g = readECLgraph(argv[1]);    
    Graph g1(g.nodes);
    for (int i = 0; i < g.nodes; i++) {
        for (int j = g.nindex[i]; j < g.nindex[i+1]; j++) {
            g1.addEdge(i, g.nlist[j]);
        }
    }

    cout << "Coloring of graph 1\n";
    
    // 開始計時
    auto start = std::chrono::high_resolution_clock::now();

    g1.greedyColoring();  // 執行著色算法

    // 結束計時
    auto end = std::chrono::high_resolution_clock::now();

    // 計算執行時間（以秒為單位）
    std::chrono::duration<double> elapsed = end - start;
    printf("runtime:    %.6f ms\n", elapsed.count()*1000);

    return 0;
}