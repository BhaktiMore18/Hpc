#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

vector<vector<int>> graph;  // adjacency list
vector<bool> visited;       // track visited nodes

void dfs(int node) {
    visited[node] = true;
    cout << "Visited " << node << " by thread " << omp_get_thread_num() << endl;

    // Explore all neighbors
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            // Create a new task for each unvisited neighbor
            #pragma omp task
            dfs(neighbor);
        }
    }
}

int main() {
    int n = 6;  // number of nodes
    graph.resize(n);
    visited.resize(n, false);

    // Undirected graph example
    graph[0] = {1, 2};
    graph[1] = {0, 3, 4};
    graph[2] = {0, 5};
    graph[3] = {1};
    graph[4] = {1, 5};
    graph[5] = {2, 4};

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        dfs(0); // start DFS from node 0
    }

    double end = omp_get_wtime();

    cout << "\nTotal time: " << end - start << " seconds" << endl;
    return 0;
}
