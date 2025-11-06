#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

void bfs_parallel(int start, vector<vector<int>> &graph)
{
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty())
    {
        int level_size = q.size();

        // Collect all nodes in the current level
        vector<int> current_level;
        for (int i = 0; i < level_size; i++)
        {
            current_level.push_back(q.front());
            q.pop();
        }

// Explore neighbors of all nodes in parallel
#pragma omp parallel for
        for (int i = 0; i < current_level.size(); i++)
        {
            int node = current_level[i];
            cout << "Visited " << node << " by thread " << omp_get_thread_num() << endl;

            for (int neighbor : graph[node])
            {
// Use a critical section to avoid race conditions
#pragma omp critical
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
                        q.push(neighbor); // enqueue for next level
                    }
                }
            }
        }
    }
}

int main()
{
    // Example graph (adjacency list)
    vector<vector<int>> graph = {
        {1, 2},    // 0
        {0, 3, 4}, // 1
        {0, 5},    // 2
        {1},       // 3
        {1, 5},    // 4
        {2, 4}     // 5
    };

    double start_time = omp_get_wtime();
    bfs_parallel(0, graph);
    double end_time = omp_get_wtime();

    cout << "BFS completed in " << end_time - start_time << " seconds" << endl;

    return 0;
}

// g++ bfs_openmp.cpp -fopenmp -o bfs_openmp
//./bfs_openmp
