#include <mpi.h>
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

void bfs_local(int start, vector<vector<int>> &graph, vector<int> &local_visited)
{
    queue<int> q;
    q.push(start);
    local_visited[start] = 1;

    while (!q.empty())
    {
        int node = q.front();
        q.pop();

        cout << "Process " << start << " visiting node " << node << endl;

        for (int neighbor : graph[node])
        {
            if (!local_visited[neighbor])
            {
                local_visited[neighbor] = 1;
                q.push(neighbor);
            }
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Example graph (adjacency list)
    vector<vector<int>> graph = {
        {1, 2},    // 0
        {0, 3, 4}, // 1
        {0, 5},    // 2
        {1},       // 3
        {1, 5},    // 4
        {2, 4}     // 5
    };

    int n = graph.size();
    vector<int> local_visited(n, 0);

    if (rank < n)
    {
        bfs_local(rank, graph, local_visited);
    }

    // Combine results at root (logical OR)
    vector<int> global_visited(n, 0);
    MPI_Reduce(local_visited.data(), global_visited.data(), n, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "\nFinal visited nodes: ";
        for (int i = 0; i < n; i++)
        {
            cout << global_visited[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
