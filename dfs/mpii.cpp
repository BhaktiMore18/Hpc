#include <mpi.h>
#include <iostream>
#include <vector>
#include <stack>
using namespace std;

void dfs_iterative(int start, vector<vector<int>> &graph, vector<int> &local_visited)
{
    stack<int> s;
    s.push(start);

    while (!s.empty())
    {
        int node = s.top();
        s.pop();

        if (!local_visited[node])
        {
            local_visited[node] = 1;
            cout << "Process " << start << " visiting node " << node << endl;

            for (int neighbor : graph[node])
            {
                if (!local_visited[neighbor])
                    s.push(neighbor);
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

    // Create a small graph (shared by all)
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

    // Each process gets a starting node (based on rank)
    if (rank < n)
    {
        dfs_iterative(rank, graph, local_visited);
    }

    // Combine all visited results at root (process 0)
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
