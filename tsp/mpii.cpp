#include <mpi.h>
#include <iostream>
#include <climits>
using namespace std;

#define N 4  // number of cities

int tsp(int dist[N][N], bool visited[N], int pos, int count, int cost, int minCost) {
    if (count == N && dist[pos][0] > 0) {
        int totalCost = cost + dist[pos][0];
        return (totalCost < minCost) ? totalCost : minCost;
    }

    for (int i = 0; i < N; i++) {
        if (!visited[i] && dist[pos][i] > 0) {
            visited[i] = true;
            minCost = tsp(dist, visited, i, count + 1, cost + dist[pos][i], minCost);
            visited[i] = false;
        }
    }
    return minCost;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dist[N][N] = {
        {0, 10, 15, 20},
        {10, 0, 35, 25},
        {15, 35, 0, 30},
        {20, 25, 30, 0}
    };

    int globalMin = INT_MAX;
    int localMin = INT_MAX;

    // Each process takes a different starting city (after city 0)
    for (int i = rank + 1; i < N; i += size) {
        bool visited[N] = {false};
        visited[0] = true;
        visited[i] = true;
        int cost = dist[0][i];
        localMin = tsp(dist, visited, i, 2, cost, localMin);
    }

    // Reduce all local minimums to a global minimum
    MPI_Reduce(&localMin, &globalMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "Minimum cost using MPI: " << globalMin << endl;

    MPI_Finalize();
    return 0;
}
