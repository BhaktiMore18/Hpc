
#include <stdio.h>
#include <limits.h>
// #include <omp.h>

#define N 10  // Number of cities

int tsp(int dist[N][N], int visited[N], int pos, int count, int cost, int *minCost) {
    if (count == N && dist[pos][0] > 0) { // complete tour
        #pragma omp critical
        {
            if (cost + dist[pos][0] < *minCost)
                *minCost = cost + dist[pos][0];
        }
        return *minCost;
    }

    for (int i = 0; i < N; i++) {
        if (!visited[i] && dist[pos][i] > 0) {
            visited[i] = 1;
            tsp(dist, visited, i, count + 1, cost + dist[pos][i], minCost);
            visited[i] = 0;
        }
    }
    return *minCost;
}

int main() {
    int dist[N][N] = {
        {0, 3, 5, 9, 2, 6, 10, 8, 7, 4},
    {3, 0, 4, 8, 5, 7, 9, 6, 3, 2},
    {5, 4, 0, 3, 6, 5, 8, 7, 4, 3},
    {9, 8, 3, 0, 5, 6, 7, 9, 4, 5},
    {2, 5, 6, 5, 0, 4, 6, 3, 2, 1},
    {6, 7, 5, 6, 4, 0, 3, 2, 4, 5},
    {10, 9, 8, 7, 6, 3, 0, 4, 5, 6},
    {8, 6, 7, 9, 3, 2, 4, 0, 3, 5},
    {7, 3, 4, 4, 2, 4, 5, 3, 0, 2},
    {4, 2, 3, 5, 1, 5, 6, 5, 2, 0}
    };

    int minCost = INT_MAX;
    double start, end;

    // Sequential Execution
    start = omp_get_wtime();
    int visited[N] = {0};
    visited[0] = 1; // start from city 0
    tsp(dist, visited, 0, 1, 0, &minCost);
    end = omp_get_wtime();
    printf("Sequential Min Cost: %d\n", minCost);
    printf("Sequential Time: %lf seconds\n", end - start);

    // Parallel Execution
    minCost = INT_MAX;
    start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 1; i < N; i++) {  // split first city choices among threads
        int visitedP[N] = {0};
        visitedP[0] = 1;
        visitedP[i] = 1;
        int localMin = minCost;
        tsp(dist, visitedP, i, 2, dist[0][i], &localMin);
        #pragma omp critical
        {
            if (localMin < minCost)
                minCost = localMin;
        }
    }
    end = omp_get_wtime();
    printf("Parallel Min Cost: %d\n", minCost);
    printf("Parallel Time: %lf seconds\n", end - start);

    return 0;
}