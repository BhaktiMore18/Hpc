#include <iostream>
#include <mpi.h>
using namespace std;

int main(int argc, char **argv)
{
    int rank, size;
    long long n = 10; // Example: compute 10!
    long long local_fact = 1, global_fact = 1;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide work among processes
    long long start = (n / size) * rank + 1;
    long long end = (rank == size - 1) ? n : (n / size) * (rank + 1);

    // Each process computes its partial factorial
    for (long long i = start; i <= end; i++)
    {
        local_fact *= i;
    }

    // Combine results from all processes (multiply partial factorials)
    MPI_Reduce(&local_fact, &global_fact, 1, MPI_LONG_LONG, MPI_PROD, 0, MPI_COMM_WORLD);

    // Rank 0 prints the final result
    if (rank == 0)
    {
        cout << n << "! = " << global_fact << endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
