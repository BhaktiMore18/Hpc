#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 10; // array size
    vector<int> arr(N);

    if (rank == 0)
    {
        // Initialize array
        for (int i = 0; i < N; i++)
            arr[i] = i + 1; // 1,2,3,...,10
        cout << "Array: ";
        for (int i = 0; i < N; i++)
            cout << arr[i] << " ";
        cout << endl;
    }

    // Determine chunk size per process
    int chunk_size = N / size; // assume N divisible by size
    vector<int> sub_arr(chunk_size);

    // Scatter array to all processes
    MPI_Scatter(arr.data(), chunk_size, MPI_INT,
                sub_arr.data(), chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    // Each process computes local sum
    int local_sum = 0;
    for (int i = 0; i < chunk_size; i++)
        local_sum += sub_arr[i];

    // Reduce local sums to global sum at root
    int global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "Sum of array elements = " << global_sum << endl;

    MPI_Finalize();
    return 0;
}
