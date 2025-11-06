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

    const int N = 4; // Matrix dimension (N x N)

    vector<vector<int>> A(N, vector<int>(N));
    vector<int> x(N), y(N, 0); // y = A * x

    if (rank == 0)
    {
        // Initialize matrix A
        cout << "Matrix A:\n";
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i][j] = i + j + 1;
                cout << A[i][j] << " ";
            }
            cout << endl;
        }

        // Initialize vector x
        cout << "Vector x:\n";
        for (int i = 0; i < N; i++)
        {
            x[i] = i + 1;
            cout << x[i] << " ";
        }
        cout << endl;
    }

    // Flatten matrix for MPI
    vector<int> A_flat(N * N);
    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A_flat[i * N + j] = A[i][j];
    }

    // Broadcast vector x to all processes
    MPI_Bcast(x.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine rows per process
    int rows_per_proc = N / size; // assume N divisible by size
    vector<int> sub_A(rows_per_proc * N);
    vector<int> sub_y(rows_per_proc, 0);

    // Scatter rows of A
    MPI_Scatter(A_flat.data(), rows_per_proc * N, MPI_INT,
                sub_A.data(), rows_per_proc * N, MPI_INT,
                0, MPI_COMM_WORLD);

    // Each process computes its part of y
    for (int i = 0; i < rows_per_proc; i++)
    {
        for (int j = 0; j < N; j++)
        {
            sub_y[i] += sub_A[i * N + j] * x[j];
        }
    }

    // Gather results back to root
    MPI_Gather(sub_y.data(), rows_per_proc, MPI_INT,
               y.data(), rows_per_proc, MPI_INT,
               0, MPI_COMM_WORLD);

    // Print result on root
    if (rank == 0)
    {
        cout << "Result vector y = A * x:\n";
        for (int i = 0; i < N; i++)
            cout << y[i] << " ";
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}

// mpicxx mpi_matvec.cpp -o mpi_matvec
// mpirun -np 4 ./mpi_matvec
