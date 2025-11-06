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

    int N = 4; // matrix dimension (N x N)
    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N, 0));

    // Root process initializes matrices
    if (rank == 0)
    {
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

        cout << "\nMatrix B:\n";
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                B[i][j] = (i == j) ? 1 : 2; // simple pattern
                cout << B[i][j] << " ";
            }
            cout << endl;
        }
    }

    // Flatten matrices for sending/receiving
    vector<int> A_flat(N * N), B_flat(N * N), C_flat(N * N, 0);

    // Root flattens A and B
    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                A_flat[i * N + j] = A[i][j];
                B_flat[i * N + j] = B[i][j];
            }
    }

    // Broadcast B to all processes
    MPI_Bcast(B_flat.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = N / size; // assume N divisible by size
    vector<int> sub_A(rows_per_proc * N);

    // Scatter rows of A
    MPI_Scatter(A_flat.data(), rows_per_proc * N, MPI_INT,
                sub_A.data(), rows_per_proc * N, MPI_INT,
                0, MPI_COMM_WORLD);

    // Each process computes its portion of C
    vector<int> sub_C(rows_per_proc * N, 0);
    for (int i = 0; i < rows_per_proc; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += sub_A[i * N + k] * B_flat[k * N + j];
            }
            sub_C[i * N + j] = sum;
        }
    }

    // Gather results back to root
    MPI_Gather(sub_C.data(), rows_per_proc * N, MPI_INT,
               C_flat.data(), rows_per_proc * N, MPI_INT,
               0, MPI_COMM_WORLD);

    // Root reconstructs and prints final matrix
    if (rank == 0)
    {
        cout << "\nResultant Matrix C = A × B:\n";
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << C_flat[i * N + j] << " ";
            }
            cout << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
