#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendData = rank; // Each process sends its rank
    int recvData;

    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;

    MPI_Status status;

    // Blocking send & receive (Deadlock-prone naive version)
    // Uncomment this version to see potential deadlock:
    // MPI_Send(&sendData, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
    // MPI_Recv(&recvData, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &status);

    // Safe version to avoid deadlock: alternate send/recv based on rank
    if (rank % 2 == 0)
    {
        MPI_Send(&sendData, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
        MPI_Recv(&recvData, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &status);
    }
    else
    {
        MPI_Recv(&recvData, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &status);
        MPI_Send(&sendData, 1, MPI_INT, right, 0, MPI_COMM_WORLD);
    }

    cout << "Process " << rank << " received " << recvData
         << " from process " << left << endl;

    MPI_Finalize();
    return 0;
}
