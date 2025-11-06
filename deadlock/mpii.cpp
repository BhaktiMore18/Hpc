#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int data = rank;

    if (rank == 0)
    {
        cout << "Process 0 sending to 1..." << endl;
        MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); // Blocking send
        cout << "Process 0 receiving from 1..." << endl;
        MPI_Recv(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if (rank == 1)
    {
        cout << "Process 1 sending to 0..." << endl;
        MPI_Send(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); // Blocking send
        cout << "Process 1 receiving from 0..." << endl;
        MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Finalize();
    return 0;
}
