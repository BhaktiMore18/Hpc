#include <mpi.h>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int data;

    if (rank == 0)
    {
        data = 42; // root process sets the value
        cout << "Process 0 initialized data = " << data << endl;
    }

    // Broadcast data from process 0 to all other processes
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Every process now has the same value of 'data'
    cout << "Process " << rank << " received data = " << data << endl;

    MPI_Finalize();
    return 0;
}
