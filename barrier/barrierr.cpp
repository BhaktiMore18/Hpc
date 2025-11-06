#include <mpi.h>
#include <iostream>
#include <unistd.h> // for sleep()

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Phase 1: before the barrier
    cout << "Process " << rank << " starting phase 1" << endl;

    // Simulate some variable work
    sleep(rank + 1); // process 0 sleeps 1s, process 1 sleeps 2s, etc.

    cout << "Process " << rank << " finished phase 1" << endl;

    // Barrier — all processes wait here
    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 2: after the barrier
    cout << "Process " << rank << " entering phase 2" << endl;

    MPI_Finalize();
    return 0;
}
