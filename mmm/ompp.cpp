#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

int main()
{
    int n = 500; // size of matrix (n x n)
    vector<vector<int>> A(n, vector<int>(n, 1));
    vector<vector<int>> B(n, vector<int>(n, 2));
    vector<vector<int>> C(n, vector<int>(n, 0));

    double start = omp_get_wtime();

// Parallel region for matrix multiplication
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    double end = omp_get_wtime();

    cout << "Matrix multiplication completed for size " << n << "x" << n << endl;
    cout << "Time taken: " << end - start << " seconds" << endl;

    // Optional: Print small matrices to verify correctness
    if (n <= 5)
    {
        cout << "Result Matrix (C):" << endl;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                cout << C[i][j] << " ";
            cout << endl;
        }
    }

    return 0;
}