#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;

int main()
{
    int N = 10;
    vector<int> arr(N);

    // Initialize array
    for (int i = 0; i < N; i++)
    {
        arr[i] = i + 1; // 1,2,3,...,10
    }

    int sum = 0;

// Parallel sum using OpenMP
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < N; i++)
    {
        sum += arr[i];
    }

    cout << "Sum of array elements: " << sum << endl;

    return 0;
}
