#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// GPU kernel — each thread calculates factorial of one element
__global__ void factorialKernel(int *input, unsigned long long *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        int num = input[idx];
        unsigned long long fact = 1;
        for (int i = 1; i <= num; i++)
        {
            fact *= i;
        }
        output[idx] = fact;
    }
}

int main()
{
    int n = 5; // number of elements
    int h_input[] = {1, 2, 3, 4, 5};
    unsigned long long h_output[n];

    int *d_input;
    unsigned long long *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(unsigned long long));

    // Copy input from host → device
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel (1 thread per number)
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    factorialKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, n);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy results device → host
    cudaMemcpy(h_output, d_output, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Display results
    cout << "Factorials:\n";
    for (int i = 0; i < n; i++)
    {
        cout << h_input[i] << "! = " << h_output[i] << "\n";
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
