#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA kernel for matrix multiplication
__global__ void matMulKernel(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // global column index

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main()
{
    int N = 3; // matrix size (NxN)
    int SIZE = N * N * sizeof(float);

    // Allocate host (CPU) memory
    float h_A[N * N], h_B[N * N], h_C[N * N];

    // Initialize input matrices
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] = i + j;
            h_B[i * N + j] = i - j;
        }
    }

    // Allocate device (GPU) memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, SIZE);
    cudaMalloc(&d_B, SIZE);
    cudaMalloc(&d_C, SIZE);

    // Copy data from host → device
    cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel on GPU
    matMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result from device → host
    cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost);

    // Print result matrix
    cout << "Result matrix C:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << h_C[i * N + j] << " ";
        }
        cout << "\n";
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
