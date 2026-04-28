#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(const float *a, const float *b, float *c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        c[index] = a[index] + b[index];
}

int main()
{
    int n = 1 << 24;
    size_t size = n * sizeof(float);

    float *h_a = (float *)malloc(size), *h_b = (float *)malloc(size), *h_c = (float *)malloc(size);
    for (int i = 0; i < n; ++i)
    {
        h_a[i] = i;
        h_b[i] = i * 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}