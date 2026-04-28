#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void matrixMulTiledKernel(const float *A, const float *B, float *C, int width)
{
    __shared__ float ds_A[TILE_DIM][TILE_DIM];
    __shared__ float ds_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float Cvalue = 0.0f;
    int numTiles = (width + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t)
    {
        if (row < width && t * TILE_DIM + tx < width)
            ds_A[ty][tx] = A[row * width + t * TILE_DIM + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (t * TILE_DIM + ty < width && col < width)
            ds_B[ty][tx] = B[(t * TILE_DIM + ty) * width + col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k)
        {
            if (t * TILE_DIM + k < width)
                Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }
        __syncthreads();
    }
    if (row < width && col < width)
        C[row * width + col] = Cvalue;
}

int main()
{
    int width = 1024;
    size_t size = width * width * sizeof(float);
    float *h_A = (float *)malloc(size), *h_B = (float *)malloc(size), *h_C = (float *)malloc(size);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    dim3 blockSize(TILE_DIM, TILE_DIM);
    dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (width + TILE_DIM - 1) / TILE_DIM);
    matrixMulTiledKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);

    return 0;
}