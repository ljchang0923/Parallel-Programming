#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY,  int maxIterations, int width, int* img) {
    // To avoid error caused by the floating number, use the following pseudo code
    
    float thisX = blockIdx.x * blockDim.x + threadIdx.x;
    float thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    float z_re = x, z_im = y;
    int i;
    for (i = 0; i < maxIterations; ++i)
    {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
    }

    int idx = int(thisY) * width + int(thisX);
    img[idx] = i;


}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    size_t size = resX * resY * sizeof(int);
    
    int *img_gpu;
    cudaMalloc(&img_gpu, size);
    //cudaMemcpy(img_gpu, img, size, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(32,16);
    dim3 numBlocks(resX / threadPerBlock.x, resY / threadPerBlock.y);
    mandelKernel <<<numBlocks, threadPerBlock, 0>>> (stepX, stepY, lowerX, lowerY, maxIterations, resX, img_gpu);

    cudaMemcpy(img, img_gpu, size, cudaMemcpyDeviceToHost);

    cudaFree(img_gpu);
}
