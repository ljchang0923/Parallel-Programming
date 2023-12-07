#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(float stepX, float stepY, float lowerX, float lowerY,  int maxIterations, int* img, int pitch, float group_size) {
    // To avoid error caused by the floating number, use the following pseudo code
    
    float thisX = (blockIdx.x * blockDim.x + threadIdx.x)*group_size;
    float thisY = (blockIdx.y * blockDim.y + threadIdx.y)*group_size;

    for(float row = 0; row < group_size; row++){
        for(float col = 0; col < group_size; col++){
            
            float x = lowerX + (thisX + col) * stepX;
            float y = lowerY + (thisY + row) * stepY;

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

            int *rowHead;
            rowHead = (int*)((char*)img + int(thisY + row) * pitch);
            rowHead[int(thisX + col)] = i;
        }
    }
    
    // int idx = int(thisY) * width + int(thisX);
    // img[idx] = i;


}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    cudaSetDeviceFlags (cudaDeviceMapHost);

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    size_t size = resX * resY * sizeof(int);
    int *img_h;
    int *img_gpu;
    size_t pitch;

    cudaHostAlloc(&img_h, size, cudaHostAllocWriteCombined|cudaHostAllocMapped);
    // cudaMalloc(&img_gpu, size);
    
    cudaHostGetDevicePointer((void**)&img_gpu, (void*)img_h, 0);
    cudaMallocPitch((void**)&img_gpu, &pitch, sizeof(int)*resX, resY);
    
    float group_size = 2;
    dim3 threadPerBlock(32,30);
    dim3 numBlocks((resX/int(group_size)) / threadPerBlock.x, (resY/int(group_size)) / threadPerBlock.y);
    mandelKernel <<<numBlocks, threadPerBlock >>> (stepX, stepY, lowerX, lowerY, maxIterations, img_gpu, pitch, group_size);
    cudaDeviceSynchronize();

    cudaMemcpy2D(img, resX*sizeof(int), img_gpu, pitch, resX*sizeof(int), resY, cudaMemcpyHostToHost);
    // for(size_t i = 0; i < resY; ++i){
    //     memcpy(&img[i*resX], &img_gpu[i*pitch], resX * sizeof(int));
    // }

    cudaFree(img_gpu);
    cudaFreeHost(img_h);
}
