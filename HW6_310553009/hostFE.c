#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"
#define MAX_SOURCE_SIZE (0x100000)

unsigned int roundUp(unsigned int value, unsigned int multiple) {
	
  // Determine how far past the nearest multiple the value is
  unsigned int remainder = value % multiple;
  
  // Add the difference to make the value a multiple
  if(remainder != 0) {
          value += (multiple-remainder);
  }
  
  return value;
}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int ret;
    size_t filterSize = filterWidth * filterWidth;
    size_t imgSize = imageHeight * imageWidth;

    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, CL_QUEUE_PROFILING_ENABLE, &ret);

    cl_mem img_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY, 
            imgSize * sizeof(float), NULL, &ret);
    cl_mem filter_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
            filterSize * sizeof(float), NULL, &ret);
    cl_mem output_mem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, 
            imgSize * sizeof(float), NULL, &ret);


    clEnqueueWriteBuffer(command_queue, img_mem, CL_FALSE, 0,
            imgSize * sizeof(float), inputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, filter_mem, CL_FALSE, 0, 
            filterSize * sizeof(float), filter, 0, NULL, NULL);

    
    clBuildProgram(*program, 1, device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(*program, "convolution", &ret);

    int paddingPixels = (int)(filterWidth/2) * 2;
    int totalWorkItemsX = roundUp(imageWidth-paddingPixels, 10);
    int totalWorkItemsY = roundUp(imageHeight-paddingPixels, 10);
    int localWidth = 10 + paddingPixels;
    int localHeight = 10 + paddingPixels; 
    size_t localMemSize =  localWidth * localHeight * sizeof(float);  

    // setup input argument
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&img_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_mem);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&filterWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 6, localMemSize, NULL);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &localHeight); 
    clSetKernelArg(kernel, 8, sizeof(cl_int), &localWidth);

    size_t global_item_size[2] = {totalWorkItemsX, totalWorkItemsY};
//     size_t global_item_size[2] = {imageHeight, imageWidth};
    size_t local_item_size[2] = {10,10}; 
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            global_item_size, local_item_size, 0, NULL, NULL);
    
    clFinish(command_queue);
    clEnqueueReadBuffer(command_queue, output_mem, CL_FALSE, 0, 
                imgSize * sizeof(float), (void *)outputImage, 0, NULL, NULL); 
    
//     clReleaseKernel(kernel);
//     clReleaseProgram(*program);
//     clReleaseMemObject(img_mem);
//     clReleaseMemObject(filter_mem);
//     clReleaseMemObject(output_mem);
//     clReleaseCommandQueue(command_queue);
//     clReleaseContext(*context);
}