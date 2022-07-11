#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "constants.cpp"

void checkError(char *msg){

    cudaError_t error;  
    const char *error_msg; 
    error = cudaGetLastError();
    if(error == cudaSuccess){
        return;
        // printf("%s Success\n", msg);
        // return;
    }
    error_msg  =  cudaGetErrorName (error );
    if(error_msg != NULL)
        printf("%s ouucr error %s\n", msg, error_msg);
}
void DeviceMemoryAllocate(double *x, int *y, double *w, double *old_w
                        , double **local_grad, double **d_x, int **d_y, double **d_weight, double **d_old_weight){
    
    int n_sample = TRAIN_SAMPLE_SIZE;
    int feature_size = FEATURE_NUM;

    // Memory allocate
    cudaMalloc(&(*d_x), n_sample * feature_size * sizeof(double));
    cudaMalloc(&(*d_y), n_sample * sizeof(int));
    cudaMalloc(&(*d_weight), feature_size * sizeof(double));
    cudaMalloc(&(*d_old_weight), feature_size * sizeof(double));
    cudaMalloc(&(*local_grad), BATCH_SIZE * feature_size * sizeof(double));


    // init value
    cudaMemcpy(*d_x, x, n_sample  * feature_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_y, y, n_sample * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_weight, w, feature_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_old_weight, old_w, feature_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(*local_grad, 0, BATCH_SIZE * feature_size * sizeof(double));
    // cudaMemset(*d_old_weight, 0, feature_size * sizeof(double));
   
}

void DeviceMemoryReleace(double *d_x, int *d_y, double *d_weight, double *d_old_weight){

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_weight);
    cudaFree(d_old_weight);
}

__device__ double global_norm[1] = {0};

__global__  void update_weight(double *local_grad, double *weight, double lr, int batch_size) {

    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    for(int i=0; i<batch_size; ++i){
        // if(idx == 0)
        //     printf("i: %d, sum: %lf\n",i , sum);
        sum += local_grad[i * FEATURE_NUM + idx];
    }
        

    weight[idx] += sum * lr / batch_size;
}   

__global__  void vecnorm(double *old_w, double *new_w) {

    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double diff =  old_w[idx] - new_w[idx];
    double local_sum = diff * diff;

    atomicAdd(global_norm, local_sum);
   
}

__global__ void train_kernel(int start_idx, double *x, int *y, double *weight, double *local_grad, double lr, int batch_size){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    // __shared__ double local_x[THREADNUMBER][FEATURE_NUM];

    // classify
    double pred = 0;

    for (int f = 0; f < FEATURE_NUM; f++) {
        // local_x[idx][f] = x[idx*FEATURE_NUM  + f];
        // pred += local_x[idx][f] * weight[f];
        pred += x[idx*FEATURE_NUM  + f] * weight[f];
    }

    
    int label = y[idx];


    for (int w = 0; w < FEATURE_NUM; w++) {   // FEATURE 123

      local_grad[index * FEATURE_NUM + w] = (label - pred) * x[idx*FEATURE_NUM  + w];
 
    }
}


void CopyWeightFromDevice(double *weight, double *d_weight){

    int feature_size = FEATURE_NUM;
    cudaMemcpy(weight, d_weight, feature_size * sizeof(double), cudaMemcpyDeviceToHost);
    // for(int i=0; i<FEATURE_NUM; ++i)
    //     printf("%lf ", weight[i]);
    // printf("\n");
}


void train(double *x, int *y, double *weight, double *old_weight,
                         int batch_size = 2 << 11,
                         double lr = 0.001, double eps = 0.00001,
                         unsigned int maxit = 50000){
 
    double *d_x, *d_weight, *d_old_weight, *local_grad;
    int *d_y;
    DeviceMemoryAllocate(x, y, weight, old_weight, &local_grad, &d_x, &d_y, &d_weight, &d_old_weight);
    

    int n = 0;
    double norm = 1.0, zero = 0;
    const int feature_size = FEATURE_NUM;
    // const int THREADNUMBER = 64;
    int start_idx;
    
    while(norm > eps){ //norm > eps

       // d_weight = d_old_weight
        cudaMemcpy(d_old_weight, d_weight, feature_size * sizeof(double), cudaMemcpyDeviceToDevice);
        CopyWeightFromDevice(weight, d_weight);

        for (start_idx = 0; start_idx < TRAIN_SAMPLE_SIZE ; start_idx += batch_size) {
            
            const int threadNum = (TRAIN_SAMPLE_SIZE - start_idx > batch_size) ? THREADNUMBER : REMAIN_THREAD; // TRAIN_SAMPLE_SIZE - start_idx 203 / 453
            const int blockNum = (threadNum == THREADNUMBER) ? batch_size / threadNum : REMAIN_BLOCK;  // 8 / 5

            dim3 threadsPerBlock(threadNum);
            dim3 numBlocks(blockNum);

            train_kernel <<< numBlocks, threadsPerBlock >>>(start_idx, d_x, d_y, d_weight, local_grad, lr, threadNum * blockNum);
            cudaDeviceSynchronize();

            dim3 threadsPerBlock2(FEATURE_NUM/3);
            dim3 numBlocks2(3);
            update_weight <<< numBlocks2, threadsPerBlock2 >>>(local_grad, d_weight, lr, threadNum * blockNum);
            cudaDeviceSynchronize();
        }
       
        // Init global norm
        cudaMemcpyToSymbol(global_norm, &zero, sizeof(double) * 1, 0, cudaMemcpyHostToDevice);

        // CALCULATE NORM
        dim3 threadsPerBlock3(FEATURE_NUM/3);
        dim3 numBlocks3(3);
        vecnorm <<< numBlocks3, threadsPerBlock3 >>>(d_old_weight, d_weight);
        cudaDeviceSynchronize();
        
        // Get norm from global_norm
        cudaMemcpyFromSymbol(&norm, global_norm, sizeof(double) * 1, 0, cudaMemcpyDeviceToHost);
        norm = sqrt(norm);
        cudaDeviceSynchronize();
        // printf("iter %d, norm: %lf\n", n, norm);

        if (++n > maxit) {
            break;
        }
    }

    printf("# of iter: %d\n", n);
    CopyWeightFromDevice(weight, d_weight);
    // CopyWeightFromDevice(old_weight, d_old_weight);
    DeviceMemoryReleace(d_x, d_y, d_weight, d_old_weight);


    // double sum = 0;
    // for(int i=0; i<FEATURE_NUM; ++i){
    //     double diff =  old_weight[i] - weight[i];
    //     sum += diff * diff;
    // }
    // sum = sqrt(sum);
    // printf("CPU norm: %lf\n", sum);
        
}



