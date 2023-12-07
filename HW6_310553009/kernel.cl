__kernel void convolution(
    const __global float *input_img, __constant float *filter, __global float *output,
    int filter_size, int img_h, int img_w, 
    __local float *localImg, int localHeight, int localWidth
)
{
    int filterRadius = (filter_size/2);
    int padding = filterRadius * 2;

    int groupStartCol = get_group_id(0)*get_local_size(0);
    int groupStartRow = get_group_id(1)*get_local_size(1);
    
    // Determine the local ID of each work item
    int localCol = get_local_id(0);
    int localRow = get_local_id(1);
    
    // Determine the global ID of each work item.  Work items
    // representing the output region will have a unique global
    // ID
    int globalCol = groupStartCol + localCol;
    int globalRow = groupStartRow + localRow;  
    // printf("%d %d\n", globalRow, globalCol);
    for(int i = localRow; i < localHeight; i += get_local_size(1)) {
        
        int curRow = groupStartRow+i;
        
        // Step across columns
        for(int j = localCol; j < localWidth; j += get_local_size(0)) {
            
            int curCol = groupStartCol+j;
            
            // Perform the read if it is in bounds
            if(curRow < img_h && curCol < img_w) {
                localImg[i*localWidth + j] = 
                    input_img[curRow*img_w + curCol];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(globalRow < img_h - padding && globalCol < img_w - padding) {
        
        // Each work item will filter around its start location 
        //(starting from the filter radius left and up)
        float sum = 0.0f;
        int filterIdx = 0;
        
         // Not unrolled
         for(int i = localRow; i < localRow + filter_size; i++) {
            int offset = i*localWidth;
            for(int j = localCol; j < localCol + filter_size; j++){
                sum += localImg[offset+j] * 
                   filter[filterIdx++];
            }
         }
        // printf("%d %d %f\n", (globalRow+filterRadius)*img_w, (globalCol+filterRadius), sum);
        output[(globalRow+filterRadius)*img_w + (globalCol+filterRadius)] = sum;
    }
}

// __kernel void convolution(
//     const __global float *input_img, __constant float *filter, __global float *output,
//     int filter_size, int img_h, int img_w, __local float *localImg) 
// {
//     const int x = get_global_id(1);
//     const int y = get_global_id(0);
//     int half_filter_size = filter_size / 2;
//     float sum = 0.0;
//     for(int i = -half_filter_size ; i <= half_filter_size; i++){

//         for(int j = -half_filter_size ; j <= half_filter_size ; j++){
           
//             if(x+j>=0 && y+i>=0 && x+j < img_w && y+i < img_h){
//                sum += input_img[(y+i)*img_w + x + j] * filter[(i+half_filter_size)*filter_size + (j+half_filter_size)];
        
//             }
//         }
//     }
    
//     // printf("start computation!!\n");
    
//     output[y*img_w + x] = sum;
//     // printf("%d %d %f \n", x, y ,output[x*img_w + y]);
// }
