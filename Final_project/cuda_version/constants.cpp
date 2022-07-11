#define FEATURE_NUM (123)          // 123

#define LEARING_RATE (0.001)
#define EPS (0.0001)
#define MAX_ITER (50000)

#define PRINT_LOG 0

//BIG SET
#define TRAIN_SAMPLE_SIZE (30296)  // 30296
#define TEST_SAMPLE_SIZE (2265)    // 2265
#define BATCH_SIZE (2 << 12) 
#define REMAIN_THREAD (286) 
#define REMAIN_BLOCK (20)   
#define THREADNUMBER (512)  


// SMALL SET 
// #define TRAIN_SAMPLE_SIZE (2265)  
// #define TEST_SAMPLE_SIZE (30296)    
// #define BATCH_SIZE (2 << 8) 
// #define REMAIN_THREAD (217) 
// #define REMAIN_BLOCK (1)   
// #define THREADNUMBER (64)  