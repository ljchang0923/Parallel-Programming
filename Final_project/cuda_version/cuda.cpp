#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <random>
#include <thread>
#include <vector>

#include "constants.cpp"
#include "load_data.cpp"
#include "train.cpp"
#include "hostFE.h"

double *converTo1D(double **x, int row, int col){
  double *xx = new double[row*col];
  for(int i=0; i<row; ++i)
    for(int j=0; j<col; ++j){
      xx[i*col + j] = x[i][j];
      // printf("%lf ",  xx[i*col + j]);
    }
  return xx;
}

int main(int argc, char *argv[]) {
  
  test_max("adult1.train");

  double **x;
  int *y;
  fetch_dataset("adult1.train", TRAIN_SAMPLE_SIZE, &x, &y);

  double *weight, *old_weight;
  init_weight(&weight, FEATURE_NUM); // 123
  init_weight(&old_weight, FEATURE_NUM);

  // SGD
  printf("start gradient descent!!\n");
  int batch_size = BATCH_SIZE; // 2 << 11
  double lr = LEARING_RATE;
  double eps = EPS;
  unsigned int maxit = MAX_ITER;

  double *xx = converTo1D(x, TRAIN_SAMPLE_SIZE, FEATURE_NUM);
  
  struct timespec start, finish;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);
  train(xx, y, weight, old_weight, batch_size, lr, eps, maxit);
  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed =
      (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1e9;
  printf("execution time(GD part): %lf\n", elapsed);
  delete_dataset(&x, &y, TRAIN_SAMPLE_SIZE);
  delete[] old_weight;

  double **x_test;
  int *y_test;
  fetch_dataset("adult1.test", TEST_SAMPLE_SIZE, &x_test, &y_test);
  test(weight, TEST_SAMPLE_SIZE, x_test, y_test);
  delete_dataset(&x_test, &y_test, TEST_SAMPLE_SIZE);
  delete[] weight;

  return 0;
}