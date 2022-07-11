#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <random>
#include <thread>
#include <vector>

#include "constants.cpp"
#include "load_data.cpp"
#include "train.cpp"

using namespace std;

static inline void train_batch(int tid, int n_threads, double **x, int *y,
                               double *weight, int start_idx, int end_idx,
                               atomic<double> *grad) {
  double local_grad[FEATURE_NUM] = {0};

  for (int i = tid + start_idx; i < end_idx; i += n_threads) {
    int label = y[i];
    double pred = classify(x[i], weight);
    for (int w = 0; w < FEATURE_NUM; w++) {
      local_grad[w] += (label - pred) * x[i][w];
    }
  }

  for (int w = 0; w < FEATURE_NUM; w++) {
    grad[w] = grad[w] + local_grad[w];
  }
}

static inline void train(double **x, int *y, double *weight, double *old_weight,
                         int n_threads, int batch_size = 2 << 12,
                         double lr = 0.001, double eps = 0.00001,
                         unsigned int maxit = 50000) {
  int n = 0;
  double norm = 1.0;
  atomic<double> grad[FEATURE_NUM] = {0};
  thread *t = new thread[n_threads];

  while (norm > eps) {
    copy_weight(old_weight, weight);

    for (int start_inx = 0; start_inx + batch_size < TRAIN_SAMPLE_SIZE;
         start_inx += batch_size) {
      for (int tid = 0; tid < n_threads; tid++) {
        t[tid] = thread(train_batch, tid, n_threads, x, y, weight, start_inx,
                        start_inx + batch_size, grad);
      }
      for (int tid = 0; tid < n_threads; tid++) {
        t[tid].join();
      }

      for (int w = 0; w < FEATURE_NUM; w++) {
        weight[w] += lr * grad[w] / batch_size;
        grad[w] = 0;
      }
    }

    norm = vecnorm(old_weight, weight);
#if PRINT_LOG > 0
    if (n % 100 == 0) printf("# convergence: %1.6f iterations: %i\n", norm, n);
#endif
    if (++n > maxit) {
      break;
    }
  }
  delete[] t;
}

int main(int argc, char *argv[]) {
  int n_threads = atoi(argv[1]);

  test_max("adult1.train");

  double **x;
  int *y;
  fetch_dataset("adult1.train", TRAIN_SAMPLE_SIZE, &x, &y);

  double *weight, *old_weight;
  init_weight(&weight, FEATURE_NUM);
  init_weight(&old_weight, FEATURE_NUM);

  // SGD
  printf("start gradient descent!!\n");
  int batch_size = BATCH_SIZE;
  double lr = LEARING_RATE;
  double eps = EPS;
  unsigned int maxit = MAX_ITER;

  struct timespec start, finish;
  double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);

  train(x, y, weight, old_weight, n_threads, batch_size, lr, eps, maxit);

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