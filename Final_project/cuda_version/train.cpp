#include <math.h>
#include <stdio.h>
#include <string.h>

#include "constants.cpp"

static inline double classify(double *x, double *weight) {
  double pred = 0;
  for (int f = 0; f < FEATURE_NUM; f++) {
    pred += x[f] * weight[f];
  }
  return pred;
}

static inline void copy_weight(double *old_w, double *new_w) {
  memcpy(old_w, new_w, FEATURE_NUM * sizeof(double));
}

static inline double vecnorm(double *old_w, double *new_w) {
  double sum = 0.0;
  for (int i = 0; i < FEATURE_NUM; i++) {
    double diff = old_w[i] - new_w[i];
    sum += diff * diff;
  }

  return sqrt(sum);
}

static inline void print_one_sample(double *a) {
  for (int i = 0; i < FEATURE_NUM; i++) {
    printf("%f", a[i]);
  }
}

static inline void test(double *weight, int test_sample, double **x_test,
                        int *y_test) {
  printf("start testing!!\n");
  double tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;
  for (int i = 0; i < test_sample; i++) {
    int label = y_test[i];
    double pred = classify(x_test[i], weight);

    if (((label == -1 || label == 0) && pred < 0.5) ||
        (label == 1 && pred >= 0.5)) {
      if (label == 1) {
        tp++;
      } else {
        tn++;
      }
    } else {
      if (label == 1) {
        fn++;
      } else {
        fp++;
      }
    }
  }
  printf("# accuracy:    %1.4f (%i/%i)\n", ((tp + tn) / (tp + tn + fp + fn)),
         (int)(tp + tn), (int)(tp + tn + fp + fn));
}

void delete_dataset(double ***x, int **y, int n_sample) {
  for (int i = 0; i < n_sample; i++) {
    delete[](*x)[i];
  }
  delete[] * x;
  delete[] * y;
}