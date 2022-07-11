#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constants.cpp"

void test_max(const char *data_file) {
  FILE *pfile;
  pfile = fopen(data_file, "r");
  char buffer[1000];

  if (pfile == NULL) {
    printf("open file fail!");
  }

  fgets(buffer, 1000, pfile);
  int sample = 0;
  int max_f = 0;
  while (fgets(buffer, 1000, pfile)) {
    // printf("buffer: %s\n", buffer);
    char *saveptr1 = NULL;
    char *ptr = strtok_r(buffer, " ", &saveptr1);
    int c = 0;
    while (ptr) {
      if (c != 0) {
        char *saveptr2 = NULL;
        char *ptr_x = strtok_r(ptr, ":", &saveptr2);
        // printf("ptr_x: %s\n", ptr_x);
        int idx = atoi(ptr_x) - 1;
        if (idx > max_f) max_f = idx;
      }
      c++;
      ptr = strtok_r(NULL, " ", &saveptr1);
    }
    sample++;
  }
  fclose(pfile);
  printf("feature num: %d\n", max_f);
}

void read_file(const char *data_file, int sample_num, double ***x, int **y) {
  FILE *pfile;
  pfile = fopen(data_file, "r");
  char buffer[1000];

  if (pfile == NULL) {
    printf("open file fail!");
  }

  for (int i = 0; i < sample_num; i++) {
    for (int j = 0; j < FEATURE_NUM; j++) (*x)[i][j] = 0;
  }

  fgets(buffer, 1000, pfile);
  int sample = 0;
  while (fgets(buffer, 1000, pfile)) {
    // printf("buffer: %s\n", buffer);
    char *saveptr1 = NULL;
    char *ptr = strtok_r(buffer, " ", &saveptr1);
    int c = 0;
    while (ptr) {
      if (c == 0) {
        (*y)[sample] = atoi(ptr);
        // printf("%d\n", atoi(ptr));
      } else {
        char *saveptr2 = NULL;
        char *ptr_x = strtok_r(ptr, ":", &saveptr2);
        // printf("ptr_x: %s\n", ptr_x);
        int idx = atoi(ptr_x) - 1;
        ptr_x = strtok_r(NULL, ":", &saveptr2);
        // printf("ptr_x value: %s\n", ptr_x);
        if (ptr_x == NULL) break;
        (*x)[sample][idx] = atof(ptr_x);
      }
      c++;
      ptr = strtok_r(NULL, " ", &saveptr1);
    }
    sample++;
  }
  fclose(pfile);
}

void fetch_dataset(const char *data_file, int n_sample, double ***x, int **y) {
  *x = (double **)malloc(n_sample * sizeof(double *));
  for (int i = 0; i < n_sample; i++)
    (*x)[i] = (double *)malloc(FEATURE_NUM * sizeof(double));

  *y = (int *)malloc(n_sample * sizeof(int));

  read_file(data_file, n_sample, x, y);
}

void init_weight(double **weight, int n_feature) {
  *weight = new double[n_feature];
  for (int i = 0; i < n_feature; i++) {
    (*weight)[i] = 0.0;
  }
}