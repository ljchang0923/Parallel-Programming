#ifndef __HOSTFE__
#define __HOSTFE__

void train(double *x, int *y, double *weight, double *old_weight,
                         int batch_size,
                         double lr, double eps,
                         unsigned int maxit);
#endif