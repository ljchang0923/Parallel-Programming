#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include <random>
long long int in_circle = 0;

pthread_mutex_t mutex_sum;

typedef struct{
    int pid;
    int start;
    int step;
} th_info;

//std::mt19937 generator(std::random_device{}());
//std::uniform_real_distribution<double> distribution(-1, 1);

void* toss(void* info){
    th_info *my_t = (th_info *)info;
    double max = 1.0, min = -1.0;
    long long int hit = 0;
    long long int t_toss_num;
    double x, y, distance;
    unsigned int seed = time(NULL);

    // printf("thread %d start from %d \n", my_t->pid, my_t->start);
    for(t_toss_num = my_t->start; t_toss_num < my_t->start + my_t->step; t_toss_num++){
        x = (max-min)*(double)rand_r(&seed) / (RAND_MAX) + min;
        y = (max-min)*(double)rand_r(&seed) / (RAND_MAX) + min;
        double distance = x*x + y*y;
        if (distance <= 1)
            hit++;
    }

    /* need to add critical section */
    pthread_mutex_lock(&mutex_sum);
    in_circle += hit;
    pthread_mutex_unlock(&mutex_sum);

    return NULL;
}

int main(int argc, char *argv[]){
    long thread_count;
    int thread_num = strtol(argv[1], NULL, 10);
    long long int toss_num = strtoll(argv[2], NULL, 10);
    int step = toss_num/thread_num;

    
    pthread_t thread[thread_num];
    th_info t_info[thread_num];
    pthread_mutex_init(&mutex_sum, NULL);
    
    

    for(thread_count = 0; thread_count < thread_num; thread_count++){
        t_info[thread_count].pid = thread_count;
        t_info[thread_count].start = thread_count*step;
        t_info[thread_count].step = step;

        pthread_create(&thread[thread_count], NULL, toss, (void *)&t_info[thread_count]);
    }
    
    for(thread_count = 0; thread_count < thread_num; thread_count++){
        pthread_join(thread[thread_count], NULL);
    }
    
    pthread_mutex_destroy(&mutex_sum);
    
    double est_pi = 4*in_circle/(double)toss_num;
    printf("%f\n", est_pi);
    pthread_exit(NULL);

    return 0;
}