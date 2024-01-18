#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Size of array
#define N 1048570


unsigned long Time;

/* generate a random floating point number from min to max */
double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void Reset()
{
   Time = 0;
}

double GetValue()
{
   return Time*1.0e-6;
}

void Start()
{
   Time = 0;
   struct timeval tv;
   gettimeofday( &tv, NULL );
   Time   = tv.tv_sec*1000000 + tv.tv_usec;
}

void Stop()
{
   struct timeval tv;
   gettimeofday( &tv, NULL );
   Time   = tv.tv_sec*1000000 + tv.tv_usec - Time;
}

// Kernel
__global__ void kernel(double *a, double *b, double *c, long length)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    while(id < length){
       c[id] = a[id] + b[id] + sqrt(a[id]) + sqrt(b[id]);
       c[id] = sqrt(c[id]);
       id += gridDim.x*blockDim.x;
    }

}

// Main program
int main()
{
    srand (time ( NULL));

    // Number of bytes to allocate for N doubles
    size_t bytes = N*sizeof(double);

    // Allocate memory for arrays A, B, and C on host
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Fill host arrays A and B
    for(int i=0; i<N; i++)
    {
        A[i] = randfrom(+1.0, 16.0);
        B[i] = randfrom(+1.0, 4.0);
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 1024;
    int blk_in_grid = 1;

    // Warm up
    Start();
    kernel<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, N);
    Stop();

    printf("# gridSize       blockSize        Time (sec)");

    // Launch kernel
    for ( blk_in_grid = 1; blk_in_grid<=1024; blk_in_grid++ ){
       for ( thr_per_blk = 1; thr_per_blk<=1024; thr_per_blk++ ){

          Start();
          kernel<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C, N);
          cudaDeviceSynchronize();
          Stop();
          printf("%d, %d, %e\n", blk_in_grid, thr_per_blk, GetValue());

       }
       printf("\n");
    }


    // Free CPU memory
    free(A);
    free(B);
    free(C);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    return 0;
}
