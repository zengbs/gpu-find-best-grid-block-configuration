72 streaming multiprocessors on NVIDIA A10

To understand how grid/block size affects GPU throughput, we use a CUDA kernel below to give a demonstration. The kernel adds two one-dimensional (1D) vectors on global memory and combines them to form another third 1D vector.

```cuda=
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
```

The kernel systematically iterates through a range of grid sizes and block sizes, executing the kernel function with each combinations.

Before and after each kernel execution, the script calls `Start()` and `Stop()` functions for timing the excution. The `cudaDeviceSynchronize()` function ensures the kernal function is completed before proceeding on the host side. Without calling `cudaDeviceSynchronize()`, the thread on CPU will  proceed to execute the `Stop()` immediately after the kernel function is called. This premature execution leads to inaccurate timing results, as it doesn't account for the actual completion time of the kernel's execution on the GPU.
```cuda=
// Launch kernel
for ( gridSize = 1; gridSize<=1024; gridSize++ ){
   for ( blockSize = 1; blockSize<=1024; blockSize++ ){
      Start();
      kernel<<< gridSize, blockSize >>>(d_A, d_B, d_C, N);
      cudaDeviceSynchronize();
      Stop();
      printf("%d, %d, %e\n", gridSize, blockSize, GetValue());
   }
   printf("\n");
}
```

The figure below illustrates the timing results derived from the benchmark script on NVIDIA A10, represented as a function of both grid size and block size. At first glance, the figure shows that the time spent on the kernel execution increases as either the block size or grid size decreases.

![image](https://github.com/zengbs/gpu-find-best-grid-block-configuration/blob/main/fig__3d.png?raw=true)

To conduct detailed analysis, we plot the profiles in the top row panels, the right panel shows how the kernel's execution time varies with grid size at a fixed block size, while the left panel shows the variation with block size at a fixed grid size. These two profiles are based on the time as a function of both grid and block size showing the above figure.


![image](https://github.com/zengbs/gpu-find-best-grid-block-configuration/blob/main/fig__occupancy.png?raw=true)


$$\begin{equation}
\displaystyle
\text{Warp occupancy}=\frac{\text{Number of threads per block}}{\text{ceiling(Number of threads per block, warp size)}}
\end{equation}$$




$$\begin{equation}
\displaystyle
\text{SM occupancy}=\frac{\text{Number of blocks per grid}}{\text{ceiling(Number of blocks per grid, Number of SMs per GPU)}}
\end{equation}$$
