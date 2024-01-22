72 streaming multiprocessors on NVIDIA A10



![image](https://github.com/zengbs/gpu-find-best-grid-block-configuration/blob/main/throughput_vs_block_grid_size.png?raw=true)


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
