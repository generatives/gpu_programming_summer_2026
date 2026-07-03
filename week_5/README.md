The operation appears to be memory bound on my machine. I have made two implementation, one naive one that runs a single matmul per thread, and another that handles one column of our put values per thread, an approach that improves memory coalescing. I also tried some experiments with loading the A matrix into shared memory across threads, transfering values between threads with warp operations. None of these changes have really helped a lot, even the memory coalescing change only shaves a few microseconds off the runtime for 320,000 matmuls.

The naive kernel is substantially faster than cublasSgemmStridedBatched. I run a single kernel which takes about 118us. cublas runs five kernels in series, each of which take about 315us.

Here is the math to support this operations being memory bound:
## Arithmetic Cost
### Arithmetic Cost of One Output Element
4 multiplies and 3 addss for a vector dot product
1 additional add for the +c operations
= 8 FLOPs/output element

## Arithmetic Cost of Mat4 MatMul
= 8 * 16 output elements = 128 FLOPs

## Memory Cost
Each operations requires reading the 3 input matrices and writing the one output matrix (4 matrics of data per operation)

matrix size = 16 floats * 4 bytes/float = 64 bytes
required per operation = 64 bytes/matrix * 4 matrices = 256 bytes/operation

## Arithmetic Intensity
= 128 FLOPs/operation / 256 bytes/operation = 0.5 FLOPs/byte

## My GPU
I have a 10GB RTX 3080. It has 760GB/s DRAM bandwidth and 29.77 TFLOPS of single precision, not including tensor cores.
The ridge point is 29.77 / 0.760 = 39.17 FLOPs/byte.

## Theoretical Limit
I have been measuring with 320,000 Mat4 MatMuls. The memory requirements are 320,000 operations * 256 bytes/operation = 0.08192GB.
It takes 0.08912GB / 760GB/s = 107.79us to transfer that data from DRAM to compute.
My kernel takes 117us to run currently. There is a little more to squeeze out there but not a lot.

## Conclusion
Given the low arithmetic intensity of this operation, I cannot provide enough data to the GPU to saturate both memory and compute