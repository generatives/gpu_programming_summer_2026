# High Performance CLiFF Implementation on GPU
This repo contains a high performance GPU implementation on the GPU.

I will start with experiments with plain GMM. I will benchmark the sklearn implementation and the "pomegranate" GPU implementation of GMM. If there are obvious opportunities for imporvement I will try writing a fast GMM implementation in Triton or CUDA. If not then I will move on the the CLiFF method itself.

## Multi KMeans Benchmarking Results
### Warp GPU Implementation
1000m 3c 3000n: 0.64s

Note: The numpy based init process takes about 0.22s so the actual kernel takes 0.42s

### Tiled Warp GPU Implementation
1000m 3c 3000n: 0.28s

Note: The numpy based init process takes about 0.22s so the actual kernel takes 0.06s

### Scikitlearn Implementation
1000m 3c 3000n series: ~2s

Note: The numpy based init process takes about 0.22s so the actual kernel takes 1.78s

## KMeans Benchmarking Results
Both with kinit++ from 1000 randomly sampled
### Warp GPU Implementation
100c 10_000n: 0.181s

Note: Almost all of this time is spent on the kinit++ algorithm, the actual fitting process takes about 0.01s

### Numpy Implementation
100c 10_000n: 1.34s

## Multi GMM Benchmarking Results
### Scikit Learn
Train a set of scikit-learn GaussianMixtureModels on separate pieces of data.

100m 3c 100n series: 3.09s
100m 3c 1000n series: 2.44s
1000m 3c 1000n series: 27s
100m 3c 100n parallel: 30.05s

### Naive Warp Kernel
First implementation of multi-gmm with Nvidia Warps

## GMM Benchmarkings Results
### Scikit Learn
OOM on 10,000 components and 10,000,000 million data points

10c 1_000n: 13.1ms
10c 10_000n: 99.2ms
10c 100_000n: 581ms
10c 1_000_000n: 5.47s

100c 1_000n: 363ms
100c 10_000n: 514ms
100c 100_000n: 7.18ms

1000c 10_000n: 44s

### Pomegranate
It seems to be much slower that Scikit Learn on CPU!
100c 10_000n CPU: 1m 16s (!!!)
100c 10_000n GPU: 33.5s (!!!)
100c 10_000n GPU Compiled: 57.9s (!!!)

torch.compile(model.fit, mode='reduce-overhead', fullgraph=True) leads to failures, it cannot compile properly
torch.compile(model.fit) works but performance is poor

It seems like Pomegranate is not really made high performance GMM fitting, it is an expressive general purpose framework.

### Naive Numpy Implementation
I tried making a simple implementation in numpy to get a feeling for the algorithm.

100c 10_000n Unvectorized: 2.0s
100c 10_000n Vectorized: 2.16s

### Optimized Numpy Implementation
Claude did a magic version which I don't understand, it didn't seem to be faster