#include <iostream>
#include "../tensor.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef NDEBUG
#define Debug(x) {}
#else
#define Debug(x) {x}
#endif


template<int filter_size>
__global__
void cf_baseline(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);

template<int filter_size>
__global__
void cf_padded(Tensor<float, 4> inputPadded, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);

template<int filter_size, int yBatch>
__global__
void cf_padded_work(Tensor<float, 4> inputPadded, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);

template<int filter_size, int yBatch>
__global__
void cf_padded_work_save(Tensor<float, 4> inputPadded, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);

template<int filter_size, int batch_size>
__global__
void cf_padded_reorder(Tensor<float, 4> inputPadded, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);

template<int filter_size, int batch_size, int yBatch>
__global__
void cf_padded_reorder_work(Tensor<float, 4> inputPadded, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);