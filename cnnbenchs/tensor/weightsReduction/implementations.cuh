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
void baseline(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);


template<int filter_size>
__global__
void w_padded(Tensor<float, 4> inputPadded, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size>
__global__
void w_shfl(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size>
__global__
void w_registers(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size>
__global__
void w_shfl_registers(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size>
__global__
void w_shfl_registers_reordered(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size>
__global__
void w_shfl_registers_reordered_padded(Tensor<float, 4> inputPadded, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size, int yBatch>
__global__
void w_shfl_registers_reordered_padded_work(Tensor<float, 4> inputPadded, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size, int yBatch>
__global__
void w_shfl_registers_reordered_padded_work_reuseIndex(Tensor<float, 4> inputPadded, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size, int batch_size>
__global__
void w_shfl_registers_reordered_padded_work_keep(Tensor<float, 4> inputPadded, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size, int batch_size>
__global__
void w_shfl_registers_reordered_padded_shared(Tensor<float, 4> inputPadded, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);


template<int filter_size, int yBatch>
__global__
void w_work(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size, int yBatch>
__global__
void w_work_shfl(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);

template<int filter_size, int yBatch>
__global__
void w_work_shfl_registers(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);
