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
void cb_baseline(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);

template<int filter_size>
__global__
void cb_padded(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights);

template<int filter_size>
__global__
void cb_norelu(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);

template<int filter_size>
__global__
void cb_padded_norelu(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);

template<int filter_size>
__global__
void cb_padded_norelu_keep(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);

template<int filter_size, int batch_size>
__global__
void cb_padded_norelu_reorder(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);

template<int filter_size, int yBatch>
__global__
void cb_padded_norelu_reorder_pos(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights);

template<int filter_size, int yBatch>
__global__
void cb_padded_norelu_work(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);

template<int filter_size, int yBatch>
__global__
void cb_padded_norelu_reorder_work(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);

template<int filter_size, int yBatch>
__global__
void cb_padded_norelu_work_save(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights);