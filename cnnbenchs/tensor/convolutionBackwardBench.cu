#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "convBackward/cb_implementations.cuh"
// #include "convRelu.cuh"
#include "tensor.cuh"
#include "../timer.hpp"
#include "../errorchecks.cuh"
// #include <benchmark/benchmark.h>

#define BENCH 0 
#undef NDEBUG

__global__
static void ReLU_backwards(Tensor<float, 4> error, Tensor<float, 4> output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.z * blockDim.z;

	int batch_size = error.getDim(0),
		channels = error.getDim(1),
		width = output.getDim(3),
		height = output.getDim(2);

	if (x >= width || y >= height || c >= channels)
		return;

	for (int b = 0; b < batch_size; b++) {
		error(b, c, y, x) = (output(b, c, y, x) != 0.)?error(b, c, y, x):0.;
	}
}


bool AreSame(double a, double b)
{
    return std::abs(a - b) <= 0.0001 * (std::abs(a) + std::abs(b));
}
void checkSame(Tensor<float,4> &a, Tensor<float,4> &b){
  if(a.getSize() != b.getSize()){
    std::cout << "Error: tensors have different sizes" << std::endl;
    return;
  }
  for(size_t i = 0; i < a.getSize(); ++i){
    if(!AreSame(a[i],b[i])){
      std::cout << "Error at index " << i << std::endl;
      std::cout << "a: " << a[i] << std::endl;
      std::cout << "b: " << b[i] << std::endl;
      std::cout << "a-b: " << a[i]-b[i] << std::endl;
      return;
    }
  }
  // std::cout << "All good" << std::endl;
}
void floorTensor(Tensor<float,4> tensor, float limit) {

    for (int b = 0; b < tensor.getSize(); b++)
        tensor(b) = tensor(b)<limit?0.0:tensor(b);
}
int main(){
    Timer timer;

    // params
    const size_t batch_size = 10;
    size_t channels_in = 120;
    size_t channels_out = 80;
    size_t img_size = 512;
    size_t width = img_size, height = img_size;
    size_t filter_size = 3;


    // setup
    
        dim3 gridDim;
        dim3 blockDim;
    // Tensor<float,4> inputPadded({batch_size, channels_in, img_size+filter_size/2, img_size+filter_size/2});
    Tensor<float,4> error({batch_size, channels_out, img_size, img_size});
    Tensor<float,4> next_error_ref({batch_size, channels_in, img_size, img_size});
    Tensor<float,4> next_error({batch_size, channels_in, img_size, img_size});
    Tensor<float,4> output({batch_size, channels_out, img_size, img_size});
    Tensor<float,4> weights({channels_out, channels_in, filter_size, filter_size});

    error.randomize(1234, 0.0, 1.0);
    output.randomize(5678, 0.0, 1.0);
    floorTensor(output,0.7);
    weights.randomize(4444, 0.0, 1.0);
    Tensor<float,4> errorPadded = error.copyWithPadding(filter_size/2);
    Tensor<float,4> errorRelu = error.copy();
    

    Tensor<float,4> outputPadded = output.copyWithPadding(filter_size/2);
    next_error_ref.setZero();
    next_error.setZero();

    next_error.prefetechToDevice();
    error.prefetechToDevice();
    output.prefetechToDevice();
    weights.prefetechToDevice();
    next_error_ref.prefetechToDevice();
    // gradient_weights.randomize(4444, 0.0, 1.0);
    // baseline




    int dims=16;
    blockDim.x = width < dims ? width : dims;
    blockDim.y = height < dims ? height : dims;
    blockDim.z = 1;
    gridDim.x = ((height + blockDim.x - 1) / blockDim.x) ;
    gridDim.y = ((width + blockDim.y - 1) / blockDim.y) ;
    gridDim.z = channels_in;

    size_t sm = blockDim.x * blockDim.y * sizeof(float);

    
    gridDim.z = channels_out;
    ReLU_backwards << <gridDim, blockDim >> > (errorRelu, output);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    Tensor<float,4> errorReluPadded = errorRelu.copyWithPadding(filter_size/2);
    gridDim.z = channels_in;

    timer.start();
    cb_baseline<3><<<gridDim, blockDim, sm>>>(error,next_error_ref, output,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl; 
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // padded
    std::cout << "cb_padded"<< std::endl;
    if(BENCH) cb_padded<3><<<gridDim, blockDim, sm>>>(errorPadded,next_error, outputPadded,weights ); 
    next_error.setZero();
    next_error.prefetechToDevice();
    timer.start();
    cb_padded<3><<<gridDim, blockDim, sm>>>(errorPadded,next_error, outputPadded,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(next_error_ref, next_error);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // cb_norelu
    std::cout << "cb_norelu"<< std::endl;
    if(BENCH) cb_norelu<3><<<gridDim, blockDim, sm>>>(error,next_error, output,weights ); 
    next_error.setZero();
    next_error.prefetechToDevice();
    timer.start();
    // gridDim.z = channels_out;
    // ReLU_backwards << <gridDim, blockDim >> > (error, output);
    // CHECK_LAST_CUDA_ERROR();
    // cudaDeviceSynchronize();
    // gridDim.z = channels_in;
    cb_norelu<3><<<gridDim, blockDim, sm>>>(errorRelu,next_error, output,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(next_error_ref, next_error);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    // cb_padded_norelu
    std::cout << "cb_padded_norelu"<< std::endl;
    if(BENCH) cb_padded_norelu<3><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    next_error.setZero();
    next_error.prefetechToDevice();
    timer.start();
    // gridDim.z = channels_out;
    // ReLU_backwards << <gridDim, blockDim >> > (error, output);
    // CHECK_LAST_CUDA_ERROR();
    // cudaDeviceSynchronize();
    // gridDim.z = channels_in;
    cb_padded_norelu<3><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(next_error_ref, next_error);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // cb_padded_norelu_reorder

    sm = (blockDim.x +2) * (blockDim.y+2) * sizeof(float) * 2 ;
    std::cout << "cb_padded_norelu_reorder"<< std::endl;
    if(BENCH) cb_padded_norelu_reorder<3,batch_size><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    next_error.setZero();
    next_error.prefetechToDevice();
    timer.start();
    cb_padded_norelu_reorder<3,batch_size><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(next_error_ref, next_error);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // cb_padded_norelu_reorder_pos
    sm = (blockDim.x +2) * (blockDim.y+2) * sizeof(float) * 2 ;
    std::cout << "cb_padded_norelu_reorder_pos"<< std::endl;
    if(BENCH) cb_padded_norelu_reorder_pos<3,batch_size><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    next_error.setZero();
    next_error.prefetechToDevice();
    timer.start();
    cb_padded_norelu_reorder_pos<3,batch_size><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(next_error_ref, next_error);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // cb_padded_norelu_work

    sm = (blockDim.x +2) * (blockDim.y+2) * sizeof(float) * 2 ;
    const int yBatch = 8;
    std::cout << "cb_padded_norelu_work"<< std::endl;
    if(BENCH) cb_padded_norelu_work<3,1><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    next_error.setZero();
    next_error.prefetechToDevice();
    
    gridDim.y /= yBatch;
    timer.start();
    cb_padded_norelu_work<3,yBatch><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    gridDim.y *= yBatch;
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(next_error_ref, next_error);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());


    // cb_padded_norelu_reorder_work

    sm = (blockDim.x +2) * (blockDim.y+2) * sizeof(float) * 2 ;
    std::cout << "cb_padded_norelu_reorder_work"<< std::endl;
    if(BENCH) cb_padded_norelu_reorder_work<3,1><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    next_error.setZero();
    next_error.prefetechToDevice();
    
    gridDim.y /= yBatch;
    timer.start();
    cb_padded_norelu_reorder_work<3,yBatch><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    gridDim.y *= yBatch;
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(next_error_ref, next_error);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // cb_padded_norelu_keep

    sm = (blockDim.x +2) * (blockDim.y+2) * sizeof(float) * 2 ;
    std::cout << "cb_padded_norelu_keep"<< std::endl;
    if(BENCH) cb_padded_norelu_keep<3><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    next_error.setZero();
    next_error.prefetechToDevice();
    timer.start();
    cb_padded_norelu_keep<3><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(next_error_ref, next_error);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // cb_padded_norelu_work_save
    sm = (blockDim.x +2) * (blockDim.y+2) * sizeof(float) * 2 ;
    std::cout << "cb_padded_norelu_work_save"<< std::endl;
    if(BENCH) cb_padded_norelu_work_save<3,1><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    next_error.setZero();
    next_error.prefetechToDevice();
    gridDim.y /= yBatch;
    timer.start();
    cb_padded_norelu_work_save<3,yBatch><<<gridDim, blockDim, sm>>>(errorReluPadded,next_error, outputPadded,weights ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    gridDim.y *= yBatch;
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(next_error_ref, next_error);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return 0;
}

/*
Optimizations:
 - shfl
 - more work per thread
 - use registers
 - padded 
*/