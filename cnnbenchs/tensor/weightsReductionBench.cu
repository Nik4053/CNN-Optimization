#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "weightsReduction/implementations.cuh"
// #include "../src/convRelu.cuh"
#include "tensor.cuh"
#include "../errorchecks.cuh"
// #include <benchmark/benchmark.h>

#define BENCH 0 
#undef NDEBUG

class Timer{
  cudaEvent_t _start, _stop;
public:
  Timer(){
    CHECK_CUDA_ERROR(cudaEventCreate(&_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&_stop));
  }
  ~Timer(){
    CHECK_CUDA_ERROR(cudaEventDestroy(_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(_stop));
  }
  void start(){
    CHECK_CUDA_ERROR(cudaEventRecord(_start));
  }
  void stop(){
    CHECK_CUDA_ERROR(cudaEventRecord(_stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(_stop));
  }
  float getElapsedTime(){
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, _start, _stop));
    return milliseconds;
  }
};


bool AreSame(double a, double b)
{
    return std::abs(a - b) < 0.0001 * (std::abs(a) + std::abs(b));
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

int main(){
  Timer timer;

  // params
  const size_t batch_size = 100;
  size_t channels_in = 120;
  size_t channels_out = 80;
  size_t img_size = 128;
  size_t width = img_size, height = img_size;
  size_t filter_size = 3;


  // setup
  
	dim3 gridDim;
	dim3 blockDim;
  Tensor<float,4> input({batch_size, channels_in, img_size, img_size});
  // Tensor<float,4> inputPadded({batch_size, channels_in, img_size+filter_size/2, img_size+filter_size/2});
  Tensor<float,4> error({batch_size, channels_out, img_size, img_size});
  Tensor<float,4> gradient_weights({channels_out, channels_in , filter_size, filter_size});
  Tensor<float,4> gradient_weights_ref({channels_out, channels_in, filter_size, filter_size});

  input.randomize(1234, 0.0, 1.0);
  Tensor<float,4> inputPadded = input.copyWithPadding(filter_size/2);
  error.randomize(5678, 0.0, 1.0);
  gradient_weights_ref.setZero();
  gradient_weights.setZero();

  input.prefetechToDevice();
  error.prefetechToDevice();
  gradient_weights_ref.prefetechToDevice();
  // gradient_weights.randomize(4444, 0.0, 1.0);
  // baseline




	int dims=16;
	blockDim.x = width < dims ? width : dims;
	blockDim.y = height < dims ? height : dims;
	blockDim.z = 1;
	gridDim.x = ((height + blockDim.x - 1) / blockDim.x) ;
	gridDim.y = ((width + blockDim.y - 1) / blockDim.y) ;
	gridDim.z = channels_out * channels_in;

	size_t sm = blockDim.x * blockDim.y * sizeof(float);

  timer.start();
  baseline<3><<<gridDim, blockDim, sm>>>(input, error, gradient_weights_ref); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl; 

  // w_padded
  std::cout << "w_padded"<< std::endl;
  if(BENCH) w_padded<3><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_padded<3><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);


  // w_shfl
  std::cout << "w_shfl"<< std::endl;
  if(BENCH) w_shfl<3><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_shfl<3><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);

  // w_registers
  std::cout << "w_registers"<< std::endl;
  if(BENCH) w_registers<3><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_registers<3><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);
 
  // w_shfl_registers
  std::cout << "w_shfl_registers"<< std::endl;
  if(BENCH) w_shfl_registers<3><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_shfl_registers<3><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);

  // w_shfl_registers_reordered
  std::cout << "w_shfl_registers_reordered"<< std::endl;
  if(BENCH) w_shfl_registers_reordered<3><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_shfl_registers_reordered<3><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);

  // w_shfl_registers_reordered_padded
  std::cout << "w_shfl_registers_reordered_padded"<< std::endl;
  if(BENCH) w_shfl_registers_reordered_padded<3><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_shfl_registers_reordered_padded<3><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);

  // w_shfl_registers_reordered_padded_work
  
  const int yBatch = 8;
  std::cout << "w_shfl_registers_reordered_padded_work"<< std::endl;
  if(BENCH) w_shfl_registers_reordered_padded_work<3,1><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  gridDim.y = gridDim.y / yBatch;
  w_shfl_registers_reordered_padded_work<3,yBatch><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  gridDim.y = gridDim.y * yBatch;
  checkSame(gradient_weights_ref, gradient_weights);

  // w_shfl_registers_reordered_padded_work_keep
  std::cout << "w_shfl_registers_reordered_padded_work_keep"<< std::endl;
  if(BENCH) w_shfl_registers_reordered_padded_work_keep<3,1><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  gridDim.y = gridDim.y / yBatch;
  w_shfl_registers_reordered_padded_work_keep<3,yBatch><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  gridDim.y = gridDim.y * yBatch;
  checkSame(gradient_weights_ref, gradient_weights);

  // w_shfl_registers_reordered_padded_work_reuseIndex
  std::cout << "w_shfl_registers_reordered_padded_work_reuseIndex"<< std::endl;
  if(BENCH) w_shfl_registers_reordered_padded_work_reuseIndex<3,1><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  gridDim.y = gridDim.y / yBatch;
  w_shfl_registers_reordered_padded_work_reuseIndex<3,yBatch><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  gridDim.y = gridDim.y * yBatch;
  checkSame(gradient_weights_ref, gradient_weights);

  // w_shfl_registers_reordered_padded_shared(b)
  std::cout << "w_shfl_registers_reordered_padded_shared"<< std::endl;
  if(BENCH) w_shfl_registers_reordered_padded_shared<3,1><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_shfl_registers_reordered_padded_shared<3,batch_size><<<gridDim, blockDim, sm>>>(inputPadded, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);


  // w_work 1
  std::cout << "w_work("<<1<<"): " << std::endl;
  if(BENCH) w_work<3,1><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_work<3,1><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);
  
  // w_work 16
  const int yBatchSize = 4;
  std::cout << "w_work("<<yBatchSize<<"): " << std::endl;
  gridDim.y = gridDim.y / yBatchSize;
  if(input.getDim(2) % yBatchSize != 0 ) std::cout <<  "y dim must be divisible by "<< yBatchSize << std::endl; 
  if(gridDim.y == 0) std::cout <<  "gridDim.y must not be 0" << std::endl; 
  if(BENCH) w_work<3,yBatchSize><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_work<3,yBatchSize><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);

  // w_work_shfl
  std::cout << "w_work("<<yBatchSize<<")_shfl: " << std::endl;
  if(input.getDim(2) % yBatchSize != 0 ) std::cout <<  "y dim must be divisible by "<< yBatchSize << std::endl; 
  if(gridDim.y == 0) std::cout <<  "gridDim.y must not be 0" << std::endl; 
  if(BENCH) w_work_shfl<3,yBatchSize><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_work_shfl<3,yBatchSize><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);

  // w_work_shfl_registers
  std::cout << "w_work("<<yBatchSize<<")_shfl_registers: " << std::endl;
  if(input.getDim(2) % yBatchSize != 0 ) std::cout <<  "y dim must be divisible by "<< yBatchSize << std::endl; 
  if(gridDim.y == 0) std::cout <<  "gridDim.y must not be 0" << std::endl; 
  if(BENCH) w_work_shfl_registers<3,yBatchSize><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  gradient_weights.setZero();
  gradient_weights.prefetechToDevice();
  timer.start();
  w_work_shfl_registers<3,yBatchSize><<<gridDim, blockDim, sm>>>(input, error, gradient_weights); 
  CHECK_LAST_CUDA_ERROR();
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

  checkSame(gradient_weights_ref, gradient_weights);

  return 0;
}

/*
Optimizations:
 - shfl
 - more work per thread
 - use registers
 - padded 
*/