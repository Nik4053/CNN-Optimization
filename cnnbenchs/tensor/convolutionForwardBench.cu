#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "convForward/cf_implementations.cuh"
#include "tensor.cuh"
#include "../errorchecks.cuh"

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
    
    Tensor<float,4> input({batch_size, channels_in, img_size, img_size});
    Tensor<float,4> output({batch_size, channels_out, img_size, img_size});
    Tensor<float,4> output_ref({batch_size, channels_out, img_size, img_size});
    Tensor<float,4> weights({channels_out, channels_in, filter_size, filter_size});
    Tensor<float,1> bias({channels_out});

    input.randomize(1234, 0.0, 1.0);
    Tensor<float,4> inputPadded = input.copyWithPadding(filter_size/2);
    weights.randomize(4444, 0.0, 1.0);
    bias.randomize(5678, 0.0, 1.0);

    output_ref.setZero();

   
    int dims=16;
    blockDim.x = width < dims ? width : dims;
    blockDim.y = height < dims ? height : dims;
    blockDim.z = 1;
    gridDim.x = ((height + blockDim.x - 1) / blockDim.x) ;
    gridDim.y = ((width + blockDim.y - 1) / blockDim.y) ;
    gridDim.z = channels_in;

    size_t sm = blockDim.x * blockDim.y * sizeof(float);

    timer.start();
    cf_baseline<3><<<gridDim, blockDim, sm>>>(input,output_ref, weights,bias ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl; 
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    

    // cf_padded
    std::cout << "cf_padded"<< std::endl;
    if(BENCH) cf_padded<3><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
    output.setZero();
    output.prefetechToDevice();
    timer.start();
    cf_padded<3><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(output_ref, output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // cf_padded_work
    const int yBatch = 8;

    std::cout << "cf_padded_work"<< std::endl;
    if(BENCH) cf_padded_work<3,1><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
    output.setZero();
    output.prefetechToDevice();
    gridDim.y /= yBatch;
    timer.start();
    cf_padded_work<3,yBatch><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    gridDim.y *= yBatch;
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(output_ref, output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // cf_padded_work_save
    std::cout << "cf_padded_work_save"<< std::endl;
    if(BENCH) cf_padded_work_save<3,1><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
    output.setZero();
    output.prefetechToDevice();
    gridDim.y /= yBatch;
    timer.start();
    cf_padded_work_save<3,yBatch><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    gridDim.y *= yBatch;
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(output_ref, output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        
    // cf_padded_reorder
    std::cout << "cf_padded_reorder"<< std::endl;
    if(BENCH) cf_padded_reorder<3,batch_size><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
    output.setZero();
    output.prefetechToDevice();
    timer.start();
    cf_padded_reorder<3,batch_size><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
    CHECK_LAST_CUDA_ERROR();
    timer.stop();
    std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

    checkSame(output_ref, output);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    if (batch_size <= 10) {
      // cf_padded_reorder_work
      std::cout << "cf_padded_reorder_work"<< std::endl;
      if(BENCH) cf_padded_reorder_work<3,batch_size,1><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
      output.setZero();
      output.prefetechToDevice();
      gridDim.y /= yBatch;
      timer.start();
      cf_padded_reorder_work<3,batch_size,yBatch><<<gridDim, blockDim, sm>>>(inputPadded,output, weights,bias ); 
      CHECK_LAST_CUDA_ERROR();
      timer.stop();
      gridDim.y *= yBatch;
      std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl;

      checkSame(output_ref, output);
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    return 0;
}

/*
Optimizations:
 - shfl
 - more work per thread
 - use registers
 - padded 
*/