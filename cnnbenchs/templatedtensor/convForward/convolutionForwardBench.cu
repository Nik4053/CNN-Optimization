#include <iostream>
#include <random>
// cuda runtimes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// helper files
#include "../tensor.hpp"
#include "../../timer.hpp"
#include "../../errorchecks.cuh"
// benchmark functions
#include "cf_baseline.cuh"
#include "cf_padded.cuh"
#include "cf_padded_work.cuh"

#undef NDEBUG

// defines that make Tensor use simpler
#define INPUT tel::Tensor<float,batch_size, channels_in, img_size, img_size>
#define INPUT_PADDED tel::Tensor<float,batch_size, channels_in, img_size+filter_size/2 +1, img_size+filter_size/2 + 1>
#define OUTPUT tel::Tensor<float,batch_size, channels_out, img_size, img_size>
#define WEIGHTS tel::Tensor<float,channels_out,channels_in , filter_size, filter_size>
#define BIAS tel::Tensor<float,channels_out>

void randomize(unsigned int seed, float min, float max, float *mem, size_t size) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> unif(min, max);

  for (int b = 0; b < size; b++)
    mem[b] = unif(rng);
}



template<class A, class B>
void copyWithPadding(A &input, B &input_padded, size_t padding){
  input_padded.setZero();
  // copy data over
  for (size_t b = 0; b<input.getDim(0);b++){
    for (size_t c_in =0; c_in<input.getDim(1);c_in++){
      for (size_t h =0; h<input.getDim(2);h++){
        for (size_t w =0; w<input.getDim(3);w++){
          input_padded(b,c_in,h+padding,w+padding) = input(b,c_in,h,w);
        }
      }
    }
  }
}
int main(){
  Timer timer;
  
  // params
  const size_t batch_size = 100;
  const size_t channels_in = 120;
  const size_t channels_out = 80;
  const size_t img_size = 128;
  const size_t width = img_size, height = img_size;
  const size_t filter_size = 3;

  // int host = -1;
  int device = -1;
  CHECK_CUDA_ERROR(cudaGetDevice(&device));


  // setup
  
  dim3 gridDim;
  dim3 blockDim;

  // alloc memory
  float *input_mem, *input_padded_mem, *output_mem, *output_ref_mem, *weights_mem, *bias_mem;
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&input_mem, INPUT::SIZE * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&input_padded_mem, INPUT_PADDED::SIZE * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&output_mem, OUTPUT::SIZE * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&output_ref_mem, OUTPUT::SIZE * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&weights_mem, WEIGHTS::SIZE * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMallocManaged((void**)&bias_mem, BIAS::SIZE * sizeof(float)));
  INPUT input(input_mem);
  INPUT_PADDED input_padded(input_padded_mem);
  OUTPUT output(output_mem);
  OUTPUT output_ref(output_ref_mem);
  WEIGHTS weights(weights_mem);
  BIAS bias(bias_mem);

  // init data
  randomize(1234, 0.0, 1.0, input_mem, input.SIZE);
  randomize(4444, 0.0, 1.0, weights_mem, weights.SIZE);
  randomize(5678, 0.0, 1.0, bias_mem, bias.SIZE);
  // set 0
  CHECK_CUDA_ERROR(cudaMemset(output_ref_mem, 0, OUTPUT::SIZE * sizeof(float)));
  // init padded input
  size_t padding = filter_size/2;
  copyWithPadding(input,input_padded,padding);



  

  int dims=16;
  blockDim.x = width < dims ? width : dims;
  blockDim.y = height < dims ? height : dims;
  blockDim.z = 1;
  gridDim.x = ((height + blockDim.x - 1) / blockDim.x) ;
  gridDim.y = ((width + blockDim.y - 1) / blockDim.y) ;
  gridDim.z = channels_in;

  size_t sm = blockDim.x * blockDim.y * sizeof(float);


  // prefetech to device
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(input_mem, INPUT::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(output_ref_mem, OUTPUT::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(weights_mem, WEIGHTS::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(bias_mem, BIAS::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  // start computation
  timer.start();
  cf_baseline<3,batch_size,channels_in,channels_out,height,width><<<gridDim, blockDim, sm>>>(input,output_ref, weights,bias ); 
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl; 


  
  // cf_padded
  std::cout << "cf_padded"<< std::endl;
  // prefetech to device
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(input_mem, INPUT::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(output_mem, OUTPUT::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(weights_mem, WEIGHTS::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(bias_mem, BIAS::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  // start computation
  timer.start();
  cf_padded<3,batch_size,channels_in,channels_out,height,width><<<gridDim, blockDim, sm>>>(input_padded,output, weights,bias ); 
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl; 

  if(output_ref != output){
    std::cout << "Are not the same!!" << std::endl;
  }

  // cf_padded_work
  const int yBatch = 8;
  std::cout << "cf_padded_work"<< std::endl;
  output.setZero();
  gridDim.y /= yBatch;
  // prefetech to device
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(input_mem, INPUT::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(input_mem, OUTPUT::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(weights_mem, WEIGHTS::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(bias_mem, BIAS::SIZE * sizeof(float), device, NULL));
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  // start computation
  timer.start();
  cf_padded_work<3,yBatch,batch_size,channels_in,channels_out,height,width><<<gridDim, blockDim, sm>>>(input_padded,output, weights,bias ); 
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "Time (ms): " << timer.getElapsedTime() << std::endl; 
  gridDim.y *= yBatch;
  if(output_ref != output){
    std::cout << "Are not the same!!" << std::endl;
  }



  // free
  CHECK_CUDA_ERROR(cudaFree(input_mem));
  CHECK_CUDA_ERROR(cudaFree(output_mem));
  CHECK_CUDA_ERROR(cudaFree(output_ref_mem));
  CHECK_CUDA_ERROR(cudaFree(weights_mem));
  CHECK_CUDA_ERROR(cudaFree(bias_mem));

  return 0;
}

/*
Optimizations:
 - shfl
 - more work per thread
 - use registers
 - padded 
*/