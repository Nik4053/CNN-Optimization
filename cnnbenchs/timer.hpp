#pragma once 

#include "errorchecks.cuh"
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