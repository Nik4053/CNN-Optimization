#include <stdio.h>
#include <iostream>
#include "errorchecks.cuh"
#include "cuda_runtime.h"



void checkLast(char const* const file, int const line)
{
    cudaError_t err{ cudaGetLastError() };
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
            << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}