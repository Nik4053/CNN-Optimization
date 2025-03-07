#include "tensor.cuh"

#ifdef __DEBUG__
 size_t allocedBytes = 0;
// static size_t freedBytes = 0;
 size_t allocedTensors = 0;
 size_t freedTensors = 0;
#endif