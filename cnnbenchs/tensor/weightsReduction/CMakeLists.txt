
cuda_add_library(CUDAWEIGHTS STATIC 
  "implementations.cuh"
  "baseline.cu"
  "w_shfl.cu"
  "w_registers.cu"
  "w_shfl_registers.cu"
  "w_shfl_registers_reordered.cu"
  "w_shfl_registers_reordered_padded.cu"
  "w_shfl_registers_reordered_padded_shared.cu"
  "w_shfl_registers_reordered_padded_work.cu"
  "w_shfl_registers_reordered_padded_work_keep.cu"
  "w_work.cu"
  "w_work_shfl.cu"
  "w_work_shfl_registers.cu"
  "w_padded.cu"
  "w_shfl_registers_reordered_padded_work_reuseIndex.cu"
)
target_compile_options(CUDAWEIGHTS PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       --fmad=true
                       >)
set_target_properties(CUDAWEIGHTS PROPERTIES CUDA_ARCHITECTURES "75")