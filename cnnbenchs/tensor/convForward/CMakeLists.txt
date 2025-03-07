
cuda_add_library(CUDACONVFOR STATIC 
  "cf_implementations.cuh"
  "cf_baseline.cu"
  "cf_padded.cu"
  "cf_padded_work.cu"
  "cf_padded_work_save.cu"
  "cf_padded_reorder.cu"
  "cf_padded_reorder_work.cu"
)
target_compile_options(CUDACONVFOR PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       --fmad=true
                       >)
set_target_properties(CUDACONVFOR PROPERTIES CUDA_ARCHITECTURES "75")