
cuda_add_library(CUDACONVBACK STATIC 
  "cb_implementations.cuh"
  "cb_baseline.cu"
  "cb_padded.cu"
  "cb_norelu.cu"
  "cb_padded_norelu.cu"
  "cb_padded_norelu_keep.cu"
  "cb_padded_norelu_reorder.cu"
  "cb_padded_norelu_work.cu"
  "cb_padded_norelu_reorder_work.cu"
  "cb_padded_norelu_reorder_pos.cu"
  "cb_padded_norelu_work_save.cu"
)
target_compile_options(CUDACONVBACK PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       --fmad=true
                       >)
set_target_properties(CUDACONVBACK PROPERTIES CUDA_ARCHITECTURES "75")