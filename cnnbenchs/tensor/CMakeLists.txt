cmake_minimum_required(VERSION 3.12)

project(unetbench)
set (CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)
enable_language(CUDA)
add_definitions(-DCUDA_ON)

set(CMAKE_BUILD_TYPE Release) 


cuda_add_library(CUDALIB STATIC 
  "../errorchecks.cuh"
  "../errorchecks.cu"
  "tensor.cuh"
  "tensor.cu"
)
target_compile_options(CUDALIB PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       --relocatable-device-code=true
                       >)
set_target_properties(CUDALIB PROPERTIES CUDA_ARCHITECTURES "75")



add_executable(CONVWEIGHTS "./weightsReductionBench.cu")
set_target_properties(CONVWEIGHTS PROPERTIES LINKER_LANGUAGE CUDA) 
add_subdirectory(weightsReduction)
target_link_libraries(CONVWEIGHTS CUDAWEIGHTS)
target_link_libraries(CONVWEIGHTS CUDALIB) 
set_target_properties(CONVWEIGHTS PROPERTIES CUDA_ARCHITECTURES "75")


add_executable(CONVBACK "./convolutionBackwardBench.cu")
set_target_properties(CONVBACK PROPERTIES LINKER_LANGUAGE CUDA) 
add_subdirectory(convBackward)
target_link_libraries(CONVBACK CUDACONVBACK)
target_link_libraries(CONVBACK CUDALIB) 
set_target_properties(CONVBACK PROPERTIES CUDA_ARCHITECTURES "75")


add_executable(CONVFOR "./convolutionForwardBench.cu")
set_target_properties(CONVFOR PROPERTIES LINKER_LANGUAGE CUDA) 
add_subdirectory(convForward)
target_link_libraries(CONVFOR CUDACONVFOR)
target_link_libraries(CONVFOR CUDALIB) 
set_target_properties(CONVFOR PROPERTIES CUDA_ARCHITECTURES "75")




