## Description
This project aims to try a few simple optimization techniques to try to speedup the computation of the convolutional layer of a neural net. The main focus is to learn different optimization techniques in CUDA and understand their effects on the performance of the program. 
## Implementations
The Project features two different versions:
* tensor: Is the main project. It features performance improvements for the forward, backward and reduction functions of the convolutional layer.
* templatedtensor: Implementation of the forward pass using the [templated tensor library](https://github.com/Nik4053/Templated-Cpp-Tensor)

## Installation
Choose one of the main folders and ```cd``` into it.
```bash
mkdir build
cd build
cmake ..
make -j
```
Now you can run the created executables.

In order to create profiling data usable with NVIDIA Nsight Compute call:
```bash
ncu -f -o profile <executable> 
```