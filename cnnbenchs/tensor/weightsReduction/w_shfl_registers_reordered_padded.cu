#include "implementations.cuh"



// reordered the for loops to get better memory access
template<int filter_size>
__global__
void w_shfl_registers_reordered_padded(Tensor<float, 4> inputPadded, Tensor<float, 4> error, Tensor<float, 4> gradient_weights) {
	const int block_y = blockIdx.y;
	const int block_x = blockIdx.x;
	const int thread_y = threadIdx.y;
	const int thread_x = threadIdx.x;

	// get our position in the grid. #blocks*blocksize + position in block
	const int x = block_x * blockDim.x  + thread_x;
	const int y = block_y * blockDim.y  + thread_y;
	const int z = blockIdx.z * blockDim.z;


	// asserts for debug
	// assert(input.getDim(3) == error.getDim(3) && input.getDim(2) == error.getDim(2)); // img size has to be identical 
	// assert(input.getDim(0) == error.getDim(0)); // batch size has to be identical
	// assert(input.getDim(1) == gradient_weights.getDim(1)); // input channels have to be identical
	// assert(error.getDim(1) == gradient_weights.getDim(0)); // output channels have to be identical
	// assert(gradient_weights.getDim(2) == filter_size && gradient_weights.getDim(3) == filter_size); // filter size has to be identical



	const int batch_size = inputPadded.getDim(0),
				input_channels = inputPadded.getDim(1),
				output_channels = error.getDim(1),
				height = error.getDim(2),
				width = error.getDim(3);

	// assert(x < width && y < height && z < gradient_weights.getDim(0) * gradient_weights.getDim(1)); // check if we are in bounds

	// get the channels we are working on
	const int c_in = z % input_channels;
	const int c_out = z / input_channels;

	// local id for shared memory
	const int tid = thread_y * blockDim.x + thread_x;
	// const int threads_in_block = blockDim.x * blockDim.y;
	
	assert(c_in < input_channels && c_out < output_channels);
   

	

	extern __shared__ float sm[];
	float val[filter_size][filter_size];

	// init val
	for(int j=0;j<filter_size;j++){
		for(int i=0;i<filter_size;i++){
			val[j][i]=0;
		}
	}
	for (int b = 0; b < batch_size; b++) {
		#pragma unroll
		for (int j = 0; j < filter_size; j++) {
			for (int i =0; i < filter_size; i++) {
				val[j][i] += inputPadded(b, c_in, y + j, x + i) * error(b, c_out, y, x);	
			}
		}
	}
	
	sm[tid] = 0;
	__syncthreads();
	for (int j = 0; j < filter_size; j++) {
		for (int i =0; i < filter_size; i++) {
			
			double vval = val[j][i];
			for (int offset = 16; offset > 0; offset /= 2)
				vval += __shfl_down_sync(0xffffffff, vval, offset);
			if (tid % 32 == 0) {
				sm[tid/32] = vval;
			}
			__syncthreads();

			// int n = threads_in_block;
			if (tid < 32) { // only works if threads_in_block/32 <= 32 or threads_in_block <= 1024
				double val = sm[tid];
				for (int offset = 16; offset > 0; offset /= 2)
					val += __shfl_down_sync(0xffffffff, val, offset);
				
				if (tid == 0)
					atomicAdd(&gradient_weights(c_out, c_in, j, i), val);
			}
		}
	}
}
template __global__ void  w_shfl_registers_reordered_padded<3>(Tensor<float, 4> inputPadded, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);