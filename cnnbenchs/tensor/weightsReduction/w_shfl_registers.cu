#include "implementations.cuh"




template<int filter_size>
__global__
void w_shfl_registers(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights) {
	const short block_y = blockIdx.y;
	const short block_x = blockIdx.x;
	const short thread_y = threadIdx.y;
	const short thread_x = threadIdx.x;

	// get our position in the grid. #blocks*blocksize + position in block
	const short x = block_x * blockDim.x  + thread_x;
	const short y = block_y * blockDim.y  + thread_y;
	const short z = blockIdx.z * blockDim.z;


	// asserts for debug
	// assert(input.getDim(3) == error.getDim(3) && input.getDim(2) == error.getDim(2)); // img size has to be identical 
	// assert(input.getDim(0) == error.getDim(0)); // batch size has to be identical
	// assert(input.getDim(1) == gradient_weights.getDim(1)); // input channels have to be identical
	// assert(error.getDim(1) == gradient_weights.getDim(0)); // output channels have to be identical
	// assert(gradient_weights.getDim(2) == filter_size && gradient_weights.getDim(3) == filter_size); // filter size has to be identical



	const short batch_size = input.getDim(0),
				input_channels = input.getDim(1),
				output_channels = error.getDim(1),
				height = error.getDim(2),
				width = error.getDim(3);

	// assert(x < width && y < height && z < gradient_weights.getDim(0) * gradient_weights.getDim(1)); // check if we are in bounds

	// get the channels we are working on
	const short c_in = z % input_channels;
	const short c_out = z / input_channels;

	// local id for shared memory
	const short tid = thread_y * blockDim.x + thread_x;
	const short threads_in_block = blockDim.x * blockDim.y;
	
	assert(c_in < input_channels && c_out < output_channels);
   

	

	extern __shared__ float sm[];
	float val[filter_size][filter_size];

	// init val
	for(int j=0;j<filter_size;j++){
		for(int i=0;i<filter_size;i++){
			val[j][i]=0;
		}
	}

	for (short j = (-filter_size+1)/2, jj =0; j <= filter_size/2; j++, jj++) {
		for (short i = (-filter_size+1)/2,ii =0; i <= filter_size/2; i++, ii++) {
			for (short b = 0; b < batch_size; b++) {
				
				
				float input_val =0;
				if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height) {
					
					input_val=input(b, c_in, y + j, x + i); // for each image go through the same filter for one single channel
				} 
				
				val[jj][ii] += input_val * error(b, c_out, y, x);
				
			}

		}
	}

	for (short j = 0; j < filter_size; j++) {
		for (short i =0; i < filter_size; i++) {
			
			sm[tid] = val[j][i];
			__syncthreads();

			int n = threads_in_block;
			while (n > 32) {
				n = (n + 1) / 2;
				if (tid < n){
					sm[tid] += sm[tid + n];
					sm[tid + n] = 0;
				}

				__syncthreads();
			}
			if (tid < 32) {
				double val = sm[tid];
				for (int offset = 16; offset > 0; offset /= 2)
					val += __shfl_down_sync(0xffffffff, val, offset);
				
				if (tid == 0)
					atomicAdd(&gradient_weights(c_out, c_in, j, i), val);
			}
		}
	}
}
template __global__ void  w_shfl_registers<3>(Tensor<float, 4> input, Tensor<float, 4> error, Tensor<float, 4> gradient_weights);