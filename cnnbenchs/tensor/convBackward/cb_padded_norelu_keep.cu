#include "cb_implementations.cuh"


#define SM(y,x) (sm[(y)*lw+(x)])

template<int filter_size>
__global__
void cb_padded_norelu_keep(Tensor<float, 4> errorPadded, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c_in = blockIdx.z * blockDim.z;

	int batch_size = errorPadded.getDim(0),
		input_channels = next_error.getDim(1),
		output_channels = errorPadded.getDim(1),
		height = next_error.getDim(2),
		width = next_error.getDim(3);
	// assert(errorPadded.getDim(2)==width);
	if (x >= width || y >= height || c_in >= input_channels)
		return;

	// float reg_weights[filter_size][filter_size];

	int x_start = blockIdx.x * blockDim.x;
	int y_start = blockIdx.y * blockDim.y;

	extern __shared__ float sm[];
	// for (int j = 0; j < filter_size; j++) {
	// 	for (int i = 0; i < filter_size; i++) {
	// 		sm[(threadIdx.y +j ) * blockDim.x + threadIdx.x + i] = weights(c_out, c_in, filter_size - j - 1, filter_size - i - 1);
	// 	}
	// }
	const int filter_padding = filter_size / 2;
	int lx = threadIdx.x;
	int ly = threadIdx.y ;
	int lw = (blockDim.x + 2); // local width
	
	
	for (int b = 0; b < batch_size; b++) {
		float val = 0.;

		for (int c_out = 0; c_out < output_channels; c_out++) {

			// load all data into shared memory
			// SM(0,0) = errorPadded(b, c_out, y + ly, x + lx); //load center
			// SM(ly+ 1,lx+ 1) = errorPadded(b, c_out, y + 1, x + 1); //load center
			// if (x ==0 && y == 0) {
			// 	// SM(0,0) = 0;
			// 	// SM(1,0) = 0;
			// 	SM(2,0) = 0;
				
			// 	// SM(0,1) = 0;
			// 	// SM(0,2) = 0;
			// }
			// if (threadIdx.y == 0) {
			// 	SM(0, lx + 1) = errorPadded(b, c_out, y_start + 0, x + 1); // top
			// 	SM(blockDim.y + 1, lx + 1) = errorPadded(b, c_out, y_start + blockDim.y + 1, x + 1); // bot
			// 	if (threadIdx.x == 0) {
			// 		SM(0,0) = errorPadded(b, c_out, y_start, x_start); // top left
			// 		SM(0,blockDim.x + 1) = errorPadded(b, c_out, y_start, x_start + blockDim.x + 1); // top right
			// 		SM(blockDim.y + 1,0) = errorPadded(b, c_out, y_start + blockDim.y + 1, x_start); // bot left
			// 		SM(blockDim.y + 1,blockDim.x + 1) = errorPadded(b, c_out, y_start + blockDim.y + 1, x_start + blockDim.x + 1); // bot right
			// 	}	
			// }
			// if (threadIdx.x == 0) {
			// 	SM(ly + 1, 0) = errorPadded(b, c_out, y + 1, x_start); // left
			// 	SM(ly + 1, blockDim.x + 1) = errorPadded(b, c_out, y + 1, x_start + blockDim.x + 1); // right
			// }



			sm[ (threadIdx.y +1 )  * blockDim.x + threadIdx.x + 1 ] = errorPadded(b, c_out, y + 1, x + 1);
			// if (threadIdx.y == 0) {
			// 	sm[0 + threadIdx.x + 1] = errorPadded(b, c_out, 0, x + 1); // top
			// 	sm[(blockDim.y ) * blockDim.x + threadIdx.x + 1] = errorPadded(b, c_out, y + blockDim.y, x + 1); // bot
			// 	if (threadIdx.x == 0) {
			// 		sm[0] = errorPadded(b, c_out, 0, 0); // top left
			// 		sm[blockDim.x + 1] = errorPadded(b, c_out, 0, x + blockDim.x); // top right
			// 		sm[(blockDim.y  ) * blockDim.x ] = errorPadded(b, c_out, y + blockDim.y, 0); // bot left
			// 		sm[(blockDim.y  ) * blockDim.x + blockDim.x + 1] = errorPadded(b, c_out, y + blockDim.y, x + blockDim.x); // bot right
			// 	}
				
			// }
			__syncthreads();
			// for (int j = 0; j < filter_size; j++) {
			// 	for (int i = 0; i < filter_size; i++) {
			// 		reg_weights[j][i] = weights(c_out, c_in, filter_size - j - 1, filter_size - i - 1);
			// 	}
			// }
			#pragma unroll
			for (int j = 0; j < filter_size; j++) {
				for (int i = 0; i < filter_size; i++) {
					// float error_val = errorPadded(b, c_out, y + j, x + i);
					float error_val = sm[(threadIdx.y + j) * blockDim.x + threadIdx.x + i];
					// float error_val = SM(ly + j, lx + i);
					val += error_val * weights(c_out, c_in, filter_size - j - 1, filter_size - i - 1);

				}
			}
		}
		next_error(b, c_in, y, x) = val;
	}

}
template __global__ void cb_padded_norelu_keep<3>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights);