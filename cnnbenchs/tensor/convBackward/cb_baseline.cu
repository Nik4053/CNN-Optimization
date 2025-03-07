#include "cb_implementations.cuh"


template<int filter_size>
__global__
void cb_baseline(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c_in = blockIdx.z * blockDim.z;

	int batch_size = error.getDim(0),
		input_channels = next_error.getDim(1),
		output_channels = error.getDim(1),
		height = next_error.getDim(2),
		width = next_error.getDim(3);
	assert(error.getDim(2)==width);
	if (x >= width || y >= height || c_in >= input_channels)
		return;

	for (int b = 0; b < batch_size; b++) {
		float val = 0.;

		for (int c_out = 0; c_out < output_channels; c_out++) {
			// if ( output(b, c_out, y, x) == 0. ) // CORRECT????
			// 	continue;
			for (int j = (-filter_size+1)/2; j <= (filter_size)/2; j++) {
				for (int i = (-filter_size+1)/2; i <= (filter_size)/2; i++) {
					float error_val =0;
					if(x + i >= 0 && x + i < width && y + j >= 0 && y + j < height) {
						error_val=(output(b, c_out, y + j, x + i) != 0.) ? error(b, c_out, y + j, x + i) : 0.;
					} 
					val += error_val * weights(c_out, c_in, filter_size/2 - j , filter_size/2 - i );

				}
			}
		}

		next_error(b, c_in, y, x) = val;
	}

	// for (int b = 0; b < batch_size; b++) {
	// 	float val = 0.;

	// 	for (int c_out = 0; c_out < output_channels; c_out++) {
	// 		// if ( output(b, c_out, y, x) == 0. ) // CORRECT????
	// 		// 	continue;
	// 		for (int j = (-filter_size+1)/2; j <= (filter_size)/2; j++) {
	// 			for (int i = (-filter_size+1)/2; i <= (filter_size)/2; i++) {
	// 				float error_val =0;
	// 				if(x + i >= 0 && x + i < width && y + j >= 0 && y + j < height) {
    //                     float buf[2];
    //                     buf[0] = 0.;
    //                     buf[1] = error(b, c_out, y + j, x + i);
	// 					error_val=buf[output(b, c_out, y + j, x + i) != 0.];
	// 				} 
	// 				val += error_val * weights(c_out, c_in, filter_size/2 - j , filter_size/2 - i );

	// 			}
	// 		}
	// 	}

	// 	next_error(b, c_in, y, x) = val;
	// }
}
template __global__ void cb_baseline<3>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);