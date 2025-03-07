#include "cb_implementations.cuh"


template<int filter_size>
__global__
void cb_padded_norelu(Tensor<float, 4> errorPadded, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c_in = blockIdx.z * blockDim.z;

	int batch_size = errorPadded.getDim(0),
		input_channels = next_error.getDim(1),
		output_channels = errorPadded.getDim(1),
		height = next_error.getDim(2),
		width = next_error.getDim(3);
		
	if (x >= width || y >= height || c_in >= input_channels)
		return;

	for (int b = 0; b < batch_size; b++) {
		float val = 0.;

		for (int c_out = 0; c_out < output_channels; c_out++) {
			#pragma unroll
			for (int j = 0; j < filter_size; j++) {
				for (int i = 0; i < filter_size; i++) {
					float error_val = errorPadded(b, c_out, y + j, x + i);
					val += error_val * weights(c_out, c_in, filter_size - j - 1, filter_size - i -1);

				}
			}
		}
		next_error(b, c_in, y, x) = val;
	}

}
template __global__ void cb_padded_norelu<3>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights);