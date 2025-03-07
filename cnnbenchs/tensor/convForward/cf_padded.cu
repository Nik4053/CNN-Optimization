#include "cf_implementations.cuh"


template<int filter_size>
__global__
void cf_padded(Tensor<float, 4> inputPadded, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c_out = blockIdx.z * blockDim.z;

	int batch_size = inputPadded.getDim(0),
		input_channels = inputPadded.getDim(1),
		output_channels = output.getDim(1),
		width = output.getDim(3),
		height = output.getDim(2);

	if (x >= width || y >= height || c_out >= output_channels)
		return;

	float channel_bias = bias(c_out);
	for (int b = 0; b < batch_size; b++) {
		float val = channel_bias;

		for (int c_in = 0; c_in < input_channels; c_in++) {
			for (int j = 0; j < filter_size; j++) {
				for (int i = 0; i < filter_size; i++) {
					//Zero Padding
					//float input_val = access_padded<filter_size>(input, b, c_in, x + i, y + j, width, height);
					float input_val = inputPadded(b,c_in,y+j,x+i);
					val += input_val * weights(c_out, c_in, j, i);

				}
			}
		}
        
		output(b, c_out, y, x) = fmaxf(0.,val);
		
	}
}

template __global__ void cf_padded<3>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);