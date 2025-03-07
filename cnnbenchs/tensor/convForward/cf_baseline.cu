#include "cf_implementations.cuh"


template<int filter_size>
__global__
void cf_baseline(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c_out = blockIdx.z * blockDim.z;

	int batch_size = input.getDim(0),
		input_channels = input.getDim(1),
		output_channels = output.getDim(1),
		width = output.getDim(3),
		height = output.getDim(2);

	if (x >= width || y >= height || c_out >= output_channels)
		return;

	float channel_bias = bias(c_out);
	for (int b = 0; b < batch_size; b++) {
		float val = channel_bias;

		for (int c_in = 0; c_in < input_channels; c_in++) {
			for (int j = (-filter_size+1)/2; j <= (filter_size)/2; j++) {
				for (int i = (-filter_size+1)/2; i <= (filter_size)/2; i++) {

					//Zero Padding
					//float input_val = access_padded<filter_size>(input, b, c_in, x + i, y + j, width, height);
					float input_val = (x+i>=0&&x+i<width&&y+j>=0&&y+j<height)?input(b,c_in,y+j,x+i):0;
					val += input_val * weights(c_out, c_in, j + (filter_size-1) / 2, i+(filter_size-1)/2);

				}
			}
		}
        
		output(b, c_out, y, x) = fmaxf(0.,val);
		
	}
}

template __global__ void cf_baseline<3>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);