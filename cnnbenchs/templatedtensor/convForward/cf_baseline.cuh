#include "cf_implementations.cuh"


template<int filter_size, int batch_size, int input_channels, int output_channels, int height, int width>
__global__
void cf_baseline(tel::Tensor<float, batch_size,input_channels,height,width> input, tel::Tensor<float, batch_size,output_channels,height,width> output, tel::Tensor<float, output_channels, input_channels, filter_size,filter_size> weights, tel::Tensor<float, output_channels> bias) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c_out = blockIdx.z * blockDim.z;

	if (x >= width || y >= height || c_out >= output_channels)
		return;

	float channel_bias = bias(c_out);
	for (int b = 0; b < batch_size; b++) {
		float val = channel_bias;

		for (int c_in = 0; c_in < input_channels; c_in++) {
			for (int j = (-filter_size+1)/2; j <= (filter_size)/2; j++) {
				for (int i = (-filter_size+1)/2; i <= (filter_size)/2; i++) {
					//Zero Padding
					float input_val = (x+i>=0&&x+i<width&&y+j>=0&&y+j<height)?input(b,c_in,y+j,x+i):0;
					val += input_val * weights(c_out, c_in, j + (filter_size-1) / 2, i+(filter_size-1)/2);

				}
			}
		}
        
		output(b, c_out, y, x) = fmaxf(0.,val);
		
	}
}
