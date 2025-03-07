#include "cf_implementations.cuh"


template<int filter_size, int yBatch, int batch_size, int input_channels, int output_channels, int height, int width>
__global__
void cf_padded_work(tel::Tensor<float, batch_size,input_channels,height + filter_size/2+1,width+ filter_size/2+1> inputPadded, tel::Tensor<float, batch_size,output_channels,height,width> output, tel::Tensor<float, output_channels, input_channels, filter_size,filter_size> weights, tel::Tensor<float, output_channels> bias) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y * yBatch + threadIdx.y * yBatch;
	int c_out = blockIdx.z * blockDim.z;


	if (x >= width || y >= height || c_out >= output_channels)
		return;

	float reg_outputs[yBatch];
	float channel_bias = bias(c_out);
	for (int b = 0; b < batch_size; b++) {
		// float val = channel_bias;

		#pragma unroll
		for ( int yy = 0; yy < yBatch; yy++){
			reg_outputs[yy] = channel_bias;
		}
		for (int c_in = 0; c_in < input_channels; c_in++) {
			
			for ( int yy = 0; yy < yBatch; yy++){
				// Debug(if (y + yy >= height) continue;) // check if we are in bounds
				for (int j = 0; j < filter_size; j++) {
					for (int i = 0; i < filter_size; i++) {
						//Zero Padding
						//float input_val = access_padded<filter_size>(input, b, c_in, x + i, y + j, width, height);
						float input_val = inputPadded(b,c_in,y+yy+j,x+i);
						reg_outputs[yy] += input_val * weights(c_out, c_in, j, i);
					}
				}
			}
		}
        #pragma unroll
		for ( int yy = 0; yy < yBatch; yy++){
			// Debug(if (y + yy >= height) continue;) // check if we are in bounds
			output(b, c_out, y + yy, x) = fmaxf(0.,reg_outputs[yy]);
		}
		
	}
}

