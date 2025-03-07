#include "cf_implementations.cuh"


template<int filter_size, int yBatch>
__global__
void cf_padded_work_save(Tensor<float, 4> inputPadded, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y * yBatch + threadIdx.y * yBatch;
	int c_out = blockIdx.z * blockDim.z;

	int batch_size = inputPadded.getDim(0),
		input_channels = inputPadded.getDim(1),
		output_channels = output.getDim(1),
		width = output.getDim(3),
		height = output.getDim(2);

	if (x >= width || y >= height || c_out >= output_channels)
		return;

	
	float weights_reg[filter_size][filter_size];
	float input_reg[filter_size][filter_size];

	float channel_bias = bias(c_out);
	for (int b = 0; b < batch_size; b++) {
		// float val = channel_bias;

		float reg_outputs[yBatch];

		#pragma unroll
		for ( int yy = 0; yy < yBatch; yy++){
			reg_outputs[yy] = channel_bias;
		}

		for (int c_in = 0; c_in < input_channels; c_in++) {
			// init weights_reg and error_reg
			for(int j=0;j<filter_size;j++){
				for(int i=0;i<filter_size;i++){
					weights_reg[j][i]=weights(c_out, c_in, j, i);
					input_reg[j][i]=inputPadded(b, c_in,y+j, x+i);
				}
			}
			for ( int yy = 0; yy < yBatch; yy++){
				Debug(if (y + yy >= height) continue;) // check if we are in bounds
				for (int j = 0; j < filter_size; j++) {
					for (int i = 0; i < filter_size ; i++) {
						reg_outputs[yy] += input_reg[j][i] * weights_reg[j][i];
					}
				}
				for(int j=0;j<filter_size-1;j++){
					for(int i=0;i<filter_size;i++){
						input_reg[j][i]=input_reg[j+1][i];
					}
				}

				for (int i = 0; i < filter_size; i++) {
					input_reg[filter_size-1][i]= inputPadded(b, c_in, y + yy + filter_size, x + i);
				}
			}
		}
        #pragma unroll
		for ( int yy = 0; yy < yBatch; yy++){
			Debug(if (y + yy >= height) continue;) // check if we are in bounds
			output(b, c_out, y + yy, x) = fmaxf(0.,reg_outputs[yy]);
		}
		
	}
}

template __global__ void cf_padded_work_save<3,1>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
template __global__ void cf_padded_work_save<3,2>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
template __global__ void cf_padded_work_save<3,4>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
template __global__ void cf_padded_work_save<3,8>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
template __global__ void cf_padded_work_save<3,16>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);