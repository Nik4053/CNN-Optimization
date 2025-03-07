#include "cf_implementations.cuh"


template<int filter_size, int batch_size>
__global__
void cf_padded_reorder(Tensor<float, 4> inputPadded, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c_out = blockIdx.z * blockDim.z;

	int input_channels = inputPadded.getDim(1),
		output_channels = output.getDim(1),
		width = output.getDim(3),
		height = output.getDim(2);

	if (x >= width || y >= height || c_out >= output_channels)
		return;

	float reg_outputs[batch_size];
	float reg_weights[filter_size][filter_size];
	float channel_bias = bias(c_out);
	for (int b = 0; b < batch_size; b++) {
		reg_outputs[b] = channel_bias;
	}

	for (int c_in = 0; c_in < input_channels; c_in++) {
		#pragma unroll
		for (int j = 0; j < filter_size; j++) {
			for (int i = 0; i < filter_size; i++) {
				reg_weights[j][i] = weights(c_out, c_in, j, i);
			}
		}
		#pragma unroll
		for (int b = 0; b < batch_size; b++) {
			for (int j = 0; j < filter_size; j++) {
				for (int i = 0; i < filter_size; i++) {
					float input_val = inputPadded(b,c_in,y+j,x+i);
					reg_outputs[b] += input_val * reg_weights[j][i];
				}
			}
		}
	}
	
	#pragma unroll        
	for (int b = 0; b < batch_size; b++) {
		output(b, c_out, y, x) = fmaxf(0.,reg_outputs[b]);
	}
}

template __global__ void cf_padded_reorder<3,1>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
template __global__ void cf_padded_reorder<3,10>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
template __global__ void cf_padded_reorder<3,100>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);