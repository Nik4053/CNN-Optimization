#include "cf_implementations.cuh"


template<int filter_size, int batch_size, int yBatch>
__global__
void cf_padded_reorder_work(Tensor<float, 4> inputPadded, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y * yBatch+ threadIdx.y * yBatch;
	int c_out = blockIdx.z * blockDim.z;

	int input_channels = inputPadded.getDim(1),
		output_channels = output.getDim(1),
		width = output.getDim(3),
		height = output.getDim(2);

	if (x >= width || y >= height || c_out >= output_channels)
		return;

	float reg_outputs[batch_size][yBatch];
	float reg_weights[filter_size][filter_size];
	float channel_bias = bias(c_out);
	for (int b = 0; b < batch_size; b++) {
		for ( int yy = 0; yy < yBatch; yy++){
			reg_outputs[b][yy] = channel_bias;
		}
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
			for ( int yy = 0; yy < yBatch; yy++){
				Debug(if (y + yy >= height) continue;) // check if we are in bounds
				for (int j = 0; j < filter_size; j++) {
					for (int i = 0; i < filter_size; i++) {
						float input_val = inputPadded(b,c_in,y+yy+j,x+i);
						reg_outputs[b][yy] += input_val * reg_weights[j][i];
					}
				}
			}
		}
	}
	
	#pragma unroll        
	for (int b = 0; b < batch_size; b++) {
		for ( int yy = 0; yy < yBatch; yy++){
			Debug(if (y + yy >= height) continue;) // check if we are in bounds
			output(b, c_out, y+yy, x) = fmaxf(0.,reg_outputs[b][yy]);
		}
	}
}
template __global__ void cf_padded_reorder_work<3,1,1>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
template __global__ void cf_padded_reorder_work<3,1,8>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
template __global__ void cf_padded_reorder_work<3,10,1>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
template __global__ void cf_padded_reorder_work<3,10,8>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
// template __global__ void cf_padded_reorder_work<3,100,1>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);
// template __global__ void cf_padded_reorder_work<3,100,8>(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float, 1> bias);