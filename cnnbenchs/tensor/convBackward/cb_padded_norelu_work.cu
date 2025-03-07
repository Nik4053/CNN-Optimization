#include "cb_implementations.cuh"

inline __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

template<int filter_size, int yBatch>
__global__
void cb_padded_norelu_work(Tensor<float, 4> errorPadded, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y * yBatch+ threadIdx.y * yBatch;
	int c_in = blockIdx.z * blockDim.z;

	int batch_size = errorPadded.getDim(0),
		input_channels = next_error.getDim(1),
		output_channels = errorPadded.getDim(1),
		height = next_error.getDim(2),
		width = next_error.getDim(3);
	// assert(errorPadded.getDim(2)==width);
	if (x >= width || y >= height || c_in >= input_channels)
		return;

	float reg_next_errors[yBatch];
	for (int b = 0; b < batch_size; b++) {
		// float val = 0.;
		#pragma unroll
		for ( int yy = 0; yy < yBatch; yy++){
			reg_next_errors[yy] = 0.;
		}
		for (int c_out = 0; c_out < output_channels; c_out++) {
			#pragma unroll
			for ( int yy = 0; yy < yBatch; yy++){
				Debug(if (y + yy >= height) continue;) // check if we are in bounds
				for (int j = 0; j < filter_size; j++) {
					for (int i = 0; i < filter_size; i++) {
						float error_val = errorPadded(b, c_out, y + yy + j, x + i);
						reg_next_errors[yy] += error_val * weights(c_out, c_in, filter_size - j - 1, filter_size - i -1);
					}
				}
			}
		}
		#pragma unroll
		for ( int yy = 0; yy < yBatch; yy++){
			Debug(if (y + yy >= height) continue;) // check if we are in bounds
			next_error(b, c_in, y + yy, x) = reg_next_errors[yy];
		}
		// next_error(b, c_in, y, x) = val;
	}

}

template __global__ void cb_padded_norelu_work<3,1>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);
template __global__ void cb_padded_norelu_work<3,4>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);
template __global__ void cb_padded_norelu_work<3,8>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);
template __global__ void cb_padded_norelu_work<3,16>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);