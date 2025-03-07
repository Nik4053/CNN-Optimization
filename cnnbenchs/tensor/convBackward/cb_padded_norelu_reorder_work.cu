#include "cb_implementations.cuh"

inline __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

template<int filter_size, int yBatch>
__global__
void cb_padded_norelu_reorder_work(Tensor<float, 4> errorPadded, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y * yBatch + threadIdx.y * yBatch;
	int c_in = blockIdx.z * blockDim.z;

	int batch_size = errorPadded.getDim(0), 
		input_channels = next_error.getDim(1),
		output_channels = errorPadded.getDim(1),
		height = next_error.getDim(2),
		width = next_error.getDim(3);
	// assert(batch_size = errorPadded.getDim(0));
	// assert(errorPadded.getDim(2)==width);
	if (x >= width || y >= height || c_in >= input_channels)
		return;

	float reg_weights[filter_size][filter_size];
	// float reg_vals[batch_size];

	// for (int b = 0; b < batch_size; b++) {
	// 	reg_vals[b] = 0.;
	// }

	for (int c_out = 0; c_out < output_channels; c_out++) {
		#pragma unroll
		for (int j = 0; j < filter_size; j++) {
			for (int i = 0; i < filter_size; i++) {
				reg_weights[j][i] = weights(c_out, c_in, filter_size - j - 1, filter_size - i - 1);
			}
		}
		// // Alternative weights loading implementation
		// {
		// 	int lane_id = get_lane_id() ;
		// 	int local_weight_pos = weights.pos(c_out, c_in, 0, 0);
		// 	float local_weight = 0;
		// 	int filter_size_2 = filter_size*filter_size;
		// 	if(lane_id < filter_size_2)
		// 		local_weight = weights(local_weight_pos + lane_id);
		// 	#define FULL_MASK 0xffffffff
		// 	// unsigned mask = __ballot_sync(FULL_MASK, warp_id < filter_size_2);
		// 	// for (int offset = filter_size_2/2; offset > 0; offset /= 2)
		// 	// 	local_weight += __shfl_down_sync(mask, local_weight, offset);
		// 	int offset = 0;
		// 	#pragma unroll
		// 	for (int j = 0; j < filter_size; j++) {
		// 		for (int i = 0; i < filter_size; i++) {
		// 			reg_weights[filter_size - j - 1][filter_size - i - 1] =  __shfl_sync(FULL_MASK, local_weight, offset);
		// 			offset++;
		// 		}
		// 	}
		// }
		#pragma unroll
		for ( int yy = 0; yy < yBatch; yy++){
			Debug(if (y + yy >= height) continue;) // check if we are in bounds
			for (int b = 0; b < batch_size; b++) {
				float val = 0.;
				for (int j = 0; j < filter_size; j++) {
					for (int i = 0; i < filter_size; i++) {
						float error_val = errorPadded(b, c_out, y + yy + j, x + i);
						val += error_val * reg_weights[j][i];
					}
				}
				next_error(b, c_in, y+ yy, x) += val;
			}
		}
	}

	
}
template __global__ void cb_padded_norelu_reorder_work<3,1>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);
template __global__ void cb_padded_norelu_reorder_work<3,4>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);
template __global__ void cb_padded_norelu_reorder_work<3,8>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);
template __global__ void cb_padded_norelu_reorder_work<3,16>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);