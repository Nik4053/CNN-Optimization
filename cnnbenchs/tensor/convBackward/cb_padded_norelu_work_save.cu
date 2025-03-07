#include "cb_implementations.cuh"

inline __device__ unsigned get_lane_id() {
	unsigned ret;
	asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
	return ret;
}

template<int filter_size, int yBatch>
__global__
void cb_padded_norelu_work_save(Tensor<float, 4> errorPadded, Tensor<float, 4> next_error, Tensor<float,4> outputPadded,Tensor<float, 4> weights) {
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
	
	float weights_reg[filter_size][filter_size];
	float error_reg[filter_size][filter_size];
	
	int padding_size=(filter_size/2);
	for (int b = 0; b < batch_size; b++) {
		float val[yBatch];
		for(int y_num=0;y_num<yBatch;y_num++){
			val[y_num]=0.;
		}

		for (int c_out = 0; c_out < output_channels; c_out++) {
			// init weights_reg and error_reg
			for(int j=(-filter_size+1)/2;j<=filter_size/2;j++){
				for(int i=(-filter_size+1)/2;i<=filter_size/2;i++){
					weights_reg[filter_size/2+j][filter_size/2+i]=weights(c_out, c_in, filter_size/2-j, filter_size/2-i);
					error_reg[filter_size/2+j][filter_size/2+i]=errorPadded(b, c_out, y+j+1, x+i+1);

				}
			}
			for (int y_num =0; y_num < yBatch ; y_num++){
				Debug(if (y + y_num >= height) continue;) // check if we are in bounds

				for (int j = 0; j < filter_size; j++) {
					for (int i = 0; i < filter_size ; i++) {

						val[y_num] += error_reg[j][i] * weights_reg[j][i];

					}
				}
				for(int j=0;j<filter_size-1;j++){
					for(int i=0;i<filter_size;i++){
						error_reg[j][i]=error_reg[j+1][i];
					}
				}

				for (int i = 0; i < filter_size; i++) {
					error_reg[filter_size-1][i]= errorPadded(b, c_out, y + y_num + filter_size, x + i);
				}
			}
		}
		for(int yy=0;yy<yBatch;yy++){
			Debug(if (y + yy >= height) continue;) // check if we are in bounds
			next_error(b, c_in, y+yy, x) = val[yy];
		}
	}

}

template __global__ void cb_padded_norelu_work_save<3,1>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);
template __global__ void cb_padded_norelu_work_save<3,4>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);
template __global__ void cb_padded_norelu_work_save<3,8>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);
template __global__ void cb_padded_norelu_work_save<3,16>(Tensor<float, 4> error, Tensor<float, 4> next_error, Tensor<float,4> output,Tensor<float, 4> weights);