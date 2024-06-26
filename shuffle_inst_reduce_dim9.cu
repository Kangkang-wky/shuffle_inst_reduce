#include "util.cuh"

#define DIM 9

template <int size>
__device__ __forceinline__ float warpReduceSum(float partial_sum) {
  if (size >= 32)
    partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, 16);
  if (size >= 16)
    partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, 8);
  if (size >= 8)
    partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, 4);
  if (size >= 4)
    partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, 2);
  if (size >= 2)
    partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, 1);
  return partial_sum;
}

__device__ __forceinline__ void swap(float &a, float &b) {
  float tmp = a;
  a = b;
  b = tmp;
}

// shuffle 多维数组规约, 有效利用 shuffle 指令带宽, 这样的做法可以让
// 32 维数组规约用满 shuffle 指令的带宽.
__device__ __forceinline__ void reduce_shfl_one(int lane_id, float &reg1,
                                                float &reg2) {

  if ((lane_id >> 4) & 0x00000001) {
    swap(reg1, reg2);
  }
  reg1 += __shfl_xor_sync(0xffffffff, reg2, 16, 32);
}

__device__ __forceinline__ void reduce_shfl_two(int lane_id, float &reg1,
                                                float &reg2) {

  if ((lane_id >> 3) & 0x00000001) {
    swap(reg1, reg2);
  }
  reg1 += __shfl_xor_sync(0xffffffff, reg2, 8, 16);
}

__device__ __forceinline__ void reduce_shfl_three(int lane_id, float &reg1,
                                                  float &reg2) {

  if ((lane_id >> 2) & 0x00000001) {
    swap(reg1, reg2);
  }
  reg1 += __shfl_xor_sync(0xffffffff, reg2, 4, 8);
}

__device__ __forceinline__ void reduce_shfl_four(int lane_id, float &reg1,
                                                 float &reg2) {

  if ((lane_id >> 1) & 0x00000001) {
    swap(reg1, reg2);
  }
  reg1 += __shfl_xor_sync(0xffffffff, reg2, 2, 4);
}
__device__ __forceinline__ void reduce_shfl_five(int lane_id, float &reg1,
                                                 float &reg2) {

  if (lane_id & 0x00000001) {
    swap(reg1, reg2);
  }
  reg1 += __shfl_xor_sync(0xffffffff, reg2, 1, 2);
}

__global__ void test_shuffle_inst_baseimporve(float *input, float *output) {

  int thread_id = threadIdx.x;
  int lane_id = threadIdx.x & (32 - 1); // % 32  &(32 - 1)
  float reg_tmp[DIM];

  for (int i = 0; i < DIM; i++) {
    reg_tmp[i] = input[i * 32 + lane_id];
  }

  // from 1 to 4
  reduce_shfl_one(lane_id, reg_tmp[0], reg_tmp[4]);
  reduce_shfl_one(lane_id, reg_tmp[1], reg_tmp[5]);
  reduce_shfl_one(lane_id, reg_tmp[2], reg_tmp[6]);
  reduce_shfl_one(lane_id, reg_tmp[3], reg_tmp[7]);

  // from 1 to 8
  reduce_shfl_two(lane_id, reg_tmp[0], reg_tmp[2]);
  reduce_shfl_two(lane_id, reg_tmp[1], reg_tmp[3]);

  // from 1 to 4
  reduce_shfl_three(lane_id, reg_tmp[0], reg_tmp[1]);

  // from 1 to 2
  reg_tmp[0] += __shfl_xor_sync(0xffffffff, reg_tmp[0], 2, 4);

  // from 1 to 1
  reg_tmp[0] += __shfl_xor_sync(0xffffffff, reg_tmp[0], 1, 2);

  reg_tmp[8] = warpReduceSum<WARP_SIZE>(reg_tmp[8]);

  if ((lane_id & 3) == 0) {
    atomicAdd(&output[lane_id >> 2], reg_tmp[0]);
  }
  if (lane_id == 0) {
    atomicAdd(&output[8], reg_tmp[8]);
  }
}

// baseline 构建完成  // baseline 160 条指令数量
__global__ void test_shuffle_inst_baseline(float *input, float *output) {

  int thread_id = threadIdx.x;
  int lane_id = threadIdx.x & 0x0000001f; // % 32  &(32 - 1)
  float reg_tmp[DIM];

  for (int i = 0; i < DIM; i++) {
    reg_tmp[i] = input[i * 32 + lane_id];
  }

  for (int i = 0; i < DIM; i++) {
    reg_tmp[i] = warpReduceSum<WARP_SIZE>(reg_tmp[i]);
  }
  if (lane_id == 0) {
    for (int i = 0; i < DIM; i++) {
      atomicAdd(&output[i], reg_tmp[i]);
    }
  }
}

int main() {

  float *host_input = (float *)malloc(DIM * 32 * sizeof(float));
  float *host_output = (float *)malloc(DIM * 1 * sizeof(float)); // cpu
  float *host_output_baseline =
      (float *)malloc(DIM * 1 * sizeof(float)); // baseline
  float *host_output_improve =
      (float *)malloc(DIM * 1 * sizeof(float)); // improve

  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < 32; j++) {
      host_input[i * 32 + j] = j + i;
    }
  }

  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < 32; j++) {
      host_output[i] += host_input[i * 32 + j];
    }
  }
  // 以上 cpu 运行

  float *dev_input;
  float *dev_output;
  float *dev_output_improve;
  CHECK_CUDA(cudaMalloc(&dev_input, sizeof(float) * DIM * 32));
  CHECK_CUDA(cudaMemcpy(dev_input, host_input, DIM * 32 * sizeof(float),
                        cudaMemcpyHostToDevice));

  /****************************************************************************************/
  CHECK_CUDA(cudaMalloc(&dev_output, sizeof(float) * DIM * 1));
  CHECK_CUDA(cudaMalloc(&dev_output_improve, sizeof(float) * DIM * 1));

  /***************************************************************************************/
  CHECK_CUDA(cudaMemset(dev_output, 0, DIM * 1 * sizeof(float)));
  CHECK_CUDA(cudaMemset(dev_output_improve, 0, DIM * 1 * sizeof(float)));

  dim3 grid_size(1);
  dim3 block_size(32);

  test_shuffle_inst_baseline<<<grid_size, block_size>>>(dev_input, dev_output);

  CHECK_CUDA(cudaMemcpy(host_output_baseline, dev_output,
                        DIM * 1 * sizeof(float), cudaMemcpyDeviceToHost));

  test_shuffle_inst_baseimporve<<<grid_size, block_size>>>(dev_input,
                                                           dev_output_improve);

  CHECK_CUDA(cudaMemcpy(host_output_improve, dev_output_improve,
                        DIM * 1 * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < DIM; i++) {
    printf("float1: %6f, float2: %6f, float3: %6f\n", host_output[i],
           host_output_baseline[i], host_output_improve[i]);
  }

  for (int i = 0; i < 10; i++) {
    test_shuffle_inst_baseline<<<grid_size, block_size>>>(dev_input,
                                                          dev_output);
    test_shuffle_inst_baseimporve<<<grid_size, block_size>>>(
        dev_input, dev_output_improve);
  }

  return 0;
}
