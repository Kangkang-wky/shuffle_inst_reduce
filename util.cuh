#ifndef _COMMON_H_
#define _COMMON_H_

#include "cuda_runtime.h"
#include "cudnn.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <time.h>
#include <utility>
#include <vector>

const float eps = 1e-4;

#define WARP_SIZE 32

// check error
// cuda runtime error
#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t cuda_error = (func);                                           \
    if (cuda_error != cudaSuccess)                                             \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__,                           \
             cudaGetErrorString(cuda_error));                                  \
  }
// cudnn runtime error
#define CHECK_CUDNN(func)                                                      \
  {                                                                            \
    cudnnStatus_t cudnn_status = (func);                                       \
    if (cudnn_status != CUDNN_STATUS_SUCCESS)                                  \
      printf("%s %d CUDNN: %s\n", __FILE__, __LINE__,                          \
             cudnnGetErrorString(cudnn_status));                               \
  }

void init(float *input, size_t input_size);

// 做正确性检查
bool check_error(float *data1, float *data2, size_t data_size);

#endif // _COMMON_H_