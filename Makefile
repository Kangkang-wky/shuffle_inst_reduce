default:
	nvcc shuffle_inst_reduce_dim9.cu -O3 -std=c++11 -I/usr/local/cuda-11.7/include -L/usr/local/cuda-11.7/lib64 -lcudnn -lcublas -arch=sm_70 --ptxas-options=-v -o main