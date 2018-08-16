NVCC = nvcc
NVFLAGS = -O5 -keep -Xcompiler -fopenmp -Xcompiler -march=broadwell \
	-Xcompiler -Wall -Xcompiler -Wextra \
	-Xptxas -warn-spills -Xptxas -Werror -std=c++14 -D_FORCE_INLINES \
	--default-stream per-thread -D_MWAITXINTRIN_H_INCLUDED \
	-gencode arch=compute_52,code=sm_52 

CUDA_VERSION=$(shell nvcc --version|grep 'release'|sed 's/.*release \([0-9]\.[0-9]\),.*/\1/')
GCC_VERSION=$(shell g++ --version|grep GCC|sed 's/^g++ (GCC) \([0-9]\+\.[0-9]\+\.[0-9]\+\).*/\1/')

ifneq ($(CUDA_VERSION),9.0)
$(error Cuda version 9.0 required)
else
NVFLAGS += -gencode arch=compute_70,code=sm_70
endif

ifneq ($(GCC_VERSION),5.3.0)
$(error GCC 5.3.0 required)
endif

all: matmul

matmul: matmul.cu
	$(NVCC) $(NVFLAGS) -o matmul matmul.cu -lcuda
