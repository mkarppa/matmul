NVCC = nvcc
NVFLAGS = -O5 -keep -Xcompiler -fopenmp -Xcompiler -march=broadwell \
	-Xcompiler -Wall -Xcompiler -Wextra \
	-Xptxas -warn-spills -Xptxas -Werror -std=c++14 -D_FORCE_INLINES \
	--default-stream per-thread -D_MWAITXINTRIN_H_INCLUDED \
	-gencode arch=compute_37,code=sm_37 \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_70,code=sm_70

all: matmul

matmul: matmul.cu
	$(NVCC) $(NVFLAGS) -o matmul matmul.cu -lcuda
