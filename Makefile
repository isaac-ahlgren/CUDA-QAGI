
NVCC=nvcc
CUDAFLAGS= -gencode arch=compute_60,code=sm_60 -rdc=true
OPT= -g -G

all:
	${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -o qagi.o qagi.cu
clean:
	rm -rf qagi.o
