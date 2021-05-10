OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")

CUDA_PATH       ?= /usr/local/cuda-10.1
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
endif

NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# Make sure to choose the right SM for your device
GENCODE_SM75    := -gencode arch=compute_75,code=sm_75
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM75)

LDFLAGS   := -lcudart -lrt -lcurand -lm
CFLAGS := -O3 -lineinfo -Xptxas="-dlcm=ca -v"
CUB_DIR = ./external/cub/
SOURCE_DIR = ./src/
TEST_DIR = ./test/
INCLUDES := -I$(CUB_DIR) -I$(SOURCE_DIR) -I$(TEST_DIR)

#obj/%.o: src/%.cu $(DEPS)
#	$(NVCC) $(CFLAGS) -I. $(INCLUDES) -g $(GENCODE_FLAGS) $< -o $@

compareTopKAlgorithms: test/compareTopKAlgorithms.cu src/bitonicTopK.cuh src/radixSelectTopK.cuh src/sortTopK.cuh \
                       src/bitonicUsing.cuh src/thresholdTopK.cuh src/impreciseBitonicTopK.cuh \
                       test/sort.cuh test/generateProblems.cuh
	$(NVCC) $(CFLAGS) $(INCLUDES) test/compareTopKAlgorithms.cu $(LDFLAGS) -o compareTopKAlgorithms

clean:
	rm -rf *.o compareTopKAlgorithms
