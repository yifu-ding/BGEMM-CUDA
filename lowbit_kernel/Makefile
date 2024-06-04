# host compiler
HOST_COMPILER ?= g++
NVCC          := nvcc -ccbin $(HOST_COMPILER) -O0 -g -G

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     := -fPIC
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

################################################################################
ALL_CCFLAGS += -DNO_PYTORCH
ALL_CCFLAGS += --std=c++17
ALL_CCFLAGS += -maxrregcount=255
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/include/
# INCLUDES  := -I/usr/local/cuda/include/ 
LIBRARIES := -lcublas 

# Gencode arguments
SMS ?= 80
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))


HEAD_FILES = csrc/fp6_linear.cuh \
			 csrc/include/configs.h \
			 csrc/include/kernel_matmul.cuh \
			 csrc/include/kernel_reduction.cuh \
			 csrc/include/ptx_cp.async.cuh \
			 csrc/include/ptx_mma.cuh \
			 csrc/include/utils_core.cuh \
			 csrc/include/utils_gmem.cuh \
			 csrc/include/utils_parallel_dequant.cuh \
			 csrc/utils/weight_prepacking.h \
			 csrc/utils/weight_quant.h \
			 csrc/utils/weight_dequant.h \
			 csrc/utils/helper.h \
			 csrc/kernel_test.h \
			 csrc/fp6_linear.cu \
			 
			 
# csrc/bgemm/helper_cuda.h \
# 			 csrc/bgemm/helper_functions.h \
# Target rules
all: bgemm.so

libfp6.so: fp6.o
	$(EXEC) $(NVCC) --shared $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

fp6.o: csrc/fp6_llm.cu $(HEAD_FILES)
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

fp6: fp6.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) 

bgemm.so: bgemm.o
	$(EXEC) $(NVCC) --shared $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

bgemm.o: csrc/benchmark.cu $(HEAD_FILES)
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bgemm: bgemm.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) 


clean:
	rm -f libfp6.so fp6.o main bgemm fp6 bgemm.so bgemm.o