# host compiler
HOST_COMPILER ?= g++
NVCC          := nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     := -DNO_PYTORCH
LDFLAGS     := -rpath=../../lowbit_kernel

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/include/ 
LIBRARIES := -lcublas 

#
INCLUDES  += -I../../lowbit_kernel/csrc
# LIBRARIES += -L../../lowbit_kernel -lfp6

################################################################################
# Gencode arguments
SMS ?= 80
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
################################################################################


# HEAD_FILES = csrc/fp6_linear.cuh \
# 			 csrc/include/configs.h \
# 			 csrc/include/kernel_matmul.cuh \
# 			 csrc/include/kernel_reduction.cuh \
# 			 csrc/include/ptx_cp.async.cuh \
# 			 csrc/include/ptx_mma.cuh \
# 			 csrc/include/utils_core.cuh \
# 			 csrc/include/utils_gmem.cuh \
# 			 csrc/include/utils_parallel_dequant.cuh \
# 			 csrc/utils/weight_prepacking.h \
# 			 csrc/utils/weight_quant.h \
# 			 csrc/utils/weight_dequant.h \
# 			 csrc/utils/helper.h \
# 			 csrc/kernel_test.h \
# 			 csrc/fp6_linear.cu \
			 

# Target rules
all: bgemm.so

kernel_test.o:  kernel_test.cu kernel_test.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

kernel_test: kernel_test.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) 

bgemm.so: bgemm.o 
	$(EXEC) $(NVCC) --shared $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

bgemm.o: benchmark.cu $(HEAD_FILES)
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bgemm: bgemm.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) 


clean:
	rm -f kernel_test kernel_test.o bgemm bgemm.so bgemm.o