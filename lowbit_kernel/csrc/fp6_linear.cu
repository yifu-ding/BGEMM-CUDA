#include "include/kernel_matmul.cuh"
#include "include/kernel_reduction.cuh"
#include "utils/weight_prepacking.h"
#include "utils/weight_dequant.h"
#include "utils/weight_quant.h"
#include "utils/helper.h"

#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include "include/configs.h"
#include "kernel_test.h"

// #define SAVE_IO
// #define DEBUG_MODE

template<typename TilingConfig, typename OutputDataType>
static void Kernel_Ex(cudaStream_t    stream,
                      const uint4     *Weight, // 4B = 32b
                      const half      *Scales, // 16b
                      const half      *B,
                      OutputDataType  *C,
                      const size_t    M_Global,
                      const size_t    N_Global,
                      const size_t    K_Global, 
                      int             Split_K) 
{   
    #ifdef DEBUG_MODE
        printf("\n");
        printf("Launcher.cu->Kernel_Ex():\n");
        printf("M: %d, N: %d, K: %d, SplitK: %d\n", M_Global, N_Global, K_Global, Split_K);
        printf("TILE_M: %d, TILE_K: %d, TILE_N: %d\n", TilingConfig::TILE_M, TilingConfig::TILE_K, TilingConfig::TILE_N);
        // printf("Weight: %u", Weight);
    #endif
    static size_t SHMEM_SZ = max(TilingConfig::SMEM_SIZE_B_TILE+SMEM_SIZE_A1_TILE+SMEM_SIZE_A2_TILE, TilingConfig::SMEM_SIZE_C_TILE);
    cudaFuncSetAttribute(QUANT_GEMM_Kernel<TilingConfig, OutputDataType>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    size_t  dimN = (N_Global-1) / TilingConfig::TILE_N + 1;
    size_t  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3    GridDim(dimN, dimM, 1);
    dim3    BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);  // blocks, threads=128
    //
    #ifdef DEBUG_MODE
        printf("GridDim.x: %d, GridDim.y: %d, GridDim.z: %d, BlockDim.x: %d, BlockDim.y: %d, BlockDim.z: %d SHMEM_SZ: %d\n",
                GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z, SHMEM_SZ);
        printf("\n");
    #endif
    QUANT_GEMM_Kernel<TilingConfig, OutputDataType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>
                    (Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);
}


template<typename TilingConfig, typename OutputDataType>
static void Kernel_Ex_Bin(cudaStream_t    stream,
                      const uint4     *Weight, // 4B = 32b
                      const half      *Scales, // 16b
                      const half      *Scales_B, // 16b
                      const uint32_t     *B,
                    //   const half      *B,
                      OutputDataType  *C,
                      const size_t    M_Global,
                      const size_t    N_Global,
                      const size_t    K_Global, 
                      int             Split_K) 
{   
    #ifdef DEBUG_MODE
        printf("\n");
        printf("Launcher.cu->Kernel_Ex_Bin():\n");
        printf("M: %d, N: %d, K: %d, SplitK: %d\n", M_Global, N_Global, K_Global, Split_K);
        printf("TILE_M_BIN: %d, TILE_K_BIN: %d, TILE_N_BIN: %d\n", TilingConfig::TILE_M_BIN, TilingConfig::TILE_K_BIN, TilingConfig::TILE_N_BIN);
        // printf("Weight: %u\n", Weight);
    #endif
    static size_t SHMEM_SZ = max(TilingConfig::SMEM_SIZE_B_TILE+SMEM_SIZE_A1_TILE+SMEM_SIZE_A2_TILE, TilingConfig::SMEM_SIZE_C_TILE);
    cudaFuncSetAttribute(QUANT_BGEMM_Kernel<TilingConfig, OutputDataType>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    size_t  dimN = (N_Global-1) / TilingConfig::TILE_N_BIN + 1; // (256-1)/128+1 = 2
    size_t  dimM = M_Global * Split_K / TilingConfig::TILE_M_BIN; // 256*1/128 = 2
    dim3    GridDim(dimN, dimM, 1);  // 2, 2, 1
    dim3    BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1); // 128, 1, 1
    // dim3    BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS_BIN, 1, 1); // 32/64, 1, 1
    //
    #ifdef DEBUG_MODE
        printf("GridDim.x: %d, GridDim.y: %d, GridDim.z: %d, BlockDim.x: %d, BlockDim.y: %d, BlockDim.z: %d SHMEM_SZ: %d\n",
                GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z, SHMEM_SZ);
        printf("\n");
    #endif
    QUANT_BGEMM_Kernel<TilingConfig, OutputDataType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>
                    (Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K);

    /* cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    checkCudaErrors(cudaFuncSetAttribute(
      apmm_w1a1, cudaFuncAttributeMaxDynamicSharedMemorySize,
      SHMEM_SZ));
    checkKernelErrors(
              (apmm_w1a1<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                                    SHMEM_SZ>>>((int4*)Weight, (int4*)B, (int*)C, M_Global, N_Global, K_Global, 0, 0))); */
}



template<typename TilingConfig, typename OutputDataType>
static void Kernel_Ex_W1A1_Pack_MM(cudaStream_t    stream,
                      const half     *Weight, // 4B = 32b
                      const half      *Scales, // 16b
                      const half      *Scales_B, // 16b
                      const half     *B,
                    //   const half      *B,
                      OutputDataType  *C,
                      const size_t    M_Global,
                      const size_t    N_Global,
                      const size_t    K_Global, 
                      int             Split_K,
                      int             INSTR) 
{   
    #ifdef DEBUG_MODE
        printf("\n");
        printf("Launcher.cu->Kernel_Ex_Bin():\n");
        printf("M: %d, N: %d, K: %d, SplitK: %d\n", M_Global, N_Global, K_Global, Split_K);
        printf("TILE_M_BIN: %d, TILE_K_BIN: %d, TILE_N_BIN: %d\n", TilingConfig::TILE_M_BIN, TilingConfig::TILE_K_BIN, TilingConfig::TILE_N_BIN);
        // printf("Weight: %u\n", Weight);
    #endif
    // static size_t SHMEM_SZ = max(WEIGHT_PER_UNIT_BIN*2, TilingConfig::SMEM_SIZE_C_TILE);
    static size_t SHMEM_SZ = WEIGHT_PER_UNIT_BIN*3; // double buffer for both weight and act (128x128 per unit)
    cudaFuncSetAttribute(PACK_BGEMM_Kernel<TilingConfig, OutputDataType>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    size_t  dimN = (N_Global-1) / TilingConfig::TILE_N_BIN + 1; // (256-1)/128+1 = 2
    size_t  dimM = M_Global * Split_K / TilingConfig::TILE_M_BIN; // 256*1/128 = 2
    dim3    GridDim(dimN, dimM, 1);  // 2, 2, 1
    dim3    BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1); // 128, 1, 1
    // dim3    BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS_BIN, 1, 1); // 32/64, 1, 1
    //
    #ifdef DEBUG_MODE
        printf("GridDim.x: %d, GridDim.y: %d, GridDim.z: %d, BlockDim.x: %d, BlockDim.y: %d, BlockDim.z: %d SHMEM_SZ: %d\n",
                GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z, SHMEM_SZ);
        printf("\n");
    #endif
    PACK_BGEMM_Kernel<TilingConfig, OutputDataType><<<GridDim, BlockDim, SHMEM_SZ, stream>>>
                    (Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K, INSTR);

    /* cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    checkCudaErrors(cudaFuncSetAttribute(
      apmm_w1a1, cudaFuncAttributeMaxDynamicSharedMemorySize,
      SHMEM_SZ));
    checkKernelErrors(
              (apmm_w1a1<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                                    SHMEM_SZ>>>((int4*)Weight, (int4*)B, (int*)C, M_Global, N_Global, K_Global, 0, 0))); */
}

/*
 *
 */
cudaError_t fp6_linear_kernel(cudaStream_t    stream,
                              const uint4     *Weight,  // 4B = 4 * 8b
                              const half      *Scales,  // 16b
                              const half      *B,
                              half            *C,
                              const size_t    M_Global,
                              const size_t    N_Global,
                              const size_t    K_Global, 
                              float           *Reduction_Workspace,  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
                              int             Split_K)
{
    assert(M_Global % 256 == 0);
    assert(K_Global % 64 == 0);
    assert(N_Global>0);

    // Work around to support more N shapes:
    size_t N_PowerOf2;
    if(N_Global>0 &&  N_Global<=8)      N_PowerOf2 = 8;
    if(N_Global>8 &&  N_Global<=16)     N_PowerOf2 = 16;
    if(N_Global>16 && N_Global<=32)     N_PowerOf2 = 32;
    if(N_Global>32 && N_Global<=64)     N_PowerOf2 = 64;
    if(N_Global>64 && N_Global<=128)    N_PowerOf2 = 128;
    if(N_Global>128)                    N_PowerOf2 = ((N_Global-1)/128+1) * 128;

    if (Split_K == 1) {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex<TilingConfig<4, 1, 1>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_Ex<TilingConfig<4, 1, 2>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_Ex<TilingConfig<4, 1, 4>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_Ex<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_Ex<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, B, C, M_Global, N_Global, K_Global, Split_K);  break;
        }
    }
    else {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex<TilingConfig<4, 1, 1>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_Ex<TilingConfig<4, 1, 2>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_Ex<TilingConfig<4, 1, 4>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_Ex<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_Ex<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
        }
        // Reduction for SplitK
        dim3 GridDim((M_Global * N_Global) / REDUCTION_ELEMENT_PER_THREADBLOCK, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    }
    return cudaGetLastError();
}


cudaError_t bin_linear_kernel(cudaStream_t    stream,
                              const uint4     *Weight,  // 4B = 4 * 8b
                              const half      *Scales,  // 16b
                              const half      *Scales_B,  // 16b
                            //   const half      *B,
                              const uint32_t     *B,  // 4B = 4 * 8b
                              half            *C,
                              const size_t    M_Global,
                              const size_t    N_Global,
                              const size_t    K_Global, 
                              float           *Reduction_Workspace,  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
                              int             Split_K)
{
    assert(M_Global % 128 == 0);
    assert(K_Global % 128 == 0);
    assert(N_Global % 32 == 0);

    // Work around to support more N shapes:
    size_t N_PowerOf2;
    if(N_Global>0 &&  N_Global<=8)      N_PowerOf2 = 8;
    if(N_Global>8 &&  N_Global<=16)     N_PowerOf2 = 16;
    if(N_Global>16 && N_Global<=32)     N_PowerOf2 = 32;
    if(N_Global>32 && N_Global<=64)     N_PowerOf2 = 64;
    if(N_Global>64 && N_Global<=128)    N_PowerOf2 = 128;
    if(N_Global>128)                    N_PowerOf2 = ((N_Global-1)/128+1) * 128;

    if (Split_K == 1) {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex_Bin<TilingConfig<4, 1, 1>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_Ex_Bin<TilingConfig<4, 1, 2>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_Ex_Bin<TilingConfig<4, 1, 2>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_Ex_Bin<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_Ex_Bin<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex_Bin<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, Scales_B,  B, C, M_Global, N_Global, K_Global, Split_K);  break;
        }
    }
    else {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex_Bin<TilingConfig<4, 1, 1>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 16:    Kernel_Ex_Bin<TilingConfig<4, 1, 2>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 32:    Kernel_Ex_Bin<TilingConfig<4, 1, 4>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 64:    Kernel_Ex_Bin<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            case 128:   Kernel_Ex_Bin<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex_Bin<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);  break;
        }
        // Reduction for SplitK
        dim3 GridDim((M_Global * N_Global) / REDUCTION_ELEMENT_PER_THREADBLOCK, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    }
    return cudaGetLastError();
}



cudaError_t bin_pack_linear_kernel(cudaStream_t    stream,
                              const half     *Weight,  // 4B = 4 * 8b
                              const half      *Scales,  // 16b
                              const half      *Scales_B,  // 16b
                            //   const half      *B,
                              const half     *B,  // 4B = 4 * 8b
                              half            *C,
                              const size_t    M_Global,
                              const size_t    N_Global,
                              const size_t    K_Global, 
                              float           *Reduction_Workspace,  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
                              int             Split_K,
                              int             INSTR=XOR_POP)
{
    assert(M_Global % 32 == 0);
    assert(K_Global % 128 == 0);
    assert(N_Global % 32 == 0);

    // Work around to support more N shapes:
    size_t N_PowerOf2;
    if(N_Global>0 &&  N_Global<=8)      N_PowerOf2 = 8;
    if(N_Global>8 &&  N_Global<=16)     N_PowerOf2 = 16;
    if(N_Global>16 && N_Global<=32)     N_PowerOf2 = 32;
    if(N_Global>32 && N_Global<=64)     N_PowerOf2 = 64;
    if(N_Global>64 && N_Global<=128)    N_PowerOf2 = 128;
    if(N_Global>128)                    N_PowerOf2 = ((N_Global-1)/128+1) * 128;

    if (Split_K == 1) {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 1>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            case 16:    Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 2>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            case 32:    Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 2>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            case 64:    Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            case 128:   Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, Scales_B, B, C, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 8>, half>(stream, Weight, Scales, Scales_B,  B, C, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
        }
    }
    else {
        switch (N_PowerOf2) {
            case 8:     Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 1>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            case 16:    Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 2>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            case 32:    Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 4>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            case 64:    Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            case 128:   Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
            default:    if (N_PowerOf2 % 128 != 0) {
                            printf("FP6LLM_API Error: Unsupported N dimension %d!\n", N_PowerOf2);
                            return cudaErrorUnknown;
                        }
                        Kernel_Ex_W1A1_Pack_MM<TilingConfig<4, 1, 8>, float>(stream, Weight, Scales, Scales_B,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K, INSTR);  break;
        }
        // Reduction for SplitK
        dim3 GridDim((M_Global * N_Global) / REDUCTION_ELEMENT_PER_THREADBLOCK, 1, 1);
        dim3 BlockDim(WARP_SIZE, 1, 1);
        SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace, M_Global, N_Global, Split_K);
    }
    return cudaGetLastError();
}




#ifndef NO_PYTORCH
// #include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

/*
Computes FP6-FP16 GEMM (PyTorch interface).

[Mathmatical Formula]
Standard definition of linear layer:    Out = In * trans(W), where In, Out, and W are stored in row-major.
After Equivalent transformation    :    trans(Out) = W * trans(In). Note that we do not perform "transpose" during runtime, we instead interpret the In/Out as column-major matrices when calling our CUDA kernel.

[Inputs]
  _in_feats:  tensor of shape [B, IC];                  // half 
  _weights:   int tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6 weights.
  _scales:    tensor of shape [OC];                     // half
  splitK:     spliting the MatMul problem along K dimension for higher GPU utilization, default 1.
[Outputs]
  _out_feats: tensor of shape [B, OC];                  // half
*/
torch::Tensor fp6_linear_forward_cuda(torch::Tensor _in_feats,
                                      torch::Tensor _weights,
                                      torch::Tensor _scales,
                                      int           splitK=1)
{
    int num_in_feats      = _in_feats.size(0);
    int num_in_channels   = _in_feats.size(1);
    int num_out_channels  = _weights.size(0);
    assert( num_in_channels%64 == 0 );
    assert( (num_in_channels/16*3) == _weights.size(1) );    // Making sure the K dimension is matched.
    //
    int M = num_out_channels;
    int K = num_in_channels;
    int N = num_in_feats;
    // Input Tensors
    auto weight = reinterpret_cast<const uint4*>(_weights.data_ptr<int>());  // weights is [OC, IC] but in FP6.
    auto in_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
    auto scales   = reinterpret_cast<const half*>(_scales.data_ptr<at::Half>());
    // Output Tensors
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    at::Tensor _out_feats = torch::empty({num_in_feats, num_out_channels}, options);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

    options = torch::TensorOptions().dtype(torch::kFloat32).device(_in_feats.device());
    at::Tensor _workspace = torch::empty({splitK, num_in_feats, num_out_channels}, options);
    auto Reduction_Workspace = reinterpret_cast<float*>(_workspace.data_ptr<float>());  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
      
    fp6_linear_kernel(0, // Using default stream here.
                      weight,
                      scales,
                      in_feats,
                      out_feats,
                      M,
                      N,
                      K, 
                      Reduction_Workspace,  
                      splitK);

    return _out_feats;
}


/*
 * Weight prepacking (Pytorch interface).
 * [Input & Output]
 *  fp6_tensor: int tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6 weights.
 * [Output]
 *  packed_tensor: int tensor of shape [OC, IC // 16 * 3];
 */
torch::Tensor weight_matrix_prepacking_cpu(torch::Tensor fp6_tensor)
{
    size_t OC = fp6_tensor.size(0);
    size_t IC = fp6_tensor.size(1);
    assert (IC%3==0);   
    IC = IC*16/3;
    assert( (OC%256==0) && (IC%64==0) );
    auto packed_tensor = torch::empty_like(fp6_tensor);
    auto packed_tensor_ptr = reinterpret_cast<int*>(packed_tensor.data_ptr<int>());
    auto fp6_tensor_ptr = reinterpret_cast<int*>(fp6_tensor.data_ptr<int>());
    weight_matrix_prepacking(packed_tensor_ptr, fp6_tensor_ptr, OC, IC);
    return packed_tensor;
}

/*
 * Dequant a FP6 matrix to a equivalent FP16 matrix using CPUs.
 * A useful tool to construct input matrices for the FP16 GEMM baseline.
 * [Input]
 *  fp6_tensor:  int  tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6  weights.
 *  fp16_scale:  half tensor of shape [OC];                 // for row-wise quantization.
 * [Output]
 *  fp16_tensor: half tensor of shape [OC, IC].     
 */
torch::Tensor weight_matrix_dequant_cpu(torch::Tensor fp6_tensor, torch::Tensor fp16_scale) 
{
    int OC = fp6_tensor.size(0);
    assert(fp6_tensor.size(1) % 3 == 0);
    int IC = fp6_tensor.size(1) / 3 * 16;
    assert(fp16_scale.size(0)==OC);
    //
    auto fp6_tensor_ptr = reinterpret_cast<int*>(fp6_tensor.data_ptr<int>());
    auto fp16_scale_ptr = reinterpret_cast<half*>(fp16_scale.data_ptr<at::Half>());
    //
    auto options = torch::TensorOptions().dtype(fp16_scale.dtype()).device(fp16_scale.device());
    at::Tensor fp16_tensor = torch::empty({OC, IC}, options);
    auto fp16_tensor_ptr = reinterpret_cast<half*>(fp16_tensor.data_ptr<at::Half>());
    //
    DeQuantMatrix_FP6_To_FP16(fp16_tensor_ptr, (unsigned char*)fp6_tensor_ptr, OC, IC, fp16_scale_ptr);
    //
    return fp16_tensor;
}
// #endif

/*
 * Weight/act prepacking on cpu
 * [Input & Output]
 *  _tensor: half tensor of shape [OC, IC]==(M, K) or [B, IC]==(N, K)
 * [Output]
 *  packed_tensor: int tensor of shape [OC, IC // 32] or [B, IC // 32]
 * 
 */

uint32_t* binary_matrix_prepacking_cpu(torch::Tensor _tensor, torch::TensorOptions _options)
{
    size_t OC = _tensor.size(0); // M or N
    size_t IC = _tensor.size(1); // K

    uint32_t* packed_tensor = (uint32_t*)malloc(OC*IC*sizeof(uint32_t)/4);  CheckMallocCPU(packed_tensor, __LINE__); 
    // uint32_t* packed_tensor;
    // cudaMalloc(reinterpret_cast<void**>(&packed_tensor), OC*IC*sizeof(uint32_t)/4); CheckMallocCUDA(packed_tensor, __LINE__);

    auto _tensor_ptr = reinterpret_cast<half*>(_tensor.data_ptr<at::Half>());

    // if (_options.dtype()==torch::kFloat32) {
    //     _tensor_ptr = reinterpret_cast<float*>(_tensor.data_ptr<float>());
    // } else if (_options.dtype()==torch::kInt32){
    //     _tensor_ptr = reinterpret_cast<int32_t*>(_tensor.data_ptr<int32_t>());
    // } else { // default half
    //     _tensor_ptr = reinterpret_cast<half*>(_tensor.data_ptr<half>());
    // }
    
    bin_matrix_prepacking_to_uint32(packed_tensor, _tensor_ptr, OC, IC);

    return packed_tensor;
}

/* 
在cpu上pack，然后move到cuda上
 */
uint32_t* binary_matrix_prepacking_move_cuda(torch::Tensor _tensor)
{
    int OC = _tensor.size(0); // M or N
    int IC = _tensor.size(1); // K

    auto options = torch::TensorOptions().dtype(_tensor.dtype()).device(_tensor.device());
    uint32_t* packed_weights = binary_matrix_prepacking_cpu(_tensor, options);  // [OC, IC] == (M, K) 

    #ifdef SAVE_IO
        auto _weights_ptr = reinterpret_cast<half*>(_tensor.data_ptr<at::Half>());
        print_half(_weights_ptr, "weights_half", OC, IC);
        print_uint32(packed_weights, "packed_weights", OC, IC);
    #endif

    uint32_t* packed_weights_cuda;
    if (_tensor.device() != torch::kCUDA) {
        cudaMalloc(reinterpret_cast<void**>(&packed_weights_cuda), IC*OC*sizeof(uint32_t)/4); CheckMallocCUDA(packed_weights_cuda, __LINE__);
        cudaMemcpy(packed_weights_cuda,  packed_weights, IC*OC*sizeof(uint32_t)/4, cudaMemcpyHostToDevice); 
        checkLastCudaError(__LINE__);
    } else {
        packed_weights_cuda = packed_weights;
    }
    
    return packed_weights_cuda; 
}

/* 
在cuda上，分block并行pack
 */
torch::Tensor binary_weight_prepacking_cuda(torch::Tensor _tensor)  // half tensor on cuda
{
    int OC = _tensor.size(0); // M or N
    int IC = _tensor.size(1); // K

    // uint32_t* packed_tensor;
    auto options = torch::TensorOptions().dtype(at::kInt).device(_tensor.device());
    at::Tensor packed_int_tensor = torch::empty({OC, (int)(IC/32)}, options);

    auto _tensor_ptr = reinterpret_cast<half*>(_tensor.data_ptr<at::Half>());
    auto packed_int_tensor_ptr = reinterpret_cast<half*>(packed_int_tensor.data_ptr<at::Half>());

    // if (_options.dtype()==torch::kFloat32) {
    //     _tensor_ptr = reinterpret_cast<float*>(_tensor.data_ptr<float>());
    // } else if (_options.dtype()==torch::kInt32){
    //     _tensor_ptr = reinterpret_cast<int32_t*>(_tensor.data_ptr<int32_t>());
    // } else { // default half
    //     _tensor_ptr = reinterpret_cast<half*>(_tensor.data_ptr<half>());
    // }
    
    // bin_matrix_prepacking_to_uint32_cuda<4, 1, 1>(packed_tensor, _tensor_ptr, OC, IC);

    return packed_int_tensor;
}

/*
Computes FP6-FP16 GEMM (PyTorch interface).

[Mathmatical Formula]
Standard definition of linear layer:    Out = In * trans(W), where In, Out, and W are stored in row-major.
After Equivalent transformation    :    trans(Out) = W * trans(In). Note that we do not perform "transpose" during runtime, we instead interpret the In/Out as column-major matrices when calling our CUDA kernel.

[Inputs]
  _in_feats:  tensor of shape [B, IC];    (N, K)             // half
  _weights:   int tensor of shape [OC, IC];    (M, K)         // half
  _scales:    tensor of shape [OC];                     // half
  splitK:     spliting the MatMul problem along K dimension for higher GPU utilization, default 1.
[Outputs]
  _out_feats: tensor of shape [B, OC];                  // half
*/


torch::Tensor bgemm_linear_forward_cuda(torch::Tensor _in_feats,  // half tensor on CUDA
                                        torch::Tensor _weights,   // half tensor on CUDA
                                        torch::Tensor _scales,    // half tensor on CUDA
                                        int           splitK=1,
                                        int           INSTR=XOR_POP)
{
    int num_in_feats      = _in_feats.size(0);
    int num_in_channels   = _in_feats.size(1);
    int num_out_channels  = _weights.size(0);

    assert (num_in_channels%128==0); // K % 128 == 0
    assert (num_in_channels == _weights.size(1));  // Making sure the K dimension is matched.

    /* 
    auto options = torch::TensorOptions().dtype(_weights.dtype()).device(_weights.device());
    uint32_t* packed_weights = binary_matrix_prepacking_cpu(_weights, options);  // [OC, IC] == (M, K) 
    options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    uint32_t* packed_act = binary_matrix_prepacking_cpu(_in_feats, options);

    #ifdef SAVE_IO
        auto _in_feats_ptr = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
        print_half(_in_feats_ptr, "in_feats_half", num_in_feats, num_in_channels); 
        auto _weights_ptr = reinterpret_cast<half*>(_weights.data_ptr<at::Half>());
        print_half(_weights_ptr, "weights_half", num_out_channels, num_in_channels);
        print_uint32(packed_weights, "packed_weights", num_out_channels, num_in_channels);
        print_uint32(packed_feats, "packed_feats", num_in_feats, num_in_channels); 
    #endif
    */
    /* half* weights_cuda;
    half* feat_cuda;
    if (_weights.device() != torch::kCUDA) {
        cudaMalloc(reinterpret_cast<void**>(&weights_cuda), num_in_channels*num_out_channels*sizeof(uint32_t)/4); CheckMallocCUDA(packed_weights_cuda, __LINE__);
        cudaMemcpy(weights_cuda,  _weights, num_in_channels*num_out_channels*sizeof(uint32_t)/4, cudaMemcpyHostToDevice); 
        checkLastCudaError(__LINE__);

        cudaMalloc(reinterpret_cast<void**>(&packed_act_cuda), num_in_feats*num_in_channels*sizeof(uint32_t)/4); CheckMallocCUDA(packed_act_cuda, __LINE__);
        cudaMemcpy(packed_act_cuda,  packed_act, num_in_feats*num_in_channels*sizeof(uint32_t)/4, cudaMemcpyHostToDevice); 
        checkLastCudaError(__LINE__);
    } else {
        // Input Tensors
        auto ori_weight = reinterpret_cast<const half*>(_weights.data_ptr<at::Half>());
        auto ori_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
    } 
     */

    auto ori_weight = reinterpret_cast<const half*>(_weights.data_ptr<at::Half>());
    auto ori_feats = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
    auto scales = reinterpret_cast<half*>(_scales.data_ptr<at::Half>());
    
    // Output Tensors
    auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(torch::kCUDA);
    at::Tensor _out_feats = torch::empty({num_in_feats, num_out_channels}, options);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

    options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    at::Tensor _workspace = torch::empty({splitK, num_in_feats, num_out_channels}, options);
    auto Reduction_Workspace = reinterpret_cast<float*>(_workspace.data_ptr<float>());  // Reduction_Workspace_Size = Split_K * M_Global * N_Global * sizeof(fp32)
    
    int M = num_out_channels;
    int K = num_in_channels;
    int N = num_in_feats;
    
    // All tensors should be on CUDA Mem before bin_linear_kernel. 
    bin_pack_linear_kernel(0, // Using default stream here.
                      (half*)ori_weight,
                      scales,
                      scales, 
                      (half*)ori_feats,
                      out_feats,
                      M,
                      N,
                      K, 
                      Reduction_Workspace,  
                      splitK,
                      INSTR);

    #ifdef SAVE_IO
        half* out_feats_h = NULL;  // col major
        out_feats_h       = (half*)malloc(sizeof(half) * num_out_channels * num_in_feats);
        cudaMemcpy(out_feats_h, out_feats, sizeof(half) * num_out_channels * num_in_feats, cudaMemcpyDeviceToHost);  // Col Major
        cudaFree(out_feats);
        cudaFree(Reduction_Workspace);
        print_half(out_feats_h, "out_feats", num_in_feats, num_out_channels);
    #endif

    return _out_feats;
}

#endif