#ifndef UTILS_CORE_CUH
#define UTILS_CORE_CUH

#include <assert.h>

#include "configs.h"
#include "ptx_mma.cuh"
#include "utils_parallel_dequant.cuh"


#ifdef PIPELINE_LEVEL_SMEM
template<int NUM_INT_PER_THREAD>
__device__ __forceinline__ void CopyFromSharedToRegister_AFrag(uint32_t Reg[], uint32_t* SPTR, int slice_id) {
    SPTR += slice_id * (NUM_INT_PER_THREAD*WARP_SIZE);
    int     lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for(int i=0; i<NUM_INT_PER_THREAD; i++) { // regi 在每个线程不一样，且线程之间连续
        Reg[i] = SPTR[lane_id+i*WARP_SIZE];
    }
}



template <typename TilingConfig>
__device__ __forceinline__ void initialize_mma_slice(uint32_t                  (*a)[4],
                                                     uint32_t                  (*b)[4],
                                                     uint32_t* __restrict__    A1_SPTR_read,
                                                     uint32_t* __restrict__    A2_SPTR_read,
                                                     half      __restrict__    (*B_SPTR_read)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                                     uint32_t*                 RPTR_Scales)
{
    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6 register per thread;  （32 threads）
    uint32_t a_1[2];                      // NO double buffer
    uint32_t a_2[4];                      // NO double buffer
    CopyFromSharedToRegister_AFrag<2>   (a_1, A1_SPTR_read, 0);
    CopyFromSharedToRegister_AFrag<4>   (a_2, A2_SPTR_read, 0);
    Dequant_32FP6_4Way(a, a_1, a_2, RPTR_Scales);   // SIMT Dequant: dequantizing FP6 to FP16 at register level, dequantizing a slice each time 
    B_FromSharedToReg<TilingConfig>(b, B_SPTR_read, 0); // Loading B from shared to registers
}

// core_mma_slice<TilingConfig>(c, a, b, read_SPTR_Frag1, read_SPTR_Frag2, read_SPTR, Scales_RPTR, 1);
template <typename TilingConfig>
__device__ __forceinline__ void core_mma_slice(float                     c[][REG_PER_THREAD_C_TENSOR_16_16],
                                               uint32_t                  (*a)[4],
                                               uint32_t                  (*b)[4],
                                               uint32_t* __restrict__    A1_SPTR_read,
                                               uint32_t* __restrict__    A2_SPTR_read,
                                               half      __restrict__    (*B_SPTR_read)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                               uint32_t*                 RPTR_Scales,
                                               int                       slice_id)      // writing slice[slice_id] to registers, k=0 -> slice_id=1 for prefetching
{
    #ifdef DEBUG_MODE
        assert((TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0));   // if WARP_COL_MMA_TENSORS == 1, B tile in registers is padded to a 16*16 MMA block
    #endif
    const int NumRegSets_a = WARP_ROW_MMA_TENSORS;                                                                              // 1 set = 4 registers, containing a 16*16 MMA block
    const int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;                // 1 set = 4 registers, containing a 16*16 MMA block
    uint32_t (*c_uint_ptr)[REG_PER_THREAD_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_THREAD_C_TENSOR_16_16]>(c);    // Reigsters for accumulated FP32 results
    // 存的fp32, 一个线程 [][8], 也就是 x * 8 elt
    // 32thread * 16*8= 64x64

    // Setting RPTRs for double buffers
    uint32_t (*a_read )[4] = a;
    uint32_t (*a_write)[4] = a;
    uint32_t (*b_read )[4] = b;
    uint32_t (*b_write)[4] = b;
    if(slice_id%2==1)   { b_write += NumRegSets_b; a_write += NumRegSets_a;} 
    else                { b_read  += NumRegSets_b; a_read  += NumRegSets_a;}

    // Reading registers and issuing core tensor core computations (a slice of A and B tile in shared memory)
    #pragma unroll
    for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) { // 4
        if(TilingConfig::WARP_COL_MMA_TENSORS==1) {
            MMA_FP16_M16N8K16( c_uint_ptr[i], a_read[i], b_read[0] ); 
        }
        else {
            #pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS/2; j++) { // 8/2=4
                MMA_FP16_M16N8K16( c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS],     a_read[i], b_read[j]     ); // 这个线程提供的c仅需4 regs 存 4个fp结果
                MMA_FP16_M16N8K16( c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS] + 4, a_read[i], b_read[j] + 2 ); // 
            }
        }
    }

    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6 register per thread;
    uint32_t a_1[2];                      // NO double buffer
    uint32_t a_2[4];                      // NO double buffer
    CopyFromSharedToRegister_AFrag<2>   (a_1, A1_SPTR_read, slice_id);
    CopyFromSharedToRegister_AFrag<4>   (a_2, A2_SPTR_read, slice_id);
    Dequant_32FP6_4Way(a_write, a_1, a_2, RPTR_Scales);   // SIMT Dequant: dequantizing FP6 to FP16 at register level, dequantizing a slice each time 
    B_FromSharedToReg<TilingConfig>     (b_write, B_SPTR_read, slice_id); // Loading B from shared to registers
}



template<int NUM_INT_PER_THREAD, int NUM_INT_PER_MMA>
__device__ __forceinline__ void CopyFromSharedToRegister_BinaryW(uint32_t (*Reg)[1], uint32_t* SPTR, int slice_id) {
    // SPTR += slice_id * (NUM_INT_PER_THREAD*WARP_SIZE);  
    int     lane_id = threadIdx.x % WARP_SIZE;  // 32
    
    if (NUM_INT_PER_MMA==1){ // 每个线程每次mma用一个int
        #pragma unroll
        for(int i=0; i<NUM_INT_PER_THREAD; i++) { // 4
            Reg[i][0] = SPTR[lane_id + i*WARP_SIZE];
        }
    } // error
}
 
template<int NUM_INT_PER_MMA> // 4, 1
__device__ __forceinline__ void CopyFromSharedToRegister_BinaryAct(
                                                         uint32_t      __restrict__    (*Reg)[1], 
                                                         uint32_t      __restrict__    (*SPTR)[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1],
                                                         int slice_id, 
                                                         int NumIterB) {
    // SPTR += slice_id * (4*WARP_SIZE);  // 8*128/32*4 = 128
    int     lane_id = threadIdx.x % WARP_SIZE;  // 32
    if (NUM_INT_PER_MMA==1){ // 每个线程每次mma用一个int
        #pragma unroll
        for(int i=0; i<NumIterB; i++) { // 4
            Reg[i][0] = SPTR[slice_id*NumIterB*8 + i*8 + lane_id/4][lane_id%4];
        }
    } // error
}



template<int NUM_INT_PER_THREAD, int NUM_INT_PER_MMA> // 4,1
__device__ __forceinline__ void PackFromSharedToRegister_BinaryW(uint32_t (*Reg)[1], half* SPTR, int slice_id) {
    // SPTR += slice_id * (NUM_INT_PER_THREAD*WARP_SIZE);  
    int     lane_id = threadIdx.x % WARP_SIZE;  // threadIdx.x % 32
    
    if (NUM_INT_PER_MMA==1){ // 每个线程每次mma用一个int
        #pragma unroll
        for(int i=0; i<NUM_INT_PER_THREAD; i++) { // 1 int / thread, 128 uint32, 128*32 half
            // Reg[i][0] = SPTR[lane_id + i*WARP_SIZE];
            // SIGN_32_HALF_TO_UINT32(Reg[i], &SPTR[((lane_id*NUM_INT_PER_THREAD + i)*WARP_SIZE)]); // Reg[i][0]是一个uint32, 每线程每次处理32个half->1uint32
            SIGN_32_HALF_TO_UINT32(Reg[i], &SPTR[(lane_id + i*WARP_SIZE)*32]); // Reg[i][0]是一个uint32, 每线程每次处理32个half->1uint32
        }
    } // error
}
 
template<int NUM_INT_PER_MMA> // 1
__device__ __forceinline__ void PackFromSharedToRegister_BinaryAct(
                                                         uint32_t      __restrict__    (*Reg)[1], 
                                                         half          __restrict__    (*SPTR)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1],
                                                         int slice_id, 
                                                         int NumIterB) {
    // SPTR += slice_id * (4*WARP_SIZE);  // 8*128/32*4 = 128
    int     lane_id = threadIdx.x % WARP_SIZE;  // 32
    if (NUM_INT_PER_MMA==1){ // 每个线程每次mma用一个int
        #pragma unroll
        for(int i=0; i<NumIterB; i++) { // 1
            // Reg[i][0] = SPTR[slice_id*NumIterB*8 + i*8 + lane_id/4][lane_id%4]; 
            // SIGN_32_HALF_TO_UINT32(Reg[i], &SPTR[lane_id][i*WARP_SIZE]); // 32 half -> Reg[i][0]
            SIGN_32_HALF_TO_UINT32(Reg[i], &SPTR[slice_id*NumIterB*8 + i*8 + lane_id/4][(lane_id%4)*WARP_SIZE]); // 32 half -> Reg[i][0]
        }
    } // error
}


template <typename TilingConfig>
__device__ __forceinline__ void initialize_mma_slice_bin(uint32_t                  (*a)[1],
                                                        uint32_t                   (*b)[1],
                                                        uint32_t* __restrict__    W_SPTR_read, // 0x7fffd5000600
                                                        uint32_t   __restrict__  (*A_SPTR_read)[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1],
                                                        const int                        NumIterB)
{
    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6 register per thread; （32 threads）
    // Registers to store binary weights for a slice (64*32) of weight matrix => 64 bin per thread =>  1 register per thread（32 thread）
    // uint32_t a_t[4];                      // NO double buffer, address for 1 reg

    // 32个线程，每个线程的有一个a[1]，uint32_t是4B，a也就是4*4B的容量。总共初始化了 32*4*4B 的容量，即 4096b 的 weight
    CopyFromSharedToRegister_BinaryW<4, 1>   (a, W_SPTR_read, 0); 
    CopyFromSharedToRegister_BinaryAct<1>   (b, A_SPTR_read, 0, NumIterB);  // 这里有一点冗余?也可能不是冗余，就是act少于4096个元素时，会有一部分多余的地址被转换
}


template <typename TilingConfig>
__device__ __forceinline__ void initialize_mma_slice_binpack(uint32_t                  (*a)[1],
                                                            uint32_t                   (*b)[1],
                                                            half* __restrict__    W_SPTR_read, // 0x7fffd5000600
                                                            half   __restrict__  (*A_SPTR_read)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1],
                                                            const int                        NumIterB)
{
    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6 register per thread; （32 threads）
    // Registers to store binary weights for a slice (64*32) of weight matrix => 64 bin per thread =>  1 register per thread（32 thread）
    // uint32_t a_t[4];                      // NO double buffer, address for 1 reg

    // 32个线程，每个线程的有一个a[1]，uint32_t是4B，a也就是4*4B的容量。总共初始化了 32*4*4B 的容量，即 4096b 的 weight
    PackFromSharedToRegister_BinaryW<1, 1>   (a, W_SPTR_read, 0); 
    PackFromSharedToRegister_BinaryAct<1>    (b, A_SPTR_read, 0, NumIterB);  // 这里有一点冗余?也可能不是冗余，就是act少于4096个元素时，会有一部分多余的地址被转换
}


template <typename TilingConfig>
__device__ __forceinline__ void core_mma_slice_bin(uint32_t                  c[][REG_PER_THREAD_C_TENSOR_16_16],
                                               uint32_t                  (*a)[1],
                                               uint32_t                  (*b)[1],
                                               uint32_t* __restrict__    W_SPTR_read,
                                               uint32_t __restrict__    (*A_SPTR_read)[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1],
                                            //    half      __restrict__    (*A_SPTR_read)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                               uint32_t*                 RPTR_Scales_w,
                                               uint32_t*                 RPTR_Scales_a,
                                               int                       slice_id,
                                               const int                       NumIterB)      // writing slice[slice_id] to registers, k=0 -> slice_id=1 for prefetching
{
    #ifdef DEBUG_MODE
        assert((TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0));   // if WARP_COL_MMA_TENSORS == 1, B tile in registers is padded to a 16*16 MMA block
    #endif
    const int NumRegSets_w = 4;                                                                              // 1 set = 4 registers, containing a 16*16 MMA block
    const int NumRegSets_a = 4; // (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;                // 1 set = 4 registers, containing a 16*16 MMA block
    uint32_t (*c_uint_ptr)[REG_PER_THREAD_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_THREAD_C_TENSOR_16_16]>(c);    // Reigsters for accumulated FP32 results
    // c_uint_ptr += slice_id * 4*16 * 8*8/8 / 32;  // 除以32线程

    // Setting RPTRs for double buffers
    // uint32_t (*a_read )[1] = a;  // 4*1
    // uint32_t (*a_write)[1] = a; 
    uint32_t (*b_read )[1] = b;  // 4*1
    uint32_t (*b_write)[1] = b;
    if(slice_id%2==1)   { b_write += NumRegSets_a; }
    else                { b_read  += NumRegSets_a; }

    // Reading registers and issuing core tensor core computations (a slice of A and B tile in shared memory)
    #pragma unroll
    for (int i = 0; i < NumRegSets_w; i++) {
        for (int j = 0; j < NumIterB; j++) {
            MMA_B1B1_M8N8K128_AND( c_uint_ptr[i + j*4] + ((slice_id+3)%4)*2, a[i], b_read[j] );
        }
    }
    
    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6 register per thread;
//     CopyFromSharedToRegister_BinaryW<4, 1>  (a_write, W_SPTR_read, slice_id);
    CopyFromSharedToRegister_BinaryAct<1>  (b_write, A_SPTR_read, slice_id, NumIterB);
}



template <typename TilingConfig>
__device__ __forceinline__ void core_mma_slice_binpack(int32_t                  c[][REG_PER_THREAD_C_TENSOR_16_16],
                                               uint32_t                  (*a)[1],
                                               uint32_t                  (*b)[1],
                                               half* __restrict__    W_SPTR_read,
                                               half __restrict__    (*A_SPTR_read)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1],
                                            //    half      __restrict__    (*A_SPTR_read)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                               int32_t*                 RPTR_Scales_w,
                                               int32_t*                 RPTR_Scales_a,
                                               int                       slice_id,
                                               const int                 NumIterB,
                                               int                      INSTR=AND_POP)      // writing slice[slice_id] to registers, k=0 -> slice_id=1 for prefetching
{
    #ifdef DEBUG_MODE
        assert((TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0));   // if WARP_COL_MMA_TENSORS == 1, B tile in registers is padded to a 16*16 MMA block
    #endif
    const int NumRegSets_w = 1;                                                                              // 1 set = 4 registers, containing a 16*16 MMA block
    const int NumRegSets_a = 1; // (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;                // 1 set = 4 registers, containing a 16*16 MMA block
    int32_t (*c_uint_ptr)[REG_PER_THREAD_C_TENSOR_16_16] = reinterpret_cast<int32_t(*)[REG_PER_THREAD_C_TENSOR_16_16]>(c);    // Reigsters for accumulated FP32 results
    // REG_PER_THREAD_C_TENSOR_16_16 = slice_num * 2 = 8

    // Setting RPTRs for double buffers
    // uint32_t (*a_read )[1] = a;  // 4*1
    // uint32_t (*a_write)[1] = a; 
    uint32_t (*b_read )[1] = b;  // 4*1
    uint32_t (*b_write)[1] = b;
    if(slice_id%2==1)   { b_write += NumRegSets_a; }
    else                { b_read  += NumRegSets_a; }

    

    // Reading registers and issuing core tensor core computations (a slice of A and B tile in shared memory)
    if (INSTR==AND_POP){
        #pragma unroll
        for (int i = 0; i < NumRegSets_w; i++) {
            for (int j = 0; j < NumIterB; j++) {
                MMA_B1B1_M8N8K128_AND( c_uint_ptr[i + j*4] + ((slice_id+3)%4)*2, a[i], b_read[j] );
            }
        }
    } else { // xor.pop
        #pragma unroll
        for (int i = 0; i < NumRegSets_w; i++) {
            for (int j = 0; j < NumIterB; j++) {
                int32_t tmp_c[2];  // 
                MMA_B1B1_M8N8K128_XOR( tmp_c, a[i], b_read[j] );
                K_SUB_2_XORPOP(c_uint_ptr[i + j*4] + ((slice_id+3)%4)*2, tmp_c, TilingConfig::TILE_K_BIN); // 128
            }
        }
    }
    
    // Writing registers
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6 register per thread;
//     CopyFromSharedToRegister_BinaryW<4, 1>  (a_write, W_SPTR_read, slice_id);
    PackFromSharedToRegister_BinaryAct<1>  (b_write, A_SPTR_read, slice_id, NumIterB);
}


#else
// Old version with naive pipeline design
template<int NUM_INT_PER_THREAD>
__device__ __forceinline__ void CopyFromSharedToRegister_AFrag(uint32_t Reg[], uint32_t* SPTR) {
    int     lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for(int i=0; i<NUM_INT_PER_THREAD; i++) {
        Reg[i] = SPTR[lane_id+i*WARP_SIZE];
    }
}
template <typename TilingConfig>
__device__ __forceinline__ void PipelinedCoreLoop(float                     c[][REG_PER_THREAD_C_TENSOR_16_16],
                                                  half      __restrict__    (*read_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                                  uint32_t* __restrict__    read_SPTR_Frag1,
                                                  uint32_t* __restrict__    read_SPTR_Frag2,
                                                  uint32_t*                 RPTR_Scales)
{
    #ifdef DEBUG_MODE
        assert((TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0));   // if WARP_COL_MMA_TENSORS == 1, B tile in registers is padded to a 16*16 MMA block
    #endif
    const int NumRegSets_a = WARP_ROW_MMA_TENSORS;                                                                  // 1 set = 4 registers, containing a 16*16 MMA block
    const int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;    // 1 set = 4 registers, containing a 16*16 MMA block

    // Reigsters to store FP32 results
    uint32_t (*c_uint_ptr)[REG_PER_THREAD_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_THREAD_C_TENSOR_16_16]>(c);
    // Registers to store FP6 fragments for a slice (64*16) of A matrix => 32 FP6 per thread => 6 register per thread;
    uint32_t a_1[2*2];                      // double buffer is used
    uint32_t a_2[4*2];                      // double buffer is used
    // Registers to store decompressed FP6
    uint32_t a  [NumRegSets_a * 1][4];      // No double buffer
    // Register to store FP16 B matrix (a slice)
    uint32_t b  [NumRegSets_b * 2][4];      // double buffer is used

    // Overlapped Smem and TC pipeline: pre-loading from shared to registers
    CopyFromSharedToRegister_AFrag<2>   (a_1, read_SPTR_Frag1);
    CopyFromSharedToRegister_AFrag<4>   (a_2, read_SPTR_Frag2);
    B_FromSharedToReg<TilingConfig>     (b, read_SPTR, 0);

    #pragma unroll
    for (int k = 0; k < WARP_K_MMA_TENSORS; k++) {
        uint32_t (*b_read)[4]   = b;
        uint32_t (*b_write)[4]  = b;
        uint32_t *a_1_read      = a_1;
        uint32_t *a_1_write     = a_1;
        uint32_t *a_2_read      = a_2;
        uint32_t *a_2_write     = a_2;
        if(k%2==0) {
            b_write     += NumRegSets_b;
            a_1_write   += 2;
            a_2_write   += 4;
        } 
        else {
            b_read      += NumRegSets_b;
            a_1_read    += 2;
            a_2_read    += 4;
        }
        // data loading
        if (k + 1 < WARP_K_MMA_TENSORS) {
            // updating SPTR for fragment1 and fragment2
            read_SPTR_Frag1 += 2*WARP_SIZE;
            read_SPTR_Frag2 += 4*WARP_SIZE;
            CopyFromSharedToRegister_AFrag<2>(a_1_write, read_SPTR_Frag1);
            CopyFromSharedToRegister_AFrag<4>(a_2_write, read_SPTR_Frag2);
            B_FromSharedToReg<TilingConfig>(b_write, read_SPTR, (k+1)*MMA_16);
        }
        // SIMT Dequant + Tensor Core computations
        Dequant_32FP6_4Way(a, a_1_read, a_2_read, RPTR_Scales);   // Dequantizing FP6 to FP16 at register level, dequantizing a slice each time
        #pragma unroll
        for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) {
            if(TilingConfig::WARP_COL_MMA_TENSORS==1)
                MMA_FP16_M16N8K16( c_uint_ptr[i], a[i], b_read[0] );
            else {            
                #pragma unroll
                for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS/2; j++) {
                    MMA_FP16_M16N8K16( c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS],     a[i], b_read[j]     );
                    MMA_FP16_M16N8K16( c_uint_ptr[i + j * WARP_ROW_MMA_TENSORS] + 4, a[i], b_read[j] + 2 ); // c+4; b+2
                }
            }
        }
    }
}
#endif // #ifdef PIPELINE_LEVEL_SMEM

template <typename TilingConfig>
__device__ __forceinline__ void StoreToSharedMemoryFromRegister(float (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C_4],
                                                                float c[][REG_PER_THREAD_C_TENSOR_16_16])
{
    const int   lane_id             = threadIdx.x % WARP_SIZE; // [0, 31]
    const int   warpId              = threadIdx.x / WARP_SIZE; // [0, 3]
    int         warp_row_offset     = warpId * (MMA_16 * WARP_ROW_MMA_TENSORS);  // warpId * 64 ，权重取的位置(M维)
    #pragma unroll
    for (int i = 0; i < WARP_ROW_MMA_TENSORS; i++) {
        #pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_MMA_TENSORS; j++) {    // Dealing with one 16*8 Tensor
            int RegSetID            = i + (j/2)*WARP_ROW_MMA_TENSORS;     // [0, 3] + [0, 3]*4 = 0~15
            int RegOffset           = (j%2)*(REG_PER_THREAD_C_TENSOR_16_16/2);  // [0,1] * (8/2) = [0, 4]
            int Tensor_row_offset   = warp_row_offset + i * MMA_16;  // [0, 16, 32, 48] + warp_row_offset
            int Tensor_col_offset   = j * MMA_8; // [0,7] * 8 = [0, 8, 16, ..., 56]
            #pragma unroll
            for (int r = 0; r < REG_PER_THREAD_C_TENSOR_16_16/2; r++) {  // r=0~3
                int row_offset = lane_id / 4;  // [0-7] + [8-15]-> [0-15]
                if (r >= 2) row_offset += 8;   // r=2,3 row_offset += 8 = [8~15]
                int col_offset = (lane_id % 4) * 2;  // 0~3 *2 = [0, 2, 4, 6] + [1, 3, 5, 7] = [0-7]
                if (r%2==1) col_offset += 1;   // r=1,3   col_offset += 1 = [1, 3, 5, 7]
                smem_CFrag[Tensor_col_offset + col_offset][Tensor_row_offset + row_offset] = c[RegSetID][r + RegOffset]; // 32 thread * 16*8 fp16 = 64*64 fp32
                
            }
        }
    }
}

template <typename TilingConfig>
__device__ __forceinline__ void StoreToSharedMemoryFromRegister(int32_t (*smem_CFrag)[TilingConfig::TILE_M_BIN+PADDING_SHARED_MEM_FOR_C_0],
                                                                int32_t c[][REG_PER_THREAD_C_TENSOR_16_16],
                                                                int NumIterB,
                                                                int NumRegSets_w)
{
    const int   lane_id             = threadIdx.x % WARP_SIZE;  // 0~31
    const int   warpId              = threadIdx.x / WARP_SIZE;  // 0,1,2,3
    // int         warp_row_offset     = warpId * (MMA_16 * WARP_ROW_MMA_TENSORS);
    int         warp_row_offset     = warpId * WARP_SIZE/4; // [0,1,2,3]*8
    #pragma unroll
    for (int i = 0; i < NumRegSets_w; i++) {
        #pragma unroll
        for (int j = 0; j < NumIterB; j++) {    // Dealing with one 16*8 Tensor
            int RegSetID            = i + j*4;
            int Tensor_row_offset   = warp_row_offset + i * 8; // [0,8,16,24 + 0]
            int Tensor_col_offset   = j * 8;
            #pragma unroll
            for (int s = 0; s < REG_PER_THREAD_C_TENSOR_16_16/2; s++) { // s = 0, 1, 2, 3  
                // int row_offset = lane_id / 4;
                // if (r >= 2) row_offset += 8;
                // int col_offset = (lane_id % 4) * 2;
                // if (r%2==1) col_offset += 1;
                int col_offset = s * NumIterB * 8 + (lane_id % 4) * 2; // s=0: [0,2,4,6]
                int row_offset = lane_id / 4;  // [0~7]  
                smem_CFrag[Tensor_col_offset + col_offset][Tensor_row_offset + row_offset] = c[RegSetID][s*2];
                smem_CFrag[Tensor_col_offset + col_offset + 1][Tensor_row_offset + row_offset] = c[RegSetID][s*2+1];
            }
        }
    }
}


// 原来的bgemm
template <typename TilingConfig>
__device__ __forceinline__ void StoreToSharedMemoryFromRegister(uint32_t (*smem_CFrag)[TilingConfig::TILE_M_BIN+PADDING_SHARED_MEM_FOR_C_0],
                                                                uint32_t c[][REG_PER_THREAD_C_TENSOR_16_16],
                                                                int NumIterB,
                                                                int NumRegSets_w)
{
    const int   lane_id             = threadIdx.x % WARP_SIZE;  // 0~31
    const int   warpId              = threadIdx.x / WARP_SIZE;  // 0,1,2,3
    // int         warp_row_offset     = warpId * (MMA_16 * WARP_ROW_MMA_TENSORS);
    int         warp_row_offset     = warpId * WARP_SIZE; // [0,1,2,3]*32
    #pragma unroll
    for (int i = 0; i < NumRegSets_w; i++) {
        #pragma unroll
        for (int j = 0; j < NumIterB; j++) {    // Dealing with one 16*8 Tensor
            int RegSetID            = i + j*4;
            int Tensor_row_offset   = warp_row_offset + i * 8; // 
            int Tensor_col_offset   = j * 8;
            #pragma unroll
            for (int s = 0; s < REG_PER_THREAD_C_TENSOR_16_16/2; s++) { // r = 0, 1, 2, 3  
                // int row_offset = lane_id / 4;
                // if (r >= 2) row_offset += 8;
                // int col_offset = (lane_id % 4) * 2;
                // if (r%2==1) col_offset += 1;
                int col_offset = s * NumIterB * 8 + (lane_id % 4) * 2;
                int row_offset = lane_id / 4;
                smem_CFrag[Tensor_col_offset + col_offset][Tensor_row_offset + row_offset] = c[RegSetID][s*2];
                smem_CFrag[Tensor_col_offset + col_offset + 1][Tensor_row_offset + row_offset] = c[RegSetID][s*2+1];
            }
        }
    }
}


#endif