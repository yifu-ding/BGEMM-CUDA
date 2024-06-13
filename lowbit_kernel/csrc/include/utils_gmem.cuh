#ifndef UTILS_GMEM_CUH
#define UTILS_GMEM_CUH

#include <assert.h>
#include <stdio.h>
#include "configs.h"
#include "ptx_cp.async.cuh"

/* 
 * Copying A1/A2 from global memory to shared memory.
 * Usually 1024 or 2048 Bytes
 */
template<int SMEM_SIZE_IN_BYTES_PER_WARP>
__device__ __forceinline__ void CopyFromGlobalToShared_A(uint32_t* SPTR, 
                                                        const uint4* GPTR,
                                                        bool pred_guard = true) {
    #ifdef DEBUG_MODE
        static_assert(SMEM_SIZE_IN_BYTES_PER_WARP/WARP_SIZE % 16 == 0);
    #endif
    int lane_id      = threadIdx.x % WARP_SIZE;
    half* SPTR_HALF = reinterpret_cast<half*>(SPTR);
    const half* GPTR_HALF = reinterpret_cast<const half*>(GPTR);
    SPTR_HALF += lane_id*8;
    GPTR_HALF += lane_id*8;
    #pragma unroll
    for(int i=0; i<SMEM_SIZE_IN_BYTES_PER_WARP/WARP_SIZE/16; i++) {
        cp_async<16>( SPTR_HALF, GPTR_HALF, pred_guard); // 每次 copy 16B, 32线程，总共32*16B=512B
        SPTR_HALF += 256;   // Forward 512 Bytes
        GPTR_HALF += 256;   // Forward 512 Bytes
    }

}


/* 
 * Copying A1/A2 from global memory to shared memory.
 * Usually 1024 or 2048 Bytes
 */
template<int SMEM_SIZE_IN_BYTES_PER_WARP> // 2048
__device__ __forceinline__ void CopyFromGlobalToShared_BinaryWeight(uint32_t* SPTR, 
                                                        const uint4* GPTR,
                                                        const int GlobalStride, // K_GLOBAL
                                                        bool pred_guard = true) {
    #ifdef DEBUG_MODE
        static_assert(SMEM_SIZE_IN_BYTES_PER_WARP/WARP_SIZE % 16 == 0);
    #endif
    int lane_id      = threadIdx.x % WARP_SIZE;
    half* SPTR_HALF = reinterpret_cast<half*>(SPTR);
    const half* GPTR_HALF = reinterpret_cast<const half*>(GPTR);
    SPTR_HALF += lane_id*8;  // 每线程差8 half, 即16B
    GPTR_HALF += lane_id*(GlobalStride/8/2); 
    // GPTR_HALF += lane_id*8;
    #pragma unroll
    for(int i=0; i<SMEM_SIZE_IN_BYTES_PER_WARP/WARP_SIZE/16; i++) { // 4 iter = 4*512=2048B
        cp_async<16>( SPTR_HALF, GPTR_HALF, pred_guard); // 每次 copy 16B, 32线程，总共32*16B=512B
        SPTR_HALF += 256;   // Forward 512 Bytes, 512/2 half
        // GPTR_HALF += 256;   // Forward 512 Bytes, 
        GPTR_HALF += 32*GlobalStride/8/2;
    }

}


// /* 
//  * Copying A1/A2 from global memory to shared memory.
//  * Usually 1024 or 2048 Bytes
//  */
// template<int MaxNumOfLinesToCopy>
// __device__ __forceinline__ void CopyFromGlobalToShared_BinaryAct(
//     // uint32_t* SPTR, 
//                                                         uint32_t __restrict__ (*SPTR)[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1], 
//                                                         const uint4* GPTR,
//                                                         const int GlobalStride, // N_Global
//                                                         bool pred_guard = true) {
//     #ifdef DEBUG_MODE
//         static_assert(MaxNumOfLinesToCopy/WARP_SIZE % 16 == 0);
//     #endif
//     int lane_id      = threadIdx.x % WARP_SIZE;
//     half* SPTR_HALF = reinterpret_cast<half*>(SPTR);
//     const half* GPTR_HALF = reinterpret_cast<const half*>(GPTR);
//     SPTR_HALF += lane_id*8;
//     GPTR_HALF += lane_id*(GlobalStride/8/2);
//     #pragma unroll
//     for(int i=0; i<MaxNumOfLinesToCopy/32; i++) {
//         cp_async<16>( SPTR_HALF, GPTR_HALF, pred_guard); // 每次 copy 16B, 32线程，总共32*16B=512B
//         SPTR_HALF += 256;   // Forward 512 Bytes
//         GPTR_HALF += 256;   // Forward 512 Bytes
//     }

// }


// template<int SMEM_SIZE_IN_BYTES_PER_WARP>
template<int MaxNumOfLinesToCopy, int BLOCK_WARPS> // 128, 4
__device__ __forceinline__ void CopyFromGlobalToShared_B(uint32_t __restrict__ (*SharedPTR)[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1], 
                                                        const uint32_t* GlobalPTR,
                                                        const int         GlobalStride, // K_GLOBAL
                                                        const int         NumOfLinesLeft,        // To support arbitrary N dimensions.
                                                        bool              Pred = true) 
{
    #ifdef DEBUG_MODE
        static_assert(SMEM_SIZE_IN_BYTES_PER_WARP/WARP_SIZE % 16 == 0);
    #endif

    // static parameters: 1 Group (8 Threads) can copy one line 64 FP16 each time  (8*16/2=64)
    //  1 Group (1 Threads) can copy one line 128b (4 uint32 = 16B) each time 
    const int NumOfThreads  = BLOCK_WARPS * WARP_SIZE;  // 4*32=128
    const int NumOfGroups   = NumOfThreads / 4; // 128/4=32
    const int MaxIteration  = (MaxNumOfLinesToCopy-1) / NumOfGroups + 1;  // (128-1) / 32 + 1 = 4
    // runtime variables

    const int line_id = threadIdx.x % WARP_SIZE; // 32
    // const int line_offset   = (threadIdx.x%4) * 4; // 0, 4, 8, 12 每个thread差了 4 uint32，就是16B
    // const int line_offset = (threadIdx.x % 16) * 4;

    // PTR for source global memory and target shared memory
    GlobalPTR += line_id * GlobalStride/32;
    SharedPTR += line_id;

    const half* GPTR_HALF = reinterpret_cast<const half*>(GlobalPTR);
    half* SPTR_HALF = reinterpret_cast<half*>(SharedPTR);

    #pragma unroll
    for (int i = 0; i < MaxIteration; i++) { // 16B*32=512

        bool AsyncCopyPred = (line_id+i*NumOfGroups) < NumOfLinesLeft && Pred;
        cp_async<16>((SPTR_HALF), GPTR_HALF, AsyncCopyPred); // 16B
        //
        // GlobalPTR += NumOfGroups * GlobalStride/32; // + 16*K/32
        // SharedPTR += NumOfGroups; // + 16 in dim0
        GPTR_HALF += GlobalStride/32*2*32; 
        SPTR_HALF += 256; // 32*16B/2  half
    }
}

    // const int NumOfThreads  = BLOCK_WARPS * WARP_SIZE;  // 4*32 // noqa
    // const int NumOfGroups   = NumOfThreads / 8;  // 4*32/8 = 16 // 
    // const int MaxIteration  = (MaxNumOfLinesToCopy-1) / NumOfGroups + 1;  // (128-1) / 16 + 1 = 8

    // // runtime variables   
    // const int line_id       = threadIdx.x / 2;   // noqa
    // const int line_offset   = (threadIdx.x%2) * 8; // 0, 8

    // // const int line_id       = threadIdx.x / 4;   // noqa
    // // const int line_offset   = (threadIdx.x%8) * 16;  // noqa

    // // int lane_id      = threadIdx.x % WARP_SIZE;
    // half* SPTR_HALF = reinterpret_cast<half*>(SPTR);
    // const half* GPTR_HALF = reinterpret_cast<const half*>(GPTR);

    // GPTR_HALF += (line_id * GlobalStride/8/2 + line_offset);   // 
    // SPTR_HALF += line_id * (WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1)*2;  // 第几行
    // // GlobalPTR += line_id * GlobalStride + line_offset;
    // // SharedPTR += line_id;

    // #pragma unroll
    // for(int i=0; i<MaxNumOfLinesToCopy/WARP_SIZE; i++) {
    //     bool AsyncCopyPred = (line_id+i*NumOfGroups) < NumOfLinesLeft && Pred;
    //     cp_async<16>( (SPTR_HALF+line_offset), GPTR_HALF, AsyncCopyPred);  // 16B, 32thread，总共32*16B=512B
    //     // noqa: 需要增大TILE_N（MaxIteration>1）才能check是否正确，包括更换SPTR_HALF和GPTR_HALF的数据类型，不需要half，直接uint4即可
    //     GPTR_HALF += NumOfGroups * GlobalStride / 16;  // 16*Kglobal/16 = 256 half，地址差512
    //     SPTR_HALF += NumOfGroups * (WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1)*2;
    // }

/* 
 * Copying 64 Quant Scales (FP16) from global memory to shared memory.
 */
__device__ __forceinline__ void CopyFromGlobalToShared_Scales(half* SPTR_QuantScales,
                                                              const half* GPTR_A_Scales) {
    int lane_id         = threadIdx.x % WARP_SIZE;
    int Offset_Shared   = lane_id*2; 
    int Offset_Global   = lane_id/4 + (lane_id%4)*16;
    for(int i=0; i<2; i++)  SPTR_QuantScales[Offset_Shared+i] = GPTR_A_Scales[Offset_Global+i*8];
}

/* 
 * (1) Copying X  rows * 64 columns of FP16 values, originally in row    major
 * (2) Copying 64 rows * X  columns of FP16 values, originally in column major
 * 16 Bytes per thread -> 512 Bytes per WARP = 4 line per WARP = 1 line per 8 Threads
 */ // 512B / ((64 column/row of FP16) * 2B/half) = 4 column/row
template<int MaxNumOfLinesToCopy, int BLOCK_WARPS>
__device__ __forceinline__ void CopyFromGlobalToShared(half __restrict__ (*SharedPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                                       const half*       GlobalPTR,
                                                       const int         GlobalStride,  // K_Global
                                                       const int         NumOfLinesLeft,        // To support arbitrary N dimensions.
                                                       bool              Pred = true) {
    // static parameters: 1 Group (8 Threads) can copy 1 line (64 FP16) each time
    const int NumOfThreads  = BLOCK_WARPS * WARP_SIZE;
    const int NumOfGroups   = NumOfThreads / 8; // 16
    const int MaxIteration  = (MaxNumOfLinesToCopy-1) / NumOfGroups + 1; // (64-1) / 16 + 1 = 4
    // runtime variables   
    const int line_id       = threadIdx.x / 8;
    const int line_offset   = (threadIdx.x%8) * 8;
    // PTR for source global memory and target shared memory
    GlobalPTR += line_id * GlobalStride + line_offset;
    SharedPTR += line_id;
    #pragma unroll
    for (int i = 0; i < MaxIteration; i++) { // 4*16B*32 = 2048B = 1024 half
        bool AsyncCopyPred = (line_id+i*NumOfGroups) < NumOfLinesLeft && Pred;
        cp_async<16>( &(*SharedPTR)[line_offset], GlobalPTR, AsyncCopyPred);
        //
        GlobalPTR += NumOfGroups * GlobalStride;
        SharedPTR += NumOfGroups;
    }
}


template<int SMEM_SIZE_IN_BYTES_PER_WARP> // 2048 = 128*16B = 128*8 half
__device__ __forceinline__ void CopyFromGlobalToShared_W(half __restrict__ (*SharedPTR),
                                                       const half*       GlobalPTR,
                                                       const int         GlobalStride,  // K_Global
                                                       const int         NumOfLinesLeft,        // To support arbitrary N dimensions.
                                                       bool              pred_guard = true) {
    // runtime variables   
    const int line_id       = threadIdx.x % WARP_SIZE;
    // PTR for source global memory and target shared memory
    // GlobalPTR += line_id*GlobalStride;  
    GlobalPTR += line_id/4*GlobalStride + line_id % 4 * WARP_SIZE /* column */; // 32 half / thread
    SharedPTR += line_id*(SMEM_SIZE_IN_BYTES_PER_WARP/WARP_SIZE/2);  // 32 half / thread
    #pragma unroll
    for (int i = 0; i < SMEM_SIZE_IN_BYTES_PER_WARP/WARP_SIZE/16; i++) {  // 4 iter
        // bool AsyncCopyPred = (line_id+i*NumOfGroups) < NumOfLinesLeft && Pred;
        cp_async<16>( SharedPTR, GlobalPTR, pred_guard);
        //
        GlobalPTR += 8; 
        SharedPTR += 8; 
    }
}

template<int MaxNumOfLinesToCopy, int BLOCK_WARPS> // 128, 4
__device__ __forceinline__ void CopyFromGlobalToShared_A(half __restrict__ (*SharedPTR)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1],
                                                       const half*       GlobalPTR,
                                                       const int         GlobalStride,  // K_Global
                                                       const int         NumOfLinesLeft,        // To support arbitrary N dimensions.
                                                       bool              Pred = true) {
    // static parameters: 1 Group (8 Threads) can copy 1 line (64 FP16) each time
    const int NumOfThreads  = BLOCK_WARPS * WARP_SIZE; // 4*32
    const int NumOfGroups   = NumOfThreads / 4; // 128/4 = 32
    const int MaxIteration  = (MaxNumOfLinesToCopy-1) / NumOfGroups + 1; // (128-1) / 32 + 1 = 4

    // runtime variables   
    const int block_id = threadIdx.x / WARP_SIZE; // [0,1,2,3] // 
    const int line_id = threadIdx.x % WARP_SIZE;  // [0~31] 
    const int line_offset = (threadIdx.x%16) * 8;  // 0~120，8B per thread
    
    GlobalPTR += (line_id / 16 + block_id * 8) * GlobalStride + line_offset;       // [0,1]+[0, 8, 16, 24] * K_Global + [0~120]
    SharedPTR += line_id / 16 + block_id * 8;                 // [0,1] + [0, 8, 16, 24] , 每个线程copy 128 half
    // half* SPTR_HALF = reinterpret_cast<half*>(SharedPTR);

    #pragma unroll
    for (int i = 0; i < MaxIteration; i++) { // 4*16B*32 = 1024 half
        // bool AsyncCopyPred = (line_id+i*NumOfGroups) < NumOfLinesLeft && Pred;
        cp_async<16>( &(*SharedPTR)[line_offset], GlobalPTR, Pred); // 512B 
        //
        GlobalPTR += GlobalStride*2; // 2行=2*128half = 2*128*2B = 512B
        SharedPTR += 2; // 128 half * 2 = 256 half = 512B = 32*16B
    }
}




// /* 
//  * (1) Copying X  rows * 64 columns of FP16 values, originally in row    major
//  * (2) Copying 64 rows * X  columns of FP16 values, originally in column major
//  * 16 Bytes per thread -> 512 Bytes per WARP = 4 line per WARP = 1 line per 8 Threads
//  */
// template<int MaxNumOfLinesToCopy, int BLOCK_WARPS>
// __device__ __forceinline__ void CopyFromGlobalToShared_BinaryAct(
//                                                     //    half __restrict__ (*SharedPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
//                                                        uint32_t*           SharedPTR,   // 0x7fffd5001400
//                                                        const uint4*        GlobalPTR,
//                                                        const int         GlobalStride,
//                                                        const int         NumOfLinesLeft,        // To support arbitrary N dimensions.
//                                                        bool              Pred = true) {
//     // static parameters: 1 Group (8 Threads) can copy 1 line (64 FP16) each time
//     const int NumOfThreads  = BLOCK_WARPS * WARP_SIZE; // 4*32
//     const int NumOfGroups   = NumOfThreads / 8;   // 16
//     const int MaxIteration  = (MaxNumOfLinesToCopy-1) / NumOfGroups + 1; // 4
//     // runtime variables   
//     const int line_id       = threadIdx.x / 8;
//     const int line_offset   = (threadIdx.x%8) * 8;
//     // PTR for source global memory and target shared memory

//     half* SPTR_HALF = reinterpret_cast<half*>(SharedPTR);
//     const half* GPTR_HALF = reinterpret_cast<const half*>(GlobalPTR);

//     // GlobalPTR += line_id * GlobalStride + line_offset;
//     // SharedPTR += line_id;
//     GPTR_HALF += line_id * GlobalStride + line_offset;
//     SPTR_HALF += line_id;

//     #pragma unroll
//     for (int i = 0; i < MaxIteration; i++) {
//         bool AsyncCopyPred = (line_id+i*NumOfGroups) < NumOfLinesLeft && Pred;
//         // cp_async<16>( &(*SharedPTR)[line_offset], GlobalPTR, AsyncCopyPred);
//         // cp_async<16>( &(*SharedPTR)[line_offset], GPTR_HALF, AsyncCopyPred);
//         cp_async<16>( (SPTR_HALF+line_offset), GPTR_HALF, AsyncCopyPred);
//         //
//         // GlobalPTR += NumOfGroups * GlobalStride;
//         // SharedPTR += NumOfGroups;
//         GPTR_HALF += NumOfGroups * GlobalStride;
//         SPTR_HALF += NumOfGroups;
//     }
// }

#endif