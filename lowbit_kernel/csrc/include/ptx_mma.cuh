/***************************************************************************
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#ifndef PTX_MMA_CUH
#define PTX_MMA_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <assert.h>
#include "configs.h"


#ifdef PIPELINE_LEVEL_SMEM
template <typename TilingConfig>
__device__ __forceinline__ void B_FromSharedToReg(uint32_t  __restrict__    Reg[][4],
                                                  half      __restrict__    (*read_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                                  int                       slice_id) {
    #ifdef DEBUG_MODE
        static_assert( (TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0) );
    #endif
    
    const int   warpId  = threadIdx.x / WARP_SIZE;
    int         lane_id = threadIdx.x % WARP_SIZE;
    int WARP_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_col = TilingConfig::WARP_COL_MMA_TENSORS * MMA_8 * WARP_j;   // each warp may start from reading warp_start_col'th column of the B tile in shared memory
    #ifdef DEBUG_MODE
        assert( warp_start_col==0 );
    #endif    

    int col = (lane_id%8) + (lane_id/16)*8;
    int row = (lane_id%16) / 8 * 8;
    uint32_t smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&read_SPTR[warp_start_col+col][slice_id*MMA_16 + row])); 
    if(TilingConfig::WARP_COL_MMA_TENSORS==1) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(Reg[0][0]), "=r"(Reg[0][1])
                     : "r"(smem_local_ptr));
    }
    else {
        #pragma unroll
        for (int i = 0; i < TilingConfig::WARP_COL_MMA_TENSORS/2; i++)
        {
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(Reg[i][0]), "=r"(Reg[i][1]), "=r"(Reg[i][2]), "=r"(Reg[i][3])
                         : "r"(smem_local_ptr));
            smem_local_ptr += 16 * (WARP_K+PADDING_SHARED_MEM_FOR_B_8) * sizeof(half);
        }
    }
}

template <typename TilingConfig>
__device__ __forceinline__ void CopyFromSharedToRegister_BinaryAct2(uint32_t  __restrict__    Reg[][4],
                                                uint32_t      __restrict__    (*read_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                                int                       slice_id) {
    #ifdef DEBUG_MODE
        static_assert( (TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0) );
    #endif
    
    const int   warpId  = threadIdx.x / WARP_SIZE;
    int         lane_id = threadIdx.x % WARP_SIZE;
    int WARP_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_col = TilingConfig::WARP_COL_MMA_TENSORS * MMA_8 * WARP_j;   // each warp may start from reading warp_start_col'th column of the B tile in shared memory
    #ifdef DEBUG_MODE
        assert( warp_start_col==0 );
    #endif    

    // int col = (lane_id%8) + (lane_id/16)*8; 
    // int row = (lane_id%16) / 8 * 8;
    // int col = (lane_id%8) + (lane_id/16)*8; 
    int row = (lane_id%32) / 8 * 8;
    // int row = lane_id; 
    int col = (lane_id) >> 5 + (lane_id%8); // all 0

    // uint32_t smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&read_SPTR[warp_start_col+col][slice_id*MMA_16 + row])); 
    uint32_t smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&read_SPTR[warp_start_col+col][slice_id*MMA_16 + row])); 
    if(TilingConfig::WARP_COL_MMA_TENSORS==1) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(Reg[0][0]), "=r"(Reg[0][1])
                     : "r"(smem_local_ptr));
    }
    else {
        #pragma unroll
        for (int i = 0; i < TilingConfig::WARP_COL_MMA_TENSORS/2; i++)
        {
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(Reg[i][0]), "=r"(Reg[i][1]), "=r"(Reg[i][2]), "=r"(Reg[i][3])
                         : "r"(smem_local_ptr));
            smem_local_ptr += 16 * (WARP_K+PADDING_SHARED_MEM_FOR_B_8) * sizeof(half);
        }
    }
}

#else
// Debug: Whether ldmatrix.trans is required???
// B is in column-major
template <typename TilingConfig>
__device__ __forceinline__ void B_FromSharedToReg(uint32_t  __restrict__    Reg[][4],
                                                  half      __restrict__    (*read_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8],
                                                  int                       k_offset) {
    #ifdef DEBUG_MODE
        static_assert( (TilingConfig::WARP_COL_MMA_TENSORS==1) || (TilingConfig::WARP_COL_MMA_TENSORS%2==0) );
    #endif
    
    const int   warpId  = threadIdx.x / WARP_SIZE;
    int         lane_id = threadIdx.x % WARP_SIZE;
    int WARP_j = warpId % TilingConfig::BLOCK_COL_WARPS;
    int warp_start_col = TilingConfig::WARP_COL_MMA_TENSORS * MMA_8 * WARP_j;   // each warp may start from reading warp_start_col'th column of the B tile in shared memory
    #ifdef DEBUG_MODE
        assert( warp_start_col==0 );
    #endif    

    int col = (lane_id%8) + (lane_id/16)*8;
    int row = (lane_id%16) / 8 * 8;
    uint32_t smem_local_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&read_SPTR[warp_start_col+col][k_offset + row]));
    if(TilingConfig::WARP_COL_MMA_TENSORS==1) {
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(Reg[0][0]), "=r"(Reg[0][1])
                     : "r"(smem_local_ptr));
    }
    else {
        #pragma unroll
        for (int i = 0; i < TilingConfig::WARP_COL_MMA_TENSORS/2; i++)
        {
            asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(Reg[i][0]), "=r"(Reg[i][1]), "=r"(Reg[i][2]), "=r"(Reg[i][3])
                         : "r"(smem_local_ptr));
            smem_local_ptr += 16 * (WARP_K+PADDING_SHARED_MEM_FOR_B_8) * sizeof(half);
        }
    }
}
#endif

__device__ __forceinline__ void
MMA_FP16_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__ *a, uint32_t __restrict__ *b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5, %6, %7 },"
                 "{ %8, %9 },"
                 "{ %10, %11, %12, %13 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                   "r"(b[0]), "r"(b[1]),
                   "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}


__device__ __forceinline__ void
MMA_B1B1_M16N8K256(uint32_t __restrict__ c[], uint32_t __restrict__ *a, uint32_t __restrict__ *b)
{
//     uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
//     uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

//     int const *C = reinterpret_cast<int const *>(&c);
//     int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), 
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}


__device__ __forceinline__ void
MMA_B1B1_M8N8K128_AND(uint32_t __restrict__ c[], uint32_t __restrict__ *a, uint32_t __restrict__ *b)
{

    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0,%1}, "
        "{%2}, "
        "{%3}, "
        "{%4,%5};\n"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(a[0]), 
          "r"(b[0]), 
          "r"(c[0]), "r"(c[1]));
}

__device__ __forceinline__ void
MMA_B1B1_M8N8K128_AND(int32_t __restrict__ c[], uint32_t __restrict__ *a, uint32_t __restrict__ *b)
{

    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0,%1}, "
        "{%2}, "
        "{%3}, "
        "{%4,%5};\n"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(a[0]), 
          "r"(b[0]), 
          "r"(c[0]), "r"(c[1]));
}

__device__ __forceinline__ void
K_SUB_2_XORPOP(int32_t __restrict__ c[], int32_t __restrict__ tmp[], int32_t __restrict__ K)
{

    asm volatile("{                     \n\t"
            ".reg .s32      a;          \n\t"
            "mul.lo.s32     a,2,%1;     \n\t"   // a = 2 * tmp[0]
            "sub.s32        a,%2,a;     \n\t"   // a = K - a = K - 2 * tmp[0]
            "add.s32        %0,%0,a;    \n\t"   // c[0] = c[0] + a
            "}"
            : "+r"(c[0]) : "r"(tmp[0]), "r"(K));
    asm volatile("{                     \n\t"
            ".reg .s32      a;          \n\t"
            "mul.lo.s32     a,2,%1;     \n\t"   // a = 2 * tmp[1]
            "sub.s32        a,%2,a;     \n\t"   // c[0] = K - a = K - 2 * tmp[1]
            "add.s32        %0,%0,a;    \n\t"   // c[0] = c[0] + a
            "}"
            : "+r"(c[1]) : "r"(tmp[1]), "r"(K));

}

__device__ __forceinline__ void
MMA_B1B1_M8N8K128_XOR(int32_t __restrict__ c[], uint32_t __restrict__ *a, uint32_t __restrict__ *b)
{   
    // asm volatile("mov.b32 %0,0x0; \n\t": "=r"(c[0]));
    // asm volatile("mov.b32 %0,0x0; \n\t": "=r"(c[1]));

    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
        "{%0,%1}, "
        "{%2}, "
        "{%3}, "
        "{%4,%5};\n"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(a[0]), 
          "r"(b[0]),
          "r"(0x0), "r"(0x0));
        //   "r"(c[0]), "r"(c[1]));
}

__device__ __forceinline__ void
ATTN_MM_PTX(int32_t __restrict__ c[], uint32_t __restrict__ *a, uint32_t __restrict__ *b)
{   

   /*  // uint32_t notA;
    // uint32_t notB;

    // asm volatile("not.b32   %0,%1; \n\t": "=r"(notA) : "r"(a[0]));
    // asm volatile("not.b32   %0,%1; \n\t": "=r"(notB) : "r"(b[0]));

    uint32_t tmp1[2];

    // asm volatile(  // notA & B
    //     "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
    //     "{%0,%1}, "
    //     "{%2}, "
    //     "{%3}, "
    //     "{%4,%5};\n"
    //     : "=r"(tmp1[0]), "=r"(tmp1[1])
    //     : "r"(notA), 
    //       "r"(b[0]), 
    //       "r"(0x0), "r"(0x0));

    // // asm volatile("sub.s32   %0,0x0,%0;\n\t":"+r"(tmp1[0]));  // - (notA & B)
    // // asm volatile("sub.s32   %0,0x0,%0;\n\t":"+r"(tmp1[1]));
    uint32_t tmp2[2];

    asm volatile(  // notA & notB
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0,%1}, "
        "{%2}, "
        "{%3}, "
        "{%4,%5};\n"
        : "=r"(tmp2[0]), "=r"(tmp2[1])
        : "r"(a[0]), 
          "r"(b[0]), 
          "r"(0x0), "r"(0x0));

    asm volatile("not.b32   %0,%1; \n\t": "=r"(tmp1[0]) : "r"(tmp2[0]));
    asm volatile("not.b32   %0,%1; \n\t": "=r"(tmp1[1]) : "r"(tmp2[1]));

    asm volatile("add.s32   %0,%0,%1; \n\t" : "+r"(c[0]) : "r"(tmp2[0]));  // 
    asm volatile("sub.s32   %0,%0,%1; \n\t" : "+r"(c[0]) : "r"(tmp1[0]));  // 
    asm volatile("add.s32   %0,%0,%1; \n\t" : "+r"(c[1]) : "r"(tmp2[1]));  // 
    asm volatile("sub.s32   %0,%0,%1; \n\t" : "+r"(c[1]) : "r"(tmp1[1]));  //  */


    uint32_t notA;
    uint32_t notB;

    asm volatile("not.b32   %0,%1; \n\t": "=r"(notA) : "r"(a[0]));
    asm volatile("not.b32   %0,%1; \n\t": "=r"(notB) : "r"(b[0]));

    uint32_t tmp1[2];

    asm volatile(  // notA & B
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0,%1}, "
        "{%2}, "
        "{%3}, "
        "{%4,%5};\n"
        : "=r"(tmp1[0]), "=r"(tmp1[1])
        : "r"(notA), 
          "r"(b[0]), 
          "r"(0x0), "r"(0x0));

    // asm volatile("sub.s32   %0,0x0,%0;\n\t":"+r"(tmp1[0]));  // - (notA & B)
    // asm volatile("sub.s32   %0,0x0,%0;\n\t":"+r"(tmp1[1]));
    uint32_t tmp2[2];

    asm volatile(  // notA & notB
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc "
        "{%0,%1}, "
        "{%2}, "
        "{%3}, "
        "{%4,%5};\n"
        : "=r"(tmp2[0]), "=r"(tmp2[1])
        : "r"(notA), 
          "r"(notB), 
          "r"(0x0), "r"(0x0));

    asm volatile("add.s32   %0,%0,%1; \n\t" : "+r"(c[0]) : "r"(tmp2[0]));  // 
    asm volatile("sub.s32   %0,%0,%1; \n\t" : "+r"(c[0]) : "r"(tmp1[0]));  // 
    asm volatile("add.s32   %0,%0,%1; \n\t" : "+r"(c[1]) : "r"(tmp2[1]));  // 
    asm volatile("sub.s32   %0,%0,%1; \n\t" : "+r"(c[1]) : "r"(tmp1[1]));  // 
}



__device__ __forceinline__ void
SIGN_32_HALF_TO_UINT32(uint32_t __restrict__ d[], half __restrict__ *a)
{
    int16_t const *A = reinterpret_cast<int16_t const *>(a);

    asm volatile("mov.b32 %0,0x0; \n\t": "=r"(d[0]));
    // uint16_t tmp1 = 0x8765;
    // uint16_t tmp2 = 0x4321;

    for (uint32_t i = 0; i < 16; i++) { // num of half
        /* asm volatile(
            "{      \n\t"
            // ".reg .u16 m,n;                 \n\t"
            ".reg .u32 a,b,c,d;             \n\t" 
            // "mov.u16 m,%1;                  \n\t"
            // "mov.u16 n,%2;                  \n\t"
            // "mov.b32 a,{m,n};               \n\t" // pack 2 half to u32
            "mov.b32 a,{%1,%2};             \n\t" // pack 2 half to u32
            "and.b32 b,a,0x00008000;        \n\t"  // sign of %2
            "shr.b32 c,b,%3;                \n\t"  // right shift %3 bits and save to c
            "and.b32 b,a,0x80000000;        \n\t"  // sign of %1    
            "shr.b32 d,b,%3;                \n\t"  // right shift %4 bits and save to d
            "or.b32  d,d,c;                 \n\t"
            // "or.b32  %0,%0,d;               \n\t"
            "mov.b32 c,%0;                  \n\t"
            "or.b32  d,d,c;                 \n\t"
            "mov.b32 %0,d;                  \n\t"    
            "}"
            : "=r"(d[0]) : "h"(A[i]), "h"(A[i+16]), "r"(i)); */

            asm volatile(
                "{      \n\t"
                ".reg .b32 a;                      \n\t" 
                "mov.b32 a,{%1,%2};                \n\t"  // pack 2 half to u32, order: %2%1
                "and.b32 a,a,0x80008000;           \n\t"  // sign of two half
                "shr.b32 a,a,%3;                   \n\t"  // a = a >> i
                "or.b32  %0,%4,a;                  \n\t"  // d[0] = d[0] | a
                "}"
                : "=r"(d[0]) : "h"(A[i+16]), "h"(A[i]), "r"(i), "r"(d[0]));
    }
    /* for (uint32_t i = 16; i < 32; i++) { // num of half
        asm volatile("and.b16 "
                     "{ %0}, "
                     "{ %1, %2};\n\t"
                    : "=h"(sign_16[i])
                    : "h"(A[i]), "h"(b));
        asm volatile("shr.b32 "
                    "{%0}, "
                    "{%1, %2};\n\t"
                    : "=r"(sign_32[i])
                    : "h"(sign_16[i]), "r"(i-16));
        asm volatile("or.b32 "
                     "{ %0}, "
                     "{ %1, %2};\n\t"
                    : "=r"(d[0])
                    : "r"(d[0]), "r"(sign_32[i]));
    } */

}

#endif

