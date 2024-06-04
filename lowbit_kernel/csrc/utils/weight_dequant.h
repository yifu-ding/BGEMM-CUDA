#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void DeQuantMatrix_FP6_To_FP16(half* A_16bit_h, unsigned char* A_6bit_h, size_t M, size_t K, half* scale) {
    assert(M%64==0);                 // Currently, M must be a multiple of 64.
    assert(K%64==0);                 // Currently, K must be a multiple of 64.
    size_t TotalSizeInByte = M*K*6/8;
    //
    half* OutPTR = A_16bit_h;
    for(size_t i=0; i<TotalSizeInByte/3; i++) {    // 4 FP6 = 3 Bytes for each Loop
        unsigned char   B1  = A_6bit_h[i*3+0] & 0xfc;
                        B1  = (B1&0x80) | ((B1>>2)&0x1f);
        // unsigned char   B2  = (A_6bit_h[i*3+0]<<6) | ((A_6bit_h[i*3+1]>>2)&0xfc);
        unsigned char   B2  = (A_6bit_h[i*3+0]<<6) | ((A_6bit_h[i*3+1]>>2)&0xfc);
                        B2  = (B2&0x80) | ((B2>>2)&0x1f);
        unsigned char   B3  = (A_6bit_h[i*3+1]<<4) | ((A_6bit_h[i*3+2]>>4)&0xfc);
                        B3  = (B3&0x80) | ((B3>>2)&0x1f);
        unsigned char   B4  = A_6bit_h[i*3+2]<<2;
                        B4  = (B4&0x80) | ((B4>>2)&0x1f);
        half            FP1, FP2, FP3, FP4;
        unsigned char   *PTR1, *PTR2, *PTR3, *PTR4;
        PTR1 = reinterpret_cast<unsigned char*>(&FP1);
        PTR2 = reinterpret_cast<unsigned char*>(&FP2);
        PTR3 = reinterpret_cast<unsigned char*>(&FP3);
        PTR4 = reinterpret_cast<unsigned char*>(&FP4);
        PTR1[0] = 0;    PTR1[1] = B1;   // small endian for X86 CPU
        PTR2[0] = 0;    PTR2[1] = B2;
        PTR3[0] = 0;    PTR3[1] = B3;
        PTR4[0] = 0;    PTR4[1] = B4;
        OutPTR[0] = __float2half_rn ( __half2float(FP1) * 4096.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[1] = __float2half_rn ( __half2float(FP2) * 4096.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[2] = __float2half_rn ( __half2float(FP3) * 4096.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[3] = __float2half_rn ( __half2float(FP4) * 4096.0f * __half2float(scale[(4*i)/K]) );
        //
        OutPTR +=4;
    }
}

void DeQuantMatrix_B1_To_FP16(half* A_16bit_h, unsigned char* A_1bit_h, size_t M, size_t K, half* scale) {
    // assert(M%64==0);                 // Currently, M must be a multiple of 64.
    // assert(K%64==0);                 // Currently, K must be a multiple of 64.
    
    size_t TotalSizeInByte = M*K*1/8;
    //
    half* OutPTR = A_16bit_h;
    for(size_t i=0; i<TotalSizeInByte; i++) {    // 4 FP6 = 3 Bytes for each Loop, 8 bin = 1 Bytes for each loop
        // unsigned char   B1  = A_1bit_h[i*3+0] & 0xfc;  // 取前6bit
        //                 B1  = (B1&0x80) | ((B1>>2)&0x1f);  
        // unsigned char   B2  = (A_1bit_h[i*3+0]<<6) | ((A_1bit_h[i*3+1]>>2)&0xfc);
        //                 B2  = (B2&0x80) | ((B2>>2)&0x1f);
        // unsigned char   B3  = (A_1bit_h[i*3+1]<<4) | ((A_1bit_h[i*3+2]>>4)&0xfc);
        //                 B3  = (B3&0x80) | ((B3>>2)&0x1f);
        // unsigned char   B4  = A_1bit_h[i*3+2]<<2;
        //                 B4  = (B4&0x80) | ((B4>>2)&0x1f);
        unsigned char   B1  = (A_1bit_h[i] & 0x80) >> 7;
        unsigned char   B2  = (A_1bit_h[i] & 0x40) >> 6;
        unsigned char   B3  = (A_1bit_h[i] & 0x20) >> 5;
        unsigned char   B4  = (A_1bit_h[i] & 0x10) >> 4;
        unsigned char   B5  = (A_1bit_h[i] & 0x8) >> 3;
        unsigned char   B6  = (A_1bit_h[i] & 0x4) >> 2;
        unsigned char   B7  = (A_1bit_h[i] & 0x2) >> 1;
        unsigned char   B8  = A_1bit_h[i] & 0x1;

        // unsigned char   B2  = (A_1bit_h[i*3+0]<<6) | ((A_1bit_h[i*3+1]>>2)&0xfc);
        //                 B2  = (B2&0x80) | ((B2>>2)&0x1f);
        // unsigned char   B3  = (A_1bit_h[i*3+1]<<4) | ((A_1bit_h[i*3+2]>>4)&0xfc);
        //                 B3  = (B3&0x80) | ((B3>>2)&0x1f);
        // unsigned char   B4  = A_1bit_h[i*3+2]<<2;
        //                 B4  = (B4&0x80) | ((B4>>2)&0x1f);
        half            FP1, FP2, FP3, FP4, FP5, FP6, FP7, FP8; // 16bit
        unsigned char   *PTR1, *PTR2, *PTR3, *PTR4,  *PTR5,  *PTR6,  *PTR7,  *PTR8; // 8bit
        PTR1 = reinterpret_cast<unsigned char*>(&FP1);
        PTR2 = reinterpret_cast<unsigned char*>(&FP2);
        PTR3 = reinterpret_cast<unsigned char*>(&FP3);
        PTR4 = reinterpret_cast<unsigned char*>(&FP4);
        PTR5 = reinterpret_cast<unsigned char*>(&FP5);
        PTR6 = reinterpret_cast<unsigned char*>(&FP6);
        PTR7 = reinterpret_cast<unsigned char*>(&FP7);
        PTR8 = reinterpret_cast<unsigned char*>(&FP8);
        PTR1[0] = 0;    PTR1[1] = B1;   // small endian for X86 CPU
        PTR2[0] = 0;    PTR2[1] = B2;
        PTR3[0] = 0;    PTR3[1] = B3;
        PTR4[0] = 0;    PTR4[1] = B4;
        PTR5[0] = 0;    PTR5[1] = B5;
        PTR6[0] = 0;    PTR6[1] = B6;
        PTR7[0] = 0;    PTR7[1] = B7;
        PTR8[0] = 0;    PTR8[1] = B8;

        OutPTR[0] = __float2half_rn ( __half2float(FP1) * 65536.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[1] = __float2half_rn ( __half2float(FP2) * 65536.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[2] = __float2half_rn ( __half2float(FP3) * 65536.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[3] = __float2half_rn ( __half2float(FP4) * 65536.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[4] = __float2half_rn ( __half2float(FP5) * 65536.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[5] = __float2half_rn ( __half2float(FP6) * 65536.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[6] = __float2half_rn ( __half2float(FP7) * 65536.0f * __half2float(scale[(4*i)/K]) );
        OutPTR[7] = __float2half_rn ( __half2float(FP8) * 65536.0f * __half2float(scale[(4*i)/K]) );
        //
        OutPTR +=8;
    }
}