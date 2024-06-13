#include <stdio.h>
#include <assert.h>
#include <vector>

using namespace std;

void Padding_8_FP6_To_8_Bytes(unsigned char Padded_FP6[], unsigned char* FP6_Array) // padding 0 to the lowerest bit location
{
    Padded_FP6[0] = FP6_Array[0] & 0xfc;
    Padded_FP6[1] = (FP6_Array[0]<<6) | ((FP6_Array[1]>>2) & 0xfc);
    Padded_FP6[2] = (FP6_Array[1]<<4) | ((FP6_Array[2]>>4) & 0xfc );
    Padded_FP6[3] = FP6_Array[2]<<2;
    Padded_FP6[4] = FP6_Array[3] & 0xfc;
    Padded_FP6[5] = (FP6_Array[3]<<6) | ((FP6_Array[4]>>2) & 0xfc);
    Padded_FP6[6] = (FP6_Array[4]<<4) | ((FP6_Array[5]>>4) & 0xfc);
    Padded_FP6[7] = FP6_Array[5]<<2;
}

unsigned char Extract_2_Bits_From_4_PaddedFP6(unsigned char B1, unsigned char B2, unsigned char B3, unsigned char B4)
{
    unsigned char out;
    out = (B1&0xc0) | ( (B2&0xc0) >> 2 ) | ( (B3&0xc0) >> 4 ) | ( (B4&0xc0) >> 6 );
    return out;
}

unsigned char Extract_4_Bits_From_2_PaddedFP6(unsigned char B1, unsigned char B2)   // The highest two bits are already extracted by Extract_2_Bits_From_4_PaddedFP6();
{
    unsigned char out;
    out = ( (B1<<2) & 0xf0 ) | ( (B2>>2) & 0x0f );
    return out;
}

// dealing with 4 1*8 blocks of FP6
void Assign_32_FP6_To_4_Thread(vector<unsigned char> Seg_2bit[], vector<unsigned char> Seg_4bit[], unsigned char* PTR_1, unsigned char* PTR_2, unsigned char* PTR_3, unsigned char* PTR_4) 
{
    unsigned char Padded_8_FP8[4][8];
    Padding_8_FP6_To_8_Bytes(Padded_8_FP8[0], PTR_1);
    Padding_8_FP6_To_8_Bytes(Padded_8_FP8[1], PTR_2);
    Padding_8_FP6_To_8_Bytes(Padded_8_FP8[2], PTR_3);
    Padding_8_FP6_To_8_Bytes(Padded_8_FP8[3], PTR_4);
    //
    unsigned char Seg1_Byte1_T[4];
    unsigned char Seg1_Byte2_T[4];
    unsigned char Seg2_Byte1_T[4];
    unsigned char Seg2_Byte2_T[4];
    unsigned char Seg2_Byte3_T[4];
    unsigned char Seg2_Byte4_T[4];
    for(int t=0; t<4; t++)
    {
        Seg1_Byte1_T[t] = Extract_2_Bits_From_4_PaddedFP6(Padded_8_FP8[0][0+t*2], Padded_8_FP8[0][1+t*2], Padded_8_FP8[1][0+t*2], Padded_8_FP8[1][1+t*2]);
        Seg1_Byte2_T[t] = Extract_2_Bits_From_4_PaddedFP6(Padded_8_FP8[2][0+t*2], Padded_8_FP8[2][1+t*2], Padded_8_FP8[3][0+t*2], Padded_8_FP8[3][1+t*2]);
        Seg2_Byte1_T[t] = Extract_4_Bits_From_2_PaddedFP6(Padded_8_FP8[0][0+t*2], Padded_8_FP8[0][1+t*2]);
        Seg2_Byte2_T[t] = Extract_4_Bits_From_2_PaddedFP6(Padded_8_FP8[1][0+t*2], Padded_8_FP8[1][1+t*2]);
        Seg2_Byte3_T[t] = Extract_4_Bits_From_2_PaddedFP6(Padded_8_FP8[2][0+t*2], Padded_8_FP8[2][1+t*2]);
        Seg2_Byte4_T[t] = Extract_4_Bits_From_2_PaddedFP6(Padded_8_FP8[3][0+t*2], Padded_8_FP8[3][1+t*2]);
    }
    //
    for(int t=0; t<4; t++)
    {
        Seg_2bit[t].push_back(Seg1_Byte1_T[t]);
        Seg_2bit[t].push_back(Seg1_Byte2_T[t]);
        Seg_4bit[t].push_back(Seg2_Byte1_T[t]);
        Seg_4bit[t].push_back(Seg2_Byte2_T[t]);
        Seg_4bit[t].push_back(Seg2_Byte3_T[t]);
        Seg_4bit[t].push_back(Seg2_Byte4_T[t]);
    }
    return;
}

void BitInterleaving_2bit(unsigned char* PTR_4Bytes)
{
    unsigned int *PTR_UINT = reinterpret_cast<unsigned int*>(PTR_4Bytes);
    unsigned int input  = *PTR_UINT;
    //
    //int order_2bit[16] = {1,5,9,13,3,7,11,15,2,6,10,14,4,8,12,16};  // pre-defined order for bit-interleaving in FP6-LLM
    int order_2bit[16] = {2,6,10,14,4,8,12,16,1,5,9,13,3,7,11,15};  // pre-defined order for bit-interleaving in FP6-LLM
    unsigned int Frags_2bit[16];    // The highest 2 bits are used to store the extracted fragments.
    for(int i=0; i<16; i++)
        Frags_2bit[i] = ( input << 2*(order_2bit[i]-1) ) & 0xc0000000;
    //
    unsigned int output = 0x00000000;
    for(int i=0; i<16; i++)
        output |= ( Frags_2bit[i] >> (i*2) );
    //
    *PTR_UINT = output;
}

void BitInterleaving_4bit(unsigned char* PTR_4Bytes)
{
    unsigned int *PTR_UINT = reinterpret_cast<unsigned int*>(PTR_4Bytes);
    unsigned int input  = *PTR_UINT;
    //
    //int order_4bit[8] = {1,5,3,7,2,6,4,8};  // pre-defined order for bit-interleaving in FP6-LLM
    int order_4bit[8] = {2,6,4,8,1,5,3,7};  // pre-defined order for bit-interleaving in FP6-LLM
    unsigned int Frags_4bit[8];    // The highest4 bits are used to store the extracted fragments.    
    for(int i=0; i<8; i++)
        Frags_4bit[i] = ( input << 4*(order_4bit[i]-1) ) & 0xf0000000;
    //
    unsigned int output = 0x00000000;
    for(int i=0; i<8; i++)
        output |= ( Frags_4bit[i] >> (i*4) );
    //
    *PTR_UINT = output;
}

/*
 * Inputs:
 * (1) unsigned char Weight_6bit [M*K*6/8]
 * Outputs:
 * (1) unsigned char Weight_2bit [M*K*2/8]
 * (2) unsigned char Weight_4bit [M*K*4/8]
 * 
 * Assumption: Weight_6bit, Weight_2bit, Weight_4bit all stored continuously in row-major.
 * 8 FP6 = 6 Bytes
 * 8 FP4 = 4 Bytes
 * 8 FP2 = 2 Bytes
 */
void weight_matrix_prepacking(int* packed_weights, int *FP6Weights, size_t M, size_t K)
{
    assert(M % 64 == 0);
    assert(K % 64 == 0);
    //
    unsigned char* Weight_6bit = reinterpret_cast<unsigned char*>(FP6Weights);
    unsigned char* Weight_2bit = reinterpret_cast<unsigned char*>(packed_weights);
    unsigned char* Weight_4bit = Weight_2bit + M*K*2/8;
    //
    vector<unsigned char> A_Segment_2bit[32];
    vector<unsigned char> A_Segment_4bit[32];
    //
    size_t BytesPerRow = K*6/8;
    // Pass-1: (1) 2+4 split; (2) assign weights to 32 threads.
    for (size_t i = 0; i < M / 64; i++)    // 
    {
        for (size_t j = 0; j < K / 16; j++)
        {
            for(size_t k=0; k<64/16; k++)  // 4
            {
                size_t row = i*64 + k*16;
                size_t col = j*16;
                unsigned char* StartPTR_1 = Weight_6bit + row*BytesPerRow + col*6/8;
                unsigned char* StartPTR_2 = StartPTR_1 + 8*BytesPerRow;
                unsigned char* StartPTR_3 = StartPTR_1 + 8*6/8;
                unsigned char* StartPTR_4 = StartPTR_2 + 8*6/8;
                // Dealing with each 16*16 blocks then...
                for(int l=0; l<8; l++) Assign_32_FP6_To_4_Thread(&A_Segment_2bit[l*4], &A_Segment_4bit[l*4], StartPTR_1+l*BytesPerRow, StartPTR_2+l*BytesPerRow, StartPTR_3+l*BytesPerRow, StartPTR_4+l*BytesPerRow);
            }
        }
    }
    // Verifying the length of 2_bit segments and 4_bit segments
    size_t BytesPerThread_2bit = M*K*2/8/32;
    size_t BytesPerThread_4bit = M*K*4/8/32;
    for(int i=0; i<32; i++)
    {
        assert(A_Segment_2bit[i].size()==BytesPerThread_2bit);
        assert(A_Segment_4bit[i].size()==BytesPerThread_4bit);
    }
    // Pass-2: Optimizing coleasced global memory access
    for(size_t i=0; i<BytesPerThread_2bit/4; i++) 
        for(int t=0; t<32; t++)
            for(int b=0; b<4; b++)
                Weight_2bit[i*128+t*4+(3-b)] = A_Segment_2bit[t][i*4+b];    // why (3-b): special byte order within a register
    for(size_t i=0; i<BytesPerThread_4bit/4; i++) 
        for(int t=0; t<32; t++)
            for(int b=0; b<4; b++)
                Weight_4bit[i*128+t*4+(3-b)] = A_Segment_4bit[t][i*4+b];    // why (3-b):special byte order within a register
    // Pass-3: Bit-level interleaving
    for(size_t i=0; i<BytesPerThread_2bit*32/4; i++)
        BitInterleaving_2bit(Weight_2bit+4*i);
    for(size_t i=0; i<BytesPerThread_4bit*32/4; i++)
        BitInterleaving_4bit(Weight_4bit+4*i);
}



/* 
input:
ori_weight: 32 half in row
output: 
packed_weights: 1 uint32
*/
void _sign_32_half_To_1_uint32(uint32_t *packed_weights, unsigned short *ori_weight) {
    #pragma unroll
    for (size_t k = 0; k<16; k++) { // 0~15
        *(packed_weights) |= ((*(ori_weight+k)) & 0x8000) << (16-k);  // 1000 0000 0000 0000 signbit?
    }
    #pragma unroll
    for (size_t k = 16; k<32; k++) { // 16~31
        *(packed_weights) |= ((*(ori_weight+k)) & 0x8000) >> (k-16);  // 1000 0000 0000 0000 signbit?
    }
}

/*
 * Inputs:
 * (1) ori_weights: half tensor of shape (M, K) 
 * Outputs:
 * (1) packed_weights: uint32 tensor of shape [M, K // 32]
 * 
 */




void bin_matrix_prepacking_to_uint32(uint32_t* packed_weights, half *ori_weights, size_t M, size_t K)  // 
{
    // assert(M % 64 == 0);
    assert(K % 128 == 0);
    union {
        half *_weights;
        unsigned short *ori_weights_char;
    };
    _weights = ori_weights;
    //
    // unsigned char* tmp_weights = reinterpret_cast<unsigned char*>(packed_weights);
    //
    // vector<unsigned char> bin_weight_in_uint32[32]; 
    //
    // size_t BytesPerRow = K*1/8;
    #pragma unroll
    for (size_t i = 0; i < M; i++)    // 
    {   
        #pragma unroll
        for (size_t j = 0; j < K/32; j++)
        {
            unsigned short *tmp_h = ori_weights_char + i*K + j*32;
            _sign_32_half_To_1_uint32(&packed_weights[i*K/32+j], tmp_h);
            // printf("[%d, %d]\n", i, j);
        }
    }
    
}


// template<typename TilingConfig, typename OutputDataType>
// static void bin_matrix_prepacking_to_uint32_cuda(cudaStream_t stream, 
//                                                 uint32_t* packed_weights, 
//                                                 const half *ori_weights, size_t M_GLOBAL, size_t K_GLOBAL)
// {
//     static size_t SHMEM_SZ = max(M_GLOBAL*K_GLOBAL+PADDING_BYTES_16);
//     cudaFuncSetAttribute(BIN_PACK_Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
//     size_t  dimN = (K_GLOBAL-1) / TilingConfig::TILE_K_BIN + 1; // (256-1)/128+1 = 2
//     size_t  dimM = M_Global / TilingConfig::TILE_M_BIN; // 256/128 = 2
//     dim3    GridDim(dimN, dimM, 1);  // 2, 2, 1
//     dim3    BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1); // 128, 1, 1
//     // dim3    BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS_BIN, 1, 1); // 32/64, 1, 1
//     // 
//     #ifdef DEBUG_MODE
//         printf("GridDim.x: %d, GridDim.y: %d, GridDim.z: %d, BlockDim.x: %d, BlockDim.y: %d, BlockDim.z: %d SHMEM_SZ: %d\n",
//                 GridDim.x, GridDim.y, GridDim.z, BlockDim.x, BlockDim.y, BlockDim.z, SHMEM_SZ);
//         printf("\n");
//     #endif
//     BIN_PACK_Kernel<<<GridDim, BlockDim, SHMEM_SZ, stream>>>
//                     (packed_weights, ori_weights, M_GLOBAL, K_GLOBAL);

// }

// template<typename TilingConfig, typename OutputDataType>
// __global__ void BIN_PACK_Kernel(uint32_t* packed_weights, const half *ori_weights, size_t M_GLOBAL, size_t K_GLOBAL) 
// {
//     // each block deal with 128x128 tensors, pack them into 128x4 uint32 (128 threads) => 1x4 uint32 each thread => 1 row each thread
//     extern __shared__ __align__(128) uint32_t smem[];
    
//       // Thread Block Mapping, considering SplitK
//     const size_t BatchID = blockIdx.y / (M_Global/TilingConfig::TILE_M_BIN); // 256/(64*4)=1, BatchID=0
//     const size_t x = blockIdx.x;                                     // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
//     const size_t y = blockIdx.y % (M_Global/TilingConfig::TILE_M_BIN);   // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
//     const size_t Tile_Start_M = y * TilingConfig::TILE_M_BIN;
//     const size_t Tile_Start_K = x * TilingConfig::TILE_K_BIN; // x*128

//     const half *Ori_GPTR = ori_weights + Tile_Start_M * K_GLOBAL + Tile_Start_K;

//     CopyFromGlobalToShared<256>(smem, Ori_GPTR);  // 128 half per thread = 256 Bytes

//     // Register Allocation 
//     constexpr int NumRegSet = 4;
//     uint32_t   a[NumRegSet];

//     CopyFromSharedToReg()

//     SIGN_32_HALF_TO_UINT32();

// }


// /* 
//  * Copying A1/A2 from global memory to shared memory.
//  * Usually 1024 or 2048 Bytes
//  */
// template<int SMEM_SIZE_IN_BYTES_PER_THREAD>
// __device__ __forceinline__ void CopyFromGlobalToShared(uint32_t* SPTR, 
//                                                         const uint4* GPTR,
//                                                         bool pred_guard = true) {
//     #ifdef DEBUG_MODE
//         static_assert(SMEM_SIZE_IN_BYTES_PER_WARP/WARP_SIZE % 16 == 0);
//     #endif
//     int lane_id      = threadIdx.x; 
//     half* SPTR_HALF = reinterpret_cast<half*>(SPTR);
//     const half* GPTR_HALF = reinterpret_cast<const half*>(GPTR);
//     SPTR_HALF += lane_id*8;  // 每个线程差8half=16B
//     GPTR_HALF += lane_id*8;
//     #pragma unroll
//     for(int i=0; i<SMEM_SIZE_IN_BYTES_PER_THREAD/16; i++) {  // 每线程 128 half，每次16half
//         cp_async<16>( SPTR_HALF, GPTR_HALF, pred_guard); 
//         SPTR_HALF += 1024;   
//         GPTR_HALF += 1024; // 128 thread * 16B / 2  = 1024 half
//     }

// }
