#include "configs.h"
#include "utils_gmem.cuh"
#include "utils_core.cuh"

// #define DEBUG_MODE

/*
 * C = A*B
 * A: row major with ahead-of-time layout transformation, FP6
 * B: col major, FP16
 * C: col major, FP16
 */ 
 template<typename TilingConfig, typename OutputDataType>
__global__ void QUANT_GEMM_Kernel(const uint4* Weight, const half* Scales,
                                  const half *B,  // &B= 0x7fffb080b200
                                  OutputDataType* C,
                                  const size_t M_Global, const size_t N_Global, const size_t K_Global,
                                  int Split_K) 
{
  #ifdef DEBUG_MODE
    assert(K_Global%TilingConfig::TILE_K==0);
    assert(M_Global%TilingConfig::TILE_M==0);
    assert( gridDim.y == Split_K * (M_Global/TilingConfig::TILE_M));
  #endif
  // 2+4 weight split
  const uint4* Weight1 = Weight; // 2bit frag // &Weight1=0x7fffb0800000
  const uint4* Weight2 = Weight1 + M_Global*K_Global*2/128;  // &Weight2=0x7fffb0801000// weight1 + 256 -> 地址 + 256, uint4=4B
  // Dynamic shared memory for FP16 A tiles， 128 Bytes aligned
  extern __shared__ __align__(128) half smem[];   // 0x0200
  half (*smem_array)[WARP_K+PADDING_SHARED_MEM_FOR_B_8] = reinterpret_cast<half (*)[WARP_K+PADDING_SHARED_MEM_FOR_B_8]> ( smem + (SMEM_SIZE_A1_TILE+SMEM_SIZE_A2_TILE)/2 ); // Dynamic shared memory for FP16 B tiles // 0x7fffd5006400
  __shared__ half QuantScales[64*TilingConfig::BLOCK_WARPS];  // static shared memory for quantization scales, 64 row per warp * 4 warps = 512 Bytes
  // Thread Block Mapping, considering SplitK
  const size_t BatchID = blockIdx.y / (M_Global/TilingConfig::TILE_M);
  const size_t x = blockIdx.x;                                     // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t y = blockIdx.y % (M_Global/TilingConfig::TILE_M);   // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t Tile_Start_M = y * TilingConfig::TILE_M;
  const size_t Tile_Start_N = x * TilingConfig::TILE_N;
  const size_t NumColumnToCopy = (N_Global-Tile_Start_N) < TilingConfig::TILE_N ? (N_Global-Tile_Start_N) : TilingConfig::TILE_N; // 1
  const size_t NumBlock_K = K_Global/TilingConfig::TILE_K;    // =256/64=4
  const size_t AverageNumBlock_K = NumBlock_K/Split_K; // 1 
  const size_t ExtraNumBlock_K   = NumBlock_K - AverageNumBlock_K * Split_K; // 0
  size_t NumIter = AverageNumBlock_K;
  if(BatchID<ExtraNumBlock_K)       NumIter ++;
  size_t StartBlockID_K = AverageNumBlock_K*BatchID;
  if(BatchID<ExtraNumBlock_K)       StartBlockID_K += BatchID;
  else                              StartBlockID_K += ExtraNumBlock_K;
  // Warp ID.
  const int warpId = threadIdx.x / WARP_SIZE;
  int WARP_i = warpId / TilingConfig::BLOCK_COL_WARPS;  // WARP_i: row number;  WARP_j: column number
  //int WARP_j = warpId % TilingConfig::BLOCK_COL_WARPS;
  // Global Memory Address for Matrix A (Weight) /////////////////////////////////////////////////////////////////////////
  // StartPTR for each ThreadBlock(TB)
  const uint4* TB_StartGPTR_A1 = Weight1 + (y*TilingConfig::BLOCK_ROW_WARPS)*NumBlock_K * NUM_INT4_PER_UNIT_2BIT_FRAG;  // 0x7fffb0800000
  const uint4* TB_StartGPTR_A2 = Weight2 + (y*TilingConfig::BLOCK_ROW_WARPS)*NumBlock_K * NUM_INT4_PER_UNIT_4BIT_FRAG;
  // StartPTR for each WARP.
  const uint4* WARP_StartGPTR_A1  = TB_StartGPTR_A1 + WARP_i * NumBlock_K * NUM_INT4_PER_UNIT_2BIT_FRAG;
  const uint4* WARP_StartGPTR_A2  = TB_StartGPTR_A2 + WARP_i * NumBlock_K * NUM_INT4_PER_UNIT_4BIT_FRAG;

  // StartPTR for each WARP, considering SplitK
  const size_t     WARP_Start_UnitID_K = StartBlockID_K;
  WARP_StartGPTR_A1  += WARP_Start_UnitID_K * NUM_INT4_PER_UNIT_2BIT_FRAG;
  WARP_StartGPTR_A2  += WARP_Start_UnitID_K * NUM_INT4_PER_UNIT_4BIT_FRAG;
  // Copying A tile from Global to Shared, using double-buffer //////////////////////////////////////////////////////////
  // StartSPTR for each ThreadBlock
  uint32_t* AFrag_2BIT_SPTR = reinterpret_cast<uint32_t*>(smem); // &AFrag_2BIT_SPTR = 0x7fffd5000200
  uint32_t* AFrag_4BIT_SPTR = AFrag_2BIT_SPTR+SMEM_SIZE_IN_BYTES_PER_WARP_A1/4*TilingConfig::BLOCK_WARPS*PIPELINE_LEVEL_GMEM; // &AFrag_4BIT_SPTR=0x7fffd5002400  // 8 buffers including double buffers, 12 for trible buffers 
  // StartSPTR for each WARP
  AFrag_2BIT_SPTR += warpId * SMEM_SIZE_IN_BYTES_PER_WARP_A1/4;  // 1024 / 4
  AFrag_4BIT_SPTR += warpId * SMEM_SIZE_IN_BYTES_PER_WARP_A2/4;  // 2048 / 4
  // Pre-fetch of A tile
  for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) {
    CopyFromGlobalToShared_A<SMEM_SIZE_IN_BYTES_PER_WARP_A1>(AFrag_2BIT_SPTR+i*SMEM_SIZE_IN_BYTES_PER_WARP_A1/4*4, WARP_StartGPTR_A1);
    CopyFromGlobalToShared_A<SMEM_SIZE_IN_BYTES_PER_WARP_A2>(AFrag_4BIT_SPTR+i*SMEM_SIZE_IN_BYTES_PER_WARP_A2/4*4, WARP_StartGPTR_A2);
    WARP_StartGPTR_A1 += SMEM_SIZE_IN_BYTES_PER_WARP_A1/16;  // + 1024 / 16 但是地址增加了0x400就是1024 //  0x7fffb0800000 -> 0x7fffb0800400
    WARP_StartGPTR_A2 += SMEM_SIZE_IN_BYTES_PER_WARP_A2/16;  // + 2048 / 16 地址增加了2048
  }
  // Global Memory Address for Matrix A (QuantScale) /////////////////////////////////////////////////////////////////////
  const half* TB_StartGPTR_A_Scale    = Scales + (y*TilingConfig::BLOCK_ROW_WARPS) * 64; //  64 row per warp 
  const half* WARP_StartGPTR_A_Scales = TB_StartGPTR_A_Scale + WARP_i * 64;  //  64 row per warp 
  CopyFromGlobalToShared_Scales(QuantScales+WARP_i*64, WARP_StartGPTR_A_Scales);  //  64 row per warp 
  // Copying B tile from Global to Shared, considering SplitK /////////////////////////////////////////////////////////////
  const half *BTile_GPTR = B + Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K;
  for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) {
    CopyFromGlobalToShared<TilingConfig::TILE_N, TilingConfig::BLOCK_WARPS> (smem_array+i*TilingConfig::TILE_N, BTile_GPTR, K_Global, NumColumnToCopy);
    BTile_GPTR += TilingConfig::TILE_K;     // 64 增加了128地址 // 0x7fffb080b200 -> 0x7fffb080b280
  }
  // Register Allocation for A,B, and C, Initilazed to Zeros /////////////////////////////////////////////////////////////////////
  constexpr int NumRegSets_a = WARP_ROW_MMA_TENSORS;                                                                  // 1 set = 4 registers, containing a 16*16 MMA block
  constexpr int NumRegSets_b = (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;    // 1 set = 4 registers, containing a 16*16 MMA block
#ifdef PIPELINE_LEVEL_SMEM
  uint32_t a  [NumRegSets_a * PIPELINE_LEVEL_SMEM][4];      // double/Trible buffer is used // Registers to store decompressed FP6
  uint32_t b  [NumRegSets_b * PIPELINE_LEVEL_SMEM][4];      // double/Triple buffer is used // Register to store FP16 B matrix (a slice)
#endif
  float c[NumRegSets_a * NumRegSets_b][REG_PER_THREAD_C_TENSOR_16_16];  // c:16*8的矩阵，每个线程有一个c矩阵，每个存一个fp32，总共64x64个output fp32
  for(int i=0; i<NumRegSets_a * NumRegSets_b; i++)
    for(int j=0; j<REG_PER_THREAD_C_TENSOR_16_16; j++)
      c[i][j] = 0.0f;  
  //
  cp_async_wait_all();
  __syncthreads();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  uint32_t Scales_RPTR[4]; // 4 Registers per thread for Quantization Scales
  ExtractFromSharedToReg_Scales(Scales_RPTR, QuantScales + WARP_i*64);  //  64 row per warp 
#ifdef PIPELINE_LEVEL_SMEM
  // Initializing the Software Pipeline: writing registers. ////////////////////////////////////////////////////////////////////////////////////////////////
  initialize_mma_slice<TilingConfig>(a, b, AFrag_2BIT_SPTR, AFrag_4BIT_SPTR, smem_array, Scales_RPTR);
#endif
  // The outer loop. /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #pragma unroll(1)
  for (size_t tile_id_k = 0; tile_id_k < NumIter; tile_id_k++)
  {
    // Trible-Buffer for A Tile
    uint32_t* __restrict__ read_SPTR_Frag1  = AFrag_2BIT_SPTR + ((tile_id_k+0)                        % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_IN_BYTES_PER_WARP_A1/4*4; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    uint32_t* __restrict__ read_SPTR_Frag2  = AFrag_4BIT_SPTR + ((tile_id_k+0)                        % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_IN_BYTES_PER_WARP_A2/4*4; // 2048 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    // read_SPTR_Frag1 = 0x7fffd5000600, read_SPTR_Frag2 = 0x7fffd5002a00
#ifdef PIPELINE_LEVEL_SMEM
    uint32_t* __restrict__ read2_SPTR_Frag1  = AFrag_2BIT_SPTR + ((tile_id_k+1)                        % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_IN_BYTES_PER_WARP_A1/4*4;
    uint32_t* __restrict__ read2_SPTR_Frag2  = AFrag_4BIT_SPTR + ((tile_id_k+1)                        % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_IN_BYTES_PER_WARP_A2/4*4;
    // read2_SPTR_Frag1, read2_SPTR_Frag2 = 0x7fffd5001600, 0x7fffd5004a00
#endif
    uint32_t* __restrict__ write_SPTR_Frag1 = AFrag_2BIT_SPTR + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_IN_BYTES_PER_WARP_A1/4*4; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    uint32_t* __restrict__ write_SPTR_Frag2 = AFrag_4BIT_SPTR + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * SMEM_SIZE_IN_BYTES_PER_WARP_A2/4*4; // 2048 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    // write_SPTR_Frag1, write_SPTR_Frag2 = 0x7fffd5001600, 0x7fffd5004a00
    // Trible-Buffer for B Tile
    half __restrict__ (*read_SPTR )[WARP_K+PADDING_SHARED_MEM_FOR_B_8] = smem_array + ((tile_id_k+0)  % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
#ifdef PIPELINE_LEVEL_SMEM
    half __restrict__ (*read2_SPTR )[WARP_K+PADDING_SHARED_MEM_FOR_B_8] = smem_array + ((tile_id_k+1) % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
    // read_SPTR, read2_SPTR = 0x7fffd5006200, 0x7fffd5006680   // 0x480
#endif
    half __restrict__ (*write_SPTR)[WARP_K+PADDING_SHARED_MEM_FOR_B_8] = smem_array + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N;
    // write_SPTR = 0x7fffd5006680
    //
    bool GlobalCopy = (tile_id_k+PIPELINE_LEVEL_GMEM-1) < NumIter;
    // Copying A tile from Global to Register, Bypassing L1, using double-buffer  
    CopyFromGlobalToShared_A<SMEM_SIZE_IN_BYTES_PER_WARP_A1>(write_SPTR_Frag1, WARP_StartGPTR_A1, GlobalCopy);
    CopyFromGlobalToShared_A<SMEM_SIZE_IN_BYTES_PER_WARP_A2>(write_SPTR_Frag2, WARP_StartGPTR_A2, GlobalCopy);
    // copying B tile from GlobalMemory to SharedMemory
    CopyFromGlobalToShared<TilingConfig::TILE_N, TilingConfig::BLOCK_WARPS> (write_SPTR, BTile_GPTR, K_Global, NumColumnToCopy, GlobalCopy);
    cp_async_group_commit();
  #ifdef PIPELINE_LEVEL_SMEM
    core_mma_slice<TilingConfig>(c, a, b, read_SPTR_Frag1, read_SPTR_Frag2, read_SPTR, Scales_RPTR, 1); // read_SPTR_Frag1, read_SPTR_Frag2 are different for each WARP; read_SPTR is shared among WARPs
    core_mma_slice<TilingConfig>(c, a, b, read_SPTR_Frag1, read_SPTR_Frag2, read_SPTR, Scales_RPTR, 2);  // 每个线程里的read_SPTR_Frag1，write_SPTR_Frag1地址不一样，但是存的数是一样的
    core_mma_slice<TilingConfig>(c, a, b, read_SPTR_Frag1, read_SPTR_Frag2, read_SPTR, Scales_RPTR, 3);
    // Barriers and Synchronizations
    cp_async_wait_group<PIPELINE_LEVEL_GMEM-2>();
    __syncthreads();
    core_mma_slice<TilingConfig>(c, a, b, read2_SPTR_Frag1, read2_SPTR_Frag2, read2_SPTR, Scales_RPTR, 0);
    // Updating global PTRs
    WARP_StartGPTR_A1 += SMEM_SIZE_IN_BYTES_PER_WARP_A1/16;  // 4KB/16=256 (1)/16: int4*+1 = char*+16
    WARP_StartGPTR_A2 += SMEM_SIZE_IN_BYTES_PER_WARP_A2/16;  // 8KB/16=512 (1)/16: int4*+1 = char*+16
    BTile_GPTR += TilingConfig::TILE_K;
  #else
    PipelinedCoreLoop<TilingConfig>(c, read_SPTR, read_SPTR_Frag1, read_SPTR_Frag2, Scales_RPTR); // read_SPTR_Frag1, read_SPTR_Frag2 are different for each WARP; read_SPTR is shared among WARPs
    // Updating global PTRs
    WARP_StartGPTR_A1 += SMEM_SIZE_IN_BYTES_PER_WARP_A1/16;  // 4KB/16=256 (1)/16: int4*+1 = char*+16
    WARP_StartGPTR_A2 += SMEM_SIZE_IN_BYTES_PER_WARP_A2/16;  // 8KB/16=512 (1)/16: int4*+1 = char*+16
    BTile_GPTR += TilingConfig::TILE_K;
    // Barriers and Synchronizations
    cp_async_wait_group<PIPELINE_LEVEL_GMEM-2>();
    __syncthreads();
  #endif
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Store the C fragments to shared memory.
  float (*smem_CFrag) [TilingConfig::TILE_M+PADDING_SHARED_MEM_FOR_C_4] =
        reinterpret_cast <float (*)[TilingConfig::TILE_M+PADDING_SHARED_MEM_FOR_C_4]> (smem);
  StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c);
  __syncthreads();
  // Now that shared memory contains all the D tiles, stream them to global memory.
  OutputDataType* BlockGlobalPTR = C + BatchID*(M_Global*N_Global) + Tile_Start_M + Tile_Start_N*M_Global;
  for(size_t i=warpId; i<NumColumnToCopy; i+=TilingConfig::BLOCK_WARPS)    // i-th column
    #pragma unroll
    for(size_t j=threadIdx.x%WARP_SIZE; j<TilingConfig::TILE_M; j+=WARP_SIZE) // j-th row
    {
      if constexpr (std::is_same<OutputDataType, half>::value)   BlockGlobalPTR[j+i*M_Global] = __float2half_rn(smem_CFrag[i][j]);
      else                                            BlockGlobalPTR[j+i*M_Global] = smem_CFrag[i][j];
    }
}



template<typename TilingConfig, typename OutputDataType>
__global__ void QUANT_BGEMM_Kernel(const uint4* Weight, const half* S_w, const half*  S_a, 
                                  const uint32_t *Act,  // 0x7fffa0c22200
                                  // const half *Act,
                                  OutputDataType* C,
                                  const size_t M_Global, const size_t N_Global, const size_t K_Global,
                                  int Split_K) 
{
  #ifdef DEBUG_MODE
    assert(K_Global%TilingConfig::TILE_K_BIN==0);
    assert(M_Global%TilingConfig::TILE_M_BIN==0);
    assert( gridDim.y == Split_K * (M_Global/TilingConfig::TILE_M_BIN));
  #endif
  // // 2+4 weight split
  // Dynamic shared memory for FP16 A tiles， 128 Bytes aligned
  extern __shared__ __align__(128) uint32_t smem_weight_packed_bin[];   // 0x0200
  // extern __shared__ __align__(128) half smem_act[];   // 0x0200
  uint32_t (*smem_act)[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1] = reinterpret_cast<uint32_t (*)[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1]> ( smem_weight_packed_bin + (SMEM_SIZE_TILE)/2 ); // Dynamic shared memory for FP16 Act tiles  // 0x7fffd5002400
  // __shared__ half QuantScales_w[64*TilingConfig::BLOCK_WARPS];  // static shared memory for quantization scales, 64 row per warp * 4 warps = 512 Bytes
  // __shared__ half QuantScales_a[64*TilingConfig::BLOCK_WARPS];  // static shared memory for quantization scales, 64 row per warp * 4 warps = 512 Bytes
  // Thread Block Mapping, considering SplitK
  const size_t BatchID = blockIdx.y / (M_Global/TilingConfig::TILE_M_BIN); // 256/(64*4)=1, BatchID=0
  const size_t x = blockIdx.x;                                     // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t y = blockIdx.y % (M_Global/TilingConfig::TILE_M_BIN);   // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t Tile_Start_M = y * TilingConfig::TILE_M_BIN;
  const size_t Tile_Start_N = x * TilingConfig::TILE_N_BIN; // x*128
  const size_t NumColumnToCopy = (N_Global-Tile_Start_N) < TilingConfig::TILE_N_BIN ? (N_Global-Tile_Start_N) : TilingConfig::TILE_N_BIN;
  const size_t NumBlock_K = K_Global/TilingConfig::TILE_K_BIN;    // K_Global / 128 = 2 
  const size_t AverageNumBlock_K = NumBlock_K/Split_K;
  const size_t ExtraNumBlock_K   = NumBlock_K - AverageNumBlock_K * Split_K;
  size_t NumIter = AverageNumBlock_K;
  if(BatchID<ExtraNumBlock_K)       NumIter ++;
  size_t StartBlockID_K = AverageNumBlock_K*BatchID;
  if(BatchID<ExtraNumBlock_K)       StartBlockID_K += BatchID;
  else                              StartBlockID_K += ExtraNumBlock_K;
  // Warp ID.
  const int warpId = threadIdx.x / WARP_SIZE;
  int WARP_i = warpId / TilingConfig::BLOCK_COL_WARPS;  // =warpId/1 // WARP_i: row number;  WARP_j: column number
  // Global Memory Address for Matrix weight and act /////////////////////////////////////////////////////////////////////////
  // StartPTR for each ThreadBlock(TB)
  const uint4* TB_StartGPTR_W = Weight + (y*TilingConfig::BLOCK_ROW_WARPS)*NumBlock_K*NUM_INT4_PER_UNIT_BIN; // &TB_StartGPTR_W = 0x7fffa0c00000
  const uint32_t* TB_StartGPTR_A = Act + (Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K_BIN)/32; // 0x7fffa0c22200
  
  // // StartPTR for each WARP.
  const uint4* WARP_StartGPTR_W  = TB_StartGPTR_W + WARP_i * NumBlock_K * NUM_INT4_PER_UNIT_BIN; // 32  // 0x7fffa0c00000
  // // StartPTR for each WARP, considering SplitK
  const size_t     WARP_Start_UnitID_K = StartBlockID_K;  // unsigned long = size_t = 32bits
  WARP_StartGPTR_W  += WARP_Start_UnitID_K;  // +1 相当于 +128b，因此直接加 K 维上的块序号即可
  // // Copying A tile from Global to Shared, using double-buffer //////////////////////////////////////////////////////////
  // // StartSPTR for each ThreadBlock
  uint32_t* Weight_SPTR = reinterpret_cast<uint32_t*>(smem_weight_packed_bin);   // 0x7fffd5000400
  // // StartSPTR for each WARP
  Weight_SPTR += warpId * SMEM_SIZE_IN_BYTES_PER_WARP_BIN/4;  // 512/4 uint32, 128个uint32
  // // Pre-fetch of A tile
  for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) {
    CopyFromGlobalToShared_BinaryWeight<SMEM_SIZE_IN_BYTES_PER_WARP_BIN>(Weight_SPTR+i*SMEM_SIZE_IN_BYTES_PER_WARP_BIN/4*4, WARP_StartGPTR_W, K_Global);
    // 一次 copy 128bx128b的weight，即128*16B=2048B (一小块)，也就是512个uint32
    // CopyFromGlobalToShared_A<SMEM_SIZE_PER_WARP>(Act_SPTR+i*SMEM_SIZE_PER_WARP/4*4, WARP_StartGPTR_A);
    // WARP_StartGPTR_W += SMEM_SIZE_IN_BYTES_PER_WARP_BIN/8; // 128 rows
    WARP_StartGPTR_W += 1; 
    // CopyFromGlobalToShared_B<TilingConfig::TILE_N_BIN, TilingConfig::BLOCK_WARPS>(smem_act+(i*TilingConfig::TILE_N_BIN), TB_StartGPTR_A, K_Global, NumColumnToCopy);
    CopyFromGlobalToShared_B<TilingConfig::TILE_N_BIN, TilingConfig::BLOCK_WARPS>(smem_act+(i*TilingConfig::TILE_N_BIN), TB_StartGPTR_A, K_Global, NumColumnToCopy);
    // copy 128x128b的act，取128行. 打印smem_act: ((@generic uint32_t*)*smem_act)[0]
    // TB_StartGPTR_A += COPY_ACT_LINES_128;  // TilingConfig::TILE_K_BIN=64, copy了64*4个uint32即256个uint32，地址增加了0x400=1024
    TB_StartGPTR_A += TilingConfig::TILE_K_BIN/32;  // 128/32
  }
  // Global Memory Address for Matrix QuantScale for weight and act （scale开始都放global mem）/////////////////////////////////////////////////////////////////////
  // const half* TB_StartGPTR_W_Scale    = S_w + (y*TilingConfig::BLOCK_ROW_WARPS) * 64;
  // const half* TB_StartGPTR_A_Scale    = S_a + (y*TilingConfig::BLOCK_ROW_WARPS) * 64;
  // const half* WARP_StartGPTR_A_Scales = TB_StartGPTR_W_Scale + WARP_i * 64;
  // const half* WARP_StartGPTR_B_Scales = TB_StartGPTR_A_Scale + WARP_i * 64;
  // CopyFromGlobalToShared_Scales(QuantScales_w+WARP_i*64, WARP_StartGPTR_A_Scales);
  // CopyFromGlobalToShared_Scales(QuantScales_a+WARP_i*64, WARP_StartGPTR_B_Scales);
  // // Copying Act tile from Global to Shared, considering SplitK /////////////////////////////////////////////////////////////
  // const half *BTile_GPTR = Act + Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K_BIN;
  // for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) {
  //   CopyFromGlobalToShared<TilingConfig::TILE_N_BIN, TilingConfig::BLOCK_WARPS> (smem_act+i*TilingConfig::TILE_N_BIN, BTile_GPTR, K_Global, NumColumnToCopy);
    // BTile_GPTR += TilingConfig::TILE_K_BIN;     // 64
  // }

  // Register Allocation for weight, Act, and C, Initilazed to Zeros /////////////////////////////////////////////////////////////////////
  constexpr int NumRegSets_w = 4;     //   WARP_ROW_MMA_TENSORS                                                      // 1 set = 4 registers, containing a 16*16 MMA block
  constexpr int NumRegSets_a = 4;     // (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;    // 1 set = 4 registers, containing a 16*16 MMA block
  // constexpr int NumRegSets_a = (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;    // 1 set = 4 registers, containing a 16*16 MMA block
#ifdef PIPELINE_LEVEL_SMEM
  uint32_t a  [NumRegSets_w * PIPELINE_LEVEL_SMEM][1];      // double/Trible buffer is used // Registers to store decompressed FP6
  uint32_t b  [NumRegSets_a * PIPELINE_LEVEL_SMEM][1];      // double/Triple buffer is used // Register to store FP16 Act matrix (a slice)
#endif
  // float c[NumRegSets_w * NumRegSets_a][REG_PER_THREAD_C_TENSOR_16_16]; 
  uint32_t c[NumRegSets_w * NumRegSets_a][REG_PER_THREAD_C_TENSOR_16_16]; // REG_PER_THREAD_C = 2 // 
  for(int i=0; i<NumRegSets_w * NumRegSets_a; i++) 
    for(int j=0; j<REG_PER_THREAD_C_TENSOR_16_16; j++)
      c[i][j] = 0;
  //
  cp_async_wait_all();
  __syncthreads();


  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  uint32_t Scales_RPTR_w[4]; // 4 Registers per thread for Quantization S_w
  uint32_t Scales_RPTR_a[4]; // 4 Registers per thread for Quantization S_w
  // ExtractFromSharedToReg_Scales(Scales_RPTR_w, QuantScales_w + WARP_i*64);
  // ExtractFromSharedToReg_Scales(Scales_RPTR_a, QuantScales_a + WARP_i*64);
#ifdef PIPELINE_LEVEL_SMEM
  // Initializing the Software Pipeline: writing registers. ////////////////////////////////////////////////////////////////////////////////////////////////
  int NumSlices = 4;
  int NumIterB = NumColumnToCopy/NumSlices/8;
  initialize_mma_slice_bin<TilingConfig>(a, b, Weight_SPTR, smem_act, NumIterB);
#endif
  // The outer loop. /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #pragma unroll(1)
  for (size_t tile_id_k = 0; tile_id_k < NumIter; tile_id_k++)
  {
    // Trible-Buffer for A Tile
	uint32_t* __restrict__ read_SPTR_W  = Weight_SPTR + ((tile_id_k+0)                     % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_K_BIN/32; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    // read_SPTR_W = 0x7fffd5000400
#ifdef PIPELINE_LEVEL_SMEM
    uint32_t* __restrict__ read2_SPTR_W  = Weight_SPTR + ((tile_id_k+1)                     % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_K_BIN/32;
    // read2_SPTR_W = 0x7fffd5000c00
#endif
    uint32_t* __restrict__ write_SPTR_W = Weight_SPTR + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) *TilingConfig::TILE_K_BIN/32; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    // write_SPTR_W = 0x7fffd5000c00  这个写法，write_SPTR_W在哪都行，因为weight已经全部load到reg了
    // Trible-Buffer for Act Tile
    uint32_t  __restrict__ (*read_SPTR_A )[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1] = smem_act + ((tile_id_k+0)  % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N_BIN;
// #ifdef PIPELINE_LEVEL_SMEM
    uint32_t  __restrict__ (*read2_SPTR_A )[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1] = smem_act + ((tile_id_k+1) % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N_BIN;// 0x7fffd5002d00
//  0x7fffd5002d00 - 0x7fffd5002400 = 2304 = 72*8*4B
// #endif
    uint32_t  __restrict__ (*write_SPTR_A)[WARP_K_BIN/32+PADDING_SHARED_MEM_FOR_B_1] = smem_act + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * TilingConfig::TILE_N_BIN;
    // write_SPTR = 0x7fffd5002d00
    bool GlobalCopy = (tile_id_k+PIPELINE_LEVEL_GMEM-1) < NumIter;
    // Copying A tile from Global to Register, Bypassing L1, using double-buffer   
    CopyFromGlobalToShared_BinaryWeight<SMEM_SIZE_IN_BYTES_PER_WARP_BIN>(write_SPTR_W, WARP_StartGPTR_W, K_Global);
    // copying Act tile from GlobalMemory to SharedMemory
	  CopyFromGlobalToShared_B<TilingConfig::TILE_N_BIN, TilingConfig::BLOCK_WARPS>(write_SPTR_A, TB_StartGPTR_A, K_Global, NumColumnToCopy, GlobalCopy);
    cp_async_group_commit();
  #ifdef PIPELINE_LEVEL_SMEM
    uint32_t (*a_read )[1] = a;  // 4*1
    uint32_t (*a_write)[1] = a; 
    if (tile_id_k%2==1) {a_read += NumRegSets_w; } else {a_write += NumRegSets_w;}
    core_mma_slice_bin<TilingConfig>(c, a_read, b, read_SPTR_W, read_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  1, NumIterB); // read_SPTR_W, read_SPTR_Frag2 are different for each WARP; read_SPTR is shared among WARPs
    core_mma_slice_bin<TilingConfig>(c, a_read, b, read_SPTR_W, read_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  2, NumIterB);
    core_mma_slice_bin<TilingConfig>(c, a_read, b, read_SPTR_W, read_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  3, NumIterB);
    // Barriers and Synchronizations
    cp_async_wait_group<PIPELINE_LEVEL_GMEM-2>();
    __syncthreads();
    core_mma_slice_bin<TilingConfig>(c, a_read, b, read2_SPTR_W, read2_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  0, NumIterB);
    CopyFromSharedToRegister_BinaryW<4, 1>   (a_write, read2_SPTR_W, 0);
    // Updating global PTRs
    WARP_StartGPTR_W += 1; // 128 columns
    TB_StartGPTR_A += TilingConfig::TILE_K_BIN/32;
  #else
    // did not update for bgemm
    PipelinedCoreLoop<TilingConfig>(c, read_SPTR_A, read_SPTR_W, Scales_RPTR_w, Scales_RPTR_a); // read_SPTR_W, read_SPTR_Frag2 are different for each WARP; read_SPTR is shared among WARPs
    // Updating global PTRs
    WARP_StartGPTR_W += SMEM_SIZE_PER_WARP/16;  // 4KB/16=256 (1)/16: int4*+1 = char*+16
    WARP_StartGPTR_A2 += SMEM_SIZE_IN_BYTES_PER_WARP_A2/16;  // 8KB/16=512 (1)/16: int4*+1 = char*+16
    BTile_GPTR += TilingConfig::TILE_K_BIN;
    // Barriers and Synchronizations
    cp_async_wait_group<PIPELINE_LEVEL_GMEM-2>();
    __syncthreads();
  #endif
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Store the C fragments to shared memory.
  uint32_t (*smem_CFrag) [TilingConfig::TILE_M_BIN+PADDING_SHARED_MEM_FOR_C_0] =
        reinterpret_cast <uint32_t (*)[TilingConfig::TILE_M_BIN+PADDING_SHARED_MEM_FOR_C_0]> (smem_weight_packed_bin);
  StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c, NumIterB, NumRegSets_w);
  __syncthreads();
  // Now that shared memory contains all the D tiles, stream them to global memory.
  OutputDataType* BlockGlobalPTR = C + BatchID*(M_Global*N_Global) + Tile_Start_M + Tile_Start_N*M_Global;
  for(size_t i=warpId; i<NumColumnToCopy; i+=TilingConfig::BLOCK_WARPS)    // i-th column
    #pragma unroll
    for(size_t j=threadIdx.x%WARP_SIZE; j<TilingConfig::TILE_M_BIN; j+=WARP_SIZE) // j-th row
    {
      if constexpr (std::is_same<OutputDataType, half>::value) {
        BlockGlobalPTR[j+i*M_Global] = __uint2half_rn(smem_CFrag[i][j]);
      // } else if constexpr (std::is_same<OutputDataType, float>::value) {                                
      //   BlockGlobalPTR[j+i*M_Global] = __uint2float_rn(smem_CFrag[i][j]);
      } else {
        BlockGlobalPTR[j+i*M_Global] = smem_CFrag[i][j];
      }
    }
}

template<typename TilingConfig, typename OutputDataType>
__global__ void PACK_BGEMM_Kernel(const half* Weight, const half* S_w, const half*  S_a, 
                                  const half *Act,  // 0x7fffa0c22200
                                  // const half *Act,
                                  OutputDataType* C,
                                  const size_t M_Global, const size_t N_Global, const size_t K_Global,
                                  int Split_K,
                                  int INSTR) 
{
  #ifdef DEBUG_MODE
    assert(K_Global%TilingConfig::TILE_K_BIN==0);
    assert(M_Global%TilingConfig::TILE_M_BIN==0);
    assert( gridDim.y == Split_K * (M_Global/TilingConfig::TILE_M_BIN));
  #endif
  // // 2+4 weight split
  // Dynamic shared memory for FP16 A tiles， 128 Bytes aligned
  extern __shared__ __align__(128) half smem_weight_half[];   // 0x0200
  half (*smem_act)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1] = reinterpret_cast<half (*)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1]> ( smem_weight_half + (HALF_WEIGHT_PER_UNIT)*2); // Dynamic shared memory for FP16 Act tiles  // 0x7fffd5002400
  // __shared__ half QuantScales_w[64*TilingConfig::BLOCK_WARPS];  // static shared memory for quantization scales, 64 row per warp * 4 warps = 512 Bytes
  // __shared__ half QuantScales_a[64*TilingConfig::BLOCK_WARPS];  // static shared memory for quantization scales, 64 row per warp * 4 warps = 512 Bytes
  // Thread Block Mapping, considering SplitK
  const size_t BatchID = blockIdx.y / (M_Global/TilingConfig::TILE_M_BIN); // 256/(64*4)=1, BatchID=0
  const size_t x = blockIdx.x;                                     // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t y = blockIdx.y % (M_Global/TilingConfig::TILE_M_BIN);   // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t Tile_Start_M = y * TilingConfig::TILE_M_BIN;
  const size_t Tile_Start_N = x * TilingConfig::TILE_N_BIN; 
  const size_t NumColumnToCopy = (N_Global-Tile_Start_N) < TilingConfig::TILE_N_BIN ? (N_Global-Tile_Start_N) : TilingConfig::TILE_N_BIN;
  const size_t NumBlock_K = K_Global/TilingConfig::TILE_K_BIN;    // K_Global / 128 = 2 
  const size_t AverageNumBlock_K = NumBlock_K/Split_K;
  const size_t ExtraNumBlock_K   = NumBlock_K - AverageNumBlock_K * Split_K;
  size_t NumIter = AverageNumBlock_K;
  if(BatchID<ExtraNumBlock_K)       NumIter ++;
  size_t StartBlockID_K = AverageNumBlock_K*BatchID;
  if(BatchID<ExtraNumBlock_K)       StartBlockID_K += BatchID;
  else                              StartBlockID_K += ExtraNumBlock_K;
  // Warp ID.
  const int warpId = threadIdx.x / WARP_SIZE; // 0,1,2,3
  int WARP_i = warpId / TilingConfig::BLOCK_COL_WARPS;  // =warpId/1 // WARP_i: row number;  WARP_j: column number
  // Global Memory Address for Matrix weight and act /////////////////////////////////////////////////////////////////////////
  // StartPTR for each ThreadBlock(TB)
  const half* TB_StartGPTR_W = Weight + (y*TilingConfig::BLOCK_ROW_WARPS)*NumBlock_K*HALF_WEIGHT_PER_UNIT/4; // &TB_StartGPTR_W = 0x7fffa0c00000
  const half* TB_StartGPTR_A = Act + (Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K_BIN); // 0x7fffa0c22200
  
  // // StartPTR for each WARP.
  const half* WARP_StartGPTR_W  = TB_StartGPTR_W + WARP_i * NumBlock_K * HALF_WEIGHT_PER_UNIT/4;   // 0x7fffa0c00000
  // // StartPTR for each WARP, considering SplitK
  const size_t     WARP_Start_UnitID_K = StartBlockID_K;  // unsigned long = size_t = 32bits
  WARP_StartGPTR_W  += WARP_Start_UnitID_K * 128;  // noqa
  // // Copying A tile from Global to Shared, using double-buffer //////////////////////////////////////////////////////////
  // // StartSPTR for each ThreadBlock
  half* Weight_SPTR = reinterpret_cast<half*>(smem_weight_half);   // 0x7fffd5000400
  // // StartSPTR for each WARP
  Weight_SPTR += warpId * HALF_WEIGHT_PER_UNIT/4; // {0,1,2,3}
  // // Pre-fetch of A tile
  for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) { // 
    CopyFromGlobalToShared_W<2048>(Weight_SPTR+(i*WEIGHT_PER_UNIT_BIN), WARP_StartGPTR_W, K_Global, NumColumnToCopy); // 128*128
    WARP_StartGPTR_W += TilingConfig::TILE_K_BIN;  // half
    // CopyFromGlobalToShared_B<TilingConfig::TILE_N_BIN, TilingConfig::BLOCK_WARPS>(smem_act+(i*TilingConfig::TILE_N_BIN), TB_StartGPTR_A, K_Global, NumColumnToCopy);
    CopyFromGlobalToShared_A<TilingConfig::TILE_K_BIN, TilingConfig::BLOCK_WARPS>(smem_act+(i*TilingConfig::TILE_K_BIN), TB_StartGPTR_A, K_Global, NumColumnToCopy);
    TB_StartGPTR_A += TilingConfig::TILE_K_BIN;  // half
  }
  // Global Memory Address for Matrix QuantScale for weight and act （scale开始都放global mem）/////////////////////////////////////////////////////////////////////
  // const half* TB_StartGPTR_W_Scale    = S_w + (y*TilingConfig::BLOCK_ROW_WARPS) * 64;
  // const half* TB_StartGPTR_A_Scale    = S_a + (y*TilingConfig::BLOCK_ROW_WARPS) * 64;
  // const half* WARP_StartGPTR_A_Scales = TB_StartGPTR_W_Scale + WARP_i * 64;
  // const half* WARP_StartGPTR_B_Scales = TB_StartGPTR_A_Scale + WARP_i * 64;
  // CopyFromGlobalToShared_Scales(QuantScales_w+WARP_i*64, WARP_StartGPTR_A_Scales);
  // CopyFromGlobalToShared_Scales(QuantScales_a+WARP_i*64, WARP_StartGPTR_B_Scales);
  // // Copying Act tile from Global to Shared, considering SplitK /////////////////////////////////////////////////////////////
  // const half *BTile_GPTR = Act + Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K_BIN;
  // for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) {
  //   CopyFromGlobalToShared<TilingConfig::TILE_N_BIN, TilingConfig::BLOCK_WARPS> (smem_act+i*TilingConfig::TILE_N_BIN, BTile_GPTR, K_Global, NumColumnToCopy);
    // BTile_GPTR += TilingConfig::TILE_K_BIN;     // 64
  // }

  // Register Allocation for weight, Act, and C, Initilazed to Zeros /////////////////////////////////////////////////////////////////////
  constexpr int NumRegSets_w = 1;     //   WARP_ROW_MMA_TENSORS                                                      // 1 set = 4 registers, containing a 16*16 MMA block
  constexpr int NumRegSets_a = 1;     // (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;    // 1 set = 4 registers, containing a 16*16 MMA block
  // constexpr int NumRegSets_a = (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;    // 1 set = 4 registers, containing a 16*16 MMA block
#ifdef PIPELINE_LEVEL_SMEM
  uint32_t a  [NumRegSets_w * PIPELINE_LEVEL_SMEM][1];      // double/Trible buffer is used // Registers to store decompressed FP6
  uint32_t b  [NumRegSets_a * PIPELINE_LEVEL_SMEM][1];      // double/Triple buffer is used // Register to store FP16 Act matrix (a slice)
#endif
  int32_t c[NumRegSets_w * NumRegSets_a][REG_PER_THREAD_C_TENSOR_16_16]; // REG_PER_THREAD_C = 2 // 
  for(int i=0; i<NumRegSets_w * NumRegSets_a; i++) 
    for(int j=0; j<REG_PER_THREAD_C_TENSOR_16_16; j++)
      c[i][j] = 0;
  //
  cp_async_wait_all();
  __syncthreads();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int32_t Scales_RPTR_w[4]; // 4 Registers per thread for Quantization S_w
  int32_t Scales_RPTR_a[4]; // 4 Registers per thread for Quantization S_w
  // ExtractFromSharedToReg_Scales(Scales_RPTR_w, QuantScales_w + WARP_i*64);
  // ExtractFromSharedToReg_Scales(Scales_RPTR_a, QuantScales_a + WARP_i*64);
#ifdef PIPELINE_LEVEL_SMEM
  // Initializing the Software Pipeline: writing registers. ////////////////////////////////////////////////////////////////////////////////////////////////
  int NumSlices = 4;
  int NumIterB = WARP_N_BIN/NumSlices/8; // =32/4/8=1
  initialize_mma_slice_binpack<TilingConfig>(a, b, Weight_SPTR, smem_act, NumIterB);
#endif
  // The outer loop. /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #pragma unroll(1)
  for (size_t tile_id_k = 0; tile_id_k < NumIter; tile_id_k++)
  {
    // Trible-Buffer for A Tile
	  half* __restrict__ read_SPTR_W  = Weight_SPTR + ((tile_id_k+0)                     % PIPELINE_LEVEL_GMEM) * HALF_WEIGHT_PER_UNIT; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    // read_SPTR_W = 0x7fffd5000400
#ifdef PIPELINE_LEVEL_SMEM
    half* __restrict__ read2_SPTR_W  = Weight_SPTR + ((tile_id_k+1)                     % PIPELINE_LEVEL_GMEM) * HALF_WEIGHT_PER_UNIT;
    // read2_SPTR_W = 0x7fffd5000c00
#endif
    half* __restrict__ write_SPTR_W = Weight_SPTR + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * HALF_WEIGHT_PER_UNIT; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    // write_SPTR_W = 0x7fffd5000c00 
    // Trible-Buffer for Act Tile
    half  __restrict__ (*read_SPTR_A )[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1] = smem_act + ((tile_id_k+0)  % PIPELINE_LEVEL_GMEM) * WARP_N_BIN;
// #ifdef PIPELINE_LEVEL_SMEM
    half  __restrict__ (*read2_SPTR_A )[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1] = smem_act + ((tile_id_k+1) % PIPELINE_LEVEL_GMEM) * WARP_N_BIN;// 0x7fffd5002d00
//  0x7fffd5002d00 - 0x7fffd5002400 = 2304 = 72*8*4B
// #endif
    half  __restrict__ (*write_SPTR_A)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1] = smem_act + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * WARP_N_BIN;
    // write_SPTR = 0x7fffd5002d00
    bool GlobalCopy = (tile_id_k+PIPELINE_LEVEL_GMEM-1) < NumIter;
    // Copying A tile from Global to Register, Bypassing L1, using double-buffer   
    CopyFromGlobalToShared_W<2048>(write_SPTR_W, WARP_StartGPTR_W, K_Global, NumColumnToCopy);
    // copying Act tile from GlobalMemory to SharedMemory
	  CopyFromGlobalToShared_A<TilingConfig::TILE_K_BIN, TilingConfig::BLOCK_WARPS>(write_SPTR_A, TB_StartGPTR_A, K_Global, NumColumnToCopy, GlobalCopy);
    cp_async_group_commit();
  #ifdef PIPELINE_LEVEL_SMEM
    uint32_t (*a_read )[1] = a;  // 4*1
    uint32_t (*a_write)[1] = a; 
    if (tile_id_k%2==1) {a_read += NumRegSets_w; } else {a_write += NumRegSets_w;}
    core_mma_slice_binpack<TilingConfig>(c, a_read, b, read_SPTR_W, read_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  1, NumIterB, INSTR); // read_SPTR_W, read_SPTR_Frag2 are different for each WARP; read_SPTR is shared among WARPs
    core_mma_slice_binpack<TilingConfig>(c, a_read, b, read_SPTR_W, read_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  2, NumIterB, INSTR);
    core_mma_slice_binpack<TilingConfig>(c, a_read, b, read_SPTR_W, read_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  3, NumIterB, INSTR);
    // Barriers and Synchronizations
    cp_async_wait_group<PIPELINE_LEVEL_GMEM-2>();
    __syncthreads();
    core_mma_slice_binpack<TilingConfig>(c, a_read, b, read2_SPTR_W, read2_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  0, NumIterB, INSTR);
    PackFromSharedToRegister_BinaryW<1, 1>   (a_write, read2_SPTR_W, 0);
    // Updating global PTRs
    WARP_StartGPTR_W +=  TilingConfig::TILE_K_BIN; // 128 half / block
    TB_StartGPTR_A += TilingConfig::TILE_K_BIN;    // 128 half
  #else
    // did not update for bgemm
    PipelinedCoreLoop<TilingConfig>(c, read_SPTR_A, read_SPTR_W, Scales_RPTR_w, Scales_RPTR_a); // read_SPTR_W, read_SPTR_Frag2 are different for each WARP; read_SPTR is shared among WARPs
    // Updating global PTRs
    WARP_StartGPTR_W += SMEM_SIZE_PER_WARP/16;  // 4KB/16=256 (1)/16: int4*+1 = char*+16
    WARP_StartGPTR_A2 += SMEM_SIZE_IN_BYTES_PER_WARP_A2/16;  // 8KB/16=512 (1)/16: int4*+1 = char*+16
    BTile_GPTR += TilingConfig::TILE_K_BIN;
    // Barriers and Synchronizations
    cp_async_wait_group<PIPELINE_LEVEL_GMEM-2>();
    __syncthreads();
  #endif
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Store the C fragments to shared memory.
  int32_t (*smem_CFrag) [TilingConfig::TILE_M_BIN+PADDING_SHARED_MEM_FOR_C_0] =
        reinterpret_cast <int32_t (*)[TilingConfig::TILE_M_BIN+PADDING_SHARED_MEM_FOR_C_0]> (smem_weight_half);
  StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c, NumIterB, NumRegSets_w);
  __syncthreads();
  // Now that shared memory contains all the D tiles, stream them to global memory.
  OutputDataType* BlockGlobalPTR = C + BatchID*(M_Global*N_Global) + Tile_Start_M + Tile_Start_N*M_Global;
  for(size_t i=warpId; i<NumColumnToCopy; i+=TilingConfig::BLOCK_WARPS)    // i-th column
    #pragma unroll
    for(size_t j=threadIdx.x%WARP_SIZE; j<TilingConfig::TILE_M_BIN; j+=WARP_SIZE) // j-th row
    {
      if constexpr (std::is_same<OutputDataType, half>::value) {
        BlockGlobalPTR[j+i*M_Global] = __int2half_rn(smem_CFrag[i][j]);
      // } else if constexpr (std::is_same<OutputDataType, float>::value) {                                
      //   BlockGlobalPTR[j+i*M_Global] = __uint2float_rn(smem_CFrag[i][j]);
      } else {
        BlockGlobalPTR[j+i*M_Global] = smem_CFrag[i][j];
      }
    }
}


template<typename TilingConfig, typename OutputDataType>
__global__ void PACK_W2A3_Kernel(const half* Weight, const half* S_w, const half*  S_a, 
                                  const half *Act,  // 0x7fffa0c22200
                                  // const half *Act,
                                  OutputDataType* C,
                                  const size_t M_Global, const size_t N_Global, const size_t K_Global,
                                  int Split_K,
                                  int INSTR) 
{
  #ifdef DEBUG_MODE
    assert(K_Global%TilingConfig::TILE_K_BIN==0);
    assert(M_Global%TilingConfig::TILE_M_BIN==0);
    assert( gridDim.y == Split_K * (M_Global/TilingConfig::TILE_M_BIN));
  #endif
  // // 2+4 weight split
  // Dynamic shared memory for FP16 A tiles， 128 Bytes aligned
  extern __shared__ __align__(128) half smem_weight_half[];   // 0x0200
  half (*smem_act)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1] = reinterpret_cast<half (*)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1]> ( smem_weight_half + (HALF_WEIGHT_PER_UNIT)*2); // Dynamic shared memory for FP16 Act tiles  // 0x7fffd5002400
  // __shared__ half QuantScales_w[64*TilingConfig::BLOCK_WARPS];  // static shared memory for quantization scales, 64 row per warp * 4 warps = 512 Bytes
  // __shared__ half QuantScales_a[64*TilingConfig::BLOCK_WARPS];  // static shared memory for quantization scales, 64 row per warp * 4 warps = 512 Bytes
  // Thread Block Mapping, considering SplitK
  const size_t BatchID = blockIdx.y / (M_Global/TilingConfig::TILE_M_BIN); // 256/(64*4)=1, BatchID=0
  const size_t x = blockIdx.x;                                     // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t y = blockIdx.y % (M_Global/TilingConfig::TILE_M_BIN);   // Output Block ID: (BlockID_Row = y; BlockID_Col = x )
  const size_t Tile_Start_M = y * TilingConfig::TILE_M_BIN;
  const size_t Tile_Start_N = x * TilingConfig::TILE_N_BIN; 
  const size_t NumColumnToCopy = (N_Global-Tile_Start_N) < TilingConfig::TILE_N_BIN ? (N_Global-Tile_Start_N) : TilingConfig::TILE_N_BIN;
  const size_t NumBlock_K = K_Global/TilingConfig::TILE_K_BIN;    // K_Global / 128 = 2 
  const size_t AverageNumBlock_K = NumBlock_K/Split_K;
  const size_t ExtraNumBlock_K   = NumBlock_K - AverageNumBlock_K * Split_K;
  size_t NumIter = AverageNumBlock_K;
  if(BatchID<ExtraNumBlock_K)       NumIter ++;
  size_t StartBlockID_K = AverageNumBlock_K*BatchID;
  if(BatchID<ExtraNumBlock_K)       StartBlockID_K += BatchID;
  else                              StartBlockID_K += ExtraNumBlock_K;
  // Warp ID.
  const int warpId = threadIdx.x / WARP_SIZE; // 0,1,2,3
  int WARP_i = warpId / TilingConfig::BLOCK_COL_WARPS;  // =warpId/1 // WARP_i: row number;  WARP_j: column number
  // Global Memory Address for Matrix weight and act /////////////////////////////////////////////////////////////////////////
  // StartPTR for each ThreadBlock(TB)
  const half* TB_StartGPTR_W = Weight + (y*TilingConfig::BLOCK_ROW_WARPS)*NumBlock_K*HALF_WEIGHT_PER_UNIT/4; // &TB_StartGPTR_W = 0x7fffa0c00000
  const half* TB_StartGPTR_A = Act + (Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K_BIN); // 0x7fffa0c22200
  
  // // StartPTR for each WARP.
  const half* WARP_StartGPTR_W  = TB_StartGPTR_W + WARP_i * NumBlock_K * HALF_WEIGHT_PER_UNIT/4;   // 0x7fffa0c00000
  // // StartPTR for each WARP, considering SplitK
  const size_t     WARP_Start_UnitID_K = StartBlockID_K;  // unsigned long = size_t = 32bits
  WARP_StartGPTR_W  += WARP_Start_UnitID_K * 128;  // noqa
  // // Copying A tile from Global to Shared, using double-buffer //////////////////////////////////////////////////////////
  // // StartSPTR for each ThreadBlock
  half* Weight_SPTR = reinterpret_cast<half*>(smem_weight_half);   // 0x7fffd5000400
  // // StartSPTR for each WARP
  Weight_SPTR += warpId * HALF_WEIGHT_PER_UNIT/4; // {0,1,2,3}
  // // Pre-fetch of A tile
  for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) { // 
    CopyFromGlobalToShared_W<2048>(Weight_SPTR+(i*WEIGHT_PER_UNIT_BIN), WARP_StartGPTR_W, K_Global, NumColumnToCopy); // 128*128
    WARP_StartGPTR_W += TilingConfig::TILE_K_BIN;  // half
    // CopyFromGlobalToShared_B<TilingConfig::TILE_N_BIN, TilingConfig::BLOCK_WARPS>(smem_act+(i*TilingConfig::TILE_N_BIN), TB_StartGPTR_A, K_Global, NumColumnToCopy);
    CopyFromGlobalToShared_A<TilingConfig::TILE_K_BIN, TilingConfig::BLOCK_WARPS>(smem_act+(i*TilingConfig::TILE_K_BIN), TB_StartGPTR_A, K_Global, NumColumnToCopy);
    TB_StartGPTR_A += TilingConfig::TILE_K_BIN;  // half
  }
  // Global Memory Address for Matrix QuantScale for weight and act （scale开始都放global mem）/////////////////////////////////////////////////////////////////////
  // const half* TB_StartGPTR_W_Scale    = S_w + (y*TilingConfig::BLOCK_ROW_WARPS) * 64;
  // const half* TB_StartGPTR_A_Scale    = S_a + (y*TilingConfig::BLOCK_ROW_WARPS) * 64;
  // const half* WARP_StartGPTR_A_Scales = TB_StartGPTR_W_Scale + WARP_i * 64;
  // const half* WARP_StartGPTR_B_Scales = TB_StartGPTR_A_Scale + WARP_i * 64;
  // CopyFromGlobalToShared_Scales(QuantScales_w+WARP_i*64, WARP_StartGPTR_A_Scales);
  // CopyFromGlobalToShared_Scales(QuantScales_a+WARP_i*64, WARP_StartGPTR_B_Scales);
  // // Copying Act tile from Global to Shared, considering SplitK /////////////////////////////////////////////////////////////
  // const half *BTile_GPTR = Act + Tile_Start_N * K_Global + StartBlockID_K * TilingConfig::TILE_K_BIN;
  // for(int i=0; i<PIPELINE_LEVEL_GMEM-1; i++) {
  //   CopyFromGlobalToShared<TilingConfig::TILE_N_BIN, TilingConfig::BLOCK_WARPS> (smem_act+i*TilingConfig::TILE_N_BIN, BTile_GPTR, K_Global, NumColumnToCopy);
    // BTile_GPTR += TilingConfig::TILE_K_BIN;     // 64
  // }

  // Register Allocation for weight, Act, and C, Initilazed to Zeros /////////////////////////////////////////////////////////////////////
  constexpr int NumRegSets_w = 1;     //   WARP_ROW_MMA_TENSORS                                                      // 1 set = 4 registers, containing a 16*16 MMA block
  constexpr int NumRegSets_a = 1;     // (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;    // 1 set = 4 registers, containing a 16*16 MMA block
  // constexpr int NumRegSets_a = (TilingConfig::WARP_COL_MMA_TENSORS==1) ? 1 : TilingConfig::WARP_COL_MMA_TENSORS/2;    // 1 set = 4 registers, containing a 16*16 MMA block
#ifdef PIPELINE_LEVEL_SMEM
  uint32_t a  [NumRegSets_w * PIPELINE_LEVEL_SMEM][1];      // double/Trible buffer is used // Registers to store decompressed FP6
  uint32_t b  [NumRegSets_a * PIPELINE_LEVEL_SMEM][2];      // double/Triple buffer is used // Register to store FP16 Act matrix (a slice)
#endif
  int32_t c[NumRegSets_w * NumRegSets_a][REG_PER_THREAD_C_TENSOR_16_16]; // REG_PER_THREAD_C = 2 // 
  for(int i=0; i<NumRegSets_w * NumRegSets_a; i++) 
    for(int j=0; j<REG_PER_THREAD_C_TENSOR_16_16; j++)
      c[i][j] = 0;
  //
  cp_async_wait_all();
  __syncthreads();

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int32_t Scales_RPTR_w[4]; // 4 Registers per thread for Quantization S_w
  int32_t Scales_RPTR_a[4]; // 4 Registers per thread for Quantization S_w
  // ExtractFromSharedToReg_Scales(Scales_RPTR_w, QuantScales_w + WARP_i*64);
  // ExtractFromSharedToReg_Scales(Scales_RPTR_a, QuantScales_a + WARP_i*64);
#ifdef PIPELINE_LEVEL_SMEM
  // Initializing the Software Pipeline: writing registers. ////////////////////////////////////////////////////////////////////////////////////////////////
  int NumSlices = 4;
  int NumIterB = WARP_N_BIN/NumSlices/8; // =32/4/8=1
  initialize_mma_slice_binpack_w2a3<TilingConfig>(a, b, Weight_SPTR, smem_act, NumIterB);
#endif
  // The outer loop. /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #pragma unroll(1)
  for (size_t tile_id_k = 0; tile_id_k < NumIter; tile_id_k++)
  {
    // Trible-Buffer for A Tile
	  half* __restrict__ read_SPTR_W  = Weight_SPTR + ((tile_id_k+0)                     % PIPELINE_LEVEL_GMEM) * HALF_WEIGHT_PER_UNIT; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    // read_SPTR_W = 0x7fffd5000400
#ifdef PIPELINE_LEVEL_SMEM
    half* __restrict__ read2_SPTR_W  = Weight_SPTR + ((tile_id_k+1)                     % PIPELINE_LEVEL_GMEM) * HALF_WEIGHT_PER_UNIT;
    // read2_SPTR_W = 0x7fffd5000c00
#endif
    half* __restrict__ write_SPTR_W = Weight_SPTR + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * HALF_WEIGHT_PER_UNIT; // 1024 (1)*4: 4 WARPs; (2)/4: int*+1 = char*+16
    // write_SPTR_W = 0x7fffd5000c00 
    // Trible-Buffer for Act Tile
    half  __restrict__ (*read_SPTR_A )[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1] = smem_act + ((tile_id_k+0)  % PIPELINE_LEVEL_GMEM) * WARP_N_BIN;
// #ifdef PIPELINE_LEVEL_SMEM
    half  __restrict__ (*read2_SPTR_A )[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1] = smem_act + ((tile_id_k+1) % PIPELINE_LEVEL_GMEM) * WARP_N_BIN;// 0x7fffd5002d00
//  0x7fffd5002d00 - 0x7fffd5002400 = 2304 = 72*8*4B
// #endif
    half  __restrict__ (*write_SPTR_A)[WARP_K_BIN+PADDING_SHARED_MEM_FOR_B_1] = smem_act + ((tile_id_k+(PIPELINE_LEVEL_GMEM-1))  % PIPELINE_LEVEL_GMEM) * WARP_N_BIN;
    // write_SPTR = 0x7fffd5002d00
    bool GlobalCopy = (tile_id_k+PIPELINE_LEVEL_GMEM-1) < NumIter;
    // Copying A tile from Global to Register, Bypassing L1, using double-buffer   
    CopyFromGlobalToShared_W<2048>(write_SPTR_W, WARP_StartGPTR_W, K_Global, NumColumnToCopy);
    // copying Act tile from GlobalMemory to SharedMemory
	  CopyFromGlobalToShared_A<TilingConfig::TILE_K_BIN, TilingConfig::BLOCK_WARPS>(write_SPTR_A, TB_StartGPTR_A, K_Global, NumColumnToCopy, GlobalCopy);
    cp_async_group_commit();
  #ifdef PIPELINE_LEVEL_SMEM
    uint32_t (*a_read )[1] = a;  // 4*1
    uint32_t (*a_write)[1] = a; 
    if (tile_id_k%2==1) {a_read += NumRegSets_w; } else {a_write += NumRegSets_w;}
    core_mma_slice_binpack_w2a3<TilingConfig>(c, a_read, b, read_SPTR_W, read_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  1, NumIterB, INSTR); // read_SPTR_W, read_SPTR_Frag2 are different for each WARP; read_SPTR is shared among WARPs
    core_mma_slice_binpack_w2a3<TilingConfig>(c, a_read, b, read_SPTR_W, read_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  2, NumIterB, INSTR);
    core_mma_slice_binpack_w2a3<TilingConfig>(c, a_read, b, read_SPTR_W, read_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  3, NumIterB, INSTR);
    // Barriers and Synchronizations
    cp_async_wait_group<PIPELINE_LEVEL_GMEM-2>();
    __syncthreads();
    core_mma_slice_binpack_w2a3<TilingConfig>(c, a_read, b, read2_SPTR_W, read2_SPTR_A, Scales_RPTR_w, Scales_RPTR_a,  0, NumIterB, INSTR);
    PackFromSharedToRegister_BinaryW<1, 1>   (a_write, read2_SPTR_W, 0);
    // Updating global PTRs
    WARP_StartGPTR_W +=  TilingConfig::TILE_K_BIN; // 128 half / block
    TB_StartGPTR_A += TilingConfig::TILE_K_BIN;    // 128 half
  #else
    // did not update for bgemm
    PipelinedCoreLoop<TilingConfig>(c, read_SPTR_A, read_SPTR_W, Scales_RPTR_w, Scales_RPTR_a); // read_SPTR_W, read_SPTR_Frag2 are different for each WARP; read_SPTR is shared among WARPs
    // Updating global PTRs
    WARP_StartGPTR_W += SMEM_SIZE_PER_WARP/16;  // 4KB/16=256 (1)/16: int4*+1 = char*+16
    WARP_StartGPTR_A2 += SMEM_SIZE_IN_BYTES_PER_WARP_A2/16;  // 8KB/16=512 (1)/16: int4*+1 = char*+16
    BTile_GPTR += TilingConfig::TILE_K_BIN;
    // Barriers and Synchronizations
    cp_async_wait_group<PIPELINE_LEVEL_GMEM-2>();
    __syncthreads();
  #endif
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Store the C fragments to shared memory.
  int32_t (*smem_CFrag) [TilingConfig::TILE_M_BIN+PADDING_SHARED_MEM_FOR_C_0] =
        reinterpret_cast <int32_t (*)[TilingConfig::TILE_M_BIN+PADDING_SHARED_MEM_FOR_C_0]> (smem_weight_half);
  StoreToSharedMemoryFromRegister<TilingConfig>(smem_CFrag, c, NumIterB, NumRegSets_w);
  __syncthreads();
  // Now that shared memory contains all the D tiles, stream them to global memory.
  OutputDataType* BlockGlobalPTR = C + BatchID*(M_Global*N_Global) + Tile_Start_M + Tile_Start_N*M_Global;
  for(size_t i=warpId; i<NumColumnToCopy; i+=TilingConfig::BLOCK_WARPS)    // i-th column
    #pragma unroll
    for(size_t j=threadIdx.x%WARP_SIZE; j<TilingConfig::TILE_M_BIN; j+=WARP_SIZE) // j-th row
    {
      if constexpr (std::is_same<OutputDataType, half>::value) {
        BlockGlobalPTR[j+i*M_Global] = __int2half_rn(smem_CFrag[i][j]);
      // } else if constexpr (std::is_same<OutputDataType, float>::value) {                                
      //   BlockGlobalPTR[j+i*M_Global] = __uint2float_rn(smem_CFrag[i][j]);
      } else {
        BlockGlobalPTR[j+i*M_Global] = smem_CFrag[i][j];
      }
    }
}