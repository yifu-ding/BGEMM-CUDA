#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#ifndef NO_PYTORCH
#include <torch/extension.h>
/*
* Computes FP6-FP16 GEMM (PyTorch interface).
*/


/*
 * Weight prepacking (Pytorch interface).
 */
// torch::Tensor weight_matrix_prepacking_cpu(torch::Tensor fp6_tensor);

// /*
//  * Dequant a FP6 matrix to a equivalent FP16 matrix using CPUs.
//  * A useful tool to construct input matrices for the FP16 GEMM baseline.
//  * [Input]
//  *  fp6_tensor:  int  tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6  weights.
//  *  fp16_scale:  half tensor of shape [OC];                 // for row-wise quantization.
//  * [Output]
//  *  fp16_tensor: half tensor of shape [OC, IC].     
//  */
// torch::Tensor weight_matrix_dequant_cpu(torch::Tensor fp6_tensor, torch::Tensor fp16_scale);
#endif