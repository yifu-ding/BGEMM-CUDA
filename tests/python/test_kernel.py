import argparse
import torch
import lowbit_kernel

WARMUP = 10
REPEAT = 1000
AND_POP = 0    # use mma.and.pop instruction
XOR_POP = 1    # use mma.xor.pop instruction

parser = argparse.ArgumentParser(description='The shape of the MatMul: (M, K)*(K, N)->(M, N).')
parser.add_argument('--OC',        type=int, required=False,     default=352,   help='number of rows of the weight matrix.')
parser.add_argument('--IC',        type=int, required=False,     default=128,   help='number of columns of the weight matrix.')
parser.add_argument('--BS',        type=int, required=False,     default=32,     help='inference batch size.')
parser.add_argument('--splitK',    type=int, required=False,     default=1,      help='Split-K parameters allow users to split the GEMM computation along the K dimension so that more CTAs will be created with a better SM utilization.')
args = parser.parse_args()

assert(args.OC%32==0)
assert(args.IC%128==0)
assert(args.BS%32==0)

print(args)

# fp16_scale = (torch.rand(args.OC).to(torch.half)+0.5).cuda()   // TODO
fp16_activation = torch.randn(args.BS, args.IC).to(torch.half).cuda()
fp16_weight = torch.randn(args.OC, args.IC).to(torch.half).cuda()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# BGEMM
####################################################################################################################################
torch.cuda.synchronize()
bgemm_res = lowbit_kernel.bgemm_linear_forward_cuda(fp16_activation, fp16_weight, 1, XOR_POP) # [B, OC]

for i in range(WARMUP):
    results_lowbit_kernel = lowbit_kernel.bgemm_linear_forward_cuda(fp16_activation, fp16_weight, 1, XOR_POP)
start_event.record()
for i in range(REPEAT):
    results_lowbit_kernel = lowbit_kernel.bgemm_linear_forward_cuda(fp16_activation, fp16_weight, 1, XOR_POP)
end_event.record()
torch.cuda.synchronize()
lowbit_kernel_time_ms = start_event.elapsed_time(end_event)/REPEAT
lowbit_kernel_tflops  = args.OC*args.IC*args.BS*2/lowbit_kernel_time_ms/1e9

# baseline fp16 GEMM (cuBLAS)
####################################################################################################################################
torch.cuda.synchronize()
cuBLAS_MatMul = torch.nn.Linear(args.IC, args.OC, False)
results_cublas = None
sign_w = torch.sign(fp16_weight)
sign_a = torch.sign(fp16_activation)
with torch.no_grad():
    cuBLAS_MatMul.weight = torch.nn.Parameter(sign_w.clone().cuda())
    act_cuda = sign_a.cuda()
    for i in range(WARMUP):
        results_cublas = cuBLAS_MatMul(act_cuda)
    start_event.record()
    for i in range(REPEAT):
        results_cublas = cuBLAS_MatMul(act_cuda)
    end_event.record()
torch.cuda.synchronize()
cublas_time_ms = start_event.elapsed_time(end_event)/REPEAT
cublas_tflops  = args.OC*args.IC*args.BS*2/cublas_time_ms/1e9
####################################################################################################################################

# Performance
print( 'cuBLAS  time: {:.5f} ms \t\t cuBLAS TFLOPs: {:.5f}'.format(cublas_time_ms,  cublas_tflops) )
print( 'BGEMM time: {:.5f} ms \t\t BGEMM TFLOPs: {:.5f}'.format(lowbit_kernel_time_ms, lowbit_kernel_tflops) )
print( 'speedup: {:.5f}'.format(cublas_time_ms/lowbit_kernel_time_ms) )

# Correctness
results_cublas    = results_cublas.to(float)
results_lowbit_kernel = results_lowbit_kernel.to(float)
error             = results_cublas.cpu() - results_lowbit_kernel.cpu()
ground_truth      = results_cublas.cpu()
mean_error        = torch.mean(abs(error))
mean_ground_truth = torch.mean(abs(ground_truth))
relative_error    = mean_error.item()/mean_ground_truth.item()
print( "relative error: {:.6f}".format(relative_error) )
