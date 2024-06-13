import argparse
import torch
import lowbit_kernel

WARMUP = 10
REPEAT = 1000

parser = argparse.ArgumentParser(description='The shape of the MatMul: (M, K)*(K, N)->(M, N).')
parser.add_argument('--OC',        type=int, required=False,     default=256,   help='number of rows of the weight matrix.')
parser.add_argument('--IC',        type=int, required=False,     default=256,   help='number of columns of the weight matrix.')
parser.add_argument('--BS',        type=int, required=False,     default=256,     help='inference batch size.')
parser.add_argument('--splitK',    type=int, required=False,     default=1,      help='Split-K parameters allow users to split the GEMM computation along the K dimension so that more CTAs will be created with a better SM utilization.')
args = parser.parse_args()

# assert(args.OC%256==0)
# assert(args.IC%64==0)

print(args)

# fp6_weight = torch.randint(4294967295, (args.OC,args.IC//16*3)).to(torch.int)    # Randomly initialize each bytes. The highest value for randint() is set the the max value of uint32_t.

fp16_scale = torch.rand(args.OC).to(torch.half)+0.5
fp16_activation = torch.randn(args.BS, args.IC).to(torch.half)
fp16_weight = torch.randn(args.OC, args.IC).to(torch.half)


start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# BGEMM
####################################################################################################################################
torch.cuda.synchronize()
bgemm_res = lowbit_kernel.bgemm_linear_forward_cuda(fp16_activation, fp16_weight, fp16_scale, 1) # [B, OC]

# import pdb; pdb.set_trace()

for i in range(WARMUP):
    results_lowbit_kernel = lowbit_kernel.bgemm_linear_forward_cuda(fp16_activation, fp16_weight, fp16_scale, 1)
start_event.record()
for i in range(REPEAT):
    results_lowbit_kernel = lowbit_kernel.bgemm_linear_forward_cuda(fp16_activation, fp16_weight, fp16_scale, 1)
end_event.record()
torch.cuda.synchronize()
lowbit_kernel_time_ms = start_event.elapsed_time(end_event)/REPEAT
lowbit_kernel_tflops  = args.OC*args.IC*args.BS*2/lowbit_kernel_time_ms/1e9

# check
sign_w = torch.sign(fp16_weight)
sign_a = torch.sign(fp16_activation)
sign_w = torch.where(sign_w == 1, torch.zeros_like(sign_w), torch.ones_like(sign_w))
sign_a = torch.where(sign_a == 1, torch.zeros_like(sign_a), torch.ones_like(sign_a))

# # fp6-fp16 GEMM (fp6-llm)
# ####################################################################################################################################
# torch.cuda.synchronize()
# fp6_weight_packed = lowbit_kernel.weight_prepacking_cpu(fp6_weight)

# for i in range(WARMUP):
#     results_lowbit_kernel = lowbit_kernel.linear_forward_cuda(act_cuda, weight_cuda, scale_cuda, args.splitK);
# start_event.record()
# for i in range(REPEAT):
#     results_lowbit_kernel = lowbit_kernel.linear_forward_cuda(act_cuda, weight_cuda, scale_cuda, args.splitK);
# end_event.record()
# torch.cuda.synchronize()
# lowbit_kernel_time_ms = start_event.elapsed_time(end_event)/REPEAT
# lowbit_kernel_tflops  = args.OC*args.IC*args.BS*2/lowbit_kernel_time_ms/1e9
# ####################################################################################################################################

# baseline fp16 GEMM (cuBLAS)
####################################################################################################################################
torch.cuda.synchronize()
# fp16_weight = lowbit_kernel.weight_dequant_cpu(fp6_weight, fp16_scale)
cuBLAS_MatMul = torch.nn.Linear(args.IC, args.OC, False)
results_cublas = None
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
print( 'cuBLAS  time: {:.2f} ms \t\t cuBLAS  TFLOPs: {:.1f}'.format(cublas_time_ms,  cublas_tflops) )
print( 'fp6-llm time: {:.2f} ms \t\t fp6-llm TFLOPs: {:.1f}'.format(lowbit_kernel_time_ms, lowbit_kernel_tflops) )
print( 'speedup: {:.2f}'.format(cublas_time_ms/lowbit_kernel_time_ms) )

# Correctness
results_cublas    = results_cublas.to(float)
results_lowbit_kernel = results_lowbit_kernel.to(float)
error             = results_cublas.cpu() - results_lowbit_kernel.cpu()
ground_truth      = results_cublas.cpu()
mean_error        = torch.mean(abs(error))
mean_ground_truth = torch.mean(abs(ground_truth))
relative_error    = mean_error.item()/mean_ground_truth.item()
print( "relative error: {:.6f}".format(relative_error) )



# 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 001100 
# "00110000110000110000110000110000"       "11000011000011000011000011000011"     "00001100001100001100001100001100"
# 818089008                                 3272356035                              204522252
#fp6_weight = torch.zeros(args.OC, args.IC//16*3).to(torch.int64)
#for i in range(args.OC):
#    for j in range(args.IC//16):
#        fp6_weight[i][j*3+0] = 818089008
#        fp6_weight[i][j*3+1] = 3272356035 
#        fp6_weight[i][j*3+2] = 204522252
#fp6_weight = fp6_weight.to(torch.int)

# Ensuring that the absolute error or relative error of each matrix element is smaller than 1e-3.
#Error = [1e-2]  
#for err in Error:
#    AllClose = torch.allclose(results_lowbit_kernel.cpu(), results_cublas.cpu(), rtol=err, atol=err, equal_nan=True)
#    print("torch.allclose\t (relative/absolute_error<" + str(err) + ") \t-> " + str(AllClose))