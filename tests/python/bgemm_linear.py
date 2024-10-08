import torch
import torch.nn as nn
import lowbit_kernel
import math
import numpy as np

__all__ = ['BNNLinear', 'BGEMMLinear', 'BGEMMLinear_elastic_signed', 'bgemm_matmul', 'bgemm_attn_matmul', 'BGEMMLinear_bireal', 'NNLinear_elastic_signed']

class bgemm_matmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, matA, matB, instruction='xor'):
        """
        :param matA: matA to be binarized
        :param matB: matB to be binarized
        :param instruction: ['and', 'xor'], 'and' for {0,1} and 'xor' for {-1,1}
        :return: calculated result 
        """
        matA = matA.to(torch.half)
        matB = matB.to(torch.half)
        assert matA.dtype == torch.half and matB.dtype == torch.half

        ctx.save_for_backward(matA, matB)
        INSTRUCTION = 0 if instruction=='and' else 1 # 1 by default
        out = lowbit_kernel.bgemm_linear_forward_cuda(matA, matB.transpose(-1, -2), 1, INSTRUCTION)  
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        matA, matB = ctx.saved_tensors
        matA, matB = matA.sign(), matB.sign()
        bs = matA.shape[0]
        assert matB.dim() == 2, "matB dim must be 2, now is %d" % matB.dim()
        
        if matA.dim() == 3:
            matB_grad = torch.bmm(matA.transpose(1, 2), grad_output).transpose(1, 2)   # checked
            matA_grad = torch.bmm(grad_output, matB.repeat(bs, 1, 1))  # checked
        elif matA.dim() == 2:
            matB_grad = torch.mm(matA.T, grad_output).T # checked
            matA_grad = torch.mm(grad_output, matB)  # checked
        return matA_grad, matB_grad, None


class bgemm_attn_matmul(torch.autograd.Function):  # noqa
    
    @staticmethod
    def forward(ctx, input1, input2, instruction="attn_mm"):  # attn, value
        """
        :param input2: input2 to be binarized
        :param input1: input1 to be binarized
        :param instruction: ['and', 'xor'], 'and' for {0,1} and 'xor' for {-1,1}
        :return: calculated result 
        """
        
        input1 = input1.to(torch.half)
        input2 = input2.to(torch.half)
        ctx.save_for_backward(input1, input2)
        assert input2.dtype == torch.half and input1.dtype == torch.half
        out = lowbit_kernel.bgemm_linear_forward_cuda(input2.transpose(-1, -2).contiguous(), input1, 1, 2).T  # value, attn

        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_output = grad_output
        
        input1, input2 = ctx.saved_tensors
        input1 = (input1.sign() + 1)/2
        input1 = torch.where(input1==0, torch.ones_like(input1), input1)
        input2 = input2.sign()
        
        bs = input2.shape[0]
        
        assert input1.dim() == 2, "input1 dim must be 2, now is %d" % input1.dim()
        
        # import pdb; pdb.set_trace()
        
        if input2.dim() == 3:
            input1_grad = torch.bmm(grad_output, input2.T).transpose(1, 2) / 2
            input2_grad = torch.bmm(input1.transpose(1, 2).repeat(bs, 1, 1), grad_output)  
        elif input2.dim() == 2:
            input1_grad = torch.mm(grad_output, input2.T) / 2
            input2_grad = torch.mm(input1.T, grad_output)
        return input1_grad, input2_grad, None



class mm(torch.autograd.Function):  # noqa
    
    @staticmethod
    def forward(ctx, matB, matA, instruction="attn_mm"): # x, w
        """
        :param matA: matA to be binarized
        :param matB: matB to be binarized
        :param instruction: ['and', 'xor'], 'and' for {0,1} and 'xor' for {-1,1}
        :return: calculated result 
        """
        
        
        # matA = matA.to(torch.half)
        # matB = matB.to(torch.half)
        # assert matA.dtype == torch.half and matB.dtype == torch.half
        # out = lowbit_kernel.bgemm_linear_forward_cuda(matA.transpose(-1, -2).contiguous(), matB, 1, 2).T  # feat, (attn)

        ctx.save_for_backward(matA, matB)
        out = matB @ matA 

        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_output = grad_output.float()
        matA, matB = ctx.saved_tensors
        bs = matA.shape[0]
        assert matB.dim() == 2, "matB dim must be 2, now is %d" % matB.dim()
        
        # import pdb; pdb.set_trace()
        
        if matA.dim() == 3:
            matB_grad = torch.bmm(matA.transpose(1, 2), grad_output).transpose(1, 2)
            matA_grad = torch.bmm(grad_output, matB.repeat(bs, 1, 1))  
        elif matA.dim() == 2:
            matA_grad = torch.mm(matB.T, grad_output)
            matB_grad = torch.mm(grad_output, matA.T) 
        return matB_grad, matA_grad, None


class bgemm_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, instruction='xor'):
        """
        :param input: input to be binarized
        :param weight: weight to be binarized
        :param instruction: ['and', 'xor'], 'and' for {0,1} and 'xor' for {-1,1}
        :return: calculated result 
        """
        input = input.to(torch.half)
        weight = weight.to(torch.half)
        assert input.dtype == torch.half and weight.dtype == torch.half

        ctx.save_for_backward(input, weight)
        INSTRUCTION = 0 if instruction=='and' else 1 # 1 by default
        out = lowbit_kernel.bgemm_linear_forward_cuda(input, weight, 1, INSTRUCTION)  
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input_, weight = ctx.saved_tensors
        input_, weight = input_.sign(), weight.sign() 
        bs = input_.shape[0]
        assert weight.dim() == 2, "weight dim must be 2, now is %d" % weight.dim()
        
        # import pdb; pdb.set_trace()
        
        if input_.dim() == 3:
            weight_grad = torch.bmm(input_.transpose(1, 2), grad_output).transpose(1, 2)   # checked
            feat_grad = torch.bmm(grad_output, weight.repeat(bs, 1, 1))  # checked
        elif input_.dim() == 2:
            feat_grad = torch.mm(grad_output, weight) 
            weight_grad = torch.mm(input_.T, grad_output).T
        return feat_grad, weight_grad, None


class bgemm_linear_bireal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, instruction='xor'):
        """
        :param input: input to be binarized
        :param weight: weight to be binarized
        :param instruction: ['and', 'xor'], 'and' for {0,1} and 'xor' for {-1,1}
        :return: calculated result 
        """
        input = input.to(torch.half)
        weight = weight.to(torch.half)
        assert input.dtype == torch.half and weight.dtype == torch.half

        ctx.save_for_backward(input, weight)
        INSTRUCTION = 0 if instruction=='and' else 1 # 1 by default
        out = lowbit_kernel.bgemm_linear_forward_cuda(input, weight, 1, INSTRUCTION)  
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input, weight = ctx.saved_tensors
        input_sign, weight_sign = input.sign(), weight.sign() 
        bs = input.shape[0]
        assert weight.dim() == 2, "weight dim must be 2, now is %d" % weight.dim()
        
        # import pdb; pdb.set_trace()
        
        if input_sign.dim() == 3:
            weight_grad = torch.bmm(input_sign.transpose(1, 2), grad_output).transpose(1, 2)   # checked
            feat_grad = torch.bmm(grad_output, weight_sign.repeat(bs, 1, 1))  # checked
        elif input_sign.dim() == 2:
            feat_grad = torch.mm(grad_output, weight_sign) 
            weight_grad = torch.mm(input_sign.T, grad_output).T
            
        # STE
        # feat_grad, weight_grad = feat_grad*1, weight_grad*1
        # bireal
        '''
        dx = 2+2x  -1<x<0
             2-2x  0<x<1
             0     otherwise
        '''
        x = input.abs()
        x = torch.where(x > 1, torch.ones_like(x), x)  # clip
        dx = 2 - 2 * x
        # np.save("/disk2/results/bibert/save_npy/save_backward/x_0.npy", input.detach().cpu().numpy())
        # np.save("/disk2/results/bibert/save_npy/save_backward/dx_0.npy", dx.detach().cpu().numpy())
        # print("saved x, dx in /disk2/results/bibert/save_npy/save_backward.")
        feat_grad = feat_grad * dx

        return feat_grad, weight_grad, None


def ternary(x, is_act):
    # # norm = ((x - x.min())/ (x.max() - x.min())).to(torch.float32)
    # # import pdb; pdb.set_trace()

    # x = x.to(torch.float32)
    # # quantile_0 = torch.quantile(x, 0.9)
    # # quantile_1 = torch.quantile(x, 0.1)
    # if is_act:
    #     quantile_0, quantile_1 = 1e-1, -1e-1
    # else:
    #     quantile_0, quantile_1 = 1e-1, -1e-1
    
    # # print("range: %.5f, %.5f" %(quantile_0, quantile_1))
    # x = torch.where(x>=quantile_0, torch.ones_like(x), x)
    # x = torch.where(x<=quantile_1, torch.ones_like(x)-2, x)
    # x = torch.where(x.abs()!=1, torch.zeros_like(x), x)
    # return torch.where(x.abs()<1e-3, torch.zeros_like(x), x.sign())
    # return x

    return x.sign()

class nn_linear_elastic_signed(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, sa, sw, grad_scale, instruction="xor"):
        input = input.clone()
        weight = weight.clone()
        # input -= 0.5
        input -= input.mean().item()
        weight -= weight.mean().item()
        
        # eps = torch.tensor(0.00001).float().to(sa.device)
        # sa = torch.where(sa > eps, sa, eps)
        # sw = torch.where(sw > eps, sw, eps)
        # import pdb; pdb.set_trace()
        # assert sa > 0 and sw > 0, 'sa = {:.6f}, sw = {:.6f} '.format(sa, sw)
        
        ctx.other = grad_scale
        input = input / sa
        # weight = weight / sw
        ctx.save_for_backward(input, weight)

        input = input.to(torch.half)
        weight = weight.to(torch.half)
        
        INSTRUCTION = 0 if instruction=='and' else 1 # 1 by default
        # out = lowbit_kernel.bgemm_linear_forward_cuda(input, weight, 1, INSTRUCTION)  
        # input, weight = input.sign(), weight.sign()
        input, weight = ternary(input, True), ternary(weight, False)

        out = input @ weight.T
        out = out * sa * sw
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):

        input, weight = ctx.saved_tensors
        grad_scale = ctx.other
        # input_sign, weight_sign = input.sign(), weight.sign() 
        input_sign, weight_sign = ternary(input, True), ternary(weight, False)
        
        bs = input.shape[0]
        assert weight.dim() == 2, "weight dim must be 2, now is %d" % weight.dim()
        if input_sign.dim() == 3:
            weight_grad = torch.bmm(input_sign.transpose(1, 2), grad_output).transpose(1, 2)   # checked
            feat_grad = torch.bmm(grad_output, weight_sign.repeat(bs, 1, 1))  # checked
        elif input_sign.dim() == 2:
            weight_grad = torch.mm(input_sign.T, grad_output).T # checked
            feat_grad = torch.mm(grad_output, weight_sign)  # checked

        grad_input, grad_sa = calculate_gradient_signed(input, feat_grad, grad_scale, True)
        _, grad_sw = calculate_gradient_signed(weight, weight_grad, grad_scale, False)
        # import pdb; pdb.set_trace()
        '''
        # Constant function (STE)
        feat_grad, weight_grad = feat_grad*1, weight_grad*1
        '''
        def ste(x):
            return torch.where(x==1, torch.zeros_like(x), torch.ones_like(x))
        '''
        # Linear function
        # dx = 2+2x  -1<x<0
        #      2-2x  0<x<1
        #      0     otherwise
        '''
        def f_lin(x):   # clip must be 1
           return 2 - 2 * x 
        '''
        # Quadratic function
        # dx = -3/2 x^2 + 3/2   -1<x<1
        #      0               otherwise
        '''
        def f_quad(x):  # clip must be 1
            return -3/2 * x.pow(2) + 3/2 
        
        def f1(x):  # pass
            # 向x正半轴和x负半轴收敛到0。关于y轴对称。经过(0,2)点。函数图像与x轴围成的面积存在极限，且极限为4
            return 2 * torch.exp(-torch.abs(x))
        
        def f2(x):
            # 在x=0附近斜率为0，函数图像与x轴围成的面积存在极限，且极限为2√pi约为3.55
            return 2 * (torch.exp(-x**2))

        def sech2(x):  
            # tanh的导数
            return 2 / torch.cosh(x)**2

        clip = 1
        x = input.abs()
        x = torch.where(x > clip, torch.ones_like(x) * clip, x)  # clip
        dx = f2(x)
        grad_input = grad_input * dx
        
        x = weight.abs()
        x = torch.where(x > clip, torch.ones_like(x) * clip, x)  # clip
        dx = f2(x)
        weight_grad = weight_grad * dx
        
        # np.save("/disk2/results/bibert/save_npy/save_backward/x_0.npy", input.detach().cpu().numpy())
        # np.save("/disk2/results/bibert/save_npy/save_backward/dx_0.npy", dx.detach().cpu().numpy())
        # print("saved x, dx in /disk2/results/bibert/save_npy/save_backward.")
        # import pdb; pdb.set_trace() 
        return grad_input, weight_grad, grad_sa, grad_sw, None
        # return feat_grad, weight_grad, None, None, None


class bgemm_linear_elastic_signed(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, sa, sw, grad_scale, instruction="xor"):
        input = input.clone()
        weight = weight.clone()
        # input -= 0.5
        input -= input.mean().item()
        weight -= weight.mean().item()
        
        # eps = torch.tensor(0.00001).float().to(sa.device)
        # sa = torch.where(sa > eps, sa, eps)
        # sw = torch.where(sw > eps, sw, eps)
        # import pdb; pdb.set_trace()
        # assert sa > 0 and sw > 0, 'sa = {:.6f}, sw = {:.6f} '.format(sa, sw)
        
        ctx.other = grad_scale
        input = input / sa
        # weight = weight / sw
        ctx.save_for_backward(input, weight)

        input = input.to(torch.half)
        weight = weight.to(torch.half)
        
        INSTRUCTION = 0 if instruction=='and' else 1 # 1 by default
        out = lowbit_kernel.bgemm_linear_forward_cuda(input, weight, 1, INSTRUCTION)  
        # input, weight = input.sign(), weight.sign()
        # out = input @ weight.T
        out = out * sa * sw
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):

        input, weight = ctx.saved_tensors
        grad_scale = ctx.other
        input_sign, weight_sign = input.sign(), weight.sign() 
        
        bs = input.shape[0]
        assert weight.dim() == 2, "weight dim must be 2, now is %d" % weight.dim()
        if input_sign.dim() == 3:
            weight_grad = torch.bmm(input_sign.transpose(1, 2), grad_output).transpose(1, 2)   # checked
            feat_grad = torch.bmm(grad_output, weight_sign.repeat(bs, 1, 1))  # checked
        elif input_sign.dim() == 2:
            weight_grad = torch.mm(input_sign.T, grad_output).T # checked
            feat_grad = torch.mm(grad_output, weight_sign)  # checked

        grad_input, grad_sa = calculate_gradient_signed(input, feat_grad, grad_scale)
        _, grad_sw = calculate_gradient_signed(weight, weight_grad, grad_scale)
        # import pdb; pdb.set_trace()
        '''
        # Constant function (STE)
        feat_grad, weight_grad = feat_grad*1, weight_grad*1
        '''
        '''
        # Linear function
        # dx = 2+2x  -1<x<0
        #      2-2x  0<x<1
        #      0     otherwise
        x = input.abs()
        x = torch.where(x > 1, torch.ones_like(x), x)  # clip
        dx = 2 - 2 * x
        grad_input = grad_input * dx
        '''
        def f_lin(x):
           return 2 - 2 * x 
        '''
        # Quadratic function
        # dx = -3/2 x^2 + 3/2   -1<x<1
        #      0               otherwise
        x = input
        x = torch.where(x.abs() > 1, torch.ones_like(x), x)  # clip
        dx = -3/2 * x.pow(2) + 3/2 
        grad_input = grad_input * dx
        '''
        def f_quad(x):  # clip must be 1
            return -3/2 * x.pow(2) + 3/2 
        
        def f1(x): 
            # 向x正半轴和x负半轴收敛到0。关于y轴对称。经过(0,2)点。函数图像与x轴围成的面积存在极限，且极限为4
            return 2 * torch.exp(-torch.abs(x))
        
        def f2(x):
            # 在x=0附近斜率为0，函数图像与x轴围成的面积存在极限，且极限为2√pi约为3.55
            return 2 * (torch.exp(-x**2))

        def sech2(x):  
            # 与x轴围成的面积是4
            return 2 / torch.cosh(x)**2

        clip = 4
        x = input.abs()
        x = torch.where(x > clip, torch.ones_like(x) * clip, x)  # clip
        dx = sech2(x)
        grad_input = grad_input * dx
        
        x = weight.abs()
        x = torch.where(x > clip, torch.ones_like(x) * clip, x)  # clip
        dx = sech2(x)
        weight_grad = weight_grad * dx
        
        # np.save("/disk2/results/bibert/save_npy/save_backward/x_0.npy", input.detach().cpu().numpy())
        # np.save("/disk2/results/bibert/save_npy/save_backward/dx_0.npy", dx.detach().cpu().numpy())
        # print("saved x, dx in /disk2/results/bibert/save_npy/save_backward.")
        # import pdb; pdb.set_trace() 
        return grad_input, weight_grad, grad_sa, grad_sw, None
        # return feat_grad, weight_grad, None, None, None


class BGEMMLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=False):
        super(BGEMMLinear, self).__init__(in_channels, out_channels, bias)
        self.initialized = False
    
    def _initialize(self, x, w):
        assert not self.initialized, 'already initialized.'
        self.sw = nn.Parameter(w.norm(1, 1).div(w.nelement()), requires_grad=True)
        self.sa = nn.Parameter(2 * x.abs().mean() , requires_grad=True)
        self.zpw = torch.mean(w).detach() # nn.Parameter(torch.mean(w), requires_grad=False)
        self.zpa = torch.mean(x).detach() # nn.Parameter(torch.mean(x), requires_grad=False)
        self.initialized = True
        
    def forward(self, x):
        assert x.shape[0] % 32 == 0, "x.shape[0] is %d" % x.shape[0]
        if not self.initialized:
            self._initialize(x, self.weight)

        w = self.weight - self.zpw
        x = x - self.zpa
        # import pdb; pdb.set_trace()
        out = bgemm_linear.apply(x, w) * self.sw * self.sa
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
    

class BGEMMLinear_bireal(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=False):
        super(BGEMMLinear_bireal, self).__init__(in_channels, out_channels, bias)
        self.initialized = False
    
    def _initialize(self, x, w):
        assert not self.initialized, 'already initialized.'
        self.sw = nn.Parameter(w.norm(1, 1).div(w.nelement()), requires_grad=True)
        self.sa = nn.Parameter(2 * x.abs().mean() , requires_grad=True)
        self.zpw = torch.mean(w).detach() # nn.Parameter(torch.mean(w), requires_grad=False)
        self.zpa = torch.mean(x).detach() # nn.Parameter(torch.mean(x), requires_grad=False)
        
        self.initialized = True
        
    def forward(self, x):
        assert x.shape[0] % 32 == 0, "x.shape[0] is %d" % x.shape[0]
        if not self.initialized:
            self._initialize(x, self.weight)

        w = self.weight - self.zpw
        x = x - self.zpa

        out = bgemm_linear_bireal.apply(x, w) * self.sw * self.sa
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


def calculate_gradient_signed(qx, grad, grad_scale, is_act):
    # qx = x / sx
    Qn, Qp = -1, 1
    indicate_small = (qx < Qn).float()
    indicate_big = (qx > Qp).float()
    indicate_middle = 1.0 - indicate_small - indicate_big   # this is more cpu-friendly than torch.ones(input_.shape)
    # # import pdb; pdb.set_trace()
    # grad_sx = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-qx + qx.round())) * grad * grad_scale).sum(dim=-1).unsqueeze(dim=0)
    # grad_sx = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-qx + ternary(qx))) * grad * grad_scale).sum(dim=-1).unsqueeze(dim=0)
    # grad_sx = ((qx.sign()) * grad * grad_scale).sum(dim=-1).unsqueeze(dim=0)
    if not is_act:
        grad_sx = (ternary(qx, is_act) * grad * grad_scale).sum(dim=-1).unsqueeze(dim=0)
    else:
        grad_sx = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-qx + ternary(qx, is_act))) * grad * grad_scale).sum(dim=-1).unsqueeze(dim=0)
    # grad_x = indicate_middle * grad
    grad_x = grad
    return grad_x, grad_sx


class BGEMMLinear_elastic_signed(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=True):
        super(BGEMMLinear_elastic_signed, self).__init__(in_channels, out_channels, bias)
        self.initialized = False
        
    
    def _initialize(self, x, w):
        assert not self.initialized, 'already initialized.'
        self.sw = nn.Parameter(w.norm(1, 1).div(w.nelement()), requires_grad=True)
        self.sa = nn.Parameter(2 * x.abs().mean() , requires_grad=True)
        self.grad_scale = 1.0 / math.sqrt(x.numel())
        self.zpw = torch.mean(w).detach() # nn.Parameter(torch.mean(w), requires_grad=False)
        self.zpa = torch.mean(x).detach() # nn.Parameter(torch.mean(x), requires_grad=False)
        self.initialized = True
        
    def forward(self, x):
        assert x.shape[0] % 32 == 0, "x.shape[0] is %d" % x.shape[0]
        if not self.initialized:
            self._initialize(x, self.weight)

        # import pdb; pdb.set_trace()
        out = bgemm_linear_elastic_signed.apply(x, self.weight, self.sa, self.sw, self.grad_scale)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out




class NNLinear_elastic_signed(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=True):
        super(NNLinear_elastic_signed, self).__init__(in_channels, out_channels, bias)
        self.initialized = False
        
    
    def _initialize(self, x, w):
        assert not self.initialized, 'already initialized.'
        self.sw = nn.Parameter(w.norm(1, 1).div(w.nelement()), requires_grad=True)
        self.sa = nn.Parameter(2 * x.abs().mean() , requires_grad=True)
        self.grad_scale = 1.0 / math.sqrt(x.numel())
        self.zpw = torch.mean(w).detach() # nn.Parameter(torch.mean(w), requires_grad=False)
        self.zpa = torch.mean(x).detach() # nn.Parameter(torch.mean(x), requires_grad=False)
        self.initialized = True
        
    def forward(self, x):
        assert x.shape[0] % 32 == 0, "x.shape[0] is %d" % x.shape[0]
        if not self.initialized:
            self._initialize(x, self.weight)

        # import pdb; pdb.set_trace()
        out = nn_linear_elastic_signed.apply(x, self.weight, self.sa, self.sw, self.grad_scale)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out



class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations for ***** BNN and XNOR *****.
    '''
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        input = input.sign()
        input = torch.where(input==0, torch.ones_like(input), input)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input.ge(1)] = 0
        # grad_input[input.le(-1)] = 0
        return grad_input


class BNNLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=False):
        super(BNNLinear, self).__init__(in_channels, out_channels, bias)
        self.initialized = False

    def _initialize(self, x, w):
        assert not self.initialized, 'already initialized.'
        self.sw = nn.Parameter(w.norm(1, 1).div(w.nelement()), requires_grad=True)
        self.sa = nn.Parameter(2 * x.abs().mean() , requires_grad=True)
        self.zpw = torch.mean(w).detach() # nn.Parameter(torch.mean(w), requires_grad=False)
        self.zpa = torch.mean(x).detach() # nn.Parameter(torch.mean(x), requires_grad=False)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._initialize(x, self.weight)
            
        w = self.weight - self.zpw
        x = x - self.zpa

        bw = BinActive().apply(w)
        bx = BinActive().apply(x)
        out = nn.functional.linear(bx, bw, self.bias) * self.sw * self.sa
        # out = bx @ bw.T + self.bias

        return out
    

class test_linear_fn(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, weight, instruction='xor'):
        """
        :param input: input to be binarized
        :param weight: weight to be binarized
        :param instruction: ['and', 'xor'], 'and' for {0,1} and 'xor' for {-1,1}
        :return: calculated result 
        """
        ctx.save_for_backward(input, weight)
        INSTRUCTION = 0 if instruction=='and' else 1 # 1 by default
        # out = lowbit_kernel.bgemm_linear_forward_cuda(input, weight, 1, INSTRUCTION)  
        out = input @ weight.T
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input_, weight = ctx.saved_tensors
        bs = input_.shape[0]
        assert weight.dim() == 2, "weight dim must be 2, now is %d" % weight.dim()
        
        if input_.dim() == 3:
            weight_grad = torch.bmm(input_.transpose(1, 2), grad_output).transpose(1, 2)   # checked
            feat_grad = torch.bmm(grad_output, weight.repeat(bs, 1, 1))  # checked
        elif input_.dim() == 2:
            weight_grad = torch.mm(input_.T, grad_output).T # checked
            feat_grad = torch.mm(grad_output, weight)  # checked
        return feat_grad, weight_grad, None
    
    
class TestLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=False):
        super(TestLinear, self).__init__(in_channels, out_channels, bias)

    def forward(self, x):
        w = self.weight
        # bgemm
        # out = bgemm_linear.apply(x, w)
        # battn = BinActive.apply(x)
        # bv = BinActive.apply(w)
        # battn = (battn+1)/2
        # battn = torch.where(battn == -1., torch.zeros_like(battn), battn)

        # out = attn_matmul.apply(w.cuda(), x.cuda())
        matA = w.cuda()
        matB = x.cuda()
        matA = BinActive.apply(matA)
        matB = BinActive.apply(matB)
        matB = (matB+1) / 2
        # matB = torch.where(matB==0.5, torch.zeros_like(matB)+0.5, matB)

        # out = matB @ matA
        out = mm.apply(matB, matA)

        return out
    
    

class TestLinear2(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=False):
        super(TestLinear2, self).__init__(in_channels, out_channels, bias)

    def forward(self, x):
        w = self.weight
        # bgemm
        # out = bgemm_linear.apply(x, w)
        out = bgemm_linear_bireal.apply(x.cuda(), w.cuda())
        
        # out = torch.matmul(battn, bv).cuda()

        # bnn
        # x = BinActive.apply(x)
        # w = BinActive.apply(w)
        # out = nn.functional.linear(x, w)
        # import pdb; pdb.set_trace()

        return out



def test_linear():
    
    torch.manual_seed(0)

    x = torch.randn(32, 128).cuda()
    y = torch.randn(32).cuda()
    
    criterion = nn.BCEWithLogitsLoss()

    lin1 = nn.Sequential(
        nn.Linear(128, 128, False),        
        BNNLinear(128, 32, False),
    )
    lin1 = lin1.cuda()
    out1 = lin1(x)
    loss1 = criterion(out1.sum(dim=-1), y)

    loss1.backward()

    lin2 = nn.Sequential(
        nn.Linear(128, 128, False),        
        BGEMMLinear_bireal(128, 32, False),
    )
    lin2 = lin2.cuda()
    lin2[0].weight = nn.Parameter(lin1[0].weight.clone(), requires_grad=True)
    lin2[1].weight = nn.Parameter(lin1[1].weight.clone(), requires_grad=True)
    # lin2[2].weight = nn.Parameter(lin1[2].weight.clone(), requires_grad=True)
    out2 = lin2(x)
    loss2 = criterion(out2.sum(dim=-1), y)
    # import pdb; pdb.set_trace()
    
    loss2.backward()
    
    import pdb; pdb.set_trace()
    

def test_matmul():
    torch.manual_seed(0)
    q = torch.randn(32, 768).cuda()
    k = torch.randn(32, 768).cuda()
    bq = torch.sign(q)
    bk = torch.sign(k)
    attn1 = torch.matmul(bq, bk.transpose(-1, -2))
    attn2 = bgemm_matmul.apply(q, k.transpose(-1, -2))
    
    import pdb; pdb.set_trace()

def test_attn():
    torch.manual_seed(0)
    criterion = nn.BCEWithLogitsLoss()
    y = torch.randn(128).cuda()
    x = torch.randn(128, 32).cuda()
    
    lin1 = nn.Sequential(
        nn.Linear(32, 128, False),
        # nn.ReLU(),
        nn.Identity(),
        TestLinear(256, 128, False),
    )
    lin1.cuda()
    res1 = lin1(x)
    loss1 = criterion(res1.sum(dim=-1), y)
    loss1.backward()
    
    lin2 = nn.Sequential(
        nn.Linear(32, 128, False),
        # nn.ReLU(),
        nn.Identity(),
        TestLinear2(256, 128, False),
    )
    lin2.cuda()
    lin2[0].weight = nn.Parameter(lin1[0].weight.clone(), requires_grad=True)
    lin2[2].weight = nn.Parameter(lin1[2].weight.clone(), requires_grad=True)
    res2 = lin2(x)
    loss2 = criterion(res2.sum(dim=-1), y)
    loss2.backward()
    import pdb; pdb.set_trace()


# test_linear()
# test_matmul()
# test_attn()