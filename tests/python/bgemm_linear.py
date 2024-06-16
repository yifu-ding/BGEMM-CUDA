import torch
import torch.nn as nn
import lowbit_kernel


class bgemm_linear(torch.autograd.Function):
    """
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
    """
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
        ctx.save_for_backward(input, weight)
        INSTRUCTION = 0 if instruction=='and' else 1 # 1 by default
        out = lowbit_kernel.bgemm_linear_forward_cuda(input, weight, 1, INSTRUCTION)  
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


class BGEMMLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=True):
        super(BGEMMLinear, self).__init__(in_channels, out_channels, bias)
        # self.sw = nn.Parameter(torch.randn(out_channels), requires_grad=True)
        self.initialized = False
    
    def _initialize(self, x, w):
        assert not self.initialized, 'already initialized.'
        self.sw = nn.Parameter(w.norm(1, 1).div(w.nelement()), requires_grad=True)
        self.sa = nn.Parameter(2 * x.abs().mean() , requires_grad=True)
        self.zpw = 0  # nn.Parameter(torch.mean(w), requires_grad=False)
        self.zpa = 0.5 # nn.Parameter(torch.mean(x), requires_grad=False)
        
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


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations for ***** BNN and XNOR *****.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input
    @staticmethod
    def backward(ctx, grad_output, ):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input.ge(1)] = 0
        # grad_input[input.le(-1)] = 0
        return grad_input


class BNNLinear(nn.Linear):

    def __init__(self, in_channels, out_channels, bias=False):
        super(BNNLinear, self).__init__(in_channels, out_channels, bias)

    def forward(self, x):
        w = self.weight

        bw = BinActive().apply(w)
        bx = BinActive().apply(x)
        # import pdb; pdb.set_trace()

        output = nn.functional.linear(bx, bw, self.bias)
        out = bx @ bw.T + self.bias

        return out
    

class test_linear(torch.autograd.Function):
    
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
        out = bgemm_linear.apply(x, w)
        
        # bnn
        # x = BinActive.apply(x)
        # w = BinActive.apply(w)
        # out = nn.functional.linear(x, w)
        import pdb; pdb.set_trace()
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


def main():
    torch.manual_seed(0)

    linear = TestLinear
    criterion = nn.BCEWithLogitsLoss()

    mlp = nn.Sequential(
        linear(128, 128, False),
        nn.ReLU(),
        linear(128, 128, False),
    )
    
    mlp = mlp.cuda()
    
    # x = torch.randn(8, 2, 4)
    # y = torch.randn(8, 2)
    
    x = torch.randn(128, 128).cuda()
    y = torch.randn(512).cuda()
    
    
    out = mlp(x)
    # loss = ((y - out)**2).sum()
    loss = criterion(out, y.unsqueeze(1))
    # loss = criterion(out, y.unsqueeze(2))
    # import pdb; pdb.set_trace()
    
    loss.backward()
    # import pdb; pdb.set_trace()
    
    
# main()