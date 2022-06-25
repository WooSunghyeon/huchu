import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
from packbits import packbits_one_bits, unpackbits_one_bits
from Auxiliary_Activation_Learning import learning_indicator

cudnn_convolution = load(name="cudnn_convolution", sources=["../cudnn_conv.cpp"], verbose=True)
cudnn_batch_norm = load(name="cudnn_batchnorm", sources=["../batchnorm.cpp"], verbose=True)

#%%
'''
We make custom batchnorm layer.
Batchnorm2d can replace nn.BatchNorm2d
'''
class BatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.momentum = momentum
        self.one_minus_momentum = 1-momentum
    
    def forward(self, input):
        self._check_input_dim(input)
        return batch_norm2d.apply(input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.track_running_stats, self.momentum, self.eps)

class batch_norm2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, training, track_running_stats, momentum, eps):
        output, save_mean, save_var, reservedspace = cudnn_batch_norm.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps)
        ctx.save_for_backward(input, weight, running_mean, running_var, save_mean, save_var, reservedspace)
        ctx.eps =eps
        return output 
    
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        input, weight, running_mean, running_var, save_mean, save_var, reservedspace = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = cudnn_batch_norm.batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, eps, reservedspace)
        return grad_input, None, None, grad_weight, grad_bias, None, None, None, None

def batchnorm_forward(input, weight, bias, training, track_running_stats, running_mean, running_var, momentum, eps, backward = False):
    N = input.size(1) # channel
    input = input.permute(0,2,3,1).contiguous()
    input_shape = input.shape
    input = input.view(-1,N)
    if training:
        mu = input.mean(0)
        var = torch.var(input,0, unbiased=False)
        if track_running_stats and not(backward):
            running_mean.data = running_mean.mul(1-momentum).add(mu.mul(momentum)).data
            running_var.data = running_var.mul(1-momentum).add(var.mul(momentum)).data
        sqrt = torch.sqrt(var+eps).reciprocal()
        mu = mu.mul(sqrt)
        weight_div_sqrt = weight.mul(sqrt)
        y = input * weight_div_sqrt + bias.add(-mu*weight)
        return y.view(input_shape).permute(0,3,1,2).contiguous(), mu, weight_div_sqrt, sqrt
        
    else:
        y = input * weight.div(torch.sqrt(running_var+eps)) \
            + bias.add(-running_mean.div(torch.sqrt(running_var+eps)).mul(weight))
        return y.view(input_shape).permute(0,3,1,2).contiguous(), None, None, None


def batchnorm_backward(out, weight, bias, grad_output, mu_div_sqrt, weight_div_sqrt, sqrt, approximate_input = False):
    N = out.size(1) # channel
    out = out.permute(0,2,3,1).contiguous()
    out = out.view(-1,N)
    
    if approximate_input:
        out *= sqrt
        out -= mu_div_sqrt
    else:
        out -= bias
        out /= weight
        
    grad_out = grad_output.permute(0,2,3,1).contiguous()
    grad_shape = grad_out.shape
    grad_out = grad_out.view(-1, N)

    grad_bias = torch.sum(grad_out, 0)
    grad_weight = torch.sum(out*grad_out, 0)
    grad_input = weight_div_sqrt*(grad_out - grad_weight*out/grad_out.size(0) - grad_bias/grad_out.size(0) )
    
    grad_input = grad_input.view(grad_shape).permute(0,3,1,2).contiguous()
    return grad_input, grad_weight, grad_bias

#%%
'''
We make custom relu layer.
ReLU can replace nn.ReLU.
When packbits=True, the nn.ReLU just store packed Tensor instead of 1byte Tensor.
'''
class ReLU(nn.Module):
    def __init__(self, packbits = True):
        super(ReLU, self).__init__()
        self.packbits = packbits
        
    def forward(self, input):
        return relu.apply(input, self.packbits)

class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, packbits):
        input = input.clamp(min=0)
        output = input.clone()
        if packbits:
            input = (input>0).to(input.device)
            input, size = packbits_one_bits(input)
            ctx.size = size
        ctx.packbits = packbits
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        packbits = ctx.packbits
        input, =  ctx.saved_tensors
        if packbits:
            size = ctx.size
            input = unpackbits_one_bits(input, size)
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        return grad_input, None

#%%
class Conv2d_ET(nn.Conv2d):
    '''
    Conv2d_ABA uses auxiliary sign activation (ASA) to update weights in backward propagation.
    You can use get_li=True if you want to extract learning indicator.
    The epsilon controls the magnitude of ARA to make comparable value to output activation.
    The learning rate can be enlarged by setting lr_expansion.
    In this layer, we do not auxiliary_activation to input because it has been already added outside of the module.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, get_li=False):
        super(Conv2d_ET, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.li=None
        self.get_li=get_li
        
    def forward(self, input):
        ASA = input.sign().detach().clone()
        if self.get_li and self.training:
            self.li = learning_indicator(input, ASA, convnet=True)
        return conv2d_et.apply(input, ASA, self.weight, self.bias, self.stride, self.padding, self.groups) 

class conv2d_et(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, ASA, weight, bias, stride=1, padding=0, groups=1):
        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        #packbits
        ASA = (ASA>0).to(ASA.device)
        ASA, restore_size = packbits_one_bits(ASA)
        ctx.restore_size = restore_size
        ctx.save_for_backward(ASA, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        ASA, weight, bias = ctx.saved_tensors
        #unpackbit
        ASA = unpackbits_one_bits(ASA, ctx.restore_size, relu=True)
        grad_weight = cudnn_convolution.convolution_backward_weight(ASA*ctx.epsilon, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False,False)
        grad_weight *= ctx.lr_expansion
        grad_input = cudnn_convolution.convolution_backward_input(ASA.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False,False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        return grad_input, None, grad_weight, grad_bias, None, None, None, None,None, None, None

