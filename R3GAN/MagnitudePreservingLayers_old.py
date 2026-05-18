import torch
import torch.nn as nn
import math
from torch_utils.ops import bias_act

def Normalize(x, Dimensions=None, ε=1e-4):
    if Dimensions is None:
        Dimensions = list(range(1, x.ndim))
    Norm = torch.linalg.vector_norm(x, dim=Dimensions, keepdim=True, dtype=torch.float32)
    Norm = torch.add(ε, Norm, alpha=math.sqrt(Norm.numel() / x.numel()))
    return x / Norm.to(x.dtype)
    
class LeakyReLU(nn.Module):
    def __init__(self, α=0.2):
        super(LeakyReLU, self).__init__()
        
        self.α = α
        self.Gain = 1 / math.sqrt(((1 + α ** 2) - (1 - α) ** 2 / math.pi) / 2)

    def forward(self, x):
        return bias_act.bias_act(x, None, act='lrelu', alpha=self.α, gain=self.Gain)

class BoundedParameter(nn.Module):
    def __init__(self, Dimension, Bound=1):
        super(BoundedParameter, self).__init__()
        
        self.Value = nn.Parameter(torch.zeros(Dimension))
        self.Bound = Bound
        
    def forward(self):
        return self.Bound * torch.tanh(self.Value / self.Bound)

class NormalizedWeight(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Groups, KernelSize, Centered):
        super(NormalizedWeight, self).__init__()
        
        self.Centered = Centered
        self.Weight = nn.Parameter(torch.randn(OutputChannels, InputChannels // Groups, *KernelSize))
        
    def Evaluate(self, w):
        if self.Centered:
            w = w - torch.mean(w, axis=list(range(1, w.ndim)), keepdim=True)
        return Normalize(w)
        
    def forward(self):
        return self.Evaluate(self.Weight.to(torch.float32))
    
    def NormalizeWeight(self):
        self.Weight.copy_(self.Evaluate(self.Weight.detach()))

class WeightNormalizedConvolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Groups, EnablePadding, KernelSize, Centered):
        super(WeightNormalizedConvolution, self).__init__()
        
        self.Groups = Groups
        self.EnablePadding = EnablePadding
        self.Weight = NormalizedWeight(InputChannels, OutputChannels, Groups, KernelSize, Centered)

    def forward(self, x, Gain=1):
        w = self.Weight()
        w = w * (Gain / math.sqrt(w[0].numel()))
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        return nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,) if self.EnablePadding else 0, groups=self.Groups)
        
def Convolution(InputChannels, OutputChannels, KernelSize, Groups=1, Centered=False):
    return WeightNormalizedConvolution(InputChannels, OutputChannels, Groups, True, [KernelSize, KernelSize], Centered)

def Linear(InputDimension, OutputDimension, Centered=False):
    return WeightNormalizedConvolution(InputDimension, OutputDimension, 1, False, [], Centered)

class BiasedPointwiseConvolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Centered=False):
        super(BiasedPointwiseConvolution, self).__init__()
        
        self.Weight = NormalizedWeight(InputChannels + 1, OutputChannels, 1, [1, 1], Centered)
        
    def forward(self, x, Gain=1):
        w = self.Weight()
        w = w / math.sqrt(w[0].numel())
        b = w[:, -1, :, :].view(-1)
        w = w[:, :-1, :, :] * Gain
        
        return nn.functional.conv2d(x, w.to(x.dtype), b.to(x.dtype))
    
class BiasedPointwiseConvolutionWithNoiseInjection(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Centered=False):
        super(BiasedPointwiseConvolutionWithNoiseInjection, self).__init__()
        
        self.Weight = NormalizedWeight(InputChannels + 2, OutputChannels, 1, [1, 1], Centered)
        
    def forward(self, x, Gain=1):
        w = self.Weight()
        w = w / math.sqrt(w[0].numel())
        b = w[:, -1, :, :].view(-1)
        s = w[:, -2, :, :].view(-1)
        w = w[:, :-2, :, :] * Gain
        n = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], device=x.device)
        
        return nn.functional.conv2d(x, w.to(x.dtype), b.to(x.dtype)).add_(n * s.view(1, -1, 1, 1))

class GenerativeBasis(nn.Module):
    def __init__(self, OutputChannels, ChannelsPerGroup):
        super(GenerativeBasis, self).__init__()
        
        self.Basis = NormalizedWeight(OutputChannels, OutputChannels, OutputChannels // ChannelsPerGroup, [4, 4], True)
        self.ChannelsPerGroup = ChannelsPerGroup
        
    def forward(self, x):
        w = self.Basis()
        x = x.view(x.shape[0], -1, self.ChannelsPerGroup)
        w = w.view(x.shape[1], -1, *w.shape[1:]) / math.sqrt(x.shape[-1])
        x = torch.einsum('ngc,gochw->ngohw', x, w).contiguous()
        
        return x.view(x.shape[0], -1, *x.shape[3:])

class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels, ChannelsPerGroup):
        super(DiscriminativeBasis, self).__init__()
        
        self.Basis = WeightNormalizedConvolution(InputChannels, InputChannels, InputChannels // ChannelsPerGroup, False, [4, 4], True)
        
    def forward(self, x):
        return self.Basis(x).view(x.shape[0], -1)
    
class ClassEmbedder(nn.Module):
    def __init__(self, NumberOfClasses, EmbeddingDimension):
        super(ClassEmbedder, self).__init__()
        
        self.Weight = NormalizedWeight(EmbeddingDimension, NumberOfClasses, 1, [], True)
        self.Weight.Weight.data.copy_(NormalizedWeight(EmbeddingDimension, 1, 1, [], True)().repeat(NumberOfClasses, 1))
    
    def forward(self, x):
        return x @ self.Weight().to(x.dtype)