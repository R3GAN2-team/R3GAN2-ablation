import torch
import torch.nn as nn
import math

def Normalize(x, Dimensions=None, ε=1e-4):
    if Dimensions is None:
        Dimensions = list(range(1, x.ndim))
    Norm = torch.linalg.vector_norm(x, ord=1, dim=Dimensions, keepdim=True, dtype=torch.float32)
    Norm = torch.add(ε, Norm, alpha=Norm.numel() / x.numel())
    return x / Norm.to(x.dtype)

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
        w = w * (Gain / w[0].numel())
        w = w.to(x.dtype)

        if w.ndim == 2:
            return x @ w.t()
        return nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,) if self.EnablePadding else 0, groups=self.Groups)
        
def Convolution(InputChannels, OutputChannels, KernelSize, Groups=1, Centered=False):
    return WeightNormalizedConvolution(InputChannels, OutputChannels, Groups, True, [KernelSize, KernelSize], Centered)

class BiasedPointwiseConvolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Centered=False):
        super(BiasedPointwiseConvolution, self).__init__()
        
        self.Weight = NormalizedWeight(InputChannels + 1, OutputChannels, 1, [1, 1], Centered)
        
    def forward(self, x, Gain=1):
        w = self.Weight()
        w = w / w[0].numel()
        b = w[:, -1, :, :].view(-1)
        w = w[:, :-1, :, :] * Gain
        
        return nn.functional.conv2d(x, w.to(x.dtype), b.to(x.dtype))