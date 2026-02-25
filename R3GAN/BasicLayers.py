import torch
import torch.nn as nn
import math

class LeakyReLU(nn.Module):
    def __init__(self, α=0.2):
        super(LeakyReLU, self).__init__()
        
        self.α = α
        self.Gain = 1 / math.sqrt(((1 + α ** 2) - (1 - α) ** 2 / math.pi) / 2)

    def forward(self, x):
        return nn.functional.leaky_relu(x, negative_slope=self.α, inplace=True)

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class Convolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, KernelSize, Groups=1, ActivationGain=1):
        super(Convolution, self).__init__()
        
        self.Layer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=KernelSize, stride=1, padding=(KernelSize - 1) // 2, groups=Groups, bias=False), ActivationGain=ActivationGain)
        
    def forward(self, x):
        return nn.functional.conv2d(x, self.Layer.weight.to(x.dtype), padding=self.Layer.padding, groups=self.Layer.groups)

def Linear(InputDimension, OutputDimension, ActivationGain=1):
    return MSRInitializer(nn.Linear(InputDimension, OutputDimension, bias=False), ActivationGain=ActivationGain)

class BiasedPointwiseConvolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ActivationGain=1):
        super(BiasedPointwiseConvolution, self).__init__()

        self.Layer = MSRInitializer(nn.Conv2d(InputChannels + 1, OutputChannels, kernel_size=1, stride=1, padding=0, groups=1, bias=False), ActivationGain=ActivationGain)
        
    def forward(self, x):
        w = self.Layer.weight
        b = w[:, -1, :, :].view(-1)
        w = w[:, :-1, :, :]

        return nn.functional.conv2d(x, w.to(x.dtype), b.to(x.dtype))

class BiasedPointwiseConvolutionWithNoiseInjection(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ActivationGain=1):
        super(BiasedPointwiseConvolutionWithNoiseInjection, self).__init__()

        self.Layer = MSRInitializer(nn.Conv2d(InputChannels + 2, OutputChannels, kernel_size=1, stride=1, padding=0, groups=1, bias=False), ActivationGain=ActivationGain)
        
    def forward(self, x):
        w = self.Layer.weight
        b = w[:, -1, :, :].view(-1)
        s = w[:, -2, :, :].view(-1)
        w = w[:, :-2, :, :]
        n = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], device=x.device)

        return nn.functional.conv2d(x, w.to(x.dtype), b.to(x.dtype)).add_(n * s.view(1, -1, 1, 1))

class GenerativeBasis(nn.Module):
    def __init__(self, OutputChannels):
        super(GenerativeBasis, self).__init__()
        
        self.Basis = nn.Parameter(torch.empty(OutputChannels, 4, 4).normal_(0, 1))
        
    def forward(self, x):
        return self.Basis.view(1, -1, 4, 4) * x.view(x.shape[0], -1, 1, 1)

class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels):
        super(DiscriminativeBasis, self).__init__()
        
        self.Basis = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
        
    def forward(self, x):
        return self.Basis(x).view(x.shape[0], -1)