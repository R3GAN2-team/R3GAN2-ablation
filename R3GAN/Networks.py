import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .FusedOperators import BiasedActivation

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

class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, ChannelsPerGroup, KernelSize, VarianceScalingParameter):
        super(ResidualBlock, self).__init__()
        
        NumberOfLinearLayers = 3
        ActivationGain = BiasedActivation.Gain * VarianceScalingParameter ** (-1 / (2 * NumberOfLinearLayers - 2))
        
        self.LinearLayer1 = Convolution(InputChannels, HiddenChannels, KernelSize=1, ActivationGain=ActivationGain)
        self.LinearLayer2 = Convolution(HiddenChannels, HiddenChannels, KernelSize=KernelSize, Groups=HiddenChannels // ChannelsPerGroup, ActivationGain=ActivationGain)
        self.LinearLayer3 = Convolution(HiddenChannels, InputChannels, KernelSize=1, ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(HiddenChannels)
        self.NonLinearity2 = BiasedActivation(HiddenChannels)
        
    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return x + y
    
class ResidualGroup(nn.Module):
    def __init__(self, InputChannels, BlockConstructors):
        super(ResidualGroup, self).__init__()
        
        self.Layers = nn.ModuleList([Block(**Arguments) for Block, Arguments in BlockConstructors])

    def forward(self, x):
        for Layer in self.Layers:
            x = Layer(x)
            
        return x

class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        
    def forward(self, x):
        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        x = self.Resampler(x)
        
        return x
    
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        
    def forward(self, x):
        x = self.Resampler(x)
        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        
        return x
    
class GenerativeBasis(nn.Module):
    def __init__(self, InputDimension, OutputChannels):
        super(GenerativeBasis, self).__init__()
        
        self.Basis = nn.Parameter(torch.empty(OutputChannels, 4, 4).normal_(0, 1))
        self.LinearLayer = MSRInitializer(nn.Linear(InputDimension, OutputChannels, bias=False))
        
    def forward(self, x):
        return self.Basis.view(1, -1, 4, 4) * self.LinearLayer(x).view(x.shape[0], -1, 1, 1)
    
class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels, OutputDimension):
        super(DiscriminativeBasis, self).__init__()
        
        self.Basis = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
        self.LinearLayer = MSRInitializer(nn.Linear(InputChannels, OutputDimension, bias=False))
        
    def forward(self, x):
        return self.LinearLayer(self.Basis(x).view(x.shape[0], -1))
    
def BuildResidualGroups(WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, VarianceScalingParameter):
    ResidualGroups = []
    for Width, NumberOfBlocks in zip(WidthPerStage, BlocksPerStage):
        BlockConstructors = []
        for _ in range(NumberOfBlocks):
            BlockConstructors += [(ResidualBlock, dict(InputChannels=Width, HiddenChannels=round(Width * FFNWidthRatio), ChannelsPerGroup=ChannelsPerConvolutionGroup, KernelSize=KernelSize, VarianceScalingParameter=VarianceScalingParameter))]
        ResidualGroups += [ResidualGroup(Width, BlockConstructors)]
    return ResidualGroups
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension, OutputChannels, WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, sum(BlocksPerStage)))
        self.TransitionLayers = nn.ModuleList([UpsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])

        self.Head = GenerativeBasis(NoiseDimension + ClassEmbeddingDimension, WidthPerStage[0])
        self.AggregationLayer = Convolution(WidthPerStage[-1], OutputChannels, KernelSize=1)
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = MSRInitializer(nn.Linear(NumberOfClasses, ClassEmbeddingDimension, bias=False))
        
    def forward(self, x, y=None):
        x = torch.cat([x, self.EmbeddingLayer(y)], dim=1) if hasattr(self, 'EmbeddingLayer') else x
        x = self.Head(x).to(torch.bfloat16)
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x = Layer(x)
            x = Transition(x)
        x = self.MainLayers[-1](x)

        return self.AggregationLayer(x)

class Discriminator(nn.Module):
    def __init__(self, InputChannels, WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize, sum(BlocksPerStage)))
        self.TransitionLayers = nn.ModuleList([DownsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])

        self.Head = DiscriminativeBasis(WidthPerStage[-1], 1 if NumberOfClasses is None else ClassEmbeddingDimension)
        self.ExtractionLayer = Convolution(InputChannels, WidthPerStage[0], KernelSize=1)
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = MSRInitializer(nn.Linear(NumberOfClasses, ClassEmbeddingDimension, bias=False), ActivationGain=1 / math.sqrt(ClassEmbeddingDimension))
        
    def forward(self, x, y=None):
        x = self.ExtractionLayer(x.to(torch.bfloat16))
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x = Layer(x)
            x = Transition(x)
        x = self.MainLayers[-1](x)
        
        x = self.Head(x.to(torch.float32))
        x = (x * self.EmbeddingLayer(y)).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.view(x.shape[0])