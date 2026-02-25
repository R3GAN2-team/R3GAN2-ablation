import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .BasicLayers import LeakyReLU, Convolution, Linear, BiasedPointwiseConvolution, GenerativeBasis, DiscriminativeBasis

class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, HiddenChannels, ChannelsPerGroup, KernelSize, VarianceScalingParameter):
        super(ResidualBlock, self).__init__()
        
        NumberOfLinearLayers = 3
        ActivationGain = LeakyReLU().Gain * VarianceScalingParameter ** (-1 / (2 * NumberOfLinearLayers - 2))
        
        self.LinearLayer1 = BiasedPointwiseConvolution(InputChannels, HiddenChannels, ActivationGain=ActivationGain)
        self.LinearLayer2 = Convolution(HiddenChannels, HiddenChannels, KernelSize=KernelSize, Groups=HiddenChannels // ChannelsPerGroup, ActivationGain=ActivationGain)
        self.LinearLayer3 = Convolution(HiddenChannels, InputChannels, KernelSize=1, ActivationGain=0)
        self.NonLinearity = LeakyReLU()
        
    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity(y))
        y = self.LinearLayer3(self.NonLinearity(y))
        
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

        assert InputChannels == OutputChannels, "The current partial impl is only correct when InputChannels == OutputChannels"
        
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
    def forward(self, x):
        x = self.Resampler(x)
        
        return x
    
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()

        assert InputChannels == OutputChannels, "The current partial impl is only correct when InputChannels == OutputChannels"
        
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        
    def forward(self, x):
        x = self.Resampler(x)
        
        return x
    
class GenerativeHead(nn.Module):
    def __init__(self, InputDimension, OutputChannels, ResamplingFilter):
        super(GenerativeHead, self).__init__()
        
        self.Basis = GenerativeBasis(OutputChannels)
        self.LinearLayer = Linear(InputDimension, OutputChannels)
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
    def forward(self, x):
        return self.Resampler(self.Basis(self.LinearLayer(x)))
    
class DiscriminativeHead(nn.Module):
    def __init__(self, InputChannels, OutputDimension, ResamplingFilter):
        super(DiscriminativeHead, self).__init__()
        
        self.Basis = DiscriminativeBasis(InputChannels)
        self.LinearLayer = Linear(InputChannels, OutputDimension)
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        
    def forward(self, x):
        return self.LinearLayer(self.Basis(self.Resampler(x)))
    
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

        self.Head = GenerativeHead(NoiseDimension + ClassEmbeddingDimension, WidthPerStage[0], ResamplingFilter)
        self.AggregationLayer = Convolution(WidthPerStage[-1], OutputChannels, KernelSize=1)
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = Linear(NumberOfClasses, ClassEmbeddingDimension)
        
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

        self.Head = DiscriminativeHead(WidthPerStage[-1], 1 if NumberOfClasses is None else ClassEmbeddingDimension, ResamplingFilter)
        self.ExtractionLayer = Convolution(InputChannels, WidthPerStage[0], KernelSize=1)
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = Linear(NumberOfClasses, ClassEmbeddingDimension, ActivationGain=1 / math.sqrt(ClassEmbeddingDimension))
        
    def forward(self, x, y=None):
        x = self.ExtractionLayer(x.to(torch.bfloat16))
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x = Layer(x)
            x = Transition(x)
        x = self.MainLayers[-1](x)
        
        x = self.Head(x.to(torch.float32))
        x = (x * self.EmbeddingLayer(y)).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.view(x.shape[0])