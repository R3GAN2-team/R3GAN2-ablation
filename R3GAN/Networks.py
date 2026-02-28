import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .MagnitudePreservingLayers import LeakyReLU, Convolution, Linear, BiasedPointwiseConvolution, BiasedPointwiseConvolutionWithNoiseInjection, GenerativeBasis, DiscriminativeBasis, BoundedParameter, ClassEmbedder

class FeedForwardNetwork(nn.Module):
    def __init__(self, FirstConvolutionType, InputChannels, HiddenChannels, ChannelsPerGroup, KernelSize):
        super(FeedForwardNetwork, self).__init__()
        
        self.LinearLayer1 = FirstConvolutionType(InputChannels, HiddenChannels, Centered=True)
        self.LinearLayer2 = Convolution(HiddenChannels, HiddenChannels, KernelSize=KernelSize, Groups=HiddenChannels // ChannelsPerGroup, Centered=True)
        self.LinearLayer3 = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True)
        self.NonLinearity = LeakyReLU()
        
    def forward(self, x, InputGain, ResidualGain):
        y = self.LinearLayer1(x, Gain=InputGain.view(1, -1, 1, 1))
        y = self.LinearLayer2(self.NonLinearity(y))
        y = self.LinearLayer3(self.NonLinearity(y), Gain=ResidualGain.view(-1, 1, 1, 1))
        
        return x + y
    
class ResidualGroup(nn.Module):
    def __init__(self, InputChannels, BlockConstructors):
        super(ResidualGroup, self).__init__()
        
        self.Layers = nn.ModuleList([Block(**Arguments) for Block, Arguments in BlockConstructors])
        self.ParametrizedAlphas = nn.ModuleList([BoundedParameter(InputChannels) for _ in range(len(self.Layers))])

    def forward(self, x):
        AccumulatedVariance = torch.ones([]).to(x.device)
        for ParametrizedAlpha, Layer in zip(self.ParametrizedAlphas, self.Layers):
            Alpha = ParametrizedAlpha()
            x = Layer(x, InputGain=torch.rsqrt(AccumulatedVariance), ResidualGain=Alpha)
            AccumulatedVariance = AccumulatedVariance + Alpha * Alpha
        
        return x, AccumulatedVariance
    
class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()

        assert InputChannels == OutputChannels, "The current partial impl is only correct when InputChannels == OutputChannels"
        
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)

    def forward(self, x, Gain):
        x = x * Gain.view(1, -1, 1, 1).to(x.dtype)
        
        return self.Resampler(x)
        
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()

        assert InputChannels == OutputChannels, "The current partial impl is only correct when InputChannels == OutputChannels"
        
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)

    def forward(self, x, Gain):
        x = self.Resampler(x * Gain.view(1, -1, 1, 1).to(x.dtype))
        
        return x
        
class GenerativeHead(nn.Module):
    def __init__(self, InputDimension, OutputChannels, HiddenChannels, ChannelsPerGroup, ResamplingFilter):
        super(GenerativeHead, self).__init__()

        self.LinearLayer1 = Linear(InputDimension + 1, HiddenChannels, Centered=True)
        self.LinearLayer2 = GenerativeBasis(HiddenChannels, ChannelsPerGroup)
        self.LinearLayer3 = Convolution(HiddenChannels, OutputChannels, KernelSize=1, Centered=True)
        self.NonLinearity = LeakyReLU()
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
    def forward(self, x):
        y = self.LinearLayer1(torch.cat([x, torch.ones_like(x[:, :1])], dim=1))
        y = self.LinearLayer2(self.NonLinearity(y))
        y = self.LinearLayer3(self.NonLinearity(y))

        return self.Resampler(y)

class DiscriminativeHead(nn.Module):
    def __init__(self, InputChannels, OutputDimension, HiddenChannels, ChannelsPerGroup, ResamplingFilter):
        super(DiscriminativeHead, self).__init__()

        self.LinearLayer1 = BiasedPointwiseConvolution(InputChannels, HiddenChannels, Centered=True)
        self.LinearLayer2 = DiscriminativeBasis(HiddenChannels, ChannelsPerGroup)
        self.LinearLayer3 = Linear(HiddenChannels, OutputDimension, Centered=True)
        self.NonLinearity = LeakyReLU()
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)

    def forward(self, x, Gain):
        y = self.LinearLayer1(self.Resampler(x), Gain=Gain.view(1, -1, 1, 1))
        y = self.LinearLayer2(self.NonLinearity(y))
        y = self.LinearLayer3(self.NonLinearity(y))

        return y

def BuildResidualGroups(WidthPerStage, BlocksPerStage, FFNFirstConvolutionType, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize):
    ResidualGroups = []
    for Width, Blocks in zip(WidthPerStage, BlocksPerStage):
        BlockConstructors = []
        for BlockType in Blocks:
            if BlockType == 'FFN':
                BlockConstructors += [(FeedForwardNetwork, dict(FirstConvolutionType=FFNFirstConvolutionType, InputChannels=Width, HiddenChannels=round(Width * FFNWidthRatio), ChannelsPerGroup=ChannelsPerConvolutionGroup, KernelSize=KernelSize))]
            else:
                raise NotImplementedError('Unknown block type')
        ResidualGroups += [ResidualGroup(Width, BlockConstructors)]
    return ResidualGroups
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension, OutputChannels, WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, BiasedPointwiseConvolutionWithNoiseInjection, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize))
        self.TransitionLayers = nn.ModuleList([UpsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.Head = GenerativeHead(NoiseDimension + ClassEmbeddingDimension, WidthPerStage[0], round(WidthPerStage[0] * FFNWidthRatio), ChannelsPerConvolutionGroup, ResamplingFilter)
        self.AggregationLayer = Convolution(WidthPerStage[-1], OutputChannels, KernelSize=1)
        self.Gain = nn.Parameter(torch.ones([]))
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = ClassEmbedder(NumberOfClasses, ClassEmbeddingDimension)
        
    def forward(self, x, y=None):
        x = torch.cat([x, self.EmbeddingLayer(y)], dim=1) if hasattr(self, 'EmbeddingLayer') else x
        x = self.Head(x).to(torch.bfloat16)
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x, AccumulatedVariance = Layer(x)
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x)

        return self.AggregationLayer(x, Gain=self.Gain * torch.rsqrt(AccumulatedVariance).view(1, -1, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, InputChannels, WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, BiasedPointwiseConvolution, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize))
        self.TransitionLayers = nn.ModuleList([DownsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])
        
        self.Head = DiscriminativeHead(WidthPerStage[-1], 1 if NumberOfClasses is None else ClassEmbeddingDimension, round(WidthPerStage[-1] * FFNWidthRatio), ChannelsPerConvolutionGroup, ResamplingFilter)
        self.ExtractionLayer = Convolution(InputChannels, WidthPerStage[0], KernelSize=1)
        self.Gain = nn.Parameter(torch.ones([]))
        
        if NumberOfClasses is not None:
            self.EmbeddingLayer = ClassEmbedder(NumberOfClasses, ClassEmbeddingDimension)
        
    def forward(self, x, y=None):
        if hasattr(self, 'EmbeddingLayer'):
            y = self.EmbeddingLayer(y)
        x = self.ExtractionLayer(x.to(torch.bfloat16))
        
        for Layer, Transition in zip(self.MainLayers[:-1], self.TransitionLayers):
            x, AccumulatedVariance = Layer(x)
            x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
        x, AccumulatedVariance = self.MainLayers[-1](x)
        
        x = self.Head(x.to(torch.float32), Gain=torch.rsqrt(AccumulatedVariance))
        x = (x * y / math.sqrt(y.shape[1])).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return self.Gain * x.view(x.shape[0]), x.view(x.shape[0])