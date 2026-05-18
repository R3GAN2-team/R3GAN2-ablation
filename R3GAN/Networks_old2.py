import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .MagnitudePreservingLayers import (
    LeakyReLU,
    Convolution,
    Linear,
    BiasedPointwiseConvolution,
    BiasedPointwiseConvolutionWithNoiseInjection,
    GenerativeBasis,
    DiscriminativeBasis,
    BoundedParameter,
    ClassEmbedder,
)


class UnscaledLeakyReLU(nn.Module):
    """LeakyReLU without the magnitude-preserving gain in forward().

    The gain is folded into the following linear/convolutional layer inside
    FeedForwardNetwork.  This is mathematically equivalent to the old scaled
    LeakyReLU FFN, but in BF16 it may differ slightly because the gain is rounded
    at a different point in the computation.
    """
    def __init__(self, α=0.2):
        super(UnscaledLeakyReLU, self).__init__()

        self.α = α
        self.Gain = 1 / math.sqrt(((1 + α ** 2) - (1 - α) ** 2 / math.pi) / 2)

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.α, inplace=True)

    def Slope(self, x):
        pos = torch.full((), 1.0, dtype=x.dtype, device=x.device)
        neg = torch.full((), self.α, dtype=x.dtype, device=x.device)
        return torch.where(x >= 0, pos, neg)


class FeedForwardNetwork(nn.Module):
    def __init__(self, FirstConvolutionType, InputChannels, HiddenChannels, ChannelsPerGroup, KernelSize):
        super(FeedForwardNetwork, self).__init__()

        self.LinearLayer1 = FirstConvolutionType(InputChannels, HiddenChannels, Centered=True)
        self.LinearLayer2 = Convolution(HiddenChannels, HiddenChannels, KernelSize=KernelSize, Groups=HiddenChannels // ChannelsPerGroup, Centered=True)
        self.LinearLayer3 = Convolution(HiddenChannels, InputChannels, KernelSize=1, Centered=True)
        self.NonLinearity = UnscaledLeakyReLU()

    def forward(self, x, InputGain, ResidualGain):
        y = self.LinearLayer1(x, Gain=InputGain.view(1, -1, 1, 1))
        y = self.LinearLayer2(self.NonLinearity(y), Gain=self.NonLinearity.Gain)
        y = self.LinearLayer3(self.NonLinearity(y), Gain=self.NonLinearity.Gain * ResidualGain.view(-1, 1, 1, 1))

        return x + y

    # ---------------------------------------------------------------------
    # Helpers for explicit R1/VJP. These are intentionally not used by the
    # ordinary forward() path above, so normal G/D behavior is unchanged.
    # ---------------------------------------------------------------------

    @staticmethod
    def _pointwise_effective_weight_bias(layer, dtype, gain):
        if hasattr(layer, 'EffectiveWeightBias'):
            return layer.EffectiveWeightBias(dtype, Gain=gain)
        if hasattr(layer, 'EffectiveWeightBiasNoiseScale'):
            w, b, _ = layer.EffectiveWeightBiasNoiseScale(dtype, Gain=gain)
            return w, b
        raise TypeError(f'{type(layer).__name__} does not expose an effective pointwise weight/bias helper')

    def forward_with_cache(self, x, InputGain, ResidualGain):
        """Run forward() while caching tensors needed by explicit_vjp().

        Only cache tensors needed by explicit_vjp or already retained by the
        ordinary autograd graph.  Because NonLinearity is in-place, y1/y2 are
        post-activation tensors by the time explicit_vjp reads them; this is OK
        for LeakyReLU with α > 0 because sign is preserved.
        """
        input_gain = InputGain.view(1, -1, 1, 1)
        residual_gain = ResidualGain.view(-1, 1, 1, 1)

        y1 = self.LinearLayer1(x, Gain=input_gain)
        a1 = self.NonLinearity(y1)
        y2 = self.LinearLayer2(a1, Gain=self.NonLinearity.Gain)
        a2 = self.NonLinearity(y2)
        y3 = self.LinearLayer3(a2, Gain=self.NonLinearity.Gain * residual_gain)
        out = x + y3

        cache = dict(
            Layer=self,
            y1=y1,
            y2=y2,
            input_gain=input_gain,
            residual_gain=residual_gain,
        )
        return out, cache

    def explicit_vjp(self, v_out, cache, KeepInputDependency=False):
        """Explicit VJP of this residual FFN wrt its input.

        This implements v_in = J_forward(x)^T v_out for the piecewise-linear
        discriminator FFN. It avoids differentiating through PyTorch's generic
        conv backward graph for R1. The returned tensor remains differentiable
        wrt effective weights and gains.
        """
        dtype = v_out.dtype

        # Residual skip: y = x + branch(x), so the skip VJP is identity.
        v_x = v_out
        v = v_out

        # LinearLayer3: [hidden -> input] 1x1 convolution.
        w3 = self.LinearLayer3.EffectiveWeight(dtype, Gain=self.NonLinearity.Gain * cache['residual_gain'])
        v = F.conv_transpose2d(v, w3, padding=self.LinearLayer3.Padding(w3), groups=self.LinearLayer3.Groups)

        # Activation 2.
        v = v * self.NonLinearity.Slope(cache['y2']).to(dtype)

        # LinearLayer2: grouped 3x3 convolution.
        w2 = self.LinearLayer2.EffectiveWeight(dtype, Gain=self.NonLinearity.Gain)
        v = F.conv_transpose2d(v, w2, padding=self.LinearLayer2.Padding(w2), groups=self.LinearLayer2.Groups)

        # Activation 1.
        v = v * self.NonLinearity.Slope(cache['y1']).to(dtype)

        # LinearLayer1: biased pointwise convolution; bias has no input VJP.
        w1, _ = self._pointwise_effective_weight_bias(self.LinearLayer1, dtype, cache['input_gain'])
        v = F.conv_transpose2d(v, w1)

        v_x = v_x + v
        if KeepInputDependency and 'x' in cache:
            v_x = v_x + cache['x'] * 0
        return v_x


class ResidualGroup(nn.Module):
    def __init__(self, InputChannels, BlockConstructors):
        super(ResidualGroup, self).__init__()

        self.Layers = nn.ModuleList([Block(**Arguments) for Block, Arguments in BlockConstructors])
        self.ParametrizedAlphas = nn.ModuleList([BoundedParameter(InputChannels) for _ in range(len(self.Layers))])

    def _forward_impl(self, x):
        AccumulatedVariance = torch.ones([], device=x.device)
        for ParametrizedAlpha, Layer in zip(self.ParametrizedAlphas, self.Layers):
            Alpha = ParametrizedAlpha()
            x = Layer(x, InputGain=torch.rsqrt(AccumulatedVariance), ResidualGain=Alpha)
            AccumulatedVariance = AccumulatedVariance + Alpha * Alpha

        return x, AccumulatedVariance

    def forward(self, x):
        CompiledForward = getattr(self, '_CompiledForward', None)
        if CompiledForward is not None:
            return CompiledForward(x)
        return self._forward_impl(x)

    def ForwardUncompiled(self, x):
        return self._forward_impl(x)

    def CompileForward(self, mode='default', fullgraph=False, dynamic=False):
        # Compile only the ordinary forward path. Explicit-R1 cache/VJP helpers
        # intentionally remain eager and unchanged.
        self._CompiledForward = torch.compile(
            self._forward_impl,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return self

    def ClearCompiledForward(self):
        if hasattr(self, '_CompiledForward'):
            del self._CompiledForward
        return self

    def __getstate__(self):
        # Do not pickle/deepcopy torch.compile callables into snapshots.
        state = self.__dict__.copy()
        state.pop('_CompiledForward', None)
        return state

    def forward_with_cache(self, x):
        """Forward with per-block caches for explicit_vjp_from_cache()."""
        AccumulatedVariance = torch.ones([], device=x.device)
        Caches = []
        for BlockIndex, (ParametrizedAlpha, Layer) in enumerate(zip(self.ParametrizedAlphas, self.Layers)):
            Alpha = ParametrizedAlpha()
            InputGain = torch.rsqrt(AccumulatedVariance)
            x, Cache = Layer.forward_with_cache(x, InputGain=InputGain, ResidualGain=Alpha)
            NewAccumulatedVariance = AccumulatedVariance + Alpha * Alpha
            Caches.append(Cache)
            AccumulatedVariance = NewAccumulatedVariance

        return x, AccumulatedVariance, Caches

    def explicit_vjp_from_cache(self, v, Caches, KeepInputDependency=False):
        """Propagate a cotangent through this residual group in reverse."""
        for Cache in reversed(Caches):
            v = Cache['Layer'].explicit_vjp(v, Cache, KeepInputDependency=KeepInputDependency)
        return v


class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()

        assert InputChannels == OutputChannels, "The current partial impl is only correct when InputChannels == OutputChannels"

        self.Resampler = InterpolativeUpsampler(ResamplingFilter)

    def forward(self, x, Gain):
        x = x * Gain.view(1, -1, 1, 1).to(x.dtype)
        return self.Resampler(x)

    def forward_with_cache(self, x, Gain):
        gain = Gain.view(1, -1, 1, 1).to(x.dtype)
        z = x * gain
        y = self.Resampler(z)
        cache = dict(
            x_shape=x.shape,
            z_shape=z.shape,
            y_shape=y.shape,
            Gain=Gain,
            gain=gain,
        )
        return y, cache

    def explicit_vjp(self, v, cache):
        v = self.Resampler.explicit_vjp(v, input_shape=cache['z_shape'])
        v = v * cache['gain'].to(v.dtype)
        return v


class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()

        assert InputChannels == OutputChannels, "The current partial impl is only correct when InputChannels == OutputChannels"

        self.Resampler = InterpolativeDownsampler(ResamplingFilter)

    def forward(self, x, Gain):
        x = self.Resampler(x * Gain.view(1, -1, 1, 1).to(x.dtype))
        return x

    def forward_with_cache(self, x, Gain):
        gain = Gain.view(1, -1, 1, 1).to(x.dtype)
        z = x * gain
        y = self.Resampler(z)
        cache = dict(
            x_shape=x.shape,
            z_shape=z.shape,
            y_shape=y.shape,
            Gain=Gain,
            gain=gain,
        )
        return y, cache

    def explicit_vjp(self, v, cache):
        v = self.Resampler.explicit_vjp(v, input_shape=cache['z_shape'])
        v = v * cache['gain'].to(v.dtype)
        return v


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

    def forward_with_cache(self, x, Gain):
        """Forward with cache for explicit VJP.

        Normal forward() is unchanged. This helper is for full-discriminator
        explicit R1 only.
        """
        gain = Gain.view(1, -1, 1, 1)

        r = self.Resampler(x)
        y1 = self.LinearLayer1(r, Gain=gain)
        a1 = self.NonLinearity(y1)

        y2 = self.LinearLayer2(a1)
        a2 = self.NonLinearity(y2)

        y3 = self.LinearLayer3(a2)

        cache = dict(
            x=x,
            Gain=Gain,
            gain=gain,
            r=r,
            y1=y1,
            a1=a1,
            y2=y2,
            a2=a2,
            y3=y3,
        )
        return y3, cache

    def explicit_vjp(self, v, cache, KeepInputDependency=False):
        """Explicit VJP through the discriminator head.

        Head forward:
            x -> Resampler -> biased 1x1 -> lrelu -> DiscriminativeBasis
              -> lrelu -> Linear

        This returns the cotangent wrt x.
        """
        # LinearLayer3: forward is a2 @ w.T.
        w3 = self.LinearLayer3.EffectiveWeight(v.dtype)
        v = v @ w3

        # Activation 2. Shape: [B, hidden].
        v = v * self.NonLinearity.Slope(cache['y2']).to(v.dtype)

        # DiscriminativeBasis VJP.
        # Forward: Basis(a1).view(B, -1), where Basis output is [B, C, 1, 1].
        basis = self.LinearLayer2.Basis
        w2 = basis.EffectiveWeight(cache['a1'].dtype)
        v = v.view(v.shape[0], w2.shape[0], 1, 1)
        v = F.conv_transpose2d(
            v,
            w2,
            padding=basis.Padding(w2),
            groups=basis.Groups,
        )

        # Activation 1. Shape: [B, hidden, 4, 4].
        v = v * self.NonLinearity.Slope(cache['y1']).to(v.dtype)

        # Biased pointwise convolution VJP; bias has no input VJP.
        w1, _ = self.LinearLayer1.EffectiveWeightBias(v.dtype, Gain=cache['gain'])
        v = F.conv_transpose2d(v, w1)

        # Resampler VJP: 4x4 -> 8x8.
        v = self.Resampler.explicit_vjp(v, input_shape=cache['x'].shape)

        if KeepInputDependency:
            v = v + cache['x'] * 0
        return v


def BuildResidualGroups(WidthPerStage, BlocksPerStage, FFNFirstConvolutionType, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize):
    ResidualGroups = []
    for Width, Blocks in zip(WidthPerStage, BlocksPerStage):
        BlockConstructors = []
        for BlockType in Blocks:
            if BlockType == 'FFN':
                BlockConstructors += [(FeedForwardNetwork, dict(
                    FirstConvolutionType=FFNFirstConvolutionType,
                    InputChannels=Width,
                    HiddenChannels=round(Width * FFNWidthRatio),
                    ChannelsPerGroup=ChannelsPerConvolutionGroup,
                    KernelSize=KernelSize,
                ))]
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

    def CompileMainLayers(self, mode='default', fullgraph=False, dynamic=False):
        for Layer in self.MainLayers:
            if hasattr(Layer, 'CompileForward'):
                Layer.CompileForward(mode=mode, fullgraph=fullgraph, dynamic=dynamic)
        return self

    def ClearCompiledMainLayers(self):
        for Layer in self.MainLayers:
            if hasattr(Layer, 'ClearCompiledForward'):
                Layer.ClearCompiledForward()
        return self


class Discriminator(nn.Module):
    def __init__(self, InputChannels, WidthPerStage, BlocksPerStage, FFNWidthRatio, ChannelsPerConvolutionGroup, NumberOfClasses=None, ClassEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()

        self.MainLayers = nn.ModuleList(BuildResidualGroups(WidthPerStage, BlocksPerStage, BiasedPointwiseConvolution, FFNWidthRatio, ChannelsPerConvolutionGroup, KernelSize))
        self.TransitionLayers = nn.ModuleList([DownsampleLayer(WidthPerStage[x], WidthPerStage[x + 1], ResamplingFilter) for x in range(len(WidthPerStage) - 1)])

        self.Head = DiscriminativeHead(WidthPerStage[-1], 1 if NumberOfClasses is None else ClassEmbeddingDimension, round(WidthPerStage[-1] * FFNWidthRatio), ChannelsPerConvolutionGroup, ResamplingFilter)
        self.ExtractionLayer = Convolution(InputChannels, WidthPerStage[0], KernelSize=1)

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

        return x.view(x.shape[0])

    # ------------------------------------------------------------------
    # Split-forward helpers for explicit R1.
    # Normal forward() above is unchanged.
    # ------------------------------------------------------------------

    def EmbedConditions(self, y=None):
        return self.EmbeddingLayer(y) if hasattr(self, 'EmbeddingLayer') else y

    def ForwardToStage0(self, x, y=None):
        y = self.EmbedConditions(y)
        x = self.ExtractionLayer(x.to(torch.bfloat16))
        return x, y

    def ForwardStage0WithCache(self, x):
        return self.MainLayers[0].forward_with_cache(x)

    def ForwardFromAfterStage(self, x, AccumulatedVariance, y=None, StageIndex=0, UseCompiled=True):
        """Continue discriminator forward after MainLayers[StageIndex].

        For StageIndex=0, x is the output of MainLayers[0] and the method starts
        at TransitionLayers[0].  This method remains for stage-0-only explicit R1
        and for wrappers/tests.
        """
        for Index in range(StageIndex, len(self.TransitionLayers)):
            x = self.TransitionLayers[Index](x, Gain=torch.rsqrt(AccumulatedVariance))
            Layer = self.MainLayers[Index + 1]
            if UseCompiled:
                x, AccumulatedVariance = Layer(x)
            else:
                if hasattr(Layer, 'ForwardUncompiled'):
                    x, AccumulatedVariance = Layer.ForwardUncompiled(x)
                else:
                    x, AccumulatedVariance = Layer(x)

        x = self.Head(x.to(torch.float32), Gain=torch.rsqrt(AccumulatedVariance))
        x = (x * y / math.sqrt(y.shape[1])).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        return x.view(x.shape[0])

    def ForwardWithStage0Cache(self, x, y=None):
        x, y = self.ForwardToStage0(x, y)
        x, AccumulatedVariance, Caches = self.ForwardStage0WithCache(x)
        logits = self.ForwardFromAfterStage(x, AccumulatedVariance, y, StageIndex=0)
        return logits, Caches, x, AccumulatedVariance, y

    # ------------------------------------------------------------------
    # Explicit-R1 helpers.
    # ------------------------------------------------------------------

    def ForwardFromAfterStage0(self, x, AccumulatedVariance, y=None, UseCompiled=True):
        """Alias for continuing after stage 0; useful for wrappers."""
        return self.ForwardFromAfterStage(x, AccumulatedVariance, y, StageIndex=0, UseCompiled=UseCompiled)

    def ExplicitStage0VJP(self, v, Caches, KeepInputDependency=False):
        """Explicit VJP through discriminator stage 0."""
        return self.MainLayers[0].explicit_vjp_from_cache(v, Caches, KeepInputDependency=KeepInputDependency)

    def ExplicitExtractionVJP(self, v, RealSamples=None, OutputDType=None, KeepInputDependency=False):
        """Explicit input-side VJP through ExtractionLayer.

        The discriminator extraction forward is:
            x0 = ExtractionLayer(real.to(torch.bfloat16))
        """
        w = self.ExtractionLayer.EffectiveWeight(v.dtype)
        gx = F.conv_transpose2d(v, w, padding=self.ExtractionLayer.Padding(w), groups=self.ExtractionLayer.Groups)
        if OutputDType is None and RealSamples is not None:
            OutputDType = RealSamples.dtype
        if OutputDType is not None:
            gx = gx.to(OutputDType)
        if KeepInputDependency and RealSamples is not None:
            gx = gx + RealSamples * 0
        return gx

    def ForwardAllResidualStagesWithCache(self, x, y=None):
        """Run all residual stages plus head with caches for explicit VJP.

        Input x is expected to be after ExtractionLayer. y is expected to be
        already embedded by ForwardToStage0().
        """
        StageOutputs = []
        StageCaches = []
        TransitionCaches = []
        AccumulatedVariances = []

        # Stage 0.
        x, AccumulatedVariance, Caches = self.MainLayers[0].forward_with_cache(x)
        StageOutputs.append(x)
        StageCaches.append(Caches)
        AccumulatedVariances.append(AccumulatedVariance)

        # Transitions + later residual stages.
        for Index, Transition in enumerate(self.TransitionLayers):
            TransitionGain = torch.rsqrt(AccumulatedVariance)
            x, TransitionCache = Transition.forward_with_cache(x, Gain=TransitionGain)
            TransitionCaches.append(TransitionCache)

            x, AccumulatedVariance, Caches = self.MainLayers[Index + 1].forward_with_cache(x)
            StageOutputs.append(x)
            StageCaches.append(Caches)
            AccumulatedVariances.append(AccumulatedVariance)

        # Head with cache.
        HeadOut, HeadCache = self.Head.forward_with_cache(
            x.to(torch.float32),
            Gain=torch.rsqrt(AccumulatedVariance),
        )

        logits = (HeadOut * y / math.sqrt(y.shape[1])).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else HeadOut
        logits = logits.view(logits.shape[0])

        Cache = dict(
            StageOutputs=StageOutputs,
            StageCaches=StageCaches,
            TransitionCaches=TransitionCaches,
            AccumulatedVariances=AccumulatedVariances,
            EmbeddedConditions=y,
            HeadOut=HeadOut,
            HeadCache=HeadCache,
        )
        return logits, Cache

    def ExplicitAllResidualStagesVJP(self, v, Cache, KeepInputDependency=False):
        """Explicit VJP from final residual-stage output back to stage-0 input.

        This explicitly handles all MainLayers and TransitionLayers.

        Important dtype/layout note:
            The discriminator head runs on ``FinalStageOutput.to(torch.float32)``.
            Autograd's VJP through that cast returns a cotangent in the dtype of
            ``FinalStageOutput`` (normally bfloat16).  The explicit head VJP,
            however, naturally produces float32.  If we pass that float32
            cotangent into the residual-stage VJP, the large 8/16/32-resolution
            VJP chain runs in fp32, causing a large speed and memory regression.
            Therefore this boundary always normalizes v to match the final
            residual activation.
        """
        FinalStageOutput = Cache.get('FinalStageOutput', None)
        if FinalStageOutput is None and 'StageOutputs' in Cache:
            FinalStageOutput = Cache['StageOutputs'][-1]

        if FinalStageOutput is not None:
            if v.dtype != FinalStageOutput.dtype:
                v = v.to(FinalStageOutput.dtype)
            if FinalStageOutput.ndim == 4 and v.ndim == 4:
                if FinalStageOutput.is_contiguous(memory_format=torch.channels_last):
                    v = v.contiguous(memory_format=torch.channels_last)
                else:
                    v = v.contiguous()

        StageCaches = Cache['StageCaches']
        TransitionCaches = Cache['TransitionCaches']

        # Last residual stage.
        LastIndex = len(self.MainLayers) - 1
        v = self.MainLayers[LastIndex].explicit_vjp_from_cache(
            v,
            StageCaches[LastIndex],
            KeepInputDependency=KeepInputDependency,
        )

        # Walk backward through transition + previous residual stage.
        for Index in reversed(range(len(self.TransitionLayers))):
            v = self.TransitionLayers[Index].explicit_vjp(v, TransitionCaches[Index])
            v = self.MainLayers[Index].explicit_vjp_from_cache(
                v,
                StageCaches[Index],
                KeepInputDependency=KeepInputDependency,
            )

        return v

    def ForwardWithFullCache(self, x, y=None):
        """Run the full discriminator with enough cache for explicit full VJP."""
        Stage0Input, EmbeddedConditions = self.ForwardToStage0(x, y)
        Logits, Cache = self.ForwardAllResidualStagesWithCache(Stage0Input, EmbeddedConditions)
        Cache['Stage0Input'] = Stage0Input
        return Logits, Cache

    def ExplicitFullVJP(self, v_logits, Cache, RealSamples=None, KeepInputDependency=False):
        """Explicit VJP from logits back to discriminator input.

        Handles class projection, head, all residual stages, transitions, and
        extraction. It does not handle an external preprocessor/augmentation
        boundary; the trainer can still use autograd for that boundary only.
        """
        y = Cache['EmbeddedConditions']

        # Class projection VJP: logits -> HeadOut.
        if hasattr(self, 'EmbeddingLayer'):
            v = v_logits.view(-1, 1) * y / math.sqrt(y.shape[1])
        else:
            v = v_logits.view(-1, 1)

        # Head VJP: HeadOut -> output of final residual stage.
        v = self.Head.explicit_vjp(v, Cache['HeadCache'], KeepInputDependency=KeepInputDependency)

        # The head forward consumes the final residual activation after casting
        # it to float32:
        #
        #     Head(FinalStageOutput.to(torch.float32), ...)
        #
        # Autograd's VJP through that cast returns a cotangent in the dtype of
        # FinalStageOutput, usually bfloat16.  The explicit head VJP naturally
        # produces float32 because the head runs in float32, so cast it back
        # before entering the large residual-stage VJP.  Otherwise stages 0/1/2
        # run their explicit VJP in fp32, causing a large speed and VRAM hit.
        FinalStageOutput = Cache.get('FinalStageOutput', None)
        if FinalStageOutput is None and 'StageOutputs' in Cache:
            # Older/full-cache path stores all stage outputs instead of the
            # explicit FinalStageOutput key.
            FinalStageOutput = Cache['StageOutputs'][-1]

        if FinalStageOutput is not None and v.dtype != FinalStageOutput.dtype:
            v = v.to(FinalStageOutput.dtype)

        # Match memory layout as well, after the dtype cast.
        if FinalStageOutput is not None and FinalStageOutput.ndim == 4 and v.ndim == 4:
            if FinalStageOutput.is_contiguous(memory_format=torch.channels_last):
                v = v.contiguous(memory_format=torch.channels_last)
            else:
                v = v.contiguous()

        # Residual stages + transitions: final stage output -> stage0 input.
        v = self.ExplicitAllResidualStagesVJP(v, Cache, KeepInputDependency=KeepInputDependency)

        # Extraction: stage0 input -> discriminator input.
        v = self.ExplicitExtractionVJP(
            v,
            RealSamples=RealSamples,
            KeepInputDependency=KeepInputDependency,
        )
        return v

    def CompileMainLayers(self, mode='default', fullgraph=False, dynamic=False):
        for Layer in self.MainLayers:
            if hasattr(Layer, 'CompileForward'):
                Layer.CompileForward(mode=mode, fullgraph=fullgraph, dynamic=dynamic)
        return self

    def ClearCompiledMainLayers(self):
        for Layer in self.MainLayers:
            if hasattr(Layer, 'ClearCompiledForward'):
                Layer.ClearCompiledForward()
        return self

