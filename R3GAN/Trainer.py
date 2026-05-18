import torch
import torch.nn as nn


class AdversarialTraining:
    def __init__(
        self,
        Generator,
        Discriminator,
        Preprocessor=lambda x: x,
        UseExplicitStage0R1=False,
        UseExplicitAllResidualStagesR1=False,
        UseExplicitFullDiscriminatorR1=False,
        UseExplicitExtractionR1=True,
        ExplicitR1ChannelsLastInput=False,
        NoGradGeneratorInDiscriminatorStep=True,
        FreezeDiscriminatorDuringGeneratorStep=False,
    ):
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.Preprocessor = Preprocessor
        self.UseExplicitStage0R1 = UseExplicitStage0R1
        self.UseExplicitAllResidualStagesR1 = UseExplicitAllResidualStagesR1
        self.UseExplicitFullDiscriminatorR1 = UseExplicitFullDiscriminatorR1
        self.UseExplicitExtractionR1 = UseExplicitExtractionR1
        self.ExplicitR1ChannelsLastInput = ExplicitR1ChannelsLastInput
        self.NoGradGeneratorInDiscriminatorStep = NoGradGeneratorInDiscriminatorStep
        self.FreezeDiscriminatorDuringGeneratorStep = FreezeDiscriminatorDuringGeneratorStep

    @staticmethod
    def _SetRequiresGrad(Module, Flag):
        OldFlags = []
        for p in Module.parameters():
            OldFlags.append(p.requires_grad)
            p.requires_grad_(Flag)
        return OldFlags

    @staticmethod
    def _RestoreRequiresGrad(Module, Flags):
        for p, flag in zip(Module.parameters(), Flags):
            p.requires_grad_(flag)

    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])

    @staticmethod
    def _MaybeChannelsLast(x):
        if x is not None and x.ndim == 4:
            return x.contiguous(memory_format=torch.channels_last)
        return x

    def _CallExplicitExtractionVJP(self, VStage0Input, DiscriminatorSamplesForR1):
        D = self.Discriminator
        if not hasattr(D, 'ExplicitExtractionVJP'):
            raise AttributeError('Discriminator does not expose ExplicitExtractionVJP')

        # Wrapper implementations may or may not accept keyword args.
        try:
            return D.ExplicitExtractionVJP(
                VStage0Input,
                DiscriminatorSamplesForR1,
                KeepInputDependency=False,
            )
        except TypeError:
            return D.ExplicitExtractionVJP(VStage0Input, DiscriminatorSamplesForR1)

    def _FinishR1FromStage0Cotangent(self, InputSamples, DiscriminatorSamplesForR1, VStage0Input):
        """Convert cotangent wrt D's stage-0 input into per-sample R1.

        If UseExplicitExtractionR1=True, this uses the analytic VJP through the
        discriminator extraction 1x1. If the discriminator input differs from
        the original R1 input, e.g. due to augmentation or channels-last copy, it
        then uses autograd only through that boundary.
        """
        if self.UseExplicitExtractionR1:
            VDiscriminatorSamples = self._CallExplicitExtractionVJP(VStage0Input, DiscriminatorSamplesForR1)

            if DiscriminatorSamplesForR1 is InputSamples:
                Gradient = VDiscriminatorSamples
            else:
                Gradient, = torch.autograd.grad(
                    outputs=DiscriminatorSamplesForR1,
                    inputs=InputSamples,
                    grad_outputs=VDiscriminatorSamples,
                    create_graph=True,
                    retain_graph=True,
                )
        else:
            Gradient, = torch.autograd.grad(
                outputs=self._LastStage0InputForFallback,
                inputs=InputSamples,
                grad_outputs=VStage0Input,
                create_graph=True,
                retain_graph=True,
            )

        return Gradient.square().sum([1, 2, 3])

    def ZeroCenteredGradientPenaltyExplicitStage0(self, InputSamples, DiscriminatorSamples, Conditions):
        """Compute logits and R1 with explicit VJP through discriminator stage 0.

        InputSamples is the tensor with respect to which R1 is defined. In the
        original trainer this is RealSamples before preprocessing.
        DiscriminatorSamples is the tensor actually fed into D, i.e. after
        preprocessing/augmentation.

        The expensive 32x32 discriminator stage is handled by explicit VJP.
        Optionally, the extraction 1x1 input VJP is also handled explicitly,
        avoiding another simple but slow generic conv gradgrad path.
        """
        D = self.Discriminator
        Required = [
            'ForwardToStage0',
            'ForwardStage0WithCache',
            'ForwardFromAfterStage',
        ]
        Missing = [name for name in Required if not hasattr(D, name)]
        if Missing:
            raise AttributeError(
                'Discriminator does not expose explicit-stage0-R1 helpers: ' + ', '.join(Missing)
            )
        if not hasattr(D.MainLayers[0], 'explicit_vjp_from_cache'):
            raise AttributeError('Discriminator.MainLayers[0] does not expose explicit_vjp_from_cache')

        if self.ExplicitR1ChannelsLastInput:
            DiscriminatorSamplesForR1 = AdversarialTraining._MaybeChannelsLast(DiscriminatorSamples)
        else:
            DiscriminatorSamplesForR1 = DiscriminatorSamples

        Stage0Input, EmbeddedConditions = D.ForwardToStage0(DiscriminatorSamplesForR1, Conditions)
        self._LastStage0InputForFallback = Stage0Input
        Stage0Output, AccumulatedVariance, Caches = D.ForwardStage0WithCache(Stage0Input)

        # Important for compile-safe D: the R1 suffix must be allowed to force
        # uncompiled/eager residual groups if the method supports UseCompiled.
        try:
            Logits = D.ForwardFromAfterStage(
                Stage0Output,
                AccumulatedVariance,
                EmbeddedConditions,
                StageIndex=0,
                UseCompiled=False,
            )
        except TypeError:
            Logits = D.ForwardFromAfterStage(
                Stage0Output,
                AccumulatedVariance,
                EmbeddedConditions,
                StageIndex=0,
            )

        # VJP through the suffix, using ordinary autograd. Keep the graph because
        # Logits also participates in the adversarial discriminator loss.
        VStage0Output, = torch.autograd.grad(
            outputs=Logits.sum(),
            inputs=Stage0Output,
            create_graph=True,
            retain_graph=True,
        )

        # Explicit VJP through the expensive 32x32 residual group.
        VStage0Input = D.MainLayers[0].explicit_vjp_from_cache(VStage0Output, Caches)

        R1Penalty = self._FinishR1FromStage0Cotangent(
            InputSamples,
            DiscriminatorSamplesForR1,
            VStage0Input,
        )
        return Logits, R1Penalty

    def ZeroCenteredGradientPenaltyExplicitAllResidualStages(self, InputSamples, DiscriminatorSamples, Conditions):
        """Compute logits and R1 with explicit VJP through all residual stages.

        This removes generic gradgrad through all discriminator MainLayers. The
        remaining autograd VJPs are only through:
          * the discriminator head,
          * optional preprocessor/layout-copy boundary.
        Transition/downsample layers are expected to be handled by the network's
        ExplicitAllResidualStagesVJP if Resamplers.py exposes explicit_vjp().
        """
        D = self.Discriminator
        Required = [
            'ForwardToStage0',
            'ForwardAllResidualStagesWithCache',
            'ExplicitAllResidualStagesVJP',
        ]
        Missing = [name for name in Required if not hasattr(D, name)]
        if Missing:
            raise AttributeError(
                'Discriminator does not expose explicit-all-stages-R1 helpers: ' + ', '.join(Missing)
            )

        if self.ExplicitR1ChannelsLastInput:
            DiscriminatorSamplesForR1 = AdversarialTraining._MaybeChannelsLast(DiscriminatorSamples)
        else:
            DiscriminatorSamplesForR1 = DiscriminatorSamples

        Stage0Input, EmbeddedConditions = D.ForwardToStage0(DiscriminatorSamplesForR1, Conditions)
        self._LastStage0InputForFallback = Stage0Input

        Logits, Cache = D.ForwardAllResidualStagesWithCache(Stage0Input, EmbeddedConditions)
        FinalStageOutput = Cache['StageOutputs'][-1]

        # VJP through the head only. The residual stages/transitions are handled
        # explicitly below.
        VFinalStageOutput, = torch.autograd.grad(
            outputs=Logits.sum(),
            inputs=FinalStageOutput,
            create_graph=True,
            retain_graph=True,
        )

        VStage0Input = D.ExplicitAllResidualStagesVJP(VFinalStageOutput, Cache)

        R1Penalty = self._FinishR1FromStage0Cotangent(
            InputSamples,
            DiscriminatorSamplesForR1,
            VStage0Input,
        )
        return Logits, R1Penalty

    def ZeroCenteredGradientPenaltyExplicitFullDiscriminator(self, InputSamples, DiscriminatorSamples, Conditions):
        """Compute logits and R1 with a fully explicit discriminator VJP.

        This removes torch.autograd.grad through the discriminator itself. If
        DiscriminatorSamplesForR1 differs from InputSamples, e.g. because of an
        augmentation/preprocessor or a channels-last copy, autograd is still used
        only through that boundary.
        """
        D = self.Discriminator
        Required = [
            'ForwardWithFullCache',
            'ExplicitFullVJP',
        ]
        Missing = [name for name in Required if not hasattr(D, name)]
        if Missing:
            raise AttributeError(
                'Discriminator does not expose explicit-full-R1 helpers: ' + ', '.join(Missing)
            )

        if self.ExplicitR1ChannelsLastInput:
            DiscriminatorSamplesForR1 = AdversarialTraining._MaybeChannelsLast(DiscriminatorSamples)
        else:
            DiscriminatorSamplesForR1 = DiscriminatorSamples

        Logits, Cache = D.ForwardWithFullCache(DiscriminatorSamplesForR1, Conditions)

        VLogits = torch.ones_like(Logits)
        VDiscriminatorSamples = D.ExplicitFullVJP(
            VLogits,
            Cache,
            RealSamples=DiscriminatorSamplesForR1,
            KeepInputDependency=False,
        )

        if DiscriminatorSamplesForR1 is InputSamples:
            Gradient = VDiscriminatorSamples
        else:
            # This is now only through augmentation/layout-copy boundary, not
            # through the discriminator.
            Gradient, = torch.autograd.grad(
                outputs=DiscriminatorSamplesForR1,
                inputs=InputSamples,
                grad_outputs=VDiscriminatorSamples,
                create_graph=True,
                retain_graph=True,
            )

        R1Penalty = Gradient.square().sum([1, 2, 3])
        return Logits, R1Penalty

    def AccumulateGeneratorGradients(self, Noise, RealSamples, Conditions, Scale=1):
        OldDFlags = None
        if self.FreezeDiscriminatorDuringGeneratorStep:
            OldDFlags = AdversarialTraining._SetRequiresGrad(self.Discriminator, False)
        try:
            FakeSamples = self.Generator(Noise, Conditions)
            RealSamples = RealSamples.detach()
            TransformedFakeSamples, TransformedRealSamples = self.Preprocessor([FakeSamples, RealSamples])

            FakeLogits = self.Discriminator(TransformedFakeSamples, Conditions)
            RealLogits = self.Discriminator(TransformedRealSamples, Conditions)

            RelativisticLogits = FakeLogits - RealLogits
            AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

            (Scale * AdversarialLoss.mean()).backward()

            return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]
        finally:
            if OldDFlags is not None:
                AdversarialTraining._RestoreRequiresGrad(self.Discriminator, OldDFlags)

    def AccumulateDiscriminatorGradients(self, Noise, RealSamples, Conditions, Gamma, Scale=1):
        RealSamples = RealSamples.detach().requires_grad_(True)

        if self.NoGradGeneratorInDiscriminatorStep:
            with torch.no_grad():
                FakeSamples = self.Generator(Noise, Conditions)
        else:
            FakeSamples = self.Generator(Noise, Conditions).detach()

        TransformedRealSamples, TransformedFakeSamples = self.Preprocessor([RealSamples, FakeSamples])

        if self.UseExplicitFullDiscriminatorR1:
            RealLogits, R1Penalty = self.ZeroCenteredGradientPenaltyExplicitFullDiscriminator(
                RealSamples,
                TransformedRealSamples,
                Conditions,
            )
        elif self.UseExplicitAllResidualStagesR1:
            RealLogits, R1Penalty = self.ZeroCenteredGradientPenaltyExplicitAllResidualStages(
                RealSamples,
                TransformedRealSamples,
                Conditions,
            )
        elif self.UseExplicitStage0R1:
            RealLogits, R1Penalty = self.ZeroCenteredGradientPenaltyExplicitStage0(
                RealSamples,
                TransformedRealSamples,
                Conditions,
            )
        else:
            RealLogits = self.Discriminator(TransformedRealSamples, Conditions)
            R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, RealLogits)

        FakeLogits = self.Discriminator(TransformedFakeSamples, Conditions)

        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * R1Penalty
        (Scale * DiscriminatorLoss.mean()).backward()

        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty]]

