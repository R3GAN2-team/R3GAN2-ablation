import contextlib
import torch
import torch.nn as nn


class AdversarialTraining:
    def __init__(
        self,
        Generator,
        Discriminator,
        Preprocessor=lambda x: x,
        UseExplicitStage0R1=False,
        NoGradGeneratorInDiscriminatorStep=True,
        FreezeDiscriminatorDuringGeneratorStep=False,
    ):
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.Preprocessor = Preprocessor
        self.UseExplicitStage0R1 = UseExplicitStage0R1
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

    def ZeroCenteredGradientPenaltyExplicitStage0(self, InputSamples, DiscriminatorSamples, Conditions):
        """Compute logits and R1 with explicit VJP through discriminator stage 0.

        InputSamples is the tensor with respect to which R1 is defined.  In the
        original trainer this is RealSamples before preprocessing.  DiscriminatorSamples
        is the tensor actually fed into D, i.e. after preprocessing/augmentation.

        The method is mathematically the same piecewise-linear R1 path as generic
        autograd for the stage-0 FFNs, but avoids PyTorch's expensive conv gradgrad
        lowering there.  The suffix after stage 0 and the extraction/preprocessor
        boundary are still handled by autograd.
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

        Stage0Input, EmbeddedConditions = D.ForwardToStage0(DiscriminatorSamples, Conditions)
        Stage0Output, AccumulatedVariance, Caches = D.ForwardStage0WithCache(Stage0Input)
        Logits = D.ForwardFromAfterStage(Stage0Output, AccumulatedVariance, EmbeddedConditions, StageIndex=0)

        # VJP through the suffix, using ordinary autograd.  Keep the graph because
        # Logits also participates in the adversarial discriminator loss.
        VStage0Output, = torch.autograd.grad(
            outputs=Logits.sum(),
            inputs=Stage0Output,
            create_graph=True,
            retain_graph=True,
        )

        # Explicit VJP through the expensive 32x32 residual group.  The FFN is
        # piecewise-linear, so the second derivative wrt activations is zero
        # almost everywhere; explicit_vjp_from_cache implements that structure.
        VStage0Input = D.MainLayers[0].explicit_vjp_from_cache(VStage0Output, Caches)

        # Propagate from stage-0 input back to the original R1 input.  This keeps
        # preprocessing/augmentation and the extraction layer in the graph.
        Gradient, = torch.autograd.grad(
            outputs=Stage0Input,
            inputs=InputSamples,
            grad_outputs=VStage0Input,
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

        if self.UseExplicitStage0R1:
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
