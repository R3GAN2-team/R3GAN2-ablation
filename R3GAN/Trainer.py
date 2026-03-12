import torch
import torch.nn as nn

class AdversarialTraining:
    def __init__(self, Generator, Discriminator, Preprocessor=lambda x: x):
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.Preprocessor = Preprocessor
        
    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])
        
    def AccumulateGeneratorGradients(self, Noise, RealSamples, Conditions, Scale=1):
        FakeSamples = self.Generator(Noise, Conditions)
        RealSamples = RealSamples.detach()
        TransformedFakeSamples, TransformedRealSamples = self.Preprocessor([FakeSamples, RealSamples])
        
        FakeLogits, _ = self.Discriminator(TransformedFakeSamples, Conditions)
        RealLogits, _ = self.Discriminator(TransformedRealSamples, Conditions)
        
        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        (Scale * AdversarialLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]
    
    def AccumulateDiscriminatorGradients(self, Noise, RealSamples, Conditions, Gamma, Scale=1):
        RealSamples = RealSamples.detach().requires_grad_(True)
        FakeSamples = self.Generator(Noise, Conditions).detach().requires_grad_(True)
        TransformedRealSamples, TransformedFakeSamples = self.Preprocessor([RealSamples, FakeSamples])
        
        RealLogits, ScaleDetachedRealLogits = self.Discriminator(TransformedRealSamples, Conditions)
        FakeLogits, ScaleDetachedFakeLogits = self.Discriminator(TransformedFakeSamples, Conditions)
        
        R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, ScaleDetachedRealLogits)
        R2Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(FakeSamples, ScaleDetachedFakeLogits)
        
        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
        (Scale * DiscriminatorLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]
