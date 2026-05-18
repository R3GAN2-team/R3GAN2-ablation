import contextlib
import torch
import torch.nn as nn


class AdversarialTraining:
    """
    Faster drop-in replacement for Trainer.py.

    Main exact-objective speed changes:
      1. D step runs G(fake) under torch.no_grad().
      2. G step freezes D parameters while preserving gradients through D(fake) to G.
      3. D step can be executed in micro-chunks, so the R1/gradient-penalty
         double-backward path avoids pathological large local-batch conv gradgrad
         kernels on B200.
      4. In DDP, all chunk backprops except the last are wrapped in D.no_sync()
         to avoid one all-reduce per chunk.

    Important note about preprocessing:
      - The fast chunked D path applies Preprocessor to each chunk instead of once
        to the full local batch. For standard per-sample stochastic augmentation,
        this is the same mathematical objective/distribution and allows R1 to be
        taken w.r.t. the chunk leaf tensor.
      - If your Preprocessor has intentional cross-sample coupling, set
        ChunkDiscriminatorStep=False to recover the strict full-batch behavior.

    Defaults are chosen for the observed B200 pathology:
      - DiscriminatorChunkSize=256
      - ChunkDiscriminatorStep=True
    """

    def __init__(
        self,
        Generator,
        Discriminator,
        Preprocessor=lambda x: x,
        DiscriminatorChunkSize=256,
        ChunkDiscriminatorStep=True,
        FreezeDiscriminatorInGeneratorStep=True,
        ChannelsLast=False,
        PrintOnce=False,
    ):
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.Preprocessor = Preprocessor
        self.DiscriminatorChunkSize = DiscriminatorChunkSize
        self.ChunkDiscriminatorStep = ChunkDiscriminatorStep
        self.FreezeDiscriminatorInGeneratorStep = FreezeDiscriminatorInGeneratorStep
        self.ChannelsLast = ChannelsLast
        self.PrintOnce = PrintOnce
        self._PrintedInfo = False

    @staticmethod
    def _SetRequiresGrad(Module, Flag):
        for Parameter in Module.parameters():
            Parameter.requires_grad_(Flag)

    @staticmethod
    def _SliceBatch(x, Start, End):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x[Start:End]
        if isinstance(x, tuple):
            return tuple(AdversarialTraining._SliceBatch(y, Start, End) for y in x)
        if isinstance(x, list):
            return [AdversarialTraining._SliceBatch(y, Start, End) for y in x]
        if isinstance(x, dict):
            return {k: AdversarialTraining._SliceBatch(v, Start, End) for k, v in x.items()}
        raise TypeError(f'Unsupported condition type for batch slicing: {type(x)}')

    def _MaybeChannelsLast(self, x):
        if self.ChannelsLast and torch.is_tensor(x) and x.ndim == 4:
            return x.contiguous(memory_format=torch.channels_last)
        return x

    @staticmethod
    def _NoSync(Module, Enabled):
        if Enabled and hasattr(Module, 'no_sync'):
            return Module.no_sync()
        return contextlib.nullcontext()

    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(
            outputs=Critics.sum(),
            inputs=Samples,
            create_graph=True,
        )
        return Gradient.square().sum([1, 2, 3])

    def AccumulateGeneratorGradients(self, Noise, RealSamples, Conditions, Scale=1):
        if self.FreezeDiscriminatorInGeneratorStep:
            AdversarialTraining._SetRequiresGrad(self.Discriminator, False)

        FakeSamples = self.Generator(Noise, Conditions)
        RealSamples = RealSamples.detach()
        TransformedFakeSamples, TransformedRealSamples = self.Preprocessor([FakeSamples, RealSamples])

        TransformedFakeSamples = self._MaybeChannelsLast(TransformedFakeSamples)
        TransformedRealSamples = self._MaybeChannelsLast(TransformedRealSamples)

        # Need gradient through D(fake) to FakeSamples/G, but not into D params.
        FakeLogits = self.Discriminator(TransformedFakeSamples, Conditions)

        # Real logits have no path to G.
        with torch.no_grad():
            RealLogits = self.Discriminator(TransformedRealSamples, Conditions)

        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

        (Scale * AdversarialLoss.mean()).backward()

        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]

    def _AccumulateDiscriminatorGradientsFullBatch(self, Noise, RealSamples, Conditions, Gamma, Scale):
        RealSamples = RealSamples.detach().requires_grad_(True)

        with torch.no_grad():
            FakeSamples = self.Generator(Noise, Conditions)

        TransformedRealSamples, TransformedFakeSamples = self.Preprocessor([RealSamples, FakeSamples])
        TransformedRealSamples = self._MaybeChannelsLast(TransformedRealSamples)
        TransformedFakeSamples = self._MaybeChannelsLast(TransformedFakeSamples)

        RealLogits = self.Discriminator(TransformedRealSamples, Conditions)
        FakeLogits = self.Discriminator(TransformedFakeSamples, Conditions)

        R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealSamples, RealLogits)

        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

        DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * R1Penalty
        (Scale * DiscriminatorLoss.mean()).backward()

        return [x.detach() for x in [AdversarialLoss, RelativisticLogits, R1Penalty]]

    def _AccumulateDiscriminatorGradientsChunked(self, Noise, RealSamples, Conditions, Gamma, Scale):
        RealSamples = RealSamples.detach()
        BatchSize = RealSamples.shape[0]
        ChunkSize = self.DiscriminatorChunkSize

        with torch.no_grad():
            FakeSamples = self.Generator(Noise, Conditions).detach()

        if self.PrintOnce and not self._PrintedInfo:
            print(
                f'[Trainer_fast_chunked] local batch={BatchSize}, '
                f'DiscriminatorChunkSize={ChunkSize}, '
                f'num_chunks={(BatchSize + ChunkSize - 1) // ChunkSize}, '
                f'DDP_no_sync={hasattr(self.Discriminator, "no_sync")}',
                flush=True,
            )
            self._PrintedInfo = True

        AdversarialLosses = []
        RelativisticLogitsList = []
        R1Penalties = []

        NumChunks = (BatchSize + ChunkSize - 1) // ChunkSize

        for ChunkIndex, Start in enumerate(range(0, BatchSize, ChunkSize)):
            End = min(Start + ChunkSize, BatchSize)
            ThisChunk = End - Start

            # Make the real chunk a leaf, so autograd.grad computes the R1 input
            # gradient for this chunk only instead of allocating a full-batch input
            # gradient on every chunk.
            RealChunk = RealSamples[Start:End].detach().requires_grad_(True)
            FakeChunk = FakeSamples[Start:End]
            ConditionsChunk = AdversarialTraining._SliceBatch(Conditions, Start, End)

            TransformedRealChunk, TransformedFakeChunk = self.Preprocessor([RealChunk, FakeChunk])
            TransformedRealChunk = self._MaybeChannelsLast(TransformedRealChunk)
            TransformedFakeChunk = self._MaybeChannelsLast(TransformedFakeChunk)

            # In DDP, avoid reducing gradients after every chunk. The final backward
            # outside no_sync() reduces the accumulated gradient once.
            UseNoSync = ChunkIndex + 1 < NumChunks
            with AdversarialTraining._NoSync(self.Discriminator, UseNoSync):
                RealLogits = self.Discriminator(TransformedRealChunk, ConditionsChunk)
                FakeLogits = self.Discriminator(TransformedFakeChunk, ConditionsChunk)

                R1Penalty = AdversarialTraining.ZeroCenteredGradientPenalty(RealChunk, RealLogits)

                RelativisticLogits = RealLogits - FakeLogits
                AdversarialLoss = nn.functional.softplus(-RelativisticLogits)

                DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * R1Penalty

                # Match original full-batch mean exactly modulo floating-point order:
                # Scale * mean(loss over B) = Scale * sum(chunk_loss) / B.
                (Scale * DiscriminatorLoss.sum() / BatchSize).backward()

            AdversarialLosses.append(AdversarialLoss.detach())
            RelativisticLogitsList.append(RelativisticLogits.detach())
            R1Penalties.append(R1Penalty.detach())

        return [
            torch.cat(AdversarialLosses, dim=0),
            torch.cat(RelativisticLogitsList, dim=0),
            torch.cat(R1Penalties, dim=0),
        ]

    def AccumulateDiscriminatorGradients(self, Noise, RealSamples, Conditions, Gamma, Scale=1):
        if self.FreezeDiscriminatorInGeneratorStep:
            AdversarialTraining._SetRequiresGrad(self.Discriminator, True)

        BatchSize = RealSamples.shape[0]
        ChunkSize = self.DiscriminatorChunkSize

        if (
            self.ChunkDiscriminatorStep
            and ChunkSize is not None
            and ChunkSize > 0
            and ChunkSize < BatchSize
        ):
            return self._AccumulateDiscriminatorGradientsChunked(
                Noise, RealSamples, Conditions, Gamma, Scale
            )

        if self.PrintOnce and not self._PrintedInfo:
            print(
                f'[Trainer_fast_chunked] full-batch D step, local batch={BatchSize}, '
                f'DiscriminatorChunkSize={ChunkSize}',
                flush=True,
            )
            self._PrintedInfo = True

        return self._AccumulateDiscriminatorGradientsFullBatch(
            Noise, RealSamples, Conditions, Gamma, Scale
        )
