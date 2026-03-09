import torch

def CollectGeneratorFeatures(Generator, x, y):
    x = torch.cat([x, Generator.EmbeddingLayer(y)], dim=1) if hasattr(Generator, 'EmbeddingLayer') else x
    x = Generator.Head(x).to(torch.bfloat16)
    f = []
        
    for Layer, Transition in zip(Generator.MainLayers[:-1], Generator.TransitionLayers):
        x, AccumulatedVariance = Layer(x)
        f += [x * torch.rsqrt(AccumulatedVariance).view(1, -1, 1, 1).to(x.dtype)]
        x = Transition(x, Gain=torch.rsqrt(AccumulatedVariance))
    x, AccumulatedVariance = Generator.MainLayers[-1](x)
    f += [x * torch.rsqrt(AccumulatedVariance).view(1, -1, 1, 1).to(x.dtype)]

    return f

def CollectDiscriminatorFeatures(Discriminator, x, y):
    x = Discriminator.ExtractionLayer(x.to(torch.bfloat16))
    f = []
    
    for Layer, Transition in zip(Discriminator.MainLayers[:-1], Discriminator.TransitionLayers):
        x, AccumulatedVariance = Layer(x)
        f += [x / AccumulatedVariance.view(1, -1, 1, 1).to(x.dtype)]
        x = Transition(x, Gain=1 / AccumulatedVariance)
    x, AccumulatedVariance = Discriminator.MainLayers[-1](x)
    f += [x / AccumulatedVariance.view(1, -1, 1, 1).to(x.dtype)]
    
    return f

def CollectMagnitude(x, mode='avg'):
    x = x.view(x.shape[0], x.shape[1], -1)
    M = x.shape[2]
    x = torch.sqrt(x.square().sum(dim=2) / M)
    if mode == 'avg':
        x = x.mean(dim=1)
    else:
        x = x.max(dim=1)[0]
    return float(x.mean(dim=0))