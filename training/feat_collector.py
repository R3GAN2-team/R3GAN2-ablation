import torch

def CollectGeneratorFeatures(Generator, x, y):
    x = torch.cat([x, Generator.EmbeddingLayer(y)], dim=1) if hasattr(Generator, 'EmbeddingLayer') else x
    x = Generator.Head(x).to(torch.bfloat16)
    f = []
    
    for Layer, Transition in zip(Generator.MainLayers[:-1], Generator.TransitionLayers):
        x = Layer(x)
        f += [x]
        x = Transition(x)
    x = Generator.MainLayers[-1](x)
    f += [x]

    return f

def CollectDiscriminatorFeatures(Discriminator, x, y):
    x = Discriminator.ExtractionLayer(x.to(torch.bfloat16))
    f = []
    
    for Layer, Transition in zip(Discriminator.MainLayers[:-1], Discriminator.TransitionLayers):
        x = Layer(x)
        f += [x]
        x = Transition(x)
    x = Discriminator.MainLayers[-1](x)
    f += [x]
    
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