import torch.nn as nn
import copy
import R3GAN.Networks


class Generator(nn.Module):
    def __init__(self, *args, **kw):
        super(Generator, self).__init__()

        config = copy.deepcopy(kw)
        del config['c_dim']
        del config['img_resolution']
        del config['img_channels']

        if kw['c_dim'] != 0:
            config['NumberOfClasses'] = kw['c_dim']

        config['OutputChannels'] = kw['img_channels']

        self.Model = R3GAN.Networks.Generator(*args, **config)
        self.z_dim = kw['NoiseDimension']
        self.c_dim = kw['c_dim']
        self.img_resolution = kw['img_resolution']

    def forward(self, x, c):
        return self.Model(x, c)


class Discriminator(nn.Module):
    def __init__(self, *args, **kw):
        super(Discriminator, self).__init__()

        config = copy.deepcopy(kw)
        del config['c_dim']
        del config['img_resolution']
        del config['img_channels']

        if kw['c_dim'] != 0:
            config['NumberOfClasses'] = kw['c_dim']

        config['InputChannels'] = kw['img_channels']

        self.Model = R3GAN.Networks.Discriminator(*args, **config)

    def forward(self, x, c):
        return self.Model(x, c)
        
    @property
    def MainLayers(self):
        return self.Model.MainLayers
        
    def ForwardToStage0(self, *args, **kw):
        return self.Model.ForwardToStage0(*args, **kw)
        
    def ForwardStage0WithCache(self, *args, **kw):
        return self.Model.ForwardStage0WithCache(*args, **kw)
        
    def ForwardFromAfterStage(self, *args, **kw):
        return self.Model.ForwardFromAfterStage(*args, **kw)
        
    def ExplicitExtractionVJP(self, *args, **kw):
        return self.Model.ExplicitExtractionVJP(*args, **kw)
        
    def ForwardAllResidualStagesWithCache(self, *args, **kw):
        return self.Model.ForwardAllResidualStagesWithCache(*args, **kw)
        
    def ExplicitAllResidualStagesVJP(self, *args, **kw):
        return self.Model.ExplicitAllResidualStagesVJP(*args, **kw)
        
    def ForwardWithFullCache(self, *args, **kw):
        return self.Model.ForwardWithFullCache(*args, **kw)
        
    def ExplicitFullVJP(self, *args, **kw):
        return self.Model.ExplicitFullVJP(*args, **kw)
