# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

from torch_utils import training_stats
from R3GAN.Trainer import AdversarialTraining
import torch

#----------------------------------------------------------------------------

class R3GANLoss:
    def __init__(self, G, D, augment_pipe=None):
        if augment_pipe is not None:
            preprocessor = lambda images_list: [y.to(images_list[0].dtype) for y in augment_pipe([x.to(torch.float32) for x in images_list])]
            self.trainer = AdversarialTraining(G, D, preprocessor)
        else:
            self.trainer = AdversarialTraining(G, D)
        
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gamma, gain):
        # G
        if phase == 'G':
            AdversarialLoss, RelativisticLogits = self.trainer.AccumulateGeneratorGradients(gen_z, real_img, real_c, gain)
            
            training_stats.report('Loss/scores/fake', RelativisticLogits)
            training_stats.report('Loss/signs/fake', RelativisticLogits.sign())
            training_stats.report('Loss/G/loss', AdversarialLoss)
            
            training_stats.report('Progress/generator_gain', self.trainer.Generator.Model.Gain)
            
            for i, l in enumerate(self.trainer.Generator.Model.MainLayers):
                for j, a in enumerate(l.ParametrizedAlphas):
                    Alpha = torch.abs(a())
                    training_stats.report('ResidualG/'+str(i)+'/'+str(j), Alpha.mean())
            
        # D
        if phase == 'D':
            AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty = self.trainer.AccumulateDiscriminatorGradients(gen_z, real_img, real_c, gamma, gain)
            
            training_stats.report('Loss/scores/real', RelativisticLogits)
            training_stats.report('Loss/signs/real', RelativisticLogits.sign())
            training_stats.report('Loss/D/loss', AdversarialLoss)
            training_stats.report('Loss/r1_penalty', R1Penalty)
            training_stats.report('Loss/r2_penalty', R2Penalty)

            training_stats.report('Progress/discriminator_gain', self.trainer.Discriminator.Model.Gain)
            
            for i, l in enumerate(self.trainer.Discriminator.Model.MainLayers):
                for j, a in enumerate(l.ParametrizedAlphas):
                    Alpha = torch.abs(a())
                    training_stats.report('ResidualD/'+str(i)+'/'+str(j), Alpha.mean())
            
#----------------------------------------------------------------------------
