import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register

@register('model')
class Model(nn.Module):

    def __init__(self, encoder_spec):
        super().__init__()

        self.encoder = models.make(encoder_spec)

    def forward(self, inp):
        return self.encoder(inp)
