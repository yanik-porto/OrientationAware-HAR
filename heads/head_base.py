import torch
import torch.nn as nn
import math

from encoder.encoders.utils import normal_init
from .losses.csc import Class_Specific_Contrastive_Loss

class HeadBase(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_persons=1,
                 n_views=1,
                 csc_weight=0.0,
                 **kwargs):
        self.num_classes = num_classes
        self.in_c = in_channels
        self.n_persons = n_persons
        self.n_views = n_views
        super().__init__(**kwargs)

        self.fc = nn.Linear(in_channels, num_classes)
        normal_init(self.fc, std=math.sqrt(2. / num_classes))

        self.ce = nn.CrossEntropyLoss()
        self.csc_weight = csc_weight
        if csc_weight > 0.0:
            self.csc = Class_Specific_Contrastive_Loss(self.num_classes, 17*17)

        self.bce = nn.BCELoss()

    def forward(self, x):
        x = self.fc(x)
        return x

    def loss(self, output, labels):
        logits = output[0] if type(output) is tuple else output
        ce_loss = self.ce(logits, labels)

        if self.csc_weight > 0.0:
            assert type(output) is tuple and len(output) > 1
            prn = output[1]
            return ce_loss + self.csc_weight * self.csc(prn, labels.detach(), logits.detach())
    
        return ce_loss

    def loss_bce(self, output, labels):
        if labels.dtype != torch.float32:
            labels = labels.float()

        output = nn.functional.softmax(output, dim=-1)
        return self.bce(output, labels)
