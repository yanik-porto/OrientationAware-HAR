import torch
import torch.nn as nn
import torch.nn.functional as F

from .head_base import HeadBase

class HeadPool(HeadBase):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)


    def forward(self, x):
        pool = nn.AdaptiveAvgPool3d(1)
        N, M, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(N, C, M, T, V)

        x = pool(x)

        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = x.squeeze(-1)

        cls_score = self.fc(x)

        return cls_score
