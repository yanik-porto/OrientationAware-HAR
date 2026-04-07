import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from encoder.encoders.utils import trunc_normal_, normal_init


class HeadTextEmbed(nn.Module):

    def __init__(self,
                num_classes,
                 in_channels,
                 mode='contrastive',
                 text_embed_dim = 512,
                 n_persons=1,
                 n_views=1,
                 balance_path="",
                 **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.in_c = in_channels
        self.n_persons = n_persons
        self.n_views = n_views

        self.fc_motion = nn.Linear(in_channels, in_channels)
        self.fc_text = nn.Linear(text_embed_dim, in_channels)

        self.apply(self._init_weights)

        self.mode = mode
        if self.mode == 'kl':
            self.criterion = KLDivergenceLoss()
        elif self.mode == 'contrastive':
            self.criterion = ContrastiveLoss()
        elif self.mode == 'clip':
            self.criterion = ClipLoss()
        elif self.mode == "contrastive_focal":
            assert balance_path != ""
            weights = np.load(balance_path)
            weights = torch.FloatTensor(weights)
            self.criterion = ContrastiveFocalLoss(alpha=weights)
        else:
            print(self.mode, ' not handled')

    def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, motion_and_text):
        x, text_embed = motion_and_text

        motion_feats_single, motion_feats = self.forward_motion(x)
        text_feats = self.forward_text(text_embed)

        return motion_feats_single, motion_feats, text_feats

    
    def forward_motion(self, x):

        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)
        x = pool(x)
        x = x.reshape(N, M, C)

        # mean if more than 1 person
        if self.n_persons > 1 and M == self.n_views * self.n_persons:
                x = x.reshape(N, self.n_views, self.n_persons, C)
                x = x.mean(dim=2)
                M = self.n_views

        if 'contrastive' in self.mode:
            motion_feats = self.fc_motion(x)
        else:
            motion_feats = self.fc_motion(x[:, 0:1, :])

        return motion_feats[:, 0, :], motion_feats

    def forward_text(self, text_embed):
        text_feats = self.fc_text(text_embed)
        return text_feats

    def loss(self, output, labels):
        _, motion_feats, text_feats = output

        N = len(motion_feats)
        
        if self.mode != 'contrastive' and self.mode != 'contrastive_focal':
            motion_feats_norm_p = motion_feats[:, 0, :]
            text_feats_norm = text_feats
            loss = self.criterion(motion_feats_norm_p, text_feats_norm)
            return loss


        else:
            motion_feats_norm_p = nn.functional.normalize(motion_feats[:, 0, :], dim=1)
            motion_feats_norm_n = nn.functional.normalize(motion_feats[:, 1, :], dim=1)
            text_feats_norm = nn.functional.normalize(text_feats, dim=1)

            # positives = torch.ones(N).to(motion_feats.device)
            # ploss = self.criterion(motion_feats_norm_p, text_feats_norm, positives)

            # negatives = torch.zeros(N).to(motion_feats.device)
            # nloss = self.criterion(motion_feats_norm_n, text_feats_norm, negatives)

            # return ploss + nloss

            motion_feats_norm = torch.cat((motion_feats_norm_p, motion_feats_norm_n), 0)
            text_feats_norm = torch.cat((text_feats_norm, text_feats_norm), 0)
            cl_labels = torch.cat((torch.ones(N), torch.zeros(N))).to(motion_feats.device)

            if self.mode == "contrastive_focal":
                loss = self.criterion(motion_feats_norm, text_feats_norm, cl_labels, labels)
            else:
                loss = self.criterion(motion_feats_norm, text_feats_norm, cl_labels)

            return loss

class HeadTextEmbedDualOutput(HeadTextEmbed):
    def __init__(self,
                 num_classes,
                 in_channels,
                 mode='contrastive',
                 text_embed_dim = 512,
                 n_persons=1,
                 n_views=1,
                 focal_in_classif=False,
                 balance_path="",
                 lambda_loss=0.5,
                 **kwargs):
        super().__init__(num_classes, in_channels, mode, text_embed_dim, n_persons, n_views, balance_path, **kwargs)
        self.lambda_loss = lambda_loss
        self.fc_classif = nn.Linear(in_channels, self.num_classes)
        normal_init(self.fc_classif, std=math.sqrt(2. / num_classes))

        if focal_in_classif:
            weights = np.load(balance_path)
            weights = torch.FloatTensor(weights)
            self.ce = FocalLoss(weights)
        else:
            self.ce = nn.CrossEntropyLoss()


    def forward(self, motion_and_text):
        x, text_embed = motion_and_text

        motion_feats_single, motion_feats, classif_feats = self.forward_motion(x)
        text_feats = self.forward_text(text_embed)

        return motion_feats_single, motion_feats, text_feats, classif_feats


    def forward_motion(self, x):

        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)
        x = pool(x)
        x = x.reshape(N, M, C)

        # mean if more than 1 person
        if self.n_persons > 1 and M % self.n_persons == 0:
            x = x.reshape(N, -1, self.n_persons, C)
            x = x.mean(dim=2)
            M = x.shape[1]
        elif self.n_persons > 1 and M != self.n_views * self.n_persons:
            print("wrong config : ", M, " vs ", self.n_views, " x ", self.n_persons)
        

        # text aligned features
        if 'contrastive' in self.mode:
            motion_feats = self.fc_motion(x)
        else:
            motion_feats = self.fc_motion(x[:, 0:1, :])

        # classif features
        classif_feats = self.fc_classif(x[:, 0, :])

        return motion_feats[:, 0, :], motion_feats, classif_feats
    
    def loss(self, output, labels):
        text_align_loss = super().loss(output[:-1], labels)
    
        classif_loss = self.ce(output[-1], labels)

        loss_total = self.lambda_loss * classif_loss + (1-self.lambda_loss)*text_align_loss
        return loss_total


# Define the contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, output1, output2, label):
        # euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        euclidean_distance = 1 - (self.cos(output1, output2) + 1.) / 2.
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
class ContrastiveFocalLoss(nn.Module):
    def __init__(self, margin=0.5, alpha=None, gamma=2):
        super(ContrastiveFocalLoss, self).__init__()
        self.margin = margin

        self.alpha = alpha
        self.gamma = gamma
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, output1, output2, label, targets):
        # euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        euclidean_distance = 1 - (self.cos(output1, output2) + 1.) / 2.
        loss_contrastive = (label) * torch.pow(euclidean_distance, 2) + \
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        

        pt = torch.exp(-loss_contrastive)
        self.alpha = self.alpha.to(output1.device)
        
        targets_pn = torch.cat((targets, targets), -1)
        loss = (self.alpha[targets_pn] * (1 - pt) ** self.gamma * loss_contrastive).mean()
        
        return loss


# TODO : check if interesting to use KLDivLoss Class from pytorch
class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, spatial_features, text_features):
        # Convert embeddings to probability distributions using softmax
        spatial_probs = F.log_softmax(spatial_features, dim=1)
        text_probs = F.softmax(text_features, dim=1) # TODO : check why log_softmax vs softmax

        # Compute KL divergence loss
        kl_div_loss = F.kl_div(spatial_probs, text_probs, reduction='batchmean')
        return kl_div_loss
    
class ClipLoss(nn.Module):
    def __init__(self, logit=0.07):
        super(ClipLoss, self).__init__()

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / logit)))

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )
    
    def forward(self, spatial_features, text_features):
        # normalized features
        spatial_features = spatial_features / spatial_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, spatial_features.t()) * logit_scale
        # logits_per_motion = logits_per_text.T

        caption_loss = self.contrastive_loss(logits_per_text)
        motion_loss = self.contrastive_loss(logits_per_text.t())
        return (caption_loss + motion_loss) / 2.0
    

class FocalLoss(nn.Module):
    """
    Focal Loss with optional class weights for handling class imbalance.
    
    Args:
        alpha (Tensor or list): Class weights (shape [num_classes]).
        gamma (float): Focusing parameter (default=2.0).
        reduction (str): 'mean', 'sum', or 'none' (default='mean').
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits (batch_size, num_classes).
            targets: Ground truth labels (batch_size,) with values in [0, num_classes-1].
        """
        log_probs = F.log_softmax(inputs, dim=-1)  # (B, C)
        probs = torch.exp(log_probs)               # softmax probabilities

        targets = targets.view(-1, 1)              # (B, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)  # log(p_t)
        pt = probs.gather(1, targets).squeeze(1)          # p_t

        # Apply class weights if provided
        if self.alpha is not None:
            at = self.alpha.gather(0, targets.squeeze(1))
        else:
            at = 1.0

        # Focal loss formula
        loss = -at * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
