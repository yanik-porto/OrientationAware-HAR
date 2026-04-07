import torch
import torch.nn as nn

from .unik import UNIK
from .protogcn import ProtoGCN
from .aagcn import AAGCN
from .utils import IndicesEncoding
import math

class AngleEncoder(nn.Module):
    def __init__(self, out_dim, use_pe=False):
        super().__init__()
        self.use_pe = use_pe
        if not use_pe:
            in_dim = 2
            self.encoder = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, out_dim)
            )
        else:
            in_dim = out_dim // 2
            self.max_len = 100
            self.pe = IndicesEncoding(in_dim, self.max_len)
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

    def ori_to_index(self, sin_cos):
        """ transform orientation in sin cos value, to an index in the range 0:max_len corresponding to [-179:180]"""
        angle = torch.atan2(sin_cos[:, 0], sin_cos[:, 1])
        angle = (angle + math.pi) / (2*math.pi) # normalized angle
        indices = angle * self.max_len
        # for ii, ind in enumerate(indices):
        #     if ind > (self.max_len - 1):
        #         indices[ii] = self.max_len - 1
        indices = angle * (self.max_len - 1)
        return indices.to(torch.int32)

    def forward(self, sin_cos):  # theta: (N,)
        # sin_cos = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)  # (N, 2)
        if self.use_pe:
            indices = self.ori_to_index(sin_cos)
            x = self.pe(indices)
            return self.encoder(x)  # (N, D)
        else:
            return self.encoder(sin_cos)  # (N, D)

class CrossAttentionConditioning(nn.Module):
    def __init__(self, feature_dim, angle_dim, heads=4):
        super().__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(angle_dim, feature_dim)
        self.value_proj = nn.Linear(angle_dim, feature_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, angle_embed):
        # x: (N, T*V, C), angle_embed: (N, angle_dim)
        query = self.query_proj(x)  # (N, L, C)
        key = self.key_proj(angle_embed).unsqueeze(1)  # (N, 1, C)
        value = self.value_proj(angle_embed).unsqueeze(1)  # (N, 1, C)
        out, _ = self.multihead_attn(query, key, value)  # (N, L, C)
        return self.norm(out + x)  # Residual + norm

class CrossAttentationAngleQuery(nn.Module):
    def __init__(self, input_dim, model_dim, conditioning_dim, num_queries=1, heads=4, use_pe=False):
        super().__init__()
        # --- Partie A : Query = angle, Key/Value = pose ---
        self.key_proj_pose = nn.Linear(input_dim, model_dim)
        self.value_proj_pose = nn.Linear(input_dim, model_dim)
        self.query_base_angle = nn.Parameter(torch.randn(num_queries, model_dim))
        if use_pe:
            self.angle_proj = AngleEncoder(model_dim, use_pe)
        else:
            # self.angle_proj = AngleEncoder(model_dim, use_pe)
            self.angle_proj = nn.Sequential(
                nn.Linear(conditioning_dim, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim)
            )
        self.attn_A = nn.MultiheadAttention(embed_dim=model_dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, pose_vec, angle_vec):
        B, T, _ = pose_vec.size()
        # ========== Branche A ==========
        # angle ? Query, pose ? Key/Value
        K_A = self.key_proj_pose(pose_vec)
        V_A = self.value_proj_pose(pose_vec)
        Q_A = self.query_base_angle.unsqueeze(0).expand(B, -1, -1) + self.angle_proj(angle_vec).unsqueeze(1)
        # Q_A = self.query_base_angle.unsqueeze(0).expand(B, -1, -1) + angle_vec# self.angle_proj(angle_vec).unsqueeze(1)
        # Q_A = angle_vec
        z_A, _ = self.attn_A(Q_A, K_A, V_A)  # (B, num_queries, D)
        return z_A
        return self.norm(z_A)# + pose_vec)  # Residual + norm

class DualAngleConditioning(nn.Module):
    def __init__(self, backbone, fusion='concat', model_dim=256, use_pe=False):
        super().__init__()

        self.fusion = fusion  # 'concat', 'sum', or 'mlp'

        self.backbone = backbone

        self.angle_encoder = AngleEncoder(out_dim=model_dim, use_pe=use_pe)
        self.cross_attn_pose_query = CrossAttentionConditioning(feature_dim=model_dim, angle_dim=model_dim)

        self.num_queries = 1
        self.cross_attn_angle_query = CrossAttentationAngleQuery(model_dim, model_dim, 2, num_queries=self.num_queries, heads=4, use_pe=use_pe)

        # Fusion des deux sorties
        if fusion == 'mlp':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim)
            )

    def forward(self, x_and_angle):
        if type(x_and_angle) is tuple:
            x, theta_sin_cos = x_and_angle
            Na, Ma, Ca = theta_sin_cos.shape
            theta_sin_cos = theta_sin_cos.reshape(Na*Ma, Ca)
            # breakpoint()
        else:
            x = x_and_angle
            Na, Ma, _, _, _ = x.shape
            theta_sin_cos = torch.tensor([[0., 1.0]] * (Na*Ma), requires_grad=False).to(x.device) # set orientation to 0°

        x = self.backbone(x)
        N, M, C, T, V = x.shape

        # pose query
        x_A = x.permute(0, 1, 3, 4, 2).reshape(N*M, T*V, C)
        angle_embed = self.angle_encoder(theta_sin_cos)  # (N, C)
        z_A = self.cross_attn_pose_query(x_A, angle_embed)  # (N*M, T*V, C)
        # x = x.permute(0, 2, 1) # (N*M, C, T*V)
        # x = x.reshape(N, M, C, T, V)

        # angle query
        x_B = x.reshape(N*M, C, T, V)
        pool = nn.AdaptiveAvgPool2d(1)
        x_B = pool(x_B)
        x_B = x_B.reshape(N*M, C, 1).permute(0, 2, 1) # (NM, 1, C)
        z_B = self.cross_attn_angle_query(x_B, theta_sin_cos) # (N*M, num_queries, C)
        # x = x.permute(0, 2, 1) # (N*M, C, num_queries)
        # x = x.reshape(N, M, -1, 1, 1)

        # ========== Fusion ==========
        if self.fusion == 'concat':
            # z = torch.cat([z_A, z_B], dim=-1)  # (NM, num_queries, 2C)
            z = torch.cat([z_A, z_B], dim=1)  # (NM, T*V*num_queries, C)
        elif self.fusion == 'sum':
            z = z_A + z_B  # (NM, num_queries, C)
        elif self.fusion == 'mlp':
            z = torch.cat([z_A, z_B], dim=-1)  # (NM, num_queries, 2C)
            z = self.fusion_mlp(z)  # (NM, num_queries, C)
        else:
            raise ValueError("Fusion type must be 'concat', 'sum', ork 'mlp'")

        z = z.permute(0, 2, 1) # (N*M, C, num_queries)
        z = z.reshape(N, M, C, -1, 1) # (N, M, C, -A, 1)

        return z

class DualAngleConditioningProtogcn(DualAngleConditioning):
    def __init__(self, graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 pretrained=None,
                 fusion='concat', 
                 model_dim=256,
                 use_pe=False,
                 **kwargs):
        backbone = ProtoGCN(graph_cfg, in_channels, base_channels, ch_ratio, num_stages, inflate_stages, down_stages, data_bn_type, num_person, pretrained, **kwargs)
        super().__init__(backbone, fusion, model_dim, use_pe)

class DualAngleConditioningAAGCN(DualAngleConditioning):
    def __init__(self, fusion='concat', **kwargs):
        backbone = AAGCN(**kwargs)
        super().__init__(backbone, fusion, 256)
        
class DualAngleConditioningUnik(nn.Module):
    def __init__(self, num_class=60, num_joints=25, num_person=2, tau=1, num_heads=3, in_channels=2, fusion='concat'):
        super().__init__()

        self.fusion = fusion  # 'concat', 'sum', or 'mlp'

        self.unik = UNIK(num_class, num_joints, num_person, tau, num_heads, in_channels)

        model_dim = 256
        self.angle_encoder = AngleEncoder(out_dim=model_dim)
        self.cross_attn_pose_query = CrossAttentionConditioning(feature_dim=model_dim, angle_dim=model_dim)

        self.num_queries = 1
        self.cross_attn_angle_query = CrossAttentationAngleQuery(model_dim, model_dim, 2, num_queries=self.num_queries, heads=4)

        # Fusion des deux sorties
        if fusion == 'mlp':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim)
            )

    def forward(self, x_and_angle):
        x, theta_sin_cos = x_and_angle
        x = self.unik(x)

        N, M, C, T, V = x.shape

        Na, Ma, Ca = theta_sin_cos.shape
        theta_sin_cos = theta_sin_cos.reshape(Na*Ma, Ca)

        # pose query
        x_A = x.permute(0, 1, 3, 4, 2).reshape(N*M, T*V, C)
        angle_embed = self.angle_encoder(theta_sin_cos)  # (N, C)
        z_A = self.cross_attn_pose_query(x_A, angle_embed)  # (N*M, T*V, C)
        # x = x.permute(0, 2, 1) # (N*M, C, T*V)
        # x = x.reshape(N, M, C, T, V)

        # angle query
        x_B = x.reshape(N*M, C, T, V)
        pool = nn.AdaptiveAvgPool2d(1)
        x_B = pool(x_B)
        x_B = x_B.reshape(N*M, C, 1).permute(0, 2, 1) # (NM, 1, C)
        z_B = self.cross_attn_angle_query(x_B, theta_sin_cos) # (N*M, num_queries, C)
        # x = x.permute(0, 2, 1) # (N*M, C, num_queries)
        # x = x.reshape(N, M, -1, 1, 1)

        # ========== Fusion ==========
        if self.fusion == 'concat':
            # z = torch.cat([z_A, z_B], dim=-1)  # (NM, num_queries, 2C)
            z = torch.cat([z_A, z_B], dim=1)  # (NM, T*V*num_queries, C)
        elif self.fusion == 'sum':
            z = z_A + z_B  # (NM, num_queries, C)
        elif self.fusion == 'mlp':
            z = torch.cat([z_A, z_B], dim=-1)  # (NM, num_queries, 2C)
            z = self.fusion_mlp(z)  # (NM, num_queries, C)
        else:
            raise ValueError("Fusion type must be 'concat', 'sum', or 'mlp'")

        z = z.permute(0, 2, 1) # (N*M, C, num_queries)
        z = z.reshape(N, M, C, -1, 1) # (N, M, C, -A, 1)

        return z