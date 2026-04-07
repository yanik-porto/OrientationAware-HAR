import copy as cp
import torch
import torch.nn as nn
from .utils import Graph
EPS = 1e-4

class GCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, unit_tcn_bn2d=True, **kwargs):

        super().__init__()
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value
        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0

        tcn_kwargs["ms_cfg"] = [(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']
        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = mstcn(out_channels, out_channels, stride=stride, all_bn_2d=unit_tcn_bn2d, **tcn_kwargs)
        # self.tcn = mstcn(out_channels, out_channels, stride=stride, num_joints=A.size(1), **tcn_kwargs) # TODO : check if working
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride, all_bn_2d=unit_tcn_bn2d)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x, gcl_graph = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), gcl_graph


"""
****************************************
*** Prototype Reconstruction Network ***
****************************************
"""  
class Prototype_Reconstruction_Network(nn.Module):
    
    def __init__(self, dim, n_prototype=100, dropout=0.1):
        super().__init__()
        self.query_matrix = nn.Linear(dim, n_prototype, bias = False)
        self.memory_matrix = nn.Linear(n_prototype, dim, bias = False)
        self.softmax = torch.softmax
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        query = self.softmax(self.query_matrix(x), dim=-1)
        z = self.memory_matrix(query)
        return self.dropout(z)


# @BACKBONES.register_module()
class ProtoGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 pretrained=None,
                 return_prn=False,
                 unit_tcn_bn2d=True,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        self.num_person = num_person
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        num_prototype = kwargs.pop('num_prototype', 100)
        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [GCN_Block(in_channels, base_channels, A.clone(), 1, residual=False, unit_tcn_bn2d = unit_tcn_bn2d, **lw_kwargs[0])]

        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(GCN_Block(in_channels, out_channels, A.clone(), stride, unit_tcn_bn2d = unit_tcn_bn2d, **lw_kwargs[i - 1]))
            down_times += (i in down_stages)

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained
        
        out_channels = base_channels
        norm = 'BN'
        norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        
        self.post = nn.Conv2d(out_channels, out_channels, 1)
        # self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        dim = 384   # base_channels * 4
        # dim = 144
        self.prn = Prototype_Reconstruction_Network(dim, num_prototype)
        self.return_prn = return_prn
        
    def init_weights(self):
        if isinstance(self.pretrained, str):
            # self.pretrained = cache_checkpoint(self.pretrained)
            # load_checkpoint(self, self.pretrained, strict=False)
            raise NotImplementedError()

    def forward(self, inputs):
        if type(inputs) is tuple:
            x = inputs[0]
        else:
            x = inputs

        N, M, T, V, C = x.size()

        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            if M != self.num_person :
                n_views = M // self.num_person
                x = x.reshape(N * n_views, self.num_person, T, V, C)
                x = x.permute(0, 1, 3, 4, 2).contiguous().view(N * n_views, self.num_person * V * C, T)
            else:
                x = x.permute(0, 1, 3, 4, 2).contiguous().view(N, M * V * C, T)

            x = self.data_bn(x)
        else:
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        get_graph = []
        for i in range(self.num_stages):
            x, gcl_graph = self.gcn[i](x)
            # N*M C V V
            get_graph.append(gcl_graph)
        
        x = x.reshape((N, M) + x.shape[1:])


        c_graph = x.size(2)
        
        graph = get_graph[-1]
        # c_graph = graph.shape[1]
        # N C V V -> N C V*V
        graph = graph.view(N, M, c_graph, V, V).mean(1).view(N, c_graph, V * V)
        
        the_graph_list = []
        for i in range(N):
            # V*V C
            the_graph = graph[i].permute(1, 0)
            # V*V C
            the_graph = self.prn(the_graph)
            # C V V
            the_graph = the_graph.permute(1, 0).view(c_graph, V, V)
            the_graph_list.append(the_graph)
        
        # N C V V
        re_graph = torch.stack(the_graph_list, dim=0)
        re_graph = self.post(re_graph)
        reconstructed_graph = self.relu(self.bn(re_graph))
        # N V*V
        reconstructed_graph = reconstructed_graph.mean(1).view(N, -1)

        if self.return_prn:
            x = (x, reconstructed_graph)
        
        # case where additive info is provided with input (for head)
        if type(inputs) is tuple:
            if type(x) is tuple:
                x = (*(x), *(inputs[1:]))
            else:
                x = (x, *(inputs[1:]))
        return x#, reconstructed_graph
    
class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 ratio=0.125,
                 intra_act='softmax',
                 inter_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        # self.act = build_activation_layer(self.act_cfg)
        self.act = nn.ReLU()
        self.intra_act = intra_act
        self.inter_act = inter_act

        self.A = nn.Parameter(A.clone())
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            nn.BatchNorm2d(mid_channels * num_subsets), self.act)
            # build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
        self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
                # build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        # self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        
        n, c, t, v = x.shape
        res = self.down(x)
        # K V V
        A = self.A
        A = A[None, :, None, None] 
        
        """
        ***********************************
        *** Motion Topology Enhancement ***
        ***********************************
        """
        # The shape of pre_x is N, K, C, T, V
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v) 
        x1, x2 = None, None
        # N C T V
        tmp_x = x
        # N K C T V
        x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        # N K C 1 V
        x1 = x1.mean(dim=-2, keepdim=True)
        x2 = x2.mean(dim=-2, keepdim=True)
        graph_list = []
        # N K C 1 V V = N K C 1 V 1 - N K C 1 1 V
        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        # N K C 1 V V
        inter_graph = getattr(self, self.inter_act)(diff)
        inter_graph = inter_graph * self.alpha[0]
        # N K C 1 V V = N K C 1 V V + 1 K 1 1 V V
        A = inter_graph + A
        graph_list.append(inter_graph)
        # N K C 1 V * N K C 1 V = N K 1 1 V V
        intra_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
        # N K 1 1 V V
        intra_graph = getattr(self, self.intra_act)(intra_graph)
        intra_graph = intra_graph * self.beta[0]
        # N K C 1 V V = N K 1 1 V V + N K C 1 V V
        A = intra_graph + A
        graph_list.append(intra_graph)
        A = A.squeeze(3)
        # N K C T V = N K C T V * N K C V V
        x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
        # N K C T V -> N K*C T V
        x = x.reshape(n, -1, t, v)
        x = self.post(x)
        """
        ***********************************
        ***********************************
        ***********************************
        """
        
        get_gcl_graph = graph_list[0] + graph_list[1]
        # N K C 1 V V -> N K C V V
        get_gcl_graph = get_gcl_graph.squeeze(3)
        # N K C V V -> N K*C V V
        get_gcl_graph = get_gcl_graph.reshape(n, -1, v, v)
        
        return self.act(self.bn(x) + res), get_gcl_graph
    
class mstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 num_joints=25,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1,
                 all_bn_2d=True):

        super().__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = nn.ReLU()
        self.num_joints = num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None, all_bn_2d=all_bn_2d))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)
        
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :V]
        global_feat = out[..., V]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])
        feat = local_feat + global_feat
        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)
    

class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, norm='BN', dropout=0, all_bn_2d = True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        # self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        # self.bn = nn.BatchNorm2d(out_channels)
        self.bn = nn.Identity() if norm == None and not all_bn_2d else nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))

    def init_weights(self):
        conv_init(self.conv)
        bn_init(self.bn, 1)