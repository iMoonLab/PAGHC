import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import Parameter
from model.utils.utils import degree_hyedge, degree_node, weight_init
from model.utils.utils import neighbor_distance, get_full_H, get_full_select_H, get_fusion_H
from einops import rearrange, repeat
from torch import einsum


class SelfAttention(nn.Module):
    def __init__(self, in_ch, hidden_dim,dropout=0.3,threshold=1.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.threshold = threshold
        self.scale = hidden_dim ** -0.5
        self.to_q = Parameter(torch.Tensor(in_ch, hidden_dim))
        self.to_k = Parameter(torch.Tensor(in_ch, hidden_dim))

        self.to_attn = Parameter(torch.Tensor(2000, 1))
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q)
        nn.init.xavier_uniform_(self.to_k)
        nn.init.xavier_uniform_(self.to_attn)

    def forward(self, x):
        q = einsum('nvc,co->nvo',x,self.to_q)
        k = einsum('nvc,co->nvo',x,self.to_k)
        qk = einsum('nvc,nkc->nvk',q,k) * self.scale
        attn = einsum('nvk,kt->nvt',qk,self.to_attn).squeeze(-1)
        attn = F.normalize(attn,p=1,dim=1)
        return attn


class HyConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, bias=True) -> None:
        super().__init__()
        self.drop_out = dropout  # nn.Dropout(dropout)#nn.Dropout(dropout)
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.bn = nn.LayerNorm(out_ch)
        self.relu = nn.LeakyReLU()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, H, coors=None, hyedge_weight=None):
        assert len(x.shape) == 3, 'the input of HyperConv should be N * V * C'

        y = einsum('nvc,co->nvo', x, self.theta)  # x.matmul(self.theta)

        if hyedge_weight is not None:
            Dv = torch.diag_embed(1.0 / (H * hyedge_weight.unsqueeze(1)).sum(-1))
        else:
            Dv = torch.diag_embed(1.0 / H.sum(-1))
        HDv = einsum('nkv,nve->nke', Dv, H)

        De = torch.diag_embed(1.0 / H.sum(1))
        HDe = einsum('nve,nek->nvk', H, De)
        if hyedge_weight is not None:
            HDe = einsum('nve,ne->nve', HDe, hyedge_weight)
        y = einsum('nvc,nve->nec', y, HDe)  # HDe
        y = einsum('nec,nve->nvc', y, HDv)
        y = y + self.bias.unsqueeze(0).unsqueeze(0)

        return y


class HyConv_woparam(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, H, coors=None, hyedge_weight=None):
        assert len(x.shape) == 3, 'the input of HyperConv should be N * V * C'
        y = x

        if hyedge_weight is not None:
            Dv = torch.diag_embed(1.0 / (H * hyedge_weight.unsqueeze(1)).sum(-1))
        else:
            Dv = torch.diag_embed(1.0 / H.sum(-1))
        HDv = einsum('nkv,nve->nke', Dv, H)

        De = torch.diag_embed(1.0 / H.sum(1))
        HDe = einsum('nve,nek->nvk', H, De)
        if hyedge_weight is not None:
            HDe = einsum('nve,ne->nve', HDe, hyedge_weight)
        y = einsum('nvc,nve->nec', y, HDe)  # HDe
        y = einsum('nec,nve->nvc', y, HDv)

        # y = self.pooling(y)
        # y = self.bn(y)
        return y  # self.relu(y)


class Residual_Block(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        self.bias = Parameter(torch.Tensor(out_ch))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)
    def forward(self,x):
        y = einsum('nvc,co->nvo',x,self.theta)
        y = y + self.bias.unsqueeze(0).unsqueeze(0)
        return y

class HDS_Module(nn.Module):
    def __init__(self, in_ch, out_ch, layer_num=2, step=10, dropout=0.15, alpha=0.05):
        super().__init__()
        self.layer_num = layer_num
        self.step = step
        self.alpha = alpha
        self.drop_out = nn.Dropout(dropout)
        self.theta = nn.Linear(in_ch, out_ch, bias=True)
        self.relu = nn.LeakyReLU()
        self.intra_hyconv = HyConv_woparam(out_ch, out_ch)

    def forward(self, x, H0):
        for i in range(self.layer_num):
            x = x + self.relu(self.theta(x))
            for _step in range(self.step):
                x = self.drop_out(x)
                newx = x * (1 - self.alpha) + self.intra_hyconv(x, H0) * self.alpha
                x = newx
        return x

class Hyconv_Module(nn.Module):
    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.drop_out = nn.Dropout(dropout)
        self.intra_hyconv0 = HyConv(in_ch, out_ch,dropout=dropout)
        self.bn = nn.LayerNorm(out_ch)
        self.relu = nn.LeakyReLU()
        if in_ch == out_ch:
           self.res = lambda x: x
        else:
            self.res = Residual_Block(in_ch, out_ch)

    def forward(self, x, H0): #
        res = self.res(x)

        x0 = self.intra_hyconv0(x, H0)

        x = x0
        x = self.bn(x+res)
        x = self.relu(x)
        x = self.drop_out(x)
        return x

class Model(nn.Module):
    def __init__(self, in_channels, n_target, hiddens, k_threshold=None, k_nearest=None, dropout=0.3,
                 hds_layer_num=2, hds_step=10, hds_dropout=0.15, hds_alpha=0.05):
        super().__init__()
        self.drop_out = nn.Dropout(dropout)
        _in = in_channels

        self.l1 = Hyconv_Module(in_channels, hiddens[0], dropout=dropout)
        self.HDS = HDS_Module(hiddens[0], hiddens[0], layer_num=hds_layer_num, step=hds_step,
                              dropout=hds_dropout, alpha=hds_alpha)

        self.attn = SelfAttention(hiddens[-1], hiddens[-1] // 4, dropout=dropout)
        self.last_fc = nn.Linear(hiddens[-1], n_target)

        self.k_nearest = k_nearest
        self.k_threshold = k_threshold

    def forward(self, x, coors=None, train=False):  #
        H0_ft, _ = self.get_H(x, full=True)
        H0_coor, _ = self.get_H(coors, full=True)
        H0 = torch.concat((H0_ft, H0_coor), dim=-1)

        x_reshape = x
        x_reshape = self.l1(x_reshape, H0)
        x_reshape = self.HDS(x_reshape, H0)

        attn = self.attn(x_reshape)
        x_reshape = einsum('nvc, nv -> nc', x_reshape, attn)

        x = self.drop_out(x_reshape)
        x = self.last_fc(x)

        return x, x_reshape  #

    def get_H(self, fts, full=False):
        if full:
            H, edge_weight = get_full_H(fts, k_threshold=self.k_threshold, k_nearest=self.k_nearest)
            return H, edge_weight
        else:
            return neighbor_distance(fts, k_threshold=self.k_threshold, k_nearest=self.k_nearest)

    def set_k_nearst(self, k_nearest):
        self.k_nearest = k_nearest