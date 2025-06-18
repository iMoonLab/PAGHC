import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import Parameter
from model.utils.utils import degree_hyedge, degree_node
from model.utils.utils import neighbor_distance, get_full_H, weight_init
from einops import rearrange, repeat
from torch import einsum


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """

    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class CrossAttention(nn.Module):
    def __init__(self, in_ch, hidden_dim) -> None:
        super().__init__()
        self.scale = hidden_dim ** -0.5
        self.to_q = Parameter(torch.Tensor(in_ch, hidden_dim))
        self.to_k = Parameter(torch.Tensor(in_ch, hidden_dim))
        self.to_v = Parameter(torch.Tensor(in_ch, in_ch))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q)
        nn.init.xavier_uniform_(self.to_k)
        nn.init.xavier_uniform_(self.to_v)

    def forward(self, x, y):
        q = einsum('nvc,co->nvo', x, self.to_q)
        k = einsum('nvc,co->nvo', y, self.to_k)
        v = einsum('nvc,co->nvo', y, self.to_v)
        attn = torch.softmax(einsum('nqc,nkc->nqk', q, k) * self.scale, dim=-1)
        output = einsum('nqk,nkc->nqc', attn, v)

        return output


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp1(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, n_target=2, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, 200)
        self.fc3 = nn.Linear(200, n_target)

        # self.fc1 = nn.Linear(in_features, in_features)
        # self.fc2 = nn.Linear(in_features, in_features // 2)
        # self.fc3 = nn.Linear(in_features // 2, n_target)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x_out = self.drop(x)
        x = self.fc3(x_out)
        return x, x_out


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2., drop_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.attn = CrossAttention(dim, dim // 4)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x, y):
        x = x + self.attn(x, y)
        x = x + self.mlp(self.norm(x))
        return x


class Layer(nn.Module):
    def __init__(self, in_channels, dropout=0.3):
        super(Layer, self).__init__()
        self.norm_s = nn.LayerNorm(in_channels)
        self.norm_c = nn.LayerNorm(in_channels)
        self.block_sc = Block(in_channels, drop_ratio=dropout)
        self.block_cs = Block(in_channels, drop_ratio=dropout)

    def forward(self, sagittal, coronal):
        sagittal = self.norm_s(sagittal)
        coronal = self.norm_c(coronal)

        sagittal_ = self.block_sc(sagittal, coronal)
        coronal_ = self.block_cs(coronal, sagittal)

        return sagittal_, coronal_


class Model(nn.Module):
    def __init__(self, in_channels, n_target=2, dropout=0.3):
        super().__init__()
        self.drop_out = nn.Dropout(dropout)

        self.l1 = Layer(in_channels, dropout)
        self.l2 = Layer(in_channels, dropout)
        self.l3 = Mlp1(in_channels, n_target=128, drop=dropout)
        self.l4 = Mlp1(in_channels, n_target=128, drop=dropout)
        self.norm_all = nn.LayerNorm(in_channels * 2)

        # self.fc = nn.Linear(in_channels * 2 , n_target)
        self.predict = Mlp1(128, n_target, drop=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # self.label_fc.apply(weight_init)
        self.apply(_init_vit_weights)

    def forward(self, sagittal, coronal, train=False):
        sagittal, coronal = self.l1(sagittal, coronal)
        sagittal, coronal = self.l2(sagittal, coronal)
        sagittal, _ = self.l3(sagittal)
        coronal, _ = self.l4(coronal)
        sagittal = sagittal.mean(dim=1)
        coronal = coronal.mean(dim=1)
        ct_2d_features = sagittal + coronal

        x, x_out = self.predict(ct_2d_features)
        return x, ct_2d_features


