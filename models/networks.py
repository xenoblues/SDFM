import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import math

from torch.nn.attention import sdpa_kernel, SDPBackend

from utils import *

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        '''
        平均更新Transformer模型
        '''
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.use_max_pooling = use_max_pooling
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        if use_max_pooling:
            self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, s, h = x.shape
        # y = self.squeeze(x).view(bs, s)
        # y = self.excitation(y).view(bs, s, 1)
        y = self.squeeze(x.transpose(1, 2)).view(bs, h)
        y = self.excitation(y).unsqueeze(1)
        if self.use_max_pooling:
            y_max = self.max_pool(x).view(bs, s)
            y_max = self.excitation(y_max).view(bs, s, 1)
            y = y + y_max
        y = x * y.expand_as(x)
        return y


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))
        y = x + y
        return y


class TemporalSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_head, dropout, flash_attention=False, cross_attention=False):
        super().__init__()
        self.num_head = num_head
        self.dropout_p = dropout
        self.flash_attention = flash_attention
        self.cross_attention = cross_attention
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        if cross_attention:
            self.key_mod = nn.Linear(latent_dim, latent_dim, bias=False)
            self.value_mod = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, x, mod_emb=None):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        C = D // H
        q = self.query(self.norm(x))  # B, T, D
        k = self.key(self.norm(x))
        v = self.value(self.norm(x))

        if not self.flash_attention:
            # B, T, H, C
            q_ = q.unsqueeze(2).view(B, T, H, C)
            k_ = k.unsqueeze(1).view(B, T, H, C)
            attention = torch.einsum('bnhd,bmhd->bnmh', q_, k_) / math.sqrt(C)
            weight = F.softmax(attention, dim=2)
            v_ = v.view(B, T, H, -1)
            y_s = torch.einsum('bnmh,bmhd->bnhd', weight, v_).reshape(B, T, D)
        else:
            # (B, T, D) -> (B, H, T, C)
            q_ = q.view(B, T, H, C).transpose(1, 2)
            k_ = k.view(B, T, H, C).transpose(1, 2)
            v_ = v.view(B, T, H, C).transpose(1, 2)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                y_s = F.scaled_dot_product_attention(q_, k_, v_, dropout_p=self.dropout_p).transpose(1, 2).reshape(B, T,
                                                                                                                   D)

        # cross attention
        if self.cross_attention and mod_emb is not None:
            k_mod = self.key_mod(self.norm(mod_emb))
            v_mod = self.value_mod(self.norm(mod_emb))
            if not self.flash_attention:
                q_ = q.view(B, T, H, C)
                k_mod_ = k_mod.view(B, T, H, C)
                v_mod_ = v_mod.view(B, T, H, C)
                cross_attention = torch.einsum('bnhd,bmhd->bnmh', q_, k_mod_) / math.sqrt(C)
                cross_weight = self.dropout(F.softmax(cross_attention, dim=2))
                y_c = torch.einsum('bnmh,bmhd->bnhd', cross_weight, v_mod_).reshape(B, T, D)
            else:
                q_ = q.view(B, T, H, C).transpose(1, 2)
                k_mod_ = k_mod.view(B, T, H, C).transpose(1, 2)
                v_mod_ = v_mod.view(B, T, H, C).transpose(1, 2)
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    y_c = F.scaled_dot_product_attention(q_, k_mod_, v_mod_, dropout_p=self.dropout_p).transpose(1,
                                                                                                                 2).reshape(
                        B, T, D)
            y = y_s + y_c
        else:
            y = y_s

        y = x + y
        return y


class TemporalDiffusionTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 latent_dim=32,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.5,
                 skip=False,
                 se_dim=20,
                 se_r=16,
                 flash_attention=True,
                 cross_attention=False,
                 se_layer=True,
                 skip_type='add'
                 ):
        super().__init__()
        self.se_layer = se_layer
        self.skip_type = skip_type
        if skip:
            self.skip_linear = nn.Linear(2 * latent_dim, latent_dim)
        if se_layer:
            self.se = SELayer(se_dim, r=se_r, use_max_pooling=False)
        self.sa_block = TemporalSelfAttention(latent_dim, num_head, dropout, flash_attention, cross_attention)
        self.ffn = FFN(latent_dim, ffn_dim, dropout)

    def forward(self, x, skip=None, mod_emb=None):
        if skip is not None:
            if self.skip_type == 'concat':
                x = self.skip_linear(torch.cat([x, skip], dim=-1))
            elif self.skip_type == 'add':
                x = x + skip
            elif self.skip_type == 'none':
                pass
        if self.se_layer:
            x = x + self.se(x)
        x = self.sa_block(x)
        x = self.ffn(x)
        return x


class MotionTransformer(nn.Module):
    def __init__(self,
                 input_feats,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.2,
                 activation="gelu",
                 **kargs):
        super().__init__()

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.cross_attention = kargs['cross_attention']
        self.flash_attention = kargs['flash_attention']
        self.se_layer = kargs['se_layer']
        self.skip_type = kargs['skip_type']
        self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))

        # Input Embedding
        self.joint_embed = nn.Linear(self.input_feats, self.latent_dim)

        self.cond_embed = nn.Linear(self.input_feats * self.num_frames, self.time_embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.in_blocks = nn.ModuleList([
            TemporalDiffusionTransformerDecoderLayer(
                latent_dim=latent_dim,
                time_embed_dim=self.time_embed_dim,
                ffn_dim=ff_size,
                num_head=num_heads,
                dropout=dropout,
                se_dim=latent_dim,
                cross_attention=self.cross_attention,
                flash_attention=self.flash_attention,
                se_layer=self.se_layer,
                skip_type=self.skip_type
            )
            for i in range(self.num_layers // 2)])

        self.out_blocks = nn.ModuleList([
            TemporalDiffusionTransformerDecoderLayer(
                latent_dim=latent_dim,
                time_embed_dim=self.time_embed_dim,
                ffn_dim=ff_size,
                num_head=num_heads,
                dropout=dropout,
                skip=True,
                se_dim=latent_dim,
                cross_attention=self.cross_attention,
                flash_attention=self.flash_attention,
                se_layer=self.se_layer,
                skip_type=self.skip_type
            )
            for i in range(self.num_layers // 2)])

        self.out = zero_module(nn.Linear(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mod=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim)).unsqueeze(1)

        if mod is not None:
            mod_proj = self.cond_embed(mod.reshape(B, -1)).unsqueeze(1)
            emb = emb + mod_proj

        h = self.joint_embed(x)
        # h = torch.cat([emb, h], dim=1)
        h = h + emb
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        if self.cross_attention and mod is not None:
            mod_emb = self.joint_embed(mod)
            mod_emb = mod_emb + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        else:
            mod_emb = None

        skips = []

        for blk in self.in_blocks:
            h = blk(h, mod_emb=mod_emb)
            skips.append(h)

        # h = self.mid_block(h)

        for blk in self.out_blocks:
            h = blk(h, skips.pop(), mod_emb=mod_emb)

        output = self.out(h[:, :T, :]).view(B, T, -1).contiguous()
        return output


if __name__ == '__main__':
    squeeze = nn.AdaptiveAvgPool1d(1)
    x = torch.randn(1, 20, 512).transpose(1, 2)
    y = squeeze(x)
    print(y.shape)
