import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def get_norm(norm, num_channels, num_groups=32):
    if norm == "in":
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm1d(num_channels)
    elif norm == "ln":
        return nn.LayerNorm(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")


class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class EmbedSequential(nn.Sequential):
    def forward(self, x, t=None, c=None):
        b, l, f = x.shape
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x.reshape(b * l, -1), t.reshape(b*l, -1))
                x = x.reshape(b, l, -1)
                t = t.reshape(b, l, -1)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, t, c)
            else:
                x = layer(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim), nn.GELU(), nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class TSABlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channels=None,
        cond_dim=None,
        time_emb_dim=None,
        clip_length=8,
        n_heads=1,
    ):
        super(TSABlock, self).__init__()
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.clip_length = clip_length
        self.n_heads = n_heads
        self.in_channel = in_channel

        if out_channels:
            self.out_channels = out_channels
            self.down = nn.Linear(self.in_channel, self.out_channels)
        else:
            self.out_channels = in_channel
        if time_emb_dim:
            self.time_emb_layers = nn.Linear(self.time_emb_dim, self.out_channels)
        if cond_dim:
            self.cond_emb_layers = nn.Linear(self.cond_dim, self.out_channels)
        self.to_qkv = nn.Linear(self.out_channels, self.out_channels * 3)
        self.to_out = zero_module(nn.Linear(self.out_channels, self.out_channels))
        self.norm = get_norm("ln", in_channel)

    def forward(self, x, t=None):
        if self.in_channel != self.out_channels:
            x = self.down(x)

        x = x.reshape(-1, self.clip_length, self.out_channels)
        if t is not None:
            x += self.time_emb_layers(t)

        qkv = self.to_qkv(x)
        bs, length, width = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = torch.split(qkv, self.out_channels, dim=2)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bqc,bkc->bqk",
            (q * scale).view(bs * self.n_heads, length, ch),
            (k * scale).view(bs * self.n_heads, length, ch),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = self.to_out(
            torch.einsum(
                "bqk,bkc->bqc", weight, v.reshape(bs * self.n_heads, length, ch)
            ).reshape(bs, length, -1)
        )
        return out


class TCABlock(nn.Module):
    def __init__(
        self,
        query_dim,
        inner_dim,
        cond_dim,
        time_emb_dim=None,
        clip_length=8,
        n_heads=1,
    ):
        super(TCABlock, self).__init__()
        self.clip_length = clip_length
        self.n_heads = n_heads
        self.query_dim = query_dim
        self.cond_dim = cond_dim
        self.inner_dim = inner_dim

        if time_emb_dim:
            self.time_emb_layers = nn.Linear(time_emb_dim, query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)
        self.down = nn.Linear(query_dim, inner_dim)
        self.to_out = zero_module(nn.Linear(inner_dim, query_dim))
        self.norm = get_norm("ln", query_dim)

    def extract_att(self, x, c):
        x = x.reshape(-1, self.clip_length, self.query_dim)
        c = c.reshape(-1, self.clip_length, self.cond_dim)

        q = self.to_q(x)
        k = self.to_k(c)
        v = self.to_v(c)
        bs, length, width = q.shape
        ch = width // (self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bqc,bkc->bqk",
            (q * scale).view(bs * self.n_heads, length, ch),
            (k * scale).view(bs * self.n_heads, length, ch),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = self.to_out(
            torch.einsum(
                "bqk,bkc->bqc", weight, v.reshape(bs * self.n_heads, length, ch)
            ).reshape(bs, length, -1)
        )
        return out, weight

    def forward(self, x, c):
        x = x.reshape(-1, self.clip_length, self.query_dim)
        c = c.reshape(-1, self.clip_length, self.cond_dim)

        q = self.to_q(x)
        k = self.to_k(c)
        v = self.to_v(c)
        bs, length, width = q.shape
        ch = width // (self.n_heads)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bqc,bkc->bqk",
            (q * scale).view(bs * self.n_heads, length, ch),
            (k * scale).view(bs * self.n_heads, length, ch),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = self.to_out(
            torch.einsum(
                "bqk,bkc->bqc", weight, v.reshape(bs * self.n_heads, length, ch)
            ).reshape(bs, length, -1)
        )
        return out


class TransformerBlock(nn.Module):
    def __init__(self, query_dim, inner_dim, cond_dim=None, time_emb_dim=None):
        super().__init__()
        if time_emb_dim is not None:
            self.time_emb_layers = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, query_dim),
            )
        self.cond_dim = cond_dim
        self.attn1 = TSABlock(
            in_channel=query_dim, time_emb_dim=time_emb_dim, cond_dim=cond_dim
        )  # is a self-attention
        self.norm1 = get_norm("ln", query_dim)
        if self.cond_dim is not None:
            self.attn2 = TCABlock(
                query_dim=query_dim,
                inner_dim=inner_dim,
                time_emb_dim=time_emb_dim,
                cond_dim=cond_dim,
            )  # is cross-attention
            self.norm2 = get_norm("ln", query_dim)
        self.ff = FeedForward(query_dim)
        self.norm3 = get_norm("ln", query_dim)

    def forward(self, x, t=None, c=None):
        b, l, ch = x.shape
        x = self.attn1(self.norm1(x.reshape(-1, ch)).reshape(b, l, ch), t) + x
        if self.cond_dim is not None:
            x = self.attn2(self.norm2(x.reshape(-1, ch)).reshape(b, l, ch), t) + x
        x = self.ff(self.norm3(x.reshape(-1, ch)).reshape(b, l, ch)) + x
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param time_emb_dim: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        in_channel,
        time_emb_dim=None,
        cond_dim=None,
        out_channel=None,
    ):
        super().__init__()
        self.channels = in_channel
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        self.out_channel = out_channel or in_channel
        self.updown = in_channel != out_channel

        if self.updown:
            self.x_upd = nn.Linear(in_channel, self.out_channel)

        self.in_layers = nn.Sequential(
            get_norm("ln", in_channel, 32),
            nn.GELU(),
            nn.Linear(in_channel, self.out_channel),
        )

        if self.time_emb_dim:
            self.time_emb_layers = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, self.out_channel),
            )

        if self.cond_dim:
            self.cond_emb_layers = nn.Sequential(
                nn.GELU(),
                nn.Linear(self.cond_dim, self.out_channel),
            )
        self.out_layers = nn.Sequential(
            get_norm("ln", self.out_channel, 32),
            nn.GELU(),
            zero_module(nn.Linear(self.out_channel, self.out_channel)),
        )

    def forward(self, x, t=None):
        h = self.in_layers(x)
        if self.updown:
            x = self.x_upd(x)

        if t is not None:
            h = h + self.time_emb_layers(t).type(h.dtype).reshape(-1, self.out_channel)

        # if c is not None:
        #   h = h + self.cond_emb_layers(c).type(h.dtype)

        h = self.out_layers(h)
        return x + h


class UnetModel(nn.Module):
    def __init__(
        self,
        in_channel,
        time_emb_dim=None,
        cond_dim=None,
        ch_mult=[2, 2, 2],
        num_res_blocks=2,
    ):
        super(UnetModel, self).__init__()
        self.dtype = torch.float32
        self.cond_dim = cond_dim
        self.time_emb_dim = time_emb_dim or cond_dim

        self.pos = nn.Sequential(
            PositionalEmbedding(self.time_emb_dim),
        )

        self.emb_layers = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.time_emb_dim, in_channel),
        )

        encoder_block_chans = [in_channel]
        self.encoder_blocks = nn.ModuleList([])
        for level, mult in enumerate(ch_mult):
            for i in range(num_res_blocks):
                layers = [
                    ResBlock(
                        in_channel=in_channel,
                        time_emb_dim=self.time_emb_dim,
                        cond_dim=self.cond_dim,
                    )
                ]
                if self.cond_dim is not None:
                    layers.append(
                        TransformerBlock(
                            query_dim=in_channel,
                            inner_dim=in_channel,
                            cond_dim=self.cond_dim,
                            time_emb_dim=self.time_emb_dim,
                        )
                    )
                else:
                    layers.append(
                        ResBlock(
                            in_channel=in_channel,
                            time_emb_dim=self.time_emb_dim,
                            cond_dim=self.cond_dim,
                        )
                    )
                self.encoder_blocks.append(EmbedSequential(*layers))
                encoder_block_chans.append(in_channel)
            if level != len(ch_mult) - 1:
                layers = [
                    ResBlock(
                        in_channel=in_channel,
                        time_emb_dim=self.time_emb_dim,
                        cond_dim=self.cond_dim,
                    )
                ]
                if self.cond_dim is not None:
                    layers.append(
                        TransformerBlock(
                            query_dim=in_channel,
                            inner_dim=in_channel,
                            cond_dim=self.cond_dim,
                            time_emb_dim=self.time_emb_dim,
                        )
                    )
                else:
                    layers.append(
                        ResBlock(
                            in_channel=in_channel,
                            time_emb_dim=self.time_emb_dim,
                            cond_dim=self.cond_dim,
                        )
                    )
                layers.append(
                    ResBlock(
                        in_channel=in_channel,
                        out_channel=in_channel // mult,
                        time_emb_dim=self.time_emb_dim,
                        cond_dim=self.cond_dim,
                    )
                )
                self.encoder_blocks.append(EmbedSequential(*layers))
                in_channel = in_channel // mult
                encoder_block_chans.append(in_channel)

        if self.cond_dim is not None:
            self.mid_blocks = EmbedSequential(
                ResBlock(
                    in_channel=in_channel,
                    time_emb_dim=self.time_emb_dim,
                    cond_dim=self.cond_dim,
                ),
                TransformerBlock(
                    query_dim=in_channel,
                    inner_dim=in_channel,
                    cond_dim=self.cond_dim,
                    time_emb_dim=self.time_emb_dim,
                ),
                ResBlock(
                    in_channel=in_channel,
                    time_emb_dim=self.time_emb_dim,
                    cond_dim=self.cond_dim,
                ),
            )
        else:
            self.mid_blocks = EmbedSequential(
                ResBlock(
                    in_channel=in_channel,
                    time_emb_dim=self.time_emb_dim,
                    cond_dim=self.cond_dim,
                ),
                ResBlock(
                    in_channel=in_channel,
                    time_emb_dim=self.time_emb_dim,
                    cond_dim=self.cond_dim,
                ),
                ResBlock(
                    in_channel=in_channel,
                    time_emb_dim=self.time_emb_dim,
                    cond_dim=self.cond_dim,
                ),
            )

        self.decoder_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(ch_mult))[::-1]:
            for i in range(num_res_blocks):
                ech = encoder_block_chans.pop()
                layers = [
                    ResBlock(
                        in_channel=in_channel + ech,
                        time_emb_dim=self.time_emb_dim,
                        out_channel=in_channel,
                        cond_dim=self.cond_dim,
                    )
                ]
                if self.cond_dim is not None:
                    layers.append(
                        TransformerBlock(
                            query_dim=in_channel,
                            inner_dim=in_channel,
                            cond_dim=self.cond_dim,
                            time_emb_dim=self.time_emb_dim,
                        )
                    )
                else:
                    layers.append(
                        ResBlock(
                            in_channel=in_channel,
                            time_emb_dim=self.time_emb_dim,
                            cond_dim=self.cond_dim,
                        )
                    )
                self.decoder_blocks.append(EmbedSequential(*layers))
            if level != len(ch_mult) - 1:
                ech = encoder_block_chans.pop()
                layers = [
                    ResBlock(
                        in_channel=in_channel + ech,
                        time_emb_dim=self.time_emb_dim,
                        out_channel=in_channel,
                        cond_dim=self.cond_dim,
                    )
                ]
                if self.cond_dim is not None:
                    layers.append(
                        TransformerBlock(
                            query_dim=in_channel,
                            inner_dim=in_channel,
                            cond_dim=self.cond_dim,
                            time_emb_dim=self.time_emb_dim,
                        )
                    )
                else:
                    layers.append(
                        ResBlock(
                            in_channel=in_channel,
                            time_emb_dim=self.time_emb_dim,
                            cond_dim=self.cond_dim,
                        )
                    )
                layers.append(
                    ResBlock(
                        in_channel=in_channel,
                        out_channel=in_channel * mult,
                        time_emb_dim=self.time_emb_dim,
                        cond_dim=self.cond_dim,
                    )
                )
                self.decoder_blocks.append(EmbedSequential(*layers))
                in_channel = in_channel * mult

    def forward(self, x, time, c=None):
        b, t, f = x.shape
        temb = self.pos(time)
        if c is not None:
            temb = (temb + c.reshape(b * t, -1)).reshape(b, t, -1)
        else:
            temb = temb
        h = x.type(self.dtype)
        hs = []
        for module in self.encoder_blocks:
            h = module(h, temb)
            hs.append(h)

        h = self.mid_blocks(h, temb)

        for module in self.decoder_blocks:
            h = torch.cat([h, hs.pop()], dim=-1)
            h = module(h, temb)

        return h

