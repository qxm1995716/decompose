from typing import Optional, Any, Dict, List, Type
import torch.nn as nn


# this map can be used in each step [no checking for variation cased by function forward]
def get_padding_mask(mask):
    # The 1 is for valid value and 0 is for non-data-value
    assert mask.ndim == 2, 'The mask should have two dims.'
    B, N = mask.shape
    active_map = mask.unsqueeze(2) @ mask.unsqueeze(1)
    active_map[active_map == 1] = 0
    active_map[active_map != 1] = -100
    # 此时返回的是一个矩阵，尺寸为[B, N, N]
    return active_map


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForwardModule, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention_v1(nn.Module):
    # this module does not have pos-embedding in each layer
    def __init__(self, input_dim, num_head=1, attn_drop=0., proj_drop=0.):
        super(MultiHeadSelfAttention_v1, self).__init__()
        self.dim_head = input_dim // num_head
        self.heads = num_head
        self.qkv = nn.Linear(input_dim, 3 * input_dim, bias=False)
        self.proj = nn.Linear(input_dim, input_dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_func = nn.LayerNorm(input_dim)

    def forward(self, x, mask=None):
        assert x.ndim == 3, 'The input dim show have 3.'
        B, N, C = x.shape
        x = self.norm_func(x)
        # qkv最后的尺度是[3, B, num_heads, N, dim_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale_factor
        # [B, num_heads, N, dim_head] @ [B, num_heads, dim_head, N] -> [B, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            attn_mask = mask.unsqueeze(-1)
            attn_mask = attn_mask @ attn_mask.transpose(-2, -1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(-111.0)).masked_fill(attn_mask == 1, float(0.0))
            check_mask = attn_mask.clone().detach().cpu().numpy()
            attn = attn + attn_mask.unsqueeze(1)  # 注意增加num_head维度
            attn = self.softmax(attn)
            check_ptb = attn.clone().detach().cpu().numpy()
            bb = 0

        attn = self.attn_drop(attn)
        # [B, nH, N, N] @ [B, nH, N, dim_head] -> [B, nH, N, dim_head] -> transpose(2, 1) -> [B, N, nH, dim_head]
        # -> reshape to [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MultiHeadSelfAttention_O(nn.Module):
    # Adapted from PyTorch Transformer class
    # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py
    # BSD 3-Clause License
    """
    x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x
    """
    def __init__(self, input_dim, num_head, attn_drop, proj_drop):
        super(MultiHeadSelfAttention_O, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_head, dropout=attn_drop,
                                               batch_first=True)
        self.norm_func = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        # x is [B, N, C], while mask is [B, N], 1 for valid element and 0 for non-data-value
        assert x.ndim == 3, 'The dim of input feature should be three.'
        x = self.norm_func(x)
        if mask is not None:
            bool_mask = mask == 0  # for this param, True for ignore and False for valid element
            attn, _ = self.attention(x, x, x, key_padding_mask=bool_mask)  # 根据源码来看，这里输入均输入x即可，其有内在的映射编码
        else:
            attn, _ = self.attention(x, x, x)

        return self.dropout(attn)


class TransformerEncoder_BlockV1(nn.Module):
    def __init__(self,
                 input_dim,
                 num_head,
                 ffn_hidden_dim,
                 attn_drop,
                 proj_drop,
                 ffn_drop):
        super(TransformerEncoder_BlockV1, self).__init__()
        self.mhsa = MultiHeadSelfAttention_v1(input_dim=input_dim, num_head=num_head, attn_drop=attn_drop,
                                              proj_drop=proj_drop)
        self.ffn = FeedForwardModule(dim=input_dim, hidden_dim=ffn_hidden_dim, dropout=ffn_drop)

    def forward(self, x, mask=None):
        shortcut = x
        x = self.mhsa(x, mask) + shortcut
        x = self.ffn(x) + x
        return x


class TransformerEncoder_BlockO(nn.Module):  # O mean Original，也就是原始版本的self attention
    def __init__(self,
                 input_dim,
                 num_head,
                 ffn_hidden_dim,
                 attn_drop,
                 proj_drop,
                 ffn_drop):
        super(TransformerEncoder_BlockO, self).__init__()
        self.mhsa = MultiHeadSelfAttention_O(input_dim=input_dim, num_head=num_head, attn_drop=attn_drop,
                                             proj_drop=proj_drop)
        self.ffn = FeedForwardModule(dim=input_dim, hidden_dim=ffn_hidden_dim, dropout=ffn_drop)

    def forward(self, x, mask=None):
        shortcut = x
        x = self.mhsa(x, mask) + shortcut
        x = self.ffn(x) + x
        return x


class TransformerEncoder_BlockV1_PE(nn.Module):
    def __init__(self,
                 input_dim,
                 num_head,
                 ffn_hidden_dim,
                 attn_drop,
                 proj_drop,
                 ffn_drop,
                 pos_dim=2):
        super(TransformerEncoder_BlockV1_PE, self).__init__()
        self.mhsa = MultiHeadSelfAttention_O(input_dim=input_dim, num_head=num_head, attn_drop=attn_drop,
                                             proj_drop=proj_drop)
        self.ffn = FeedForwardModule(dim=input_dim, hidden_dim=ffn_hidden_dim, dropout=ffn_drop)

    def forward(self, x, mask=None):
        shortcut = x
        x = self.mhsa(x, mask) + shortcut
        x = self.ffn(x) + x
        return x


# Transformer Encoder
class TFEncoder(nn.Module):
    def __init__(self,
                 sa_block_num,
                 input_dim,
                 num_head,
                 ffn_hidden_dim,
                 sa_block: nn.Module,
                 attn_drops: list or float,
                 proj_drops: list or float,
                 ffn_drops: list or float):
        super(TFEncoder, self).__init__()
        backbone = []
        for idx in range(sa_block_num):
            _attn_drop = attn_drops[idx] if type(attn_drops) is list else attn_drops
            _proj_drop = proj_drops[idx] if type(proj_drops) is list else proj_drops
            _ffn_drop = ffn_drops[idx] if type(ffn_drops) is list else ffn_drops
            backbone.append(
                sa_block(input_dim=input_dim, num_head=num_head, ffn_hidden_dim=ffn_hidden_dim, attn_drop=_attn_drop,
                         proj_drop=_proj_drop, ffn_drop=_ffn_drop)
            )

        self.backbone = nn.ModuleList(backbone)

    def forward(self, x, mask=None):
        for block in self.backbone:
            x = block(x, mask=mask)
        return x


