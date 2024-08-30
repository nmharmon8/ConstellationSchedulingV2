import random
import math
import inspect
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from rotary_embedding_torch import RotaryEmbedding


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, bias=False, causal=True, time_emd=True, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.causal = causal
        self.time_emd = time_emd
        self.query = nn.Linear(n_embd, n_embd, bias=bias)
        self.key = nn.Linear(n_embd, n_embd, bias=bias)
        self.value = nn.Linear(n_embd, n_embd, bias=bias)

        # output projection
        self.out = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            
        if self.time_emd:
            self.rotary_emb = RotaryEmbedding(dim = n_embd//n_head)

    def forward(self, x, kv_cache = None, T_q=None):
        B, _, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.query(x).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if kv_cache is None or self.key not in kv_cache:
            k = self.key(x).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(x).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:
            # Fire the hooks to update the kv_cache
            k = self.key(x[:, -1:, :])
            v = self.value(x[:, -1:, :])
            k = kv_cache[self.key].view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = kv_cache[self.value].view(B, -1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            # If using kv cache we are only useing the last token
            self.causal = False

        offset = T_q if T_q is not None else 0
        if self.time_emd:
            q = self.rotary_emb.rotate_queries_or_keys(q, offset=offset)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        y = y.transpose(1, 2).contiguous().view(B, -1, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.out(y))
        return y

class CrossAttention(nn.Module):
    """
    Query Sequence: The "query" sequence is typically the output sequence for which we are 
    generating the next token. For example, in a language translation task, the query 
    sequence might be the translated words generated so far.

    condition Sequence: The "condition" sequence (also known as the "key-value" sequence) is 
    typically the input sequence that the model uses as context to generate the output 
    sequence. In the language translation example, the condition sequence would be the 
    sentence in the original language that is being translated.
    """
    def __init__(self, n_embd, n_head, bias=False, causal=True, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0

        self.causal = causal
        self.query = nn.Linear(n_embd, n_embd, bias=bias)
        self.key = nn.Linear(n_embd, n_embd, bias=bias)
        self.value = nn.Linear(n_embd, n_embd, bias=bias)

        # output projection
        self.out = nn.Linear(n_embd, n_embd, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.n_head = n_head
        self.n_embd = n_embd
        self.n_embd = n_embd
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.rotary_emb = RotaryEmbedding(dim = n_embd//self.n_head)

    def forward(self, q, condition, kv_cache = None, T_q=None):
        
        B, _, C = q.size()

        q = self.query(q)
        q = q.view(B, -1, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T_q, hs)

        if kv_cache is None or self.key not in kv_cache:
            B, T, _ = condition.size()
            k = self.key(condition).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = self.value(condition).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:

            if condition is not None:
                # Fire the hooks to update the kv_cache
                k = self.key(condition[:, -1:, :])
                v = self.value(condition[:, -1:, :])

            k = kv_cache[self.key].view(B, -1, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = kv_cache[self.value].view(B, -1, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        offset = T_q if T_q is not None else 0
        q = self.rotary_emb.rotate_queries_or_keys(q, offset=offset)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        y = y.transpose(1, 2).contiguous().view(B, -1, C)

        # output projection
        y = self.resid_dropout(self.out(y))
        return y

class MLP(nn.Module):

    def __init__(self, n_embd, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, n_embd, n_head, bias=False, causal=False, time_emd=True, dropout=0.0):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = SelfAttention(n_embd, n_head, bias, causal, time_emd, dropout)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class CrossBlock(nn.Module):

    def __init__(self, n_embd, n_head, bias=False, causal=False, time_emd=True, dropout=0.0):
        super().__init__()
        self.attn = SelfAttention(n_embd, n_head, bias, causal, time_emd, dropout)
        self.attn_ln = LayerNorm(n_embd, bias=bias)
        self.cross_ln = LayerNorm(n_embd, bias=bias)
        self.cross_attn = CrossAttention(n_embd, n_head, bias, causal, dropout)
        self.mpl_ln = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x, condition, kv_cache=None, T=None):
        x = x + self.attn(self.attn_ln(x), kv_cache=kv_cache, T_q=T)
        x = x + self.cross_attn(self.cross_ln(x), condition, kv_cache, T_q=T)
        x = x + self.mlp(self.mpl_ln(x))
        return x
    
@dataclass
class TransformerConfig:
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # bias in Linears and LayerNorms, like Transformer-2. False: a bit better and faster