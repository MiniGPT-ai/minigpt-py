import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj  = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # create a buffer on CPU
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        # project to heads
        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # move mask to same device as x
        mask = self.mask[:T, :T].to(x.device)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(config.n_embd, config.ffn_hidden_size),
            nn.GELU(),
            nn.Linear(config.ffn_hidden_size, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed     = nn.Embedding(config.vocab_size, config.n_embd)
        self.config    = config
        # positional embeddings as a Parameter
        self.pos_embed = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.dropout   = nn.Dropout(config.dropout)
        self.blocks    = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f      = nn.LayerNorm(config.n_embd)
        self.head      = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.embed.weight

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.pos_embed.size(1), f"Input length {T} exceeds block size"

        # move pos_embed slice to input device
        pos = self.pos_embed[:, :T, :].to(idx.device)

        x = self.embed(idx) + pos
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
