# reference: see www.youtube.com/watch?v=l8pRSuU81PU
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    
    block_size: int = 256
    vocab_size: int = 20000
    pad_idx: int = 0
    n_layers: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_ff: int = 4 * 256 
    dropout: float = 0.1

class PositionalEncoding(nn.Module):

    def __init__(self, n_embd, max_len):
        super().__init__()
        pos_enc = torch.zeros(max_len, n_embd, dtype=torch.float32)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2, dtype=torch.float32) * (-math.log(10000.0) / n_embd)
        )
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pos_enc", pos_enc)
    
    def forward(self, T, device=None):
        pos_enc = self.pos_enc[:T]
        return pos_enc if device is None else pos_enc.to(device)

class EmbeddingLayer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0, std=0.02)

        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].zero_()
            
            def hook(grad):
                grad = grad.clone()
                grad[padding_idx].zero_()
                return grad

            self.weight.register_hook(hook)
    
    def forward(self, x):
        """
        x: torch tensor with shape [B, T], dtype long
        """
        assert x.dtype == torch.long, "expected a long type input"
        # assert x.min().item() >= 0 and x.max().item() < self.num_embeddings, "token id out of range"

        out = self.weight.index_select(0, x.view(-1)).view(*x.shape, self.embedding_dim)
        return out
    
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2) # each (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_head, T, head_size)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.dropout(out)

        return out

class FeedForwardNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_ff)
        self.fc2 = nn.Linear(config.n_ff, config.n_embd)
        self.relu = nn.ReLU() # vanilla transformer: ReLU
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out
    
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mhsa = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForwardNetwork(config)

    def forward(self, x, mask=None):
        out = x + self.mhsa(self.ln_1(x), mask=mask)
        out = out + self.ffn(self.ln_2(out))
        return out
    
class TransformerEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = EmbeddingLayer(config.vocab_size, config.n_embd, padding_idx=config.pad_idx),
            wpe = PositionalEncoding(config.n_embd, config.block_size + 1),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        ))
        self.cls = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        nn.init.normal_(self.cls, mean=0, std=0.02)
        self.classifier = nn.Linear(config.n_embd, 2)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, idx, lengths): # add lengths argument to match GRU
        B, T = idx.size()
        assert T <= self.config.block_size

        src_key_padding_mask = (idx == self.config.pad_idx)
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        cls = self.cls.expand(B, 1, self.config.n_embd)
        x = torch.cat([cls, tok_emb], dim=1)

        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=idx.device)
        mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        pos_emb = self.transformer.wpe(T + 1, device=idx.device)
        out = self.dropout(x + pos_emb.unsqueeze(0))

        for block in self.transformer.h:
            out = block(out, mask=mask)

        out = self.ln_f(out)
        cls_out = out[:, 0, :]
        logits = self.classifier(cls_out)
        return logits
