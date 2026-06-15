from multi_head_attention import MultiHeadAttention
from layer_normalization import LayerNorm
from feed_forward_network import FeedForward
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        emb_dim = cfg["emb_dim"]

        # multi-head self attention
        self.attn = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )

        # pre-layernorm
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

        # feed forward network
        self.ffn = FeedForward(cfg)

        # dropout
        self.dropout = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        # x shape: (batch, seq_len, emb_dim)

        # ----- attention block -----
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut

        # ----- feedforward block -----
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x