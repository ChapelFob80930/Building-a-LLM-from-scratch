from layer_normalization import LayerNorm
from transformer_block import TransformerBlock
import torch.nn as nn
import torch

class GPTModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        emb_dim = cfg["emb_dim"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], emb_dim)
        self.pos_emb = nn.Embedding(cfg["context_length"], emb_dim)
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, cfg["vocab_size"], bias=False)


    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape

        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits