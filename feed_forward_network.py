import torch
import torch.nn as nn
from GLEU import GELU

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), # Expansion layer
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]) # Projection layer
        )
    
    def forward(self, x):
        return self.layers(x)