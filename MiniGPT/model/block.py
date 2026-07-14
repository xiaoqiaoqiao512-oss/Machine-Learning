import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .mlp import FeedForward
from config import GPTConfig

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.ffn = FeedForward(d_model=config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = x + self.attention(x)
        x = self.norm1(x)

        x = x + self.ffn(x)
        x = self.norm2(x)

        return x