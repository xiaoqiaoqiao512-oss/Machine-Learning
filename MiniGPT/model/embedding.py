import torch
import torch.nn as nn
from config import GPTConfig

class GPTEmbedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )
        self.pos_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

    def forward(self, tokens):
        x = self.embedding(tokens)
        position = torch.arange(tokens.size(1), device= tokens.device)
        position = position.unsqueeze(0)
        pos = self.pos_embedding(position)
        x = x + pos

        return x