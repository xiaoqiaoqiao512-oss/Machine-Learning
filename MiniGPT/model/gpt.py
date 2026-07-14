import torch
import torch.nn as nn

from .embedding import GPTEmbedding
from .block import TransformerBlock
from config import GPTConfig

class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embedding = GPTEmbedding(
            config
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config=config
            )
            for _ in range(config.num_layers)
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)

        return logits