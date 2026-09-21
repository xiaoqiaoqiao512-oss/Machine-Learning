import torch
import torch.nn as nn
from config import GPTConfig
from model.gpt import MiniGPT

config = GPTConfig()

model = MiniGPT(config)

x = torch.randint(
    0,
    config.vocab_size,
    (2, config.max_seq_len)
)

logits = model(x)

print(x.shape)

print(logits.shape)