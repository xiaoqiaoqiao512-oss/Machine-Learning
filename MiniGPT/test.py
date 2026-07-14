import torch
import torch.nn as nn
from config import GPTConfig
from model.gpt import MiniGPT

config = GPTConfig()

model = MiniGPT(config)

x = torch.randint(
    0,
    config.vocab_size,
    (2, 8)
)
y = model(x)
print(y.shape)