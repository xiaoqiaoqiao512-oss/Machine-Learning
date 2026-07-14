import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import GPTConfig
from model import MiniGPT
from dataset import ToyDataset

def train():
    config = GPTConfig()
    model = MiniGPT(config)

    device = torch(config.device)
    model.to(device)

    dataset = ToyDataset(
        vocab_size=config.vocab_size,
        seq_len=config.max_seq_len,
        num_samples=1000,
    )
    dataloader = dataloader(
        dataset,
        batch_size = config.batch_size,
        shuffle=True,
    )
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.learning_rate,
    )
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            logits = logits.view(
                -1,
                config.vocab_size
            )

            y = y.view(-1)
            loss = criterion(
                logits,
                y
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss +=loss.item()
        avg_loss = total_loss / len(dataloader)
        print(
            f"Epoch [{epoch + 1}/{config.epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

if __name__ == "__main__":
    train()