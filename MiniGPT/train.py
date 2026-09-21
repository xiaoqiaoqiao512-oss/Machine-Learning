import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import GPTConfig
from model import MiniGPT
from dataset import TextDataset
from tokenizer.tokenizer import Tokenizer


def train():

    # ======================
    # 读取文本
    # ======================

    with open(
        "data.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()


    # ======================
    # tokenizer
    # ======================

    tokenizer = Tokenizer(text)

    tokens = tokenizer.encode(text)


    print(
        "vocab size:",
        tokenizer.vocab_size
    )

    print(
        "token length:",
        len(tokens)
    )


    # ======================
    # config
    # ======================

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size
    )


    # ======================
    # model
    # ======================

    model = MiniGPT(config)


    device = torch.device(
        config.device
    )

    model.to(device)


    # ======================
    # dataset
    # ======================

    dataset = TextDataset(
        tokens,
        config.max_seq_len
    )


    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )


    # ======================
    # loss optimizer
    # ======================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )


    # ======================
    # training
    # ======================

    model.train()


    for epoch in range(config.epochs):

        total_loss = 0


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


            total_loss += loss.item()


        avg_loss = total_loss / len(dataloader)


        print(
            f"Epoch [{epoch+1}/{config.epochs}] "
            f"Loss: {avg_loss:.4f}"
        )



if __name__ == "__main__":
    train()