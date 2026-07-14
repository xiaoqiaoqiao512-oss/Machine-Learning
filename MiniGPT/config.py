from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 100

    max_seq_len: int = 32

    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2

    d_ff: int = 512

    dropout: float = 0.1

    batch_size: int = 16
    learning_rate: float = 3e-4
    epochs: int = 20

    device: str = "mps"