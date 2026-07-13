import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        score = torch.matmul(Q, K.transpose(-2, -1))
        attention = torch.softmax(score / math.sqrt(self.num_heads), dim = -1)
        output = torch.matmul(attention,V)

        output = output.transpose(1, 2)
        output = output.contiguous()

        output = output.view(batch_size, seq_len, self.d_model)

        return output

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)

    mha = MultiHeadAttention(
        d_model=8,
        num_heads=2
    )

    output = mha(x)

    print(x.shape)
    print(output.shape)