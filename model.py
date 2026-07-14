import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
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
        score = score / math.sqrt(self.head_dim)
        attention = torch.softmax(score, dim = -1)
        output = torch.matmul(attention, V)

        output = output.transpose(1, 2)
        output = output.contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention(x)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)

        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )
        self.position_embedding = nn.Embedding(
            max_len,
            d_model
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self,tokens):
        x = self.embedding(tokens)
        position = torch.arange(tokens.size(1),device = tokens.device)
        position = position.unsqueeze(0)
        pos = self.position_embedding(position)
        x = x+pos
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)
        return logits
            