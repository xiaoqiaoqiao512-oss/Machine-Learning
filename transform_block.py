import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        score = torch.matmul(Q, K.transpose(-2, -1))
        score = score / math.sqrt(Q.size(-1))
        attention = torch.softmax(score, dim=-1)
        output = torch.matmul(attention,V)

        return output

class TransformBlock(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.self_attention = SelfAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model),
        )
        
    def forward(self,x):
        attention_output = self.self_attention(x)
        x = self.norm1(x+attention_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x+ffn_output)
        return x
    
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(3,4)
    transform_block = TransformBlock(d_model=4)
    output = transform_block(x)
    print(output)