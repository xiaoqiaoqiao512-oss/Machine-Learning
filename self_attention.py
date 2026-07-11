import torch
import torch.nn as nn
import math

torch.manual_seed(0)
x = torch.randn(3,4)

WQ = torch.randn(4,4)
WK = torch.randn(4,4)
WV = torch.randn(4,4)

Q = x@WQ
K = x@WK
V = x@WV

attention = torch.softmax(Q@K.T/(Q.shape[-1]** 0.5),dim = -1)
output = attention@V

print(Q)
print(K)
print(V)
print(attention)
print(output)

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

if __name__ == "__main__":
    self_attention = SelfAttention(d_model=4)
    output = self_attention(x)
    print(output)