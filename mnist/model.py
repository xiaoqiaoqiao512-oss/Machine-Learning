import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x