import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3.0, b=3.0)

    def forward(self, x):
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.W = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        
        nn.init.trunc_normal_(self.W, mean=0.0, std=1.0, a=-3.0, b=-3.0)

    def forward(self, token_ids):
        return self.W[token_ids]