import torch
from math import sqrt
from einops import einsum

class MyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        weight = torch.empty(in_features, out_features)
        sigma = sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

        self.W = torch.nn.Parameter(weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch way
        # output = x @ self.W 

        # einsum way
        output = einsum(x, self.W, "... j, j k-> ... k")

        return output
    
class MyEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

        weight = torch.empty(num_embeddings, embedding_dim)
        torch.nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)

        self.W = torch.nn.Parameter(weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch way
        output = self.W[x]

        return output

class MyRMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.device = device
        self.dtype = dtype

        weight = torch.ones(d_model)
        self.W = torch.nn.Parameter(weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch way
        output = x.to(torch.float32)
        denom = torch.sqrt(output.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = output / denom
        output = output * self.W
        output = output.to(self.dtype)

        return output
    
class MySilu(torch.nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch way
        sigm = torch.sigmoid(x)
        output = x * sigm
        return output