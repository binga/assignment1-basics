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
    
def MySoftmax(x: torch.Tensor, dim: int):
    eps = 1e-8

    # numerical stability at high values of logits
    # dim-wise max and not overall max
    max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - max

    # usual business here
    num = torch.exp(x)
    denom = torch.exp(x).sum(dim=dim, keepdim=True)
    output = num / (denom + eps)
    return output

def Myscaled_dot_product_attention(Q, K, V, mask=None):
    ## einsum approach
    attn_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    attn_scores = attn_scores / torch.sqrt(torch.tensor(Q.shape[-1]))

    if mask is not None:
        attn_weights = attn_scores.masked_fill(mask == 0, -torch.inf)

    attn_weights = MySoftmax(attn_weights, dim=-1)
    context_vec = einsum(attn_weights, V, "... queries sl, ... sl d_v -> ... queries d_v")
    return context_vec

class MySDPA(torch.nn.Module):
    def __init__(self, seq_len, emb_size):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.randn(seq_len, emb_size))
        self.W_key   = torch.nn.Parameter(torch.randn(seq_len, emb_size))
        self.W_value = torch.nn.Parameter(torch.randn(seq_len, emb_size))

    def forward(self, x):
        return Myscaled_dot_product_attention(self.W_query, self.W_key, self.W_value, None)

class MyCausalMHA(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, seq_len: int, emb_size: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.d_k = self.d_model / self.num_heads

        self.heads = torch.nn.ModuleList([MySDPA(self.seq_len, self.emb_size)
                                         for _ in range(self.num_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context_vecs = torch.cat([head(x) for head in self.heads], dim=-1)
        return context_vecs