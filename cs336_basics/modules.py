import torch
from math import sqrt
from einops import einsum, rearrange

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
    #  ## einsum approach
    attn_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    attn_weights = attn_scores / torch.sqrt(torch.tensor(Q.shape[-1]))

    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask == 0, -torch.inf)

    attn_weights = MySoftmax(attn_weights, dim=-1)
    context_vec = einsum(attn_weights, V, "... queries sl, ... sl d_v -> ... queries d_v")
    return context_vec

class MyRoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.rotation_matrix_table = self.generate_rotation_matrix(theta, d_k, max_seq_len)
        self.register_buffer('rotation_matrix', self.rotation_matrix_table, persistent=False)

    def generate_rotation_block(self, theta, block_index, seq_pos, d_k):
        angle = torch.tensor(seq_pos / (theta ** (2 * block_index / d_k)))
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        rotation_matrix = torch.Tensor([[cos, -sin], [sin, cos]])
        return rotation_matrix
    
    def generate_rotation_matrix(self, theta, d_k, max_seq_len):
        rotation_matrix_table = torch.zeros(max_seq_len, d_k, d_k)
        for i in range(max_seq_len):
            blocks = [self.generate_rotation_block(theta, k, i, d_k) for k in range(d_k // 2)]
            rotation_matrix_table[i, :, :] = torch.block_diag(*blocks)
        return rotation_matrix_table
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        *dims, seq_len, d_k = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        rotation_matrix = self.rotation_matrix_table[token_positions]
        x_rotated = rotation_matrix @ x.unsqueeze(-1)
        x_rotated = x_rotated.squeeze(-1)
        return x_rotated

class MyCausalMHA(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=10000, device=None, use_rope=False, token_positions=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_k = int(d_model / num_heads)
        self.use_rope = use_rope
        self.rope = MyRoPE(theta, self.d_k, max_seq_len) if use_rope else None
        self.token_positions = token_positions

        # self.q_proj = torch.nn.Parameter(torch.randn(d_model, d_model))
        # self.k_proj = torch.nn.Parameter(torch.randn(d_model, d_model))
        # self.v_proj = torch.nn.Parameter(torch.randn(d_model, d_model))
        # self.o_proj = torch.nn.Parameter(torch.randn(d_model, d_model))

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.o_proj = torch.nn.Linear(d_model, d_model)


    def forward(self, x):
        bs, sl, d_model = x.shape

        # queries = einsum(x, self.q_proj.weight, "... sl d_m, d_m d_m -> ... sl d_m")
        # keys = einsum(x, self.k_proj.weight, "... sl d_m, d_m d_m -> ... sl d_m")
        # values = einsum(x, self.v_proj.weight, "... sl d_m, d_m d_m -> ... sl d_m")

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Expand the head dimension into it's own dimension and transpose for self-attention soon
        queries = rearrange(queries, "... sl (h d_head) -> ... h sl d_head", h=self.num_heads)
        keys = rearrange(keys, "... sl (h d_head) -> ... h sl d_head", h=self.num_heads)
        values = rearrange(values, "... sl (h d_head) -> ... h sl d_head", h=self.num_heads)

        # use rope if needed
        if self.use_rope:
            queries = self.rope(queries, self.token_positions)
            keys = self.rope(keys, self.token_positions)

        # apply causal mask``
        causal_mask = torch.ones((sl, sl))
        causal_mask = torch.triu(causal_mask, diagonal=1).to(bool)
        context_vec = Myscaled_dot_product_attention(queries, keys, values, ~causal_mask)

        # concatenate head with o_projection & pass it through o_proj
        context_vec = rearrange(context_vec, "... h sl d_v -> ... sl (h d_v)")
        # output = einsum(context_vec, self.o_proj, "... sl d_model, d_model d_v -> ... sl d_v")
        output = self.o_proj(context_vec)
        return output


class MyCausalMHA2(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=10000, device=None, use_rope=False, token_positions=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_k = int(d_model / num_heads)
        self.use_rope = use_rope
        self.rope = MyRoPE(theta, self.d_k, max_seq_len) if use_rope else None
        self.token_positions = token_positions

        self.q_proj = torch.randn(d_model, d_model)
        self.k_proj = torch.randn(d_model, d_model)
        self.v_proj = torch.randn(d_model, d_model)
        self.o_proj = torch.randn(d_model, d_model)

    def forward(self, x):
        bs, sl, d_model = x.shape

        # einsum notation
        queries = einsum(x, self.q_proj, "... sl d_m, d_k d_m -> ... sl d_k")
        keys = einsum(x, self.k_proj, "... sl d_m, d_k d_m -> ... sl d_k")
        values = einsum(x, self.v_proj, "... sl d_m, d_k d_m -> ... sl d_k")

        # pytorch matrix notation
        # queries = x @ self.q_proj.T
        # keys = x @ self.k_proj.T
        # values = x @ self.v_proj.T

        # Expand the head dimension into it's own dimension and transpose for self-attention soon
        queries = rearrange(queries, "... sl (h d_head) -> ... h sl d_head", h=self.num_heads)
        keys = rearrange(keys, "... sl (h d_head) -> ... h sl d_head", h=self.num_heads)
        values = rearrange(values, "... sl (h d_head) -> ... h sl d_head", h=self.num_heads)

        # use rope if needed
        if self.use_rope:
            queries = self.rope(queries, self.token_positions)
            keys = self.rope(keys, self.token_positions)

        # apply causal mask
        causal_mask = torch.ones((sl, sl))
        causal_mask = torch.triu(causal_mask, diagonal=1).to(bool)
        context_vec = Myscaled_dot_product_attention(queries, keys, values, ~causal_mask)

        # concatenate head with o_projection & pass it through o_proj
        context_vec = rearrange(context_vec, "... h sl d_v -> ... sl (h d_v)")
        output = einsum(context_vec, self.o_proj, "... sl d_model, d_v d_model -> ... sl d_v")
        # output = context_vec @ self.o_proj.T
        return output