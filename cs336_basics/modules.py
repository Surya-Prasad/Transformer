import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
    
        std = math.sqrt(2 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3, b=3)

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-5, device = None, dtype = None):
        super().__init__()
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        
        x_f32 = x.to(torch.float32)
        
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        result = (x_f32 / rms) * self.weight
        
        return result.to(in_dtype)

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        
        if d_ff is None:
            d_ff = int(8.0 / 3.0 * d_model)
            d_ff = ((d_ff + 63) // 64) * 64
            
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        
        positions = torch.arange(max_seq_len, device=device).float()

        angles = torch.einsum("i,j->ij", positions, inv_freq)
        
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(start_dim=-2)
        
        return x_rotated

def softmax(x: torch, dim):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_safe = x - x_max
    
    x_exp = torch.exp(x_safe)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(q, k, v, mask = None):
    d_k = q.size(-1)

    scores = torch.einsum("... q d, ... k d -> ... q k", q, k) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
        
    attention_weights = softmax(scores, dim=-1)
    output = torch.einsum("... q k, ... k v -> ... q v", attention_weights, v)
    
    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, rope=None, token_positions=None):
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(*batch_shape, seq_len, self.num_heads, self.d_k)
        k = k.view(*batch_shape, seq_len, self.num_heads, self.d_k)
        v = v.view(*batch_shape, seq_len, self.num_heads, self.d_k)

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        
        if rope is not None and token_positions is not None:
            rope_positions = token_positions.unsqueeze(-2)
            
            q = rope(q, rope_positions)
            k = rope(k, rope_positions)

        row_indices = torch.arange(seq_len, device=x.device).unsqueeze(1)
        col_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
        causal_mask = row_indices >= col_indices

        out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        
        out = out.transpose(-2, -3).contiguous()
        out = out.view(*batch_shape, seq_len, self.d_model)
        
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device = None, dtype = None):
        super().__init__()
        
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype)
        
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope = None, token_positions = None):
        x = x + self.attn(self.ln1(x), rope=rope, token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        d_model: int, 
        num_layers: int, 
        num_heads: int, 
        d_ff: int, 
        rope_theta: float = 10000.0,
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.context_length = context_length
        
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.rope = RoPE(
            theta=rope_theta, 
            d_k=d_model // num_heads, 
            max_seq_len=context_length,
            device=device
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        batch_shape = in_indices.shape[:-1]
        seq_len = in_indices.shape[-1]
        
        assert seq_len <= self.context_length, f"Sequence length {seq_len} exceeds context length {self.context_length}"
        
        x = self.token_embeddings(in_indices)
        token_positions = torch.arange(seq_len, device=in_indices.device)
        token_positions = token_positions.expand(*batch_shape, seq_len)
        
        for layer in self.layers:
            x = layer(x, rope=self.rope, token_positions=token_positions)
            
        x = self.ln_final(x)
        
        logits = self.lm_head(x)
        
        return logits
