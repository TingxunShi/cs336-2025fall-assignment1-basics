import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import einsum, rearrange, reduce
from torch import Tensor


def silu(x: Tensor) -> Tensor:
    return x * F.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - max_val
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    d = q.size(-1)
    qk = einsum(q, k, 'b ... t1 d, b ... t2 d -> b ... t1 t2') / (d ** 0.5)
    if mask is not None:
        qk.masked_fill_(~mask, -math.inf)
    return einsum(softmax(qk, -1), v, 'b ... t1 t2, b ... t2 d -> b ... t1 d')


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features).to(device=device, dtype=dtype))
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim).to(device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding[token_ids]


class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model).to(device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_of_squares = reduce(x * x, '... d_model -> ... 1', 'mean')
        rms = torch.sqrt(mean_of_squares + self.eps)
        result = x / rms * self.gain
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        if d_ff is None:
            d_ff = round(d_model * 8 / 3 / 64) * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None,
    ):
        super().__init__()
        # calculate 1 / (Theta ^ ((2k - 2) / d))
        dims = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)  # 0, 2, 4, ... (Sequence of 2k - 2)
        dom = 1.0 / (theta ** (dims / d_k))
        token_indices = torch.arange(0, max_seq_len, dtype=torch.float32, device=device)

        angles = torch.outer(token_indices, dom)   # shape: max_seq_len, d_model / 2

        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)

        # Get the original shape info
        *ori_shape, seq_len, d_k = x.shape

        # Use einops to flatten all batch dimensions
        x_flat = rearrange(x, '... seq_len d_k -> (...) seq_len d_k')
        token_positions_flat = rearrange(token_positions, '... seq_len -> (...) seq_len')

        # Get cos and sin for the token positions using advanced indexing
        cos_angles = self.cos_cached[token_positions_flat]  # [batch_flat, seq_len, d_k//2]
        sin_angles = self.sin_cached[token_positions_flat]  # [batch_flat, seq_len, d_k//2]

        # Use einops to reshape x into pairs for rotation
        x_pairs = rearrange(x_flat, 'batch seq_len (d_half two) -> batch seq_len d_half two', two=2)

        # Create complex representations
        x_complex = torch.complex(x_pairs[..., 0], x_pairs[..., 1])  # [batch_flat, seq_len, d_k//2]
        rot_complex = torch.complex(cos_angles, sin_angles)  # [batch_flat, seq_len, d_k//2]

        # Apply rotation
        x_rotated_complex = x_complex * rot_complex  # [batch_flat, seq_len, d_k//2]

        # Convert back to real representation and use einops to stack
        x_rotated_real = torch.stack([
            x_rotated_complex.real,
            x_rotated_complex.imag
        ], dim=-1)  # [batch_flat, seq_len, d_k//2, 2]

        # Use einops to flatten the last two dimensions
        x_rotated = rearrange(x_rotated_real, 'batch seq_len d_half two -> batch seq_len (d_half two)')

        # Reshape back
        x_out = x_rotated.view(*ori_shape, seq_len, d_k)

        return x_out.to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            rope: RotaryPositionalEmbedding | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.rope = rope
        self.dim_head = d_model // num_heads
        self.num_heads = num_heads
        self.QKV = Linear(d_model, d_model * 3, device=device, dtype=dtype)
        self.O = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
            self,
            x: Tensor,
            mask: Tensor | None = None,
            need_mask: bool = True,
            token_positions: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        qkv = self.QKV(x)

        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        q_out = rearrange(q, 'b t (h dh) -> b h t dh', h=self.num_heads, dh=self.dim_head)
        k_out = rearrange(k, 'b t (h dh) -> b h t dh', h=self.num_heads, dh=self.dim_head)
        v_out = rearrange(v, 'b t (h dh) -> b h t dh', h=self.num_heads, dh=self.dim_head)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q_out = self.rope(q_out, token_positions)
            k_out = self.rope(k_out, token_positions)

        if need_mask and mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=0).bool()
            mask = rearrange(mask, 't1 t2 -> 1 1 t1 t2')
        if not need_mask and mask is not None:
            mask = None

        mha = scaled_dot_product_attention(q_out, k_out, v_out, mask)
        mha = rearrange(mha, 'b h t dh -> b t (h dh)')
        return self.O(mha)
