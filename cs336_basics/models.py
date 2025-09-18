from torch import nn, Tensor

from .modules import Embedding, Linear, MultiHeadAttention, RMSNorm, RotaryPositionalEmbedding, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope: RotaryPositionalEmbedding
    ):
        super().__init__()
        self.attn_rmsnorm = RMSNorm(d_model)
        self.swiglu_rmsnorm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, rope)
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        attn_x = x + self.attn(self.attn_rmsnorm(x))
        result = attn_x + self.swiglu(self.swiglu_rmsnorm(attn_x))
        return result


class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            theta: float,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            weight_tying: bool = False,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, context_length)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope=self.rope)
                                     for _ in range(num_layers)])
        self.lm_head_rms = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        if weight_tying:
            self.lm_head.weight = self.embedding.embedding

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.lm_head(self.lm_head_rms(x))
        return x
