import torch
from torch import nn
from torch import Tensor
from torch.nn import Module


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        # initialize the weights
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W.T


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embd = num_embeddings
        self.embd_dim = embedding_dim

        self.E = nn.Parameter(torch.empty(self.num_embd, self.embd_dim, device=device, dtype=dtype))
        std = 1
        torch.nn.init.trunc_normal_(self.E, std=std, a=-1, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.E[token_ids]


class RMSNorm(Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # scale factor
        self.alpha = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        result = x * torch.rsqrt(variance + self.eps) * self.alpha

        return result.to(in_dtype)


class SwiGLU(Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        self.W3 = nn.Linear(d_model, d_ff, bias=False)

    def _silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.W2(self._silu(self.W1(x)) * self.W3(x))


class RoPEEmbedding(Module):
    def __init__(
        self,
        theta_base: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        self.theta_base = theta_base
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1 / (theta_base ** (torch.arange(0, d_k, 2, device=device)[: d_k // 2].float() / d_k))
        positions = torch.arange(0, max_seq_len, device=device)
        # (max_seq_len, d_k // 2)
        angles = torch.outer(positions, inv_freq)

        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # (..., seq_len, d_k// 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # (.., seq_len, d_k // 2)
        cos = self.get_buffer("cos_cache")[token_positions]
        sin = self.get_buffer("sin_cache")[token_positions]

        # (..., seq_len, d_k // 2)
        roated_even = cos * x_even - sin * x_odd
        roated_odd = sin * x_even + cos * x_odd

        # (..., seq_len, d_k // 2, 2)
        rotated = torch.stack([roated_even, roated_odd], dim=-1)
        rotated = rotated.flatten(-2)

        return rotated


