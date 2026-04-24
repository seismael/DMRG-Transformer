"""Embedding utilities for the DMRG-Transformer."""
from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed).

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim, dtype=dtype)
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=dtype) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: [batch, seq_len, embed_dim]
        """
        return x + self.pe[:, :x.size(1), :]
