from typing import Callable, List, Optional

import torch
from torch import Tensor, nn

from .attention import SelfAttention
from .ffn_layers import SwiGLUFFN

class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        ffn_drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = SwiGLUFFN,
        dual_attention: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self._dual_attention = bool(dual_attention)
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            proj_drop=attn_drop,
            device=device,
        )
        self.nope_attn = (
            attn_class(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                proj_drop=attn_drop,
                device=device,
            )
            if self._dual_attention
            else None
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=ffn_drop,
            bias=ffn_bias,
            device=device,
        )

    def compute_attention_output(
        self,
        x: Tensor,
        rope: tuple[Tensor, Tensor] | None = None,
        attn_mask: Tensor | None = None,
        use_nope_attn: bool = False,
    ) -> Tensor:
        """Compute self-attention output without adding the residual."""
        if use_nope_attn:
            if self.nope_attn is None:
                raise RuntimeError("nope_attn is not initialized; instantiate with dual_attention=True")
            attn_module = self.nope_attn
        else:
            attn_module = self.attn
        return attn_module(self.norm1(x), attn_mask=attn_mask, rope=rope)

    def forward_with_attention_output(self, x: Tensor, attn_output: Tensor) -> Tensor:
        """Apply residual connection + FFN given a precomputed attention output."""
        x_attn = x + attn_output
        x_ffn = x_attn + self.mlp(self.norm2(x_attn))
        return x_ffn

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None, attn_mask: Tensor | None = None) -> Tensor:
        """Forward for batched tensor inputs only."""
        attn_output = self.compute_attention_output(x, rope=rope, attn_mask=attn_mask)
        return self.forward_with_attention_output(x, attn_output)