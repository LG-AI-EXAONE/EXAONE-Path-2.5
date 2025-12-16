from typing import Optional

from torch import Tensor, nn
import torch

__all__ = ["WSIFeatureEmbed"]


class WSIFeatureEmbed(nn.Module):
    """Linear projection for WSI patch features shaped as [B, N, C].

    - Projects features to model dimension
    - Optionally applies LayerNorm
    - Optionally concatenates CLS and storage tokens provided by caller
    """

    def __init__(self, input_dim: int, embed_dim: int, use_norm: bool = True) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Linear(self.input_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-5) if use_norm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        cls_token: Optional[Tensor] = None,
        storage_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        # x: [B, N, C], cls_token: [1,1,D], storage_tokens: [1,S,D]
        if x.dim() != 3:
            raise ValueError(f"WSIFeatureEmbed expects [B, N, C] input but received {tuple(x.shape)}")
        B = x.size(0)
        x = self.proj(x)
        x = self.norm(x)
        if cls_token is not None:
            cls = cls_token.expand(B, -1, -1)
            if storage_tokens is not None and storage_tokens.numel() > 0:
                stor = storage_tokens.expand(B, -1, -1)
            else:
                stor = x.new_empty(B, 0, x.size(-1))
            x = torch.cat([cls, stor, x], dim=1)
        return x  # [B, 1+S+N, D] if cls/storage provided else [B, N, D]
