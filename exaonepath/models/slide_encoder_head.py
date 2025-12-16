from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .slide_transformer import VisionTransformer

__all__ = ["WSIEncoderHead"]


class WSIEncoderHead(nn.Module):
    """Adapter around VisionTransformer with aggregation over patch tokens.

    Inputs:
    - patch_features: [B, N, C]
    - patch_mask: [B, N] with 1 for valid tokens (required for correct masking)
    - patch_coords: optional [B, N, 2] integer coords for RoPE

        Returns:
        - dict with exactly two keys:
            - patch_embedding: [B, N, C_in + C] concat(raw_patch_features, transformer_patch_tokens)
            - slide_embedding: [B, C_in + C] concat(masked_mean(raw_patch_features), masked_mean(transformer_patch_tokens))
    """

    def __init__(
        self,
        transformer: VisionTransformer,
        input_dim: int,
        embed_dim: int, # aggregator token channel dim
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.embed_dim = int(embed_dim)
        self.input_dim = int(input_dim)

    def _masked_mean(self, tokens: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Mask-aware mean over sequence dimension without fallback.

        - tokens: [B, N, C]
        - mask: [B, N] with 1 valid, 0 invalid; when all invalid, returns zero-vector mean (sum=0, count=1)
        """
        if mask is None:
            return tokens.mean(dim=1)
        valid = mask.to(dtype=tokens.dtype).unsqueeze(-1)  # [B, N, 1]
        sums = (tokens * valid).sum(dim=1)  # [B, C]
        counts = valid.sum(dim=1).clamp_min(1.0)  # [B, 1]
        return sums / counts

    def forward(
        self,
        patch_features: Tensor,
        patch_mask: Tensor,
        patch_coords: Optional[Tensor] = None,
        patch_contour_index: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # patch_features: [B, N, C], patch_mask: [B, N] with 1 for valid tokens
        if patch_mask is None:
            raise ValueError("WSIFeatureEncoder requires patch_mask (shape [B, N]) to be provided.")

        mask = patch_mask.to(device=patch_features.device)
        # Pass optional per-patch contour indices to restrict attention within contours when provided.
        encoded = self.transformer(
            patch_features,
            masks=mask,
            coords=patch_coords,
            contour_index=patch_contour_index,
        )
        patch_tokens = encoded["x_norm_patchtokens"]  # [B, N, C]

        # Patch-level embedding: concat(raw patch features, transformer patch tokens)
        patch_embedding = torch.cat([patch_features, patch_tokens], dim=-1)  # [B, N, C_in + C]

        # Slide-level embedding: concat(masked mean of raw patch features, masked mean of transformer patch tokens)
        raw_patch_mean = self._masked_mean(patch_features, mask)  # [B, C_in]
        token_mean = self._masked_mean(patch_tokens, mask)  # [B, C]
        slide_embedding = torch.cat([raw_patch_mean, token_mean], dim=-1)  # [B, C_in + C]

        return {
            "patch_embedding": patch_embedding,
            "slide_embedding": slide_embedding,
        }
