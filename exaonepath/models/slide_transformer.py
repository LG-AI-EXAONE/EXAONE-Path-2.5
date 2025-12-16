from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from .layers import SelfAttentionBlock, SwiGLUFFN, WSIFeatureEmbed, RopePositionEmbedding
from .model_utils import named_apply

ffn_layer_dict = {
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-5),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": partial(nn.RMSNorm, eps=1e-5),
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, nn.RMSNorm):
        module.reset_parameters()


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        # Always WSI feature input mode
        input_dim: int = 768,
        patch_size: int = 256,
        embed_use_norm: bool = True,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_dtype: str = "fp32",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: str = "layernorm",
        ffn_layer: str = "swiglu128",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        ffn_drop: float = 0.0,
        attn_drop: float = 0.0,
        n_storage_tokens: int = 0,
        nope_interval: int = 2,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.nope_interval = max(1, int(nope_interval)) # e.g., nope_interval=2 applies NOPE attention at block indices 0,2,4,...

        if input_dim is None:
            raise ValueError("VisionTransformer requires input_dim for WSI feature inputs.")
        self.patch_embed = WSIFeatureEmbed(
            input_dim=int(input_dim), embed_dim=embed_dim, use_norm=embed_use_norm
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            patch_size=patch_size,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                ffn_drop=ffn_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer_cls,
                ffn_layer=ffn_layer_cls,
                device=device,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything: CLS, registers, patch, and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        named_apply(init_weights_vit, self)
    
    def forward(
        self,
        x: Tensor,
        masks: Tensor,
        coords: Optional[Tensor] = None,
        contour_index: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # x: [B, N, C] (WSI feature tokens)
        B = x.size(0)
        # Concatenate CLS and storage tokens inside embedder
        storage_tokens = self.storage_tokens if self.n_storage_tokens > 0 else None
        x = self.patch_embed(x, cls_token=self.cls_token, storage_tokens=storage_tokens)

        # Attention key padding mask: True = valid
        attn_key_mask = masks
        if attn_key_mask.dim() > 2:
            raise ValueError("masks must be of shape [B, N] with True for valid tokens")
        if attn_key_mask.dtype != torch.bool:
            attn_key_mask = attn_key_mask.to(dtype=torch.bool)
        # prepend valid tokens for CLS/storage
        prefix = torch.ones(B, 1 + self.n_storage_tokens, dtype=torch.bool, device=x.device)
        attn_key_mask = torch.cat([prefix, attn_key_mask], dim=1)

        # Build pairwise contour-based attention mask when contour_index is provided.
        # NOTE (mask semantics): This project follows PyTorch SDPA docs where a boolean attn_mask
        # True indicates that the element should take part in attention (allowed).
        # Rules:
        # - CLS and storage tokens do NOT interact with patch tokens in contour-constrained ("rope attention") blocks.
        # - Patch tokens can attend only to patch tokens with the same contour_index.
        # - Key padding mask (invalid/padded tokens) are always excluded from attention.
        T = x.size(1)
        pre_len = 1 + self.n_storage_tokens  # CLS + storage tokens length
        attn_mask_pairwise: Optional[Tensor] = None
        if contour_index is not None:
            # Ensure contour_index shape is [B, N]
            if contour_index.dim() != 2:
                raise ValueError(f"contour_index must be of shape [B, N], received shape={tuple(contour_index.shape)}")
            contour_index = contour_index.to(device=x.device, dtype=torch.long)
            # Combine prefix (-1) with provided contour indices for patch tokens
            prefix_ci = torch.full((B, pre_len), fill_value=-1, device=x.device, dtype=torch.long)
            full_ci = torch.cat([prefix_ci, contour_index], dim=1)  # [B, T]

            # For patch-to-patch, allow only same contour_index (equal values)
            attn_mask_pairwise = full_ci.unsqueeze(2).eq(full_ci.unsqueeze(1))  # [B, T, T] # True means allowed

        # Key padding mask as broadcastable shape for SDPA: [B, 1, 1, S]
        # True = allowed (participate), False = excluded
        nope_mask_broadcast = attn_key_mask.view(B, 1, 1, -1)
        # Default to key-padding-only mask; if pairwise is available, intersect (AND) to keep only allowed entries
        attn_mask_broadcast = nope_mask_broadcast
        if attn_mask_pairwise is not None:
            pairwise_broadcast = attn_mask_pairwise.view(B, 1, T, T)
            # Combine pairwise (allowed) with key padding (allowed) using AND so only allowed stays True
            attn_mask_broadcast = pairwise_broadcast & nope_mask_broadcast  # broadcast over K dimension

        # RoPE from coords (patch tokens only). coords shape: [B, N, 2] or [N, 2]
        rope_sincos = self.rope_embed(coords=coords) if (self.rope_embed is not None and coords is not None) else None

        for idx, blk in enumerate(self.blocks):
            if (idx % self.nope_interval) == 0:
                # NOPE attention block: no RoPE, only key padding constraints
                x = blk(x, rope=None, attn_mask=nope_mask_broadcast)
            else:
                # ROPE attention block: RoPE + contour-aware pairwise mask (when provided) + key padding
                x = blk(x, rope=rope_sincos, attn_mask=attn_mask_broadcast)

        # Output packing
        x_norm = self.norm(x)
        x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
        x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]

        return {
            "x_norm_clstoken": x_norm_cls_reg[:, 0],
            "x_storage_tokens": x_norm_cls_reg[:, 1:],
            "x_norm_patchtokens": x_norm_patch,
            "x_prenorm": x,
            "masks": masks,
        }
