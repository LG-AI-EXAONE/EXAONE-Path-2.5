from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from torch import Tensor, nn
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import DictConfig, OmegaConf

from .slide_transformer import VisionTransformer
from .slide_encoder_head import WSIEncoderHead


def _build_wsi_encoder(wsi_cfg: DictConfig) -> Tuple[nn.Module, int]:
    """Construct a WSIFeatureEncoder composed of a VisionTransformer.

    This is a minimal, WSI-only factory equivalent to
    ``MultiModalMetaModel._build_wsi_encoder`` but without importing the
    full multimodal meta model.
    """

    embed_dim = int(wsi_cfg.get("embed_dim", 768))
    input_dim = int(wsi_cfg.get("input_dim", 768))

    transformer_kwargs = {
        "input_dim": input_dim,
        "patch_size": int(wsi_cfg.get("patch_size", 256)),
        "embed_use_norm": bool(wsi_cfg.get("embed_use_norm", True)),
        "embed_dim": embed_dim,
        "depth": int(wsi_cfg.get("depth", 12)),
        "num_heads": int(wsi_cfg.get("num_heads", 12)),
        "ffn_ratio": float(wsi_cfg.get("ffn_ratio", 4.0)),
        "qkv_bias": bool(wsi_cfg.get("qkv_bias", True)),
        "norm_layer": wsi_cfg.get("norm_layer", "layernorm"),
        "ffn_layer": wsi_cfg.get("ffn_layer", "swiglu128"),
        "ffn_bias": bool(wsi_cfg.get("ffn_bias", True)),
        "proj_bias": bool(wsi_cfg.get("proj_bias", True)),
        "ffn_drop": float(wsi_cfg.get("ffn_drop", 0.0)),
        "attn_drop": float(wsi_cfg.get("attn_drop", 0.0)),
        "n_storage_tokens": int(wsi_cfg.get("n_storage_tokens", 0)),
        "nope_interval": int(wsi_cfg.get("nope_interval", 2)),
        # Rope / coords related
        "pos_embed_rope_base": wsi_cfg.get("pos_embed_rope_base", 10000.0),
        "pos_embed_rope_min_period": wsi_cfg.get("pos_embed_rope_min_period"),
        "pos_embed_rope_max_period": wsi_cfg.get("pos_embed_rope_max_period"),
        "pos_embed_rope_dtype": wsi_cfg.get("pos_embed_rope_dtype", "fp32"),
    }

    # Build transformer with internal WSI patch embedding
    transformer = VisionTransformer(**transformer_kwargs)
    if hasattr(transformer, "init_weights"):
        transformer.init_weights()

    wsi_encoder = WSIEncoderHead(
        transformer,
        input_dim,
        embed_dim,
    )

    return wsi_encoder


class WSIEncoder(nn.Module, PyTorchModelHubMixin):
    """WSI slide-level encoder wrapper with Hugging Face Hub support.

    This wraps the internal :class:`WSIFeatureEncoder` (ViT + aggregation)
    used in EXAONE-Path for slide-level feature extraction, and exposes it
    as a Hub-compatible model via :class:`PyTorchModelHubMixin`.

    The minimal configuration (``wsi_cfg``)
    is stored on the instance so that it can be serialized to ``config.json``
    when calling :meth:`save_pretrained`.
    """

    def __init__(
        self,
        *,
        wsi_cfg: Dict[str, Any],
    ) -> None:
        super().__init__()

        # Store config in plain-dict form for easy JSON/YAML serialization
        self.wsi_cfg: Dict[str, Any] = dict(wsi_cfg)

        # Build encoder on CPU
        cfg_obj: DictConfig = OmegaConf.create(self.wsi_cfg)
        if isinstance(cfg_obj, DictConfig):
            OmegaConf.resolve(cfg_obj)

        wsi_encoder = _build_wsi_encoder(cfg_obj)
        self.wsi_encoder = wsi_encoder

    def forward(
        self,
        patch_features: Tensor,
        patch_mask: Tensor,
        patch_coords: Optional[Tensor] = None,
        patch_contour_index: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward to underlying WSIFeatureEncoder.

        Args:
            patch_features: [B, N, C]
            patch_mask: [B, N] with 1 for valid tokens
            patch_coords: optional [B, N, 2] coords (for RoPE)
            patch_contour_index: optional [B, N] contour indices
        """

        return self.wsi_encoder(
            patch_features=patch_features,
            patch_mask=patch_mask,
            patch_coords=patch_coords,
            patch_contour_index=patch_contour_index,
        )

    # Optional: expose a small helper to reconstruct from a minimal config dict
    @classmethod
    def from_wsi_config(
        cls,
        wsi_cfg: Dict[str, Any],
    ) -> WSIEncoder:
        return cls(wsi_cfg=wsi_cfg)