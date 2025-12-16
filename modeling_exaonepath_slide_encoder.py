from __future__ import annotations

"""Remote-code modeling file for EXAONE-Path Slide/WSI encoder.

This file is imported by Transformers when using `trust_remote_code=True`.

Important:
- This file acts as a *thin AutoModel entrypoint*.
- The actual implementation lives in `exaonepath.models.slide_encoder_hf`.
- At runtime, the repository snapshot is downloaded via `snapshot_download`
    and added to `sys.path` so that `exaonepath/` can be imported.
- Do NOT import sibling modules like `configuration_exaonepath_slide_encoder` here.
    Transformers' remote-code dependency checker treats those imports as missing
    third-party packages (e.g. it suggests `pip install configuration_exaonepath_slide_encoder`).
"""

from typing import Any, Dict, Optional
import importlib
import sys

from huggingface_hub import snapshot_download
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel


class ExaonePathSlideEncoderConfig(PretrainedConfig):
    """Self-contained Transformers config for EXAONE-Path Slide/WSI encoder.

    Keep it here (in the modeling file) so we don't need a separate
    `configuration_exaonepath_slide_encoder.py` on the Hub.
    """

    model_type = "exaonepath_slide_encoder"

    def __init__(self, wsi_cfg: Dict[str, Any] | None = None, **kwargs: Any):
        self.wsi_cfg = dict(wsi_cfg or {})
        super().__init__(**kwargs)


class ExaonePathSlideEncoderModel(PreTrainedModel):
    config_class = ExaonePathSlideEncoderConfig
    base_model_prefix = "slide_encoder"

    def __init__(self, config: ExaonePathSlideEncoderConfig):
        super().__init__(config)

        # Ensure the repo code (including `exaonepath/`) is available at runtime.
        # NOTE: config._name_or_path is usually the repo id when loaded from Hub.
        repo_id = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
        if isinstance(repo_id, str) and repo_id:
            local_root = snapshot_download(repo_id)
            if local_root not in sys.path:
                sys.path.insert(0, local_root)

        WSIEncoder = getattr(importlib.import_module("exaonepath.models.slide_encoder_hf"), "WSIEncoder")
        self.slide_encoder: nn.Module = WSIEncoder.from_wsi_config(wsi_cfg=config.wsi_cfg)

        self.post_init()

    def forward(
        self,
        patch_features: Tensor,
        patch_mask: Tensor,
        patch_coords: Optional[Tensor] = None,
        patch_contour_index: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        """Return patch- and slide-level embeddings.

        Returns a dict with exactly two keys:
        - "patch_embedding": [B, N, C_in + D]
        - "slide_embedding": [B, C_in + D]

        Note: We intentionally return a plain dict (instead of a ModelOutput)
        to make the remote-code API explicit and easy to use.
        """

        out: Dict[str, Tensor] = self.slide_encoder(
            patch_features=patch_features,
            patch_mask=patch_mask,
            patch_coords=patch_coords,
            patch_contour_index=patch_contour_index,
        )
        return out


__all__ = ["ExaonePathSlideEncoderModel"]
