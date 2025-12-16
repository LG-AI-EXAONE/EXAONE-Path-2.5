from __future__ import annotations

"""Remote-code modeling file for EXAONE-Path Patch Encoder.

Unified with slide-encoder style:
- Keep this file small.
- At runtime, download the repo snapshot and import the actual model code from
  `exaonepath/` (so we don't duplicate model definitions here).

This requires the Hub repo to include `exaonepath/`.
"""

from typing import Any, Dict, Optional
import importlib
import sys

from huggingface_hub import snapshot_download
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel


class ExaonePathPatchEncoderConfig(PretrainedConfig):
    model_type = "exaonepath_patch_encoder"

    def __init__(
        self,
        image_encoder: str = "vitb",
        patch_size: int = 14,
        img_size=(224, 224),
        extra_kwargs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.image_encoder = str(image_encoder)
        self.patch_size = int(patch_size)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = [int(img_size[0]), int(img_size[1])]
        self.extra_kwargs = dict(extra_kwargs or {})
        super().__init__(**kwargs)


class ExaonePathPatchEncoderModel(PreTrainedModel):
    config_class = ExaonePathPatchEncoderConfig
    base_model_prefix = "patch_encoder"

    def __init__(self, config: ExaonePathPatchEncoderConfig):
        super().__init__(config)

        # Ensure the repo code (including `exaonepath/`) is available at runtime.
        repo_id = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
        if isinstance(repo_id, str) and repo_id:
            local_root = snapshot_download(repo_id)
            if local_root not in sys.path:
                sys.path.insert(0, local_root)

        PatchEncoder = getattr(
            importlib.import_module("exaonepath.models.patch_encoder_hf"),
            "PatchEncoder",
        )

        extra = getattr(config, "extra_kwargs", None) or {}
        self.patch_encoder: nn.Module = PatchEncoder(
            image_encoder=config.image_encoder,
            patch_size=int(config.patch_size),
            img_size=list(config.img_size),
            **extra,
        )

        self.post_init()

    def forward(
        self,
        x: Optional[Tensor] = None,
        *,
        pixel_values: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Return patch embedding as a tensor.

        Returns:
            patch_embedding: [B, C]

        Note:
            The patch encoder produces a single embedding per input patch image.
            We return the tensor directly for the simplest user-facing API.
        """

        # Prefer the simple positional argument `x`, but also accept the
        # Hugging Face convention `pixel_values=` for compatibility.
        if x is None:
            x = pixel_values
        if x is None:
            raise ValueError("Missing input tensor. Provide `x` (positional) or `pixel_values=`.")

        return self.patch_encoder(x)


__all__ = ["ExaonePathPatchEncoderConfig", "ExaonePathPatchEncoderModel"]
