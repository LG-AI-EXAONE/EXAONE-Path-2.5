from __future__ import annotations

from typing import Any, Sequence, Union

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .patch_transformer import vit_base


class PatchEncoder(nn.Module, PyTorchModelHubMixin):
    """EXAONE-Path image patch encoder with Hugging Face Hub support.

    This class wraps the ViT backbone used for patch-level feature extraction and
    integrates with the Hub via :class:`huggingface_hub.PyTorchModelHubMixin`.

    The configuration (``image_encoder``, ``patch_size``, ``img_size``, and any
    extra keyword arguments) is automatically serialized to ``config.json`` when
    calling ``save_pretrained``.
    """

    def __init__(
        self,
        image_encoder: str = "vitb",
        patch_size: int = 14,
        img_size: Union[int, Sequence[int]] = 224,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if isinstance(img_size, int):
            img_size = [img_size, img_size]

        self.image_encoder = image_encoder
        self.patch_size = int(patch_size)
        self.img_size = [int(img_size[0]), int(img_size[1])]
        self.extra_kwargs = dict(kwargs)

        if image_encoder == "vitb":
            model_kwargs = dict(self.extra_kwargs)
            model_kwargs["img_size"] = self.img_size  # VisionTransformer expects [H, W]
            self.backbone = vit_base(patch_size=self.patch_size, **model_kwargs)
        else:
            raise ValueError(f"Unsupported image_encoder for PatchEncoder: {image_encoder}")

    def forward(self, x):
        return self.backbone(x)
