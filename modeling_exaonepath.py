from __future__ import annotations

"""Root-level config entrypoint for EXAONE-Path Hub loading.

This file defines a unified repo config that can be loaded from the model repo
root. The returned config can then generate the patch/slide sub-configs that
are passed into `AutoModel.from_pretrained(..., config=..., subfolder=...)`.
It also exposes a root `AutoModel` entrypoint that dispatches to either the
patch or slide encoder while still making the root `config.json` part of the
actual model-loading path.
"""

from typing import Any, Dict

from transformers import AutoModel, PreTrainedModel
from transformers import PretrainedConfig


PATCH_AUTO_MAP = {
    "AutoConfig": "modeling_exaonepath_patch_encoder.ExaonePathPatchEncoderConfig",
    "AutoModel": "modeling_exaonepath_patch_encoder.ExaonePathPatchEncoderModel",
}

SLIDE_AUTO_MAP = {
    "AutoConfig": "modeling_exaonepath_slide_encoder.ExaonePathSlideEncoderConfig",
    "AutoModel": "modeling_exaonepath_slide_encoder.ExaonePathSlideEncoderModel",
}


class ExaonePathPatchEncoderConfig(PretrainedConfig):
    model_type = "exaonepath_patch_encoder"

    def __init__(
        self,
        image_encoder: str = "vitb",
        patch_size: int = 14,
        img_size=(224, 224),
        extra_kwargs: Dict[str, Any] | None = None,
        auto_map: Dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self.image_encoder = str(image_encoder)
        self.patch_size = int(patch_size)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_size = [int(img_size[0]), int(img_size[1])]
        self.extra_kwargs = dict(extra_kwargs or {})
        self.auto_map = dict(auto_map or PATCH_AUTO_MAP)
        super().__init__(**kwargs)


class ExaonePathSlideEncoderConfig(PretrainedConfig):
    model_type = "exaonepath_slide_encoder"

    def __init__(
        self,
        wsi_cfg: Dict[str, Any] | None = None,
        auto_map: Dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self.wsi_cfg = dict(wsi_cfg or {})
        self.auto_map = dict(auto_map or SLIDE_AUTO_MAP)
        super().__init__(**kwargs)


class ExaonePathConfig(PretrainedConfig):
    model_type = "exaonepath"

    def __init__(
        self,
        patch_encoder_config: Dict[str, Any] | None = None,
        slide_encoder_config: Dict[str, Any] | None = None,
        patch_encoder_subfolder: str = "patch-encoder",
        slide_encoder_subfolder: str = "slide-encoder",
        **kwargs: Any,
    ):
        self.patch_encoder_config = dict(patch_encoder_config or {})
        self.slide_encoder_config = dict(slide_encoder_config or {})
        self.patch_encoder_subfolder = str(patch_encoder_subfolder)
        self.slide_encoder_subfolder = str(slide_encoder_subfolder)
        super().__init__(**kwargs)

    def get_patch_config(self) -> ExaonePathPatchEncoderConfig:
        if not self.patch_encoder_config:
            raise ValueError("Root config does not contain `patch_encoder_config`.")
        patch_cfg = dict(self.patch_encoder_config)
        patch_cfg.setdefault("auto_map", dict(PATCH_AUTO_MAP))
        return ExaonePathPatchEncoderConfig(**patch_cfg)

    def get_slide_config(self) -> ExaonePathSlideEncoderConfig:
        if not self.slide_encoder_config:
            raise ValueError("Root config does not contain `slide_encoder_config`.")
        slide_cfg = dict(self.slide_encoder_config)
        slide_cfg.setdefault("auto_map", dict(SLIDE_AUTO_MAP))
        return ExaonePathSlideEncoderConfig(**slide_cfg)


class ExaonePathModel(PreTrainedModel):
    config_class = ExaonePathConfig
    base_model_prefix = "exaonepath"

    def __init__(self, config: ExaonePathConfig):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any):
        config = kwargs.pop("config", None)
        if config is None:
            raise ValueError("Missing root config. Load with AutoModel.from_pretrained(..., trust_remote_code=True).")
        if not isinstance(config, ExaonePathConfig):
            raise TypeError(f"Expected ExaonePathConfig, got {type(config).__name__}.")

        component = kwargs.pop("component", None) or kwargs.pop("encoder_type", None)
        component_subfolder = kwargs.pop("component_subfolder", None)
        if component is None:
            raise ValueError("Specify `component=\"patch\"` or `component=\"slide\"` when loading from the repo root.")

        component = str(component).strip().lower()
        if component in {"patch", "patch-encoder", "patch_encoder"}:
            subfolder = component_subfolder or config.patch_encoder_subfolder
            subconfig = config.get_patch_config()
        elif component in {"slide", "slide-encoder", "slide_encoder", "wsi"}:
            subfolder = component_subfolder or config.slide_encoder_subfolder
            subconfig = config.get_slide_config()
        else:
            raise ValueError(f"Unsupported component: {component!r}. Use `patch` or `slide`.")

        return AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            subfolder=subfolder,
            config=subconfig,
            **kwargs,
        )


__all__ = [
    "ExaonePathConfig",
    "ExaonePathModel",
    "ExaonePathPatchEncoderConfig",
    "ExaonePathSlideEncoderConfig",
]
