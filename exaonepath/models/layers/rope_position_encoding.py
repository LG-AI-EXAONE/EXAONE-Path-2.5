import math

import torch
from torch import Tensor, nn


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    # NOTE: Modified to index by patch_size instead of dataset-wise max H/W.
    #       - During __init__, provide patch_size and store as member.
    #       - In forward(), pass coords tensor with shape [B, N, 2];
    #         coords are converted to patch indices using (coords + patch_size//2) / patch_size.
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        patch_size: int = 256,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        # Store patch size for converting pixel coords to patch indices
        self.patch_size = int(patch_size)

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Compute RoPE values for given coordinates.

    Args:
        coords: Tensor of shape [B, N, 2] representing (h, w) pixel coordinates.
            Converted to patch indices using (coord + patch_size//2) / patch_size.
        Returns:
            Tuple (sin, cos):
              - Outputs are [B, 1, N, D_head] to broadcast across heads.
        """
        device = self.periods.device
        dtype = self.dtype

        if coords.device != device:
            coords = coords.to(device)
        if dtype is not None and coords.dtype != dtype:
            coords = coords.to(dtype)
        # Enforce batched coords for consistent behavior with attention and sampling
        assert coords.ndim == 3 and coords.shape[-1] == 2, f"coords must be [B, N, 2], got shape {tuple(coords.shape)}"

        # Convert pixel coordinates to patch indices centered at patch centers
        # index = (coord + patch_size//2) / patch_size
        patch_size_tensor = torch.tensor(self.patch_size, device=device, dtype=dtype)  # for broadcasting
        center_offset = torch.tensor(self.patch_size // 2, device=device, dtype=dtype)
        coords_norm = (coords + center_offset) / patch_size_tensor

        # Prepare angles and sin/cos for [B, N, 2]
        angles = 2 * math.pi * coords_norm[:, :, :, None] / self.periods[None, None, None, :]  # [B, N, 2, D//4]
        angles = angles.flatten(2, 3)  # [B, N, D//2]
        angles = angles.tile((1, 1, 2))  # [B, N, D]
        cos = torch.cos(angles)  # [B, N, D]
        sin = torch.sin(angles)  # [B, N, D]
        # Expand head dimension to broadcast across heads: [B, 1, N, D]
        return (sin.unsqueeze(1), cos.unsqueeze(1))

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods