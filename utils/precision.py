from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AmpMode:
    requested_precision: str
    use_mixed_precision: bool
    autocast_dtype: torch.dtype | None
    scaler_enabled: bool


class NullGradScaler:
    def __init__(self) -> None:
        self._enabled = False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def is_enabled(self) -> bool:
        return self._enabled


def resolve_amp_mode(precision: str | int | None, device_type: str) -> AmpMode:
    normalized = str(precision if precision is not None else "32").lower()
    if normalized in {"32", "32-true", "fp32"}:
        return AmpMode(
            requested_precision=normalized,
            use_mixed_precision=False,
            autocast_dtype=None,
            scaler_enabled=False,
        )
    if normalized == "16-mixed":
        enabled = device_type == "cuda"
        return AmpMode(
            requested_precision=normalized,
            use_mixed_precision=enabled,
            autocast_dtype=torch.float16 if enabled else None,
            scaler_enabled=enabled,
        )
    raise ValueError(f"Unsupported precision: {precision}")


def create_autocast_context(mode: AmpMode, device_type: str):
    if not mode.use_mixed_precision or mode.autocast_dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type=device_type, dtype=mode.autocast_dtype)


def create_grad_scaler(mode: AmpMode):
    if not mode.scaler_enabled:
        return NullGradScaler()
    return torch.amp.GradScaler("cuda", enabled=True)
