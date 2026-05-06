from __future__ import annotations

from utils.precision import create_autocast_context, create_grad_scaler, resolve_amp_mode
from utils.progress import create_progress


class BaseTrainer:
    @staticmethod
    def create_progress(*, total: int | None, desc: str, enabled: bool = True, progress_factory=None):
        return create_progress(total=total, desc=desc, enabled=enabled, factory=progress_factory)

    @staticmethod
    def resolve_amp_mode(*, precision: str | int | None, device_type: str):
        return resolve_amp_mode(precision, device_type)

    @staticmethod
    def create_autocast_context(*, mode, device_type: str):
        return create_autocast_context(mode, device_type)

    @staticmethod
    def create_grad_scaler(*, mode):
        return create_grad_scaler(mode)
