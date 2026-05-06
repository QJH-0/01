from __future__ import annotations

from trainers.base_trainer import BaseTrainer


def train(*args, progress_enabled: bool = True, progress_factory=None, **kwargs):
    progress = BaseTrainer.create_progress(
        total=None,
        desc="E2 Train Setup",
        enabled=progress_enabled,
        progress_factory=progress_factory,
    )
    progress.set_postfix(status="pending")
    progress.close()
    raise NotImplementedError("E2 trainer is not implemented yet.")
