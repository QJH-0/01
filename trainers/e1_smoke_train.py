from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import warnings
from typing import Iterator

import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm as LightningTqdm
from torch.utils.data import DataLoader
import yaml

from data import MiniLibriMixDataset, collate_mini_librimix_batch, librimix_train_json_dir, librimix_valid_json_dir
from models.convtasnet import ConvTasNet
from models.teacher import DEFAULT_MODEL_ID, get_model_args_from_config, load_pretrained_package_path, load_teacher_checkpoint
from trainers.eval_utils import best_pairwise_sisnr, si_snr
from utils.helper import apply_cli_overrides, get_by_path, load_yaml, require_by_path, resolve_device, set_seed
from tqdm import tqdm

from utils.logger import (
    append_epoch_history_csv,
    append_experiment_record,
    build_training_run_paths,
    make_run_timestamp,
    save_model_weights,
    save_training_checkpoint,
    write_json,
)


@contextmanager
def _suppress_lightning_rank_zero_info() -> Iterator[None]:
    """Hide Lightning's routine ``Trainer.fit stopped: ...`` INFO lines during ``fit``."""
    log = logging.getLogger("lightning_utilities.core.rank_zero")
    previous = log.level
    log.setLevel(max(previous, logging.WARNING))
    try:
        yield
    finally:
        log.setLevel(previous)


def compute_batch_pit_sisnr_loss(
    estimates: torch.Tensor, references: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    scores = []
    for idx in range(estimates.shape[0]):
        valid_len = int(lengths[idx].item())
        est_i = estimates[idx, :, :valid_len]
        ref_i = references[idx, :, :valid_len]
        scores.append(best_pairwise_sisnr(est_i, ref_i))
    return -torch.stack(scores).mean()


def compute_batch_sisnri(
    estimates: torch.Tensor, references: torch.Tensor, mixtures: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    improvements = []
    for idx in range(estimates.shape[0]):
        valid_len = int(lengths[idx].item())
        est_i = estimates[idx, :, :valid_len]
        ref_i = references[idx, :, :valid_len]
        mix_i = mixtures[idx, :valid_len]
        input_score = (si_snr(mix_i, ref_i[0]) + si_snr(mix_i, ref_i[1])) / 2
        output_score = best_pairwise_sisnr(est_i, ref_i)
        improvements.append(output_score - input_score)
    return torch.stack(improvements).mean()


def summarize_scalar_history(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_last": None, f"{prefix}_mean": None}
    return {
        f"{prefix}_last": values[-1],
        f"{prefix}_mean": sum(values) / len(values),
    }


def load_trainable_teacher(model_id: str = DEFAULT_MODEL_ID, device: str | torch.device = "cpu") -> torch.nn.Module:
    package_path = load_pretrained_package_path(model_id)
    model = ConvTasNet.from_pretrained_path(package_path)
    model.train()
    return model.to(device)


def _console_line(message: str) -> None:
    """Print on its own line without breaking tqdm (stdout for pytest capsys)."""
    tqdm.write(message, file=sys.stdout)


def build_epoch_summary(epoch_record: dict[str, float | int | None]) -> str:
    epoch = int(epoch_record["epoch"])
    return (
        f"Epoch {epoch}: train_loss={float(epoch_record['train_loss_mean']):.4f}, "
        f"val_loss={float(epoch_record['val_loss_mean']):.4f}"
    )


def _metric_to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_lightning_precision(precision: object, accelerator: str) -> str:
    precision_value = str(precision)
    if accelerator == "cpu" and precision_value in {"16-mixed", "bf16-mixed"}:
        return "32-true"
    return precision_value


def _resolve_accelerator(config: dict[str, object]) -> str:
    resolved_device = resolve_device(str(require_by_path(config, "runtime.device")))
    return "gpu" if resolved_device == "cuda" else "cpu"


def _resolve_trainer_devices(config: dict[str, object], accelerator: str) -> int | list[int]:
    """Lightning ``devices``; GPU 缺省 1，可设 -1 表示全部可见 GPU；CPU 恒为 1。"""
    if accelerator != "gpu":
        return 1
    raw = get_by_path(config, "runtime.devices", None)
    if raw is None:
        return 1
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return int(raw)


def _device_count_for_strategy(devices: int | list[int]) -> int:
    if isinstance(devices, list):
        return len(devices)
    return devices


def _resolve_trainer_strategy(
    config: dict[str, object], accelerator: str, devices: int | list[int]
) -> str | None:
    explicit = get_by_path(config, "runtime.strategy", None)
    if explicit is not None and str(explicit).strip() != "":
        return str(explicit).strip()
    if accelerator != "gpu":
        return None
    n = _device_count_for_strategy(devices)
    if n > 1 or n == -1:
        return "ddp"
    return None


class E1TQDMProgressBar(TQDMProgressBar):
    """Avoid Lightning's transient ``Training 0/?`` line before the first epoch begins."""

    def on_train_start(self, *args: object) -> None:
        self.train_progress_bar = LightningTqdm(disable=True)

    def on_train_epoch_start(self, trainer: Trainer, *args: object) -> None:
        if self.train_progress_bar.disable:
            self.train_progress_bar.close()
            self.train_progress_bar = self.init_train_tqdm()
        super().on_train_epoch_start(trainer, *args)


def build_lightning_trainer(config: dict[str, object]) -> tuple[Trainer, ModelCheckpoint, SimpleNamespace]:
    output_root = Path(str(require_by_path(config, "output.root_dir")))
    monitor = str(require_by_path(config, "train.monitor"))
    monitor_mode = str(require_by_path(config, "train.monitor_mode"))
    accelerator = _resolve_accelerator(config)
    devices = _resolve_trainer_devices(config, accelerator)
    strategy = _resolve_trainer_strategy(config, accelerator, devices)
    precision = _resolve_lightning_precision(get_by_path(config, "runtime.precision", "32"), accelerator)
    limit_train_batches = get_by_path(config, "train.max_train_steps_per_epoch", None)
    limit_val_batches = get_by_path(config, "train.max_val_steps", None)
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=1,
        save_last=True,
        filename=f"{{epoch}}-{{step}}-{{{monitor}:.4f}}",
        auto_insert_metric_name=False,
    )
    logger = SimpleNamespace(save_dir=str(output_root))
    progress_enabled = bool(get_by_path(config, "runtime.progress_bar", True))
    callback_list: list[Callback] = []
    if progress_enabled:
        callback_list.append(E1TQDMProgressBar(refresh_rate=1, leave=True))
    warnings.filterwarnings(
        "ignore",
        message=r"The '.*_dataloader' does not have many workers.*",
    )
    trainer_kwargs: dict[str, object] = dict(
        max_epochs=int(require_by_path(config, "train.epochs")),
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=False,
        callbacks=callback_list,
        enable_checkpointing=False,
        enable_progress_bar=progress_enabled,
        limit_train_batches=1.0 if limit_train_batches is None else int(limit_train_batches),
        limit_val_batches=1.0 if limit_val_batches is None else int(limit_val_batches),
        deterministic=True,
        log_every_n_steps=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    trainer = Trainer(**trainer_kwargs)
    trainer.enable_progress_bar = progress_enabled
    return trainer, checkpoint_cb, logger


class LegacyArtifactCallback(Callback):
    def __init__(self, config: dict[str, object], paths: dict[str, Path], run_id: str) -> None:
        super().__init__()
        self.config = config
        self.paths = paths
        self.run_id = run_id
        self.monitor = str(require_by_path(config, "train.monitor"))
        self.monitor_mode = str(require_by_path(config, "train.monitor_mode"))
        self.best_metric: float | None = None
        self.best_archived_checkpoint: Path | None = None
        self.epoch_records: list[dict[str, float | int | None]] = []

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self.paths["run_checkpoint_dir"].mkdir(parents=True, exist_ok=True)
        with self.paths["conf_yaml"].open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False, allow_unicode=True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return

        train_loss = pl_module.get_epoch_train_loss()
        val_loss = pl_module.get_epoch_val_loss()
        val_si_snri = pl_module.get_epoch_val_si_snri()
        # Match Lightning / tqdm: first epoch is 0 (trainer.current_epoch).
        epoch_record = {
            "epoch": int(trainer.current_epoch),
            "train_loss_last": train_loss,
            "train_loss_mean": train_loss,
            "val_loss_last": val_loss,
            "val_loss_mean": val_loss,
            "val_si_snri_last": val_si_snri,
            "val_si_snri_mean": val_si_snri,
        }
        history_row = {
            "epoch": int(trainer.current_epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_si_snri": val_si_snri,
        }
        self.epoch_records.append(epoch_record)
        append_epoch_history_csv(self.paths["run_history_csv"], history_row)
        if not bool(get_by_path(self.config, "runtime.progress_bar", True)):
            _console_line(build_epoch_summary(epoch_record))

        monitored_value = epoch_record.get(f"{self.monitor}_mean")
        monitored_float = float(monitored_value) if monitored_value is not None else float("nan")
        metrics_for_ckpt = {
            "epoch": epoch_record["epoch"],
            "train_loss_last": epoch_record["train_loss_last"],
            "train_loss_mean": epoch_record["train_loss_mean"],
            "val_loss_last": epoch_record["val_loss_last"],
            "val_loss_mean": epoch_record["val_loss_mean"],
            "val_si_snri_last": epoch_record["val_si_snri_last"],
            "val_si_snri_mean": epoch_record["val_si_snri_mean"],
        }
        archived_checkpoint = (
            self.paths["run_checkpoint_dir"]
            / f"epoch={int(epoch_record['epoch']):03d}-step={int(trainer.global_step):06d}-{self.monitor}={monitored_float:.4f}.ckpt"
        )
        optimizer = trainer.optimizers[0]
        save_training_checkpoint(
            checkpoint_path=archived_checkpoint,
            model=pl_module.model,
            optimizer=optimizer,
            epoch=int(epoch_record["epoch"]),
            global_step=int(trainer.global_step),
            metrics=metrics_for_ckpt,
            config=self.config,
        )
        save_training_checkpoint(
            checkpoint_path=self.paths["last_ckpt"],
            model=pl_module.model,
            optimizer=optimizer,
            epoch=int(epoch_record["epoch"]),
            global_step=int(trainer.global_step),
            metrics=metrics_for_ckpt,
            config=self.config,
        )

        is_better = False
        if monitored_value is not None:
            if self.best_metric is None:
                is_better = True
            elif self.monitor_mode == "min":
                is_better = float(monitored_value) < float(self.best_metric)
            else:
                is_better = float(monitored_value) > float(self.best_metric)
        if is_better:
            self.best_metric = float(monitored_value)
            self.best_archived_checkpoint = archived_checkpoint
            save_training_checkpoint(
                checkpoint_path=self.paths["best_ckpt"],
                model=pl_module.model,
                optimizer=optimizer,
                epoch=int(epoch_record["epoch"]),
                global_step=int(trainer.global_step),
                metrics=metrics_for_ckpt,
                config=self.config,
            )
            save_model_weights(self.paths["best_weights"], pl_module.model)
            write_json(self.paths["best_k_models_json"], {str(archived_checkpoint): self.best_metric})

    def build_final_metrics(self, trainer: Trainer, pl_module: LightningModule) -> dict[str, object]:
        callback_metrics = trainer.callback_metrics
        result = {
            "run_id": self.run_id,
            "experiment_name": str(require_by_path(self.config, "experiment.name")),
            "model_id": getattr(pl_module, "model_source", None),
            "data_root": _resolve_data_root_for_metrics(self.config, trainer),
            "train_dir": str(librimix_train_json_dir(self.config)),
            "valid_dir": str(librimix_valid_json_dir(self.config)),
            "train_split": librimix_train_json_dir(self.config).name,
            "val_split": librimix_valid_json_dir(self.config).name,
            "epochs": int(require_by_path(self.config, "train.epochs")),
            "steps_completed": int(trainer.global_step),
            "batch_size": int(require_by_path(self.config, "runtime.batch_size")),
            "progress_bar": bool(get_by_path(self.config, "runtime.progress_bar", True)),
            "precision": str(get_by_path(self.config, "runtime.precision", "32")),
            "lr": float(require_by_path(self.config, "train.lr")),
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "best_metric": self.best_metric,
            "epoch_records": self.epoch_records,
            "final_train_loss": _metric_to_float(callback_metrics.get("train_loss")),
            "final_val_loss": _metric_to_float(callback_metrics.get("val_loss")),
            "final_val_si_snri": _metric_to_float(callback_metrics.get("val_si_snri")),
            "run_checkpoint_dir": str(self.paths["run_checkpoint_dir"]),
            "run_results_dir": str(self.paths["run_results_dir"]),
            "run_dir": str(self.paths["run_dir"]),
            "experiment_dir": str(self.paths["experiment_root"]),
            "best_checkpoint_path": str(self.paths["best_ckpt"]),
            "last_checkpoint_path": str(self.paths["last_ckpt"]),
            "best_weights_path": str(self.paths["best_weights"]),
            "history_csv_path": str(self.paths["run_history_csv"]),
            "best_model_path": str(self.paths["best_weights"]),
            "best_model_score": self.best_metric,
            "logged_metrics": {
                "train_loss": _metric_to_float(callback_metrics.get("train_loss")),
                "val_loss": _metric_to_float(callback_metrics.get("val_loss")),
                "val_si_snri": _metric_to_float(callback_metrics.get("val_si_snri")),
            },
        }
        return result


def _resolve_data_root_for_metrics(config: dict[str, object], trainer: Trainer) -> str:
    configured = get_by_path(config, "data.root", None)
    if configured is not None and str(configured).strip() != "":
        return str(configured)
    try:
        dl = trainer.train_dataloader
        if dl is not None:
            ds = dl.dataset
            inferred = getattr(ds, "inferred_dataset_root", "") or ""
            if inferred:
                return str(inferred)
    except Exception:
        pass
    return ""


def _build_dataloader(
    config: dict[str, object],
    *,
    role: str,
    start_index_key: str,
    max_examples_key: str,
    shuffle_key: str,
    device_type: str,
) -> DataLoader:
    pin_memory = get_by_path(config, "runtime.pin_memory", None)
    speakers_cfg = get_by_path(config, "data.speakers", None)
    data_root = get_by_path(config, "data.root", None)
    if data_root is not None and str(data_root).strip() == "":
        data_root = None
    json_dir = librimix_train_json_dir(config) if role == "train" else librimix_valid_json_dir(config)
    dataset = MiniLibriMixDataset(
        json_dir=json_dir,
        dataset_root=data_root,
        max_examples=get_by_path(config, max_examples_key),
        start_index=int(get_by_path(config, start_index_key, 0)),
        speakers=tuple(str(x) for x in speakers_cfg) if speakers_cfg is not None else None,
    )
    return DataLoader(
        dataset,
        batch_size=int(require_by_path(config, "runtime.batch_size")),
        shuffle=bool(get_by_path(config, shuffle_key, False)),
        num_workers=int(require_by_path(config, "runtime.num_workers")),
        pin_memory=device_type == "gpu" if pin_memory is None else bool(pin_memory),
        collate_fn=collate_mini_librimix_batch,
    )


class E1LightningModule(LightningModule):
    def __init__(self, config: dict[str, object]) -> None:
        super().__init__()
        self.config = config
        self.model_source, self.model = self._build_model(config)
        self._train_epoch_losses: list[float] = []
        self._val_epoch_losses: list[float] = []
        self._val_epoch_sisnri: list[float] = []

    @staticmethod
    def _build_model(config: dict[str, object]) -> tuple[str, torch.nn.Module]:
        model_args = get_model_args_from_config(config)
        checkpoint_path = get_by_path(config, "model.checkpoint_path")
        if checkpoint_path:
            model = load_teacher_checkpoint(str(checkpoint_path), device="cpu", model_args=model_args)
            model.train()
            return str(checkpoint_path), model
        model_source = str(require_by_path(config, "model.pretrained_id"))
        return model_source, load_trainable_teacher(model_source, device="cpu")

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        return self.model(mix)

    def on_train_epoch_start(self) -> None:
        self._train_epoch_losses = []

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_losses = []
        self._val_epoch_sisnri = []

    def training_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        mix = batch["mix"]
        sources = batch["sources"]
        lengths = batch["lengths"]
        estimates = self(mix).float()
        loss = compute_batch_pit_sisnr_loss(estimates, sources, lengths)
        self._train_epoch_losses.append(float(loss.detach().cpu().item()))
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=mix.shape[0],
        )
        return loss

    def validation_step(self, batch: dict[str, object], batch_idx: int) -> None:
        mix = batch["mix"]
        sources = batch["sources"]
        lengths = batch["lengths"]
        estimates = self(mix).float()
        val_loss = compute_batch_pit_sisnr_loss(estimates, sources, lengths)
        val_si_snri = compute_batch_sisnri(estimates, sources, mix, lengths)
        self._val_epoch_losses.append(float(val_loss.detach().cpu().item()))
        self._val_epoch_sisnri.append(float(val_si_snri.detach().cpu().item()))
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=mix.shape[0])
        self.log("val_si_snri", val_si_snri, on_step=False, on_epoch=True, prog_bar=False, batch_size=mix.shape[0])

    def get_epoch_train_loss(self) -> float | None:
        if not self._train_epoch_losses:
            return None
        return sum(self._train_epoch_losses) / len(self._train_epoch_losses)

    def get_epoch_val_loss(self) -> float | None:
        if not self._val_epoch_losses:
            return None
        return sum(self._val_epoch_losses) / len(self._val_epoch_losses)

    def get_epoch_val_si_snri(self) -> float | None:
        if not self._val_epoch_sisnri:
            return None
        return sum(self._val_epoch_sisnri) / len(self._val_epoch_sisnri)

    def configure_optimizers(self):
        optimizer_name = str(get_by_path(self.config, "train.optimizer.name", "adam")).lower()
        optimizer_lr = float(require_by_path(self.config, "train.lr"))
        optimizer_weight_decay = float(get_by_path(self.config, "train.optimizer.weight_decay", 0.0))
        if optimizer_name != "adam":
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return torch.optim.Adam(
            self.model.parameters(),
            lr=optimizer_lr,
            weight_decay=optimizer_weight_decay,
            betas=tuple(get_by_path(self.config, "train.optimizer.betas", [0.9, 0.999])),
            eps=float(get_by_path(self.config, "train.optimizer.eps", 1.0e-8)),
        )


def run_smoke_train_from_config(config: dict[str, object], progress_factory=None) -> dict[str, object]:
    del progress_factory
    seed = int(require_by_path(config, "runtime.seed"))
    set_seed(seed)
    seed_everything(seed, workers=True)

    run_id = make_run_timestamp()
    paths = build_training_run_paths(
        experiment_name=str(require_by_path(config, "experiment.name")),
        output_root=Path(str(require_by_path(config, "output.root_dir"))),
        run_id=run_id,
    )
    trainer, checkpoint_cb, logger = build_lightning_trainer(config)
    legacy_callback = LegacyArtifactCallback(config, paths, run_id)
    trainer.callbacks.append(legacy_callback)
    accelerator = _resolve_accelerator(config)
    train_loader = _build_dataloader(
        config,
        role="train",
        start_index_key="data.train_start_index",
        max_examples_key="data.train_max_examples",
        shuffle_key="runtime.train_shuffle",
        device_type=accelerator,
    )
    val_loader = _build_dataloader(
        config,
        role="valid",
        start_index_key="data.val_start_index",
        max_examples_key="data.val_max_examples",
        shuffle_key="runtime.val_shuffle",
        device_type=accelerator,
    )
    module = E1LightningModule(config)
    with _suppress_lightning_rank_zero_info():
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.is_global_zero:
        result = legacy_callback.build_final_metrics(trainer, module)
        write_json(paths["summary_json"], result)
        write_json(paths["run_summary_json"], result)
        record = {
            "run_id": run_id,
            "experiment_name": str(require_by_path(config, "experiment.name")),
            "train_split": librimix_train_json_dir(config).name,
            "val_split": librimix_valid_json_dir(config).name,
            "steps_completed": result["steps_completed"],
            "final_train_loss": result["final_train_loss"],
            "final_val_loss": result["final_val_loss"],
            "final_val_si_snri": result["final_val_si_snri"],
            "best_metric": result["best_metric"],
            "history_csv_path": str(paths["run_history_csv"]),
            "summary_json_path": str(paths["summary_json"]),
            "run_summary_json_path": str(paths["run_summary_json"]),
            "best_checkpoint_path": str(paths["best_ckpt"]),
            "last_checkpoint_path": str(paths["last_ckpt"]),
            "best_weights_path": str(paths["best_weights"]),
            "experiment_dir": str(paths["experiment_root"]),
        }
        append_experiment_record(paths["manifest"], record)
        result.update(record)
        result["monitor"] = checkpoint_cb.monitor
        result["best_model_path"] = str(paths["best_weights"])
        result["experiment_dir"] = str(paths["experiment_root"])
        result["run_id"] = run_id
        result["is_global_zero"] = True
        return result

    result = {
        "run_id": run_id,
        "experiment_name": str(require_by_path(config, "experiment.name")),
        "is_global_zero": False,
        "logged_metrics": {
            "train_loss": _metric_to_float(trainer.callback_metrics.get("train_loss")),
            "val_loss": _metric_to_float(trainer.callback_metrics.get("val_loss")),
            "val_si_snri": _metric_to_float(trainer.callback_metrics.get("val_si_snri")),
        },
        "best_model_path": "",
        "experiment_dir": str(paths["experiment_root"]),
        "monitor": checkpoint_cb.monitor,
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run a short smoke-training loop on the pretrained teacher.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_known_args()


def main() -> None:
    args, overrides = parse_args()
    config = load_yaml(args.config)
    config = apply_cli_overrides(config, overrides)
    result = run_smoke_train_from_config(config)
    if not result.get("is_global_zero", True):
        return
    tl = result["logged_metrics"]["train_loss"]
    vl = result["logged_metrics"]["val_loss"]
    tl_s = f"{tl:.4f}" if tl is not None else "n/a"
    vl_s = f"{vl:.4f}" if vl is not None else "n/a"
    print(f"Done: train_loss={tl_s}, val_loss={vl_s}, best_model_path={result['best_model_path']}")


if __name__ == "__main__":
    main()
