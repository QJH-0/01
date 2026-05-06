from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import torch


DEFAULT_ROUND_DIGITS = 6


def make_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def build_eval_run_paths(output_root: Path | str, experiment_name: str, run_id: str) -> dict[str, Path]:
    experiment_root = Path(output_root) / experiment_name
    results_dir = experiment_root / "results"
    experiment_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "experiment_root": experiment_root,
        "run_dir": experiment_root,
        "metrics": experiment_root / "final_metrics.json",
        "manifest": experiment_root / "history.jsonl",
        "results_dir": results_dir,
        "run_metrics": results_dir / f"{run_id}_metrics.json",
    }


def append_experiment_record(manifest_path: Path | str, record: dict) -> None:
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(round_nested(record), ensure_ascii=False) + "\n")


def round_nested(value, digits: int = DEFAULT_ROUND_DIGITS):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, Mapping):
        return {key: round_nested(inner, digits) for key, inner in value.items()}
    if isinstance(value, tuple):
        return tuple(round_nested(item, digits) for item in value)
    if isinstance(value, list):
        return [round_nested(item, digits) for item in value]
    return value


def write_json(path: Path | str, payload: dict | list, digits: int = DEFAULT_ROUND_DIGITS) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(round_nested(payload, digits), indent=2, ensure_ascii=False), encoding="utf-8")


def append_epoch_history_csv(history_path: Path | str, row: dict) -> None:
    history = Path(history_path)
    history.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history.exists()
    rounded_row = round_nested(row)
    with history.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rounded_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(rounded_row)


def save_training_checkpoint(
    checkpoint_path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    metrics: dict,
    config: dict,
) -> None:
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, checkpoint)


def save_model_weights(weights_path: Path | str, model: torch.nn.Module) -> None:
    path = Path(weights_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def build_training_run_paths(
    experiment_name: str,
    output_root: Path | str,
    run_id: str,
) -> dict[str, Path]:
    experiment_root = Path(output_root) / experiment_name
    checkpoint_root = experiment_root / "checkpoints"
    results_dir = experiment_root / "results"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "experiment_root": experiment_root,
        "run_dir": experiment_root,
        "checkpoint_root": checkpoint_root,
        "run_checkpoint_dir": results_dir / run_id / "checkpoints",
        "run_results_dir": results_dir / run_id,
        "history_csv": experiment_root / "history.csv",
        "summary_json": experiment_root / "final_metrics.json",
        "manifest": experiment_root / "history.jsonl",
        "last_ckpt": checkpoint_root / "last.ckpt",
        "best_ckpt": checkpoint_root / "best.ckpt",
        "best_weights": experiment_root / "best_model.pth",
        "latest_summary_json": experiment_root / "final_metrics.json",
        "best_k_models_json": experiment_root / "best_k_models.json",
        "conf_yaml": experiment_root / "conf.yml",
        "results_dir": results_dir,
        "run_summary_json": results_dir / run_id / "summary.json",
        "run_history_csv": results_dir / run_id / "history.csv",
    }
