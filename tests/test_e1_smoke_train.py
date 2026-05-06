import sys
from pathlib import Path
from io import StringIO

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainers import e1_smoke_train
from trainers.e1_smoke_train import build_epoch_summary, compute_batch_pit_sisnr_loss, summarize_scalar_history


class DummyMiniLibriMixDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        self.items = [
            {
                "mix": torch.tensor([0.8, -0.2, 0.1, 0.0], dtype=torch.float32),
                "sources": torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                "length": 4,
                "sample_rate": 8000,
            }
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]


class DummySeparationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        return torch.stack((mix * self.scale, mix * (1.0 - self.scale)), dim=1)


class RecordingProgressBar:
    def __init__(self) -> None:
        self.train_updates: list[dict[str, object]] = []

    def get_metrics(self, trainer, model):
        metrics = super().get_metrics(trainer, model)
        self.train_updates.append(dict(metrics))
        return metrics


def test_compute_batch_pit_sisnr_loss_returns_finite_scalar() -> None:
    estimates = torch.tensor(
        [
            [
                [0.9, -0.1, 0.2, 0.0],
                [0.0, 0.8, -0.2, 0.1],
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    references = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    lengths = torch.tensor([4], dtype=torch.long)

    loss = compute_batch_pit_sisnr_loss(estimates, references, lengths)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert estimates.grad is not None


def test_summarize_scalar_history_reports_last_and_mean() -> None:
    summary = summarize_scalar_history([1.0, 2.0, 3.0], prefix="val")

    assert summary["val_last"] == 3.0
    assert summary["val_mean"] == 2.0


def test_build_epoch_summary_uses_mean_metrics() -> None:
    text = build_epoch_summary(
        {
            "epoch": 2,
            "train_loss_mean": -0.25,
            "val_loss_mean": -0.5,
            "val_si_snri_mean": 1.75,
        }
    )

    assert "Epoch 2:" in text
    assert "train_loss=-0.2500" in text
    assert "val_loss=-0.5000" in text


def test_build_lightning_trainer_uses_monitor_config(tmp_path: Path) -> None:
    config = {
        "runtime": {
            "device": "cpu",
            "precision": "32",
            "progress_bar": False,
        },
        "train": {
            "epochs": 3,
            "monitor": "val_loss",
            "monitor_mode": "min",
        },
        "experiment": {
            "name": "e1_lightning_test",
        },
        "output": {
            "root_dir": str(tmp_path / "Experiments"),
        },
    }

    trainer, checkpoint_cb, logger = e1_smoke_train.build_lightning_trainer(config)

    assert trainer.max_epochs == 3
    assert checkpoint_cb.monitor == "val_loss"
    assert checkpoint_cb.mode == "min"
    assert trainer.enable_progress_bar is False
    assert logger.save_dir == str(tmp_path / "Experiments")


def test_run_smoke_train_from_config_uses_lightning(tmp_path: Path, monkeypatch) -> None:
    def fake_collate(batch):
        item = batch[0]
        return {
            "mix": item["mix"].unsqueeze(0),
            "sources": item["sources"].unsqueeze(0),
            "lengths": torch.tensor([item["length"]], dtype=torch.long),
            "sample_rate": item["sample_rate"],
        }

    monkeypatch.setattr(e1_smoke_train, "MiniLibriMixDataset", DummyMiniLibriMixDataset)
    monkeypatch.setattr(e1_smoke_train, "collate_mini_librimix_batch", fake_collate)
    monkeypatch.setattr(e1_smoke_train, "load_trainable_teacher", lambda *args, **kwargs: DummySeparationModel())

    config = {
        "data": {
            "root": str(tmp_path),
            "index_root": str(tmp_path / "DataIndex"),
            "index_name": "mini_debug",
            "train_split": "train",
            "val_split": "val",
            "train_start_index": 0,
            "val_start_index": 0,
            "train_max_examples": 1,
            "val_max_examples": 1,
        },
        "model": {
            "pretrained_id": "dummy",
            "checkpoint_path": None,
        },
        "runtime": {
            "device": "cpu",
            "seed": 1,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "train_shuffle": False,
            "val_shuffle": False,
            "progress_bar": True,
            "precision": "16-mixed",
        },
        "train": {
            "lr": 1.0e-3,
            "epochs": 1,
            "monitor": "val_loss",
            "monitor_mode": "min",
            "optimizer": {
                "name": "adam",
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
            },
        },
        "experiment": {
            "name": "e1_progress_test",
        },
        "output": {
            "root_dir": str(tmp_path / "Experiments"),
        },
    }

    result = e1_smoke_train.run_smoke_train_from_config(config)

    assert result["epochs"] == 1
    assert result["steps_completed"] >= 1
    assert result["best_model_path"] is not None
    assert result["monitor"] == "val_loss"
    assert result["logged_metrics"]["val_loss"] is not None
    assert result["logged_metrics"]["val_si_snri"] is not None


def test_run_smoke_train_preserves_legacy_experiment_artifacts(tmp_path: Path, monkeypatch, capsys) -> None:
    def fake_collate(batch):
        item = batch[0]
        return {
            "mix": item["mix"].unsqueeze(0),
            "sources": item["sources"].unsqueeze(0),
            "lengths": torch.tensor([item["length"]], dtype=torch.long),
            "sample_rate": item["sample_rate"],
        }

    monkeypatch.setattr(e1_smoke_train, "MiniLibriMixDataset", DummyMiniLibriMixDataset)
    monkeypatch.setattr(e1_smoke_train, "collate_mini_librimix_batch", fake_collate)
    monkeypatch.setattr(e1_smoke_train, "load_trainable_teacher", lambda *args, **kwargs: DummySeparationModel())
    monkeypatch.setattr(e1_smoke_train, "make_run_timestamp", lambda: "20260506-180000")

    config = {
        "data": {
            "root": str(tmp_path),
            "index_root": str(tmp_path / "DataIndex"),
            "index_name": "mini_debug",
            "train_split": "train",
            "val_split": "val",
            "train_start_index": 0,
            "val_start_index": 0,
            "train_max_examples": 1,
            "val_max_examples": 1,
        },
        "model": {
            "pretrained_id": "dummy",
            "checkpoint_path": None,
        },
        "runtime": {
            "device": "cpu",
            "seed": 1,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "train_shuffle": False,
            "val_shuffle": False,
            "progress_bar": False,
            "precision": "32",
        },
        "train": {
            "lr": 1.0e-3,
            "epochs": 1,
            "monitor": "val_loss",
            "monitor_mode": "min",
            "optimizer": {
                "name": "adam",
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
            },
        },
        "experiment": {
            "name": "e1_legacy_artifacts",
        },
        "output": {
            "root_dir": str(tmp_path / "Experiments"),
        },
    }

    result = e1_smoke_train.run_smoke_train_from_config(config)
    out = capsys.readouterr().out
    assert "Epoch 0:" in out
    assert "train_loss=" in out
    assert "val_loss=" in out
    experiment_dir = Path(result["experiment_dir"])
    run_dir = experiment_dir / "results" / "20260506-180000"

    assert (experiment_dir / "conf.yml").exists()
    assert (experiment_dir / "final_metrics.json").exists()
    assert (experiment_dir / "history.jsonl").exists()
    assert (experiment_dir / "best_k_models.json").exists()
    assert (experiment_dir / "best_model.pth").exists()
    assert (experiment_dir / "checkpoints" / "best.ckpt").exists()
    assert (experiment_dir / "checkpoints" / "last.ckpt").exists()
    assert not (experiment_dir / "history.csv").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert not any(experiment_dir.glob("version_*"))

    history_text = (run_dir / "history.csv").read_text(encoding="utf-8")
    history_lines = history_text.splitlines()
    assert history_lines[0] == "epoch,train_loss,val_loss,val_si_snri"
    assert history_lines[1].split(",")[1] != ""


def test_run_smoke_train_prints_epoch_summary_and_updates_train_loss_in_progress_bar(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def fake_collate(batch):
        item = batch[0]
        return {
            "mix": item["mix"].unsqueeze(0),
            "sources": item["sources"].unsqueeze(0),
            "lengths": torch.tensor([item["length"]], dtype=torch.long),
            "sample_rate": item["sample_rate"],
        }

    monkeypatch.setattr(e1_smoke_train, "MiniLibriMixDataset", DummyMiniLibriMixDataset)
    monkeypatch.setattr(e1_smoke_train, "collate_mini_librimix_batch", fake_collate)
    monkeypatch.setattr(e1_smoke_train, "load_trainable_teacher", lambda *args, **kwargs: DummySeparationModel())

    config = {
        "data": {
            "root": str(tmp_path),
            "index_root": str(tmp_path / "DataIndex"),
            "index_name": "mini_debug",
            "train_split": "train",
            "val_split": "val",
            "train_start_index": 0,
            "val_start_index": 0,
            "train_max_examples": 1,
            "val_max_examples": 1,
        },
        "model": {
            "pretrained_id": "dummy",
            "checkpoint_path": None,
        },
        "runtime": {
            "device": "cpu",
            "seed": 1,
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "train_shuffle": False,
            "val_shuffle": False,
            "progress_bar": True,
            "precision": "32",
        },
        "train": {
            "lr": 1.0e-3,
            "epochs": 1,
            "monitor": "val_loss",
            "monitor_mode": "min",
            "optimizer": {
                "name": "adam",
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
            },
        },
        "experiment": {
            "name": "e1_epoch_summary_test",
        },
        "output": {
            "root_dir": str(tmp_path / "Experiments"),
        },
    }

    result = e1_smoke_train.run_smoke_train_from_config(config)

    captured = capsys.readouterr()
    assert "Epoch 0:" in captured.out
    assert "train_loss=" in captured.out
    assert "val_loss=" in captured.out
    assert result["logged_metrics"]["train_loss"] is not None
