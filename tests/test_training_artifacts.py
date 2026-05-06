import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import append_epoch_history_csv, save_training_checkpoint


def test_append_epoch_history_csv_creates_header_and_rows(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"

    append_epoch_history_csv(history_path, {"epoch": 1, "train_loss": -1.0, "val_loss": -0.5})
    append_epoch_history_csv(history_path, {"epoch": 2, "train_loss": -1.1, "val_loss": -0.6})

    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    assert rows[1]["val_loss"] == "-0.6"


def test_save_training_checkpoint_stores_resume_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "last.ckpt"

    save_training_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=3,
        metrics={"train_loss": -1.0, "val_loss": -0.5},
        config={"experiment_name": "demo"},
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 1
    assert payload["global_step"] == 3
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert payload["metrics"]["val_loss"] == -0.5
