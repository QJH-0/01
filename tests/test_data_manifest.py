import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.libri_mix import (
    MiniLibriMixDataset,
    build_mini_librimix_index,
    ensure_json_index,
    librimix_train_json_dir,
    librimix_valid_json_dir,
)


def _write_wav(path: Path, value: float, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(16, value, dtype=np.float32), sample_rate)


def test_librimix_json_dir_helpers_match_tiger_config_keys(tmp_path: Path) -> None:
    cfg = {
        "data": {
            "train_dir": str(tmp_path / "train_fold"),
            "valid_dir": str(tmp_path / "val_fold"),
        }
    }
    assert librimix_train_json_dir(cfg) == (tmp_path / "train_fold").resolve()
    assert librimix_valid_json_dir(cfg) == (tmp_path / "val_fold").resolve()


def test_build_mini_librimix_index_creates_local_json_index(tmp_path: Path) -> None:
    root = tmp_path / "MiniLibriMix"
    split_dir = root / "test"
    _write_wav(split_dir / "mix_both" / "a.wav", 0.1)
    _write_wav(split_dir / "s1" / "a.wav", 1.1)
    _write_wav(split_dir / "s2" / "a.wav", 2.1)
    index_dir = tmp_path / "DataIndex" / "default"

    build_mini_librimix_index(root, index_dir, splits=("test",))

    mix_json = index_dir / "test" / "mix_both.json"
    assert mix_json.is_file()
    rows = json.loads(mix_json.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert Path(rows[0][0]).stem == "a"
    assert rows[0][1] == 16

    ds = MiniLibriMixDataset(json_dir=index_dir / "test")
    assert len(ds) == 1
    assert ds[0]["id"] == "a"


def test_dataset_can_limit_examples_through_json_index(tmp_path: Path) -> None:
    root = tmp_path / "MiniLibriMix"
    split_dir = root / "train"
    for idx in range(3):
        name = f"{idx}.wav"
        _write_wav(split_dir / "mix_both" / name, 0.1 + idx)
        _write_wav(split_dir / "s1" / name, 1.1 + idx)
        _write_wav(split_dir / "s2" / name, 2.1 + idx)
    index_root = tmp_path / "DataIndex"
    ensure_json_index(root, index_root, "default", "train")

    dataset = MiniLibriMixDataset(root=root, split="train", index_root=index_root, index_name="default", max_examples=2)

    assert len(dataset) == 2


def test_build_mini_librimix_index_explicit_mix_clean_speakers(tmp_path: Path) -> None:
    """仅当数据目录为 mix_clean 时需显式传入 speakers。"""
    root = tmp_path / "MiniLibriMix"
    split_dir = root / "test"
    _write_wav(split_dir / "mix_clean" / "a.wav", 0.1)
    _write_wav(split_dir / "s1" / "a.wav", 1.1)
    _write_wav(split_dir / "s2" / "a.wav", 2.1)
    index_dir = tmp_path / "DataIndex" / "legacy_clean"

    build_mini_librimix_index(
        root,
        index_dir,
        splits=("test",),
        speakers=("mix_clean", "s1", "s2"),
    )

    mix_json = index_dir / "test" / "mix_clean.json"
    assert mix_json.is_file()
    dataset = MiniLibriMixDataset(
        root=root,
        split="test",
        index_root=tmp_path / "DataIndex",
        index_name="legacy_clean",
        speakers=("mix_clean", "s1", "s2"),
    )
    assert len(dataset) == 1
    assert dataset[0]["id"] == "a"


def test_build_mini_librimix_index_can_limit_each_split_size(tmp_path: Path) -> None:
    root = tmp_path / "MiniLibriMix"
    for split, count in (("train", 4), ("val", 3), ("test", 2)):
        split_dir = root / split
        for idx in range(count):
            name = f"{idx}.wav"
            _write_wav(split_dir / "mix_both" / name, 0.1 + idx)
            _write_wav(split_dir / "s1" / name, 1.1 + idx)
            _write_wav(split_dir / "s2" / name, 2.1 + idx)

    index_dir = tmp_path / "DataIndex" / "mini_debug"
    build_mini_librimix_index(
        root,
        index_dir,
        split_counts={"train": 2, "val": 1, "test": 2},
    )

    train_rows = json.loads((index_dir / "train" / "mix_both.json").read_text(encoding="utf-8"))
    val_rows = json.loads((index_dir / "val" / "mix_both.json").read_text(encoding="utf-8"))
    test_rows = json.loads((index_dir / "test" / "mix_both.json").read_text(encoding="utf-8"))

    assert len(train_rows) == 2
    assert len(val_rows) == 1
    assert len(test_rows) == 2
