import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.libri_mix import MiniLibriMixDataset


def _write_wav(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full(16, value, dtype=np.float32)
    sf.write(path, audio, 8000)


def test_mini_librimix_dataset_pairs_split_files_in_sorted_order(tmp_path: Path) -> None:
    split_dir = tmp_path / "test"
    _write_wav(split_dir / "mix_both" / "b.wav", 0.1)
    _write_wav(split_dir / "mix_both" / "a.wav", 0.2)
    _write_wav(split_dir / "s1" / "b.wav", 1.1)
    _write_wav(split_dir / "s1" / "a.wav", 1.2)
    _write_wav(split_dir / "s2" / "b.wav", 2.1)
    _write_wav(split_dir / "s2" / "a.wav", 2.2)

    dataset = MiniLibriMixDataset(root=tmp_path, split="test", index_root=tmp_path / "DataIndex", index_name="default")

    assert len(dataset) == 2

    first = dataset[0]
    second = dataset[1]

    assert first["id"] == "a"
    assert second["id"] == "b"
    assert first["mix"].shape[0] == 16
    assert first["sources"].shape == (2, 16)


def test_mini_librimix_dataset_raises_when_source_file_is_missing(tmp_path: Path) -> None:
    split_dir = tmp_path / "test"
    _write_wav(split_dir / "mix_both" / "a.wav", 0.2)
    _write_wav(split_dir / "s1" / "a.wav", 1.2)

    try:
        MiniLibriMixDataset(root=tmp_path, split="test", index_root=tmp_path / "DataIndex", index_name="default")
    except FileNotFoundError as exc:
        assert "s2" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError when a paired source file is missing")
