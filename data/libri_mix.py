from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from utils.helper import get_by_path

# 与 TIGER look2hear/datas/Libri2Mix16.py 一致：默认读取 mix_both.json + s1.json + s2.json。
DEFAULT_SPEAKERS = ("mix_both", "s1", "s2")


def _index_progress_disabled() -> bool:
    """pytest 等场景可设 INDEX_BUILD_QUIET=1 关闭进度条。"""
    return os.environ.get("INDEX_BUILD_QUIET", "").strip().lower() in ("1", "true", "yes")


def _maybe_tqdm(iterable: Iterable[Path], *, desc: str) -> Iterable[Path]:
    if _index_progress_disabled():
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        unit="wav",
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
    )


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    split: str
    mix_path: Path
    s1_path: Path
    s2_path: Path
    num_frames: int
    sample_rate: int


def _sorted_wavs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".wav")


def _select_aligned_files(split_dir: Path, count: int, speakers: tuple[str, ...]) -> dict[str, list[Path]]:
    """与 TIGER `DataPreProcess/build_mini_librimix_index.py` 一致：各 speaker 取前 count 个 wav 且文件名对齐。"""
    selected = {speaker: _sorted_wavs(split_dir / speaker)[:count] for speaker in speakers}

    if any(len(paths) < count for paths in selected.values()):
        raise ValueError(f"Split '{split_dir.name}' does not contain at least {count} wav files per speaker")

    reference_names = [path.name for path in selected[speakers[0]]]
    for speaker in speakers[1:]:
        if [path.name for path in selected[speaker]] != reference_names:
            raise ValueError(f"Split '{split_dir.name}' has misaligned filenames across speakers")

    return selected


def _write_index_file(paths: list[Path], out_path: Path, *, desc: str) -> None:
    """与 TIGER 一致：用 SoundFile 取帧数，JSON indent=4。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_infos: list[list[object]] = []
    for wav_path in _maybe_tqdm(paths, desc=desc):
        samples = sf.SoundFile(str(wav_path))
        file_infos.append([str(wav_path.resolve()), int(len(samples))])
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(file_infos, handle, indent=4, ensure_ascii=False)


def build_mini_librimix_index(
    in_dir: str | Path,
    out_dir: str | Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    split_counts: dict[str, int | None] | None = None,
    speakers: tuple[str, ...] = DEFAULT_SPEAKERS,
) -> None:
    raw_root = Path(in_dir)
    output_root = Path(out_dir)

    for split in splits:
        split_dir = raw_root / split
        for speaker in speakers:
            speaker_dir = split_dir / speaker
            if not speaker_dir.is_dir():
                raise FileNotFoundError(f"Expected dataset directory: {speaker_dir}")

        split_limit = None if split_counts is None else split_counts.get(split)
        if split_limit is not None:
            if split_limit <= 0:
                raise ValueError(f"Split count for '{split}' must be positive, got {split_limit}")
            selected = _select_aligned_files(split_dir, split_limit, speakers)
        else:
            selected = {speaker: _sorted_wavs(split_dir / speaker) for speaker in speakers}
            reference_names = [path.name for path in selected[speakers[0]]]
            if not reference_names:
                raise FileNotFoundError(f"No wav files found in {split_dir / speakers[0]}")
            for speaker in speakers:
                if [path.name for path in selected[speaker]] != reference_names:
                    raise ValueError(f"Split '{split}' has misaligned filenames across speakers")

        for speaker, paths in selected.items():
            _write_index_file(paths, output_root / split / f"{speaker}.json", desc=f"{split}/{speaker}")


def ensure_json_index(
    root: Path | str | None,
    index_root: Path | str,
    index_name: str,
    split: str,
    speakers: tuple[str, ...] = DEFAULT_SPEAKERS,
) -> Path:
    index_dir = Path(index_root) / index_name
    required_files = [index_dir / split / f"{speaker}.json" for speaker in speakers]
    if all(path.is_file() for path in required_files):
        return index_dir
    if root is None or str(root).strip() == "":
        missing = [str(p) for p in required_files if not p.is_file()]
        raise FileNotFoundError(
            "索引 JSON 不存在且未配置 data.root，无法自动生成索引。请先运行 data.build_index，或在配置中设置 data.root。"
            f" 缺失: {missing}"
        )
    build_mini_librimix_index(Path(root), index_dir, splits=(split,), split_counts=None, speakers=speakers)
    return index_dir


def _require_mix_jsons(json_dir: Path, speakers: tuple[str, str, str]) -> None:
    mix_key, s1_key, s2_key = speakers
    for key in (mix_key, s1_key, s2_key):
        p = json_dir / f"{key}.json"
        if not p.is_file():
            raise FileNotFoundError(
                f"未找到 {p}。TIGER 方案下 json_dir 应为含 mix_both.json、s1.json、s2.json 的目录（与 Libri2MixDataset 的 json_dir 一致）。"
            )


def _maybe_build_split_from_root(
    json_dir: Path,
    dataset_root: Path | str | None,
    speakers: tuple[str, ...],
) -> None:
    """若 json_dir 下缺少 JSON 且提供了原始 MiniLibriMix 根目录，则为该 split 生成索引（写入 json_dir 的父目录为 index 根）。"""
    mix_key = speakers[0]
    if (json_dir / f"{mix_key}.json").is_file():
        return
    if dataset_root is None or str(dataset_root).strip() == "":
        raise FileNotFoundError(
            f"目录 {json_dir} 下缺少索引 JSON，且未设置 data.root，无法自动生成。"
        )
    split_name = json_dir.name
    index_base = json_dir.parent
    build_mini_librimix_index(
        Path(dataset_root),
        index_base,
        splits=(split_name,),
        split_counts=None,
        speakers=speakers,
    )


def librimix_train_json_dir(config: dict) -> Path:
    """训练集：优先 data.train_dir（TIGER train_dir），否则 index_root/index_name/train_split。"""
    explicit = get_by_path(config, "data.train_dir")
    if explicit is not None and str(explicit).strip():
        return Path(str(explicit)).expanduser().resolve()
    ir = get_by_path(config, "data.index_root")
    if ir is None:
        raise KeyError("请设置 data.train_dir（与 TIGER Libri2MixModuleRemix 一致）或 legacy：data.index_root + data.train_split")
    iname = str(get_by_path(config, "data.index_name", "default"))
    split = get_by_path(config, "data.train_split")
    if split is None:
        raise KeyError("legacy 模式需要 data.train_split")
    return (Path(str(ir)).expanduser() / iname / str(split)).resolve()


def librimix_valid_json_dir(config: dict) -> Path:
    """验证集：优先 data.valid_dir（TIGER valid_dir），否则 index_root/index_name/val_split。"""
    explicit = get_by_path(config, "data.valid_dir")
    if explicit is not None and str(explicit).strip():
        return Path(str(explicit)).expanduser().resolve()
    ir = get_by_path(config, "data.index_root")
    if ir is None:
        raise KeyError("请设置 data.valid_dir（与 TIGER 一致）或 legacy：data.index_root + data.val_split")
    iname = str(get_by_path(config, "data.index_name", "default"))
    split = get_by_path(config, "data.val_split")
    if split is None:
        raise KeyError("legacy 模式需要 data.val_split")
    return (Path(str(ir)).expanduser() / iname / str(split)).resolve()


def librimix_eval_json_dir(
    *,
    json_dir: str | Path | None,
    index_root: str | Path | None,
    index_name: str,
    split: str,
) -> Path:
    """评估 CLI：优先 --json_dir，否则 index_root/index_name/split。"""
    if json_dir is not None and str(json_dir).strip():
        return Path(str(json_dir)).expanduser().resolve()
    if index_root is None or str(index_root).strip() == "":
        raise ValueError("请提供 --json_dir（TIGER）或 --index_root + --index_name + --split（legacy）")
    return (Path(str(index_root)).expanduser() / index_name / split).resolve()


def _load_index_file(index_path: Path) -> list[list[object]]:
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Index file must contain a JSON list: {index_path}")
    return payload


class MiniLibriMixDataset(Dataset):
    """与 TIGER `Libri2MixDataset` 一致：json_dir 为含 mix_both.json / s1.json / s2.json 的目录。"""

    def __init__(
        self,
        json_dir: str | Path | None = None,
        *,
        max_examples: int | None = None,
        start_index: int = 0,
        speakers: Sequence[str] | None = None,
        dataset_root: str | Path | None = None,
        root: str | Path | None = None,
        split: str | None = None,
        index_root: str | Path | None = None,
        index_name: str = "default",
    ) -> None:
        self.speakers: tuple[str, str, str] = (
            tuple(str(x) for x in speakers) if speakers is not None else DEFAULT_SPEAKERS
        )
        if len(self.speakers) != 3:
            raise ValueError("speakers must be length-3: (mix_condition, s1, s2)")

        legacy_root = root
        if legacy_root is not None and str(legacy_root).strip() == "":
            legacy_root = None
        build_root = dataset_root
        if build_root is not None and str(build_root).strip() == "":
            build_root = None
        build_root = build_root or legacy_root

        if json_dir is not None and str(json_dir).strip():
            self.json_dir = Path(str(json_dir)).expanduser().resolve()
            _maybe_build_split_from_root(self.json_dir, build_root, self.speakers)
            _require_mix_jsons(self.json_dir, self.speakers)
        elif split is not None and index_root is not None:
            ensure_json_index(
                legacy_root,
                Path(str(index_root)).expanduser(),
                index_name,
                split,
                speakers=self.speakers,
            )
            self.json_dir = (Path(str(index_root)).expanduser() / index_name / split).resolve()
            _require_mix_jsons(self.json_dir, self.speakers)
        else:
            raise ValueError(
                "MiniLibriMixDataset: 传入 json_dir=...（与 TIGER Libri2MixDataset 一致），"
                "或 legacy：index_root + index_name + split。"
            )

        self.split = self.json_dir.name
        self.root: Path | None = Path(legacy_root).resolve() if legacy_root is not None else None
        self.samples = self._load_samples(max_examples=max_examples, start_index=start_index)

    def _load_samples(self, max_examples: int | None, start_index: int) -> list[SamplePaths]:
        mix_key, s1_key, s2_key = self.speakers
        mix_entries = _load_index_file(self.json_dir / f"{mix_key}.json")
        s1_entries = _load_index_file(self.json_dir / f"{s1_key}.json")
        s2_entries = _load_index_file(self.json_dir / f"{s2_key}.json")
        if not (len(mix_entries) == len(s1_entries) == len(s2_entries)):
            raise ValueError(f"Mismatched index lengths under {self.json_dir}")

        rows = list(zip(mix_entries, s1_entries, s2_entries))[start_index:]
        if max_examples is not None:
            rows = rows[:max_examples]

        samples: list[SamplePaths] = []
        for mix_entry, s1_entry, s2_entry in rows:
            mix_path = Path(str(mix_entry[0]))
            s1_path = Path(str(s1_entry[0]))
            s2_path = Path(str(s2_entry[0]))
            sample_id = mix_path.stem
            if s1_path.stem != sample_id or s2_path.stem != sample_id:
                raise ValueError(f"Mismatched sample ids inside index for split '{self.split}': {sample_id}")
            if not mix_path.is_file():
                raise FileNotFoundError(f"Missing indexed mix file: {mix_path}")
            if not s1_path.is_file():
                raise FileNotFoundError(f"Missing indexed source file in s1: {s1_path}")
            if not s2_path.is_file():
                raise FileNotFoundError(f"Missing indexed source file in s2: {s2_path}")
            info = sf.info(str(mix_path))
            samples.append(
                SamplePaths(
                    sample_id=sample_id,
                    split=self.split,
                    mix_path=mix_path,
                    s1_path=s1_path,
                    s2_path=s2_path,
                    num_frames=int(mix_entry[1]),
                    sample_rate=int(info.samplerate),
                )
            )

        if not samples:
            raise FileNotFoundError(f"No indexed samples found in {self.json_dir}")
        return samples

    @property
    def inferred_dataset_root(self) -> str:
        """数据集集合根目录（train/val/test 的上一级），与 TIGER 中 train_dir 的父目录一致。"""
        if self.root is not None:
            return str(self.root)
        try:
            return str(self.json_dir.parent.resolve())
        except (OSError, ValueError):
            return ""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        mix, mix_sr = sf.read(sample.mix_path, dtype="float32")
        s1, s1_sr = sf.read(sample.s1_path, dtype="float32")
        s2, s2_sr = sf.read(sample.s2_path, dtype="float32")
        if len({mix_sr, s1_sr, s2_sr}) != 1:
            raise ValueError(f"Sample rate mismatch for {sample.sample_id}")
        return {
            "id": sample.sample_id,
            "sample_rate": mix_sr,
            "mix": torch.from_numpy(mix),
            "sources": torch.stack((torch.from_numpy(s1), torch.from_numpy(s2)), dim=0),
        }
