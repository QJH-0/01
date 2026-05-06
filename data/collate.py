from __future__ import annotations

import torch


def collate_mini_librimix_batch(batch: list[dict[str, torch.Tensor | str | int]]) -> dict[str, object]:
    lengths = [int(item["mix"].shape[-1]) for item in batch]
    max_len = max(lengths)
    mixes = []
    sources = []
    ids = []
    sample_rates = set()

    for item in batch:
        mix = item["mix"]
        src = item["sources"]
        sample_rates.add(int(item["sample_rate"]))
        pad_len = max_len - int(mix.shape[-1])
        if pad_len > 0:
            mix = torch.nn.functional.pad(mix, (0, pad_len))
            src = torch.nn.functional.pad(src, (0, pad_len))
        mixes.append(mix)
        sources.append(src)
        ids.append(item["id"])

    if len(sample_rates) != 1:
        raise ValueError(f"Mixed sample rates inside a batch: {sample_rates}")

    return {
        "ids": ids,
        "sample_rate": sample_rates.pop(),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "mix": torch.stack(mixes, dim=0),
        "sources": torch.stack(sources, dim=0),
    }
