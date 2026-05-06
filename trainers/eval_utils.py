from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def si_snr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    proj = torch.sum(estimate * reference, dim=-1, keepdim=True) * reference
    proj = proj / (torch.sum(reference**2, dim=-1, keepdim=True) + eps)
    noise = estimate - proj
    ratio = (torch.sum(proj**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return 10 * torch.log10(ratio)


def best_pairwise_sisnr(estimated_sources: torch.Tensor, reference_sources: torch.Tensor) -> torch.Tensor:
    direct = (si_snr(estimated_sources[0], reference_sources[0]) + si_snr(estimated_sources[1], reference_sources[1])) / 2
    swapped = (si_snr(estimated_sources[0], reference_sources[1]) + si_snr(estimated_sources[1], reference_sources[0])) / 2
    return torch.maximum(direct, swapped)


def evaluate_model(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    collate_fn,
    shuffle: bool = False,
    pin_memory: bool | None = None,
) -> dict[str, float | int | str]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda" if pin_memory is None else pin_memory,
        collate_fn=collate_fn,
    )

    total_improvement = 0.0
    utterance_count = 0
    sample_rate = None

    with torch.no_grad():
        for batch in loader:
            sample_rate = int(batch["sample_rate"])
            mix = batch["mix"].to(device)
            sources = batch["sources"].to(device)
            lengths = batch["lengths"]
            estimates = model(mix)

            for idx in range(mix.shape[0]):
                valid_len = int(lengths[idx].item())
                mix_i = mix[idx, :valid_len]
                ref_i = sources[idx, :, :valid_len]
                est_i = estimates[idx, :, :valid_len]

                input_score = (si_snr(mix_i, ref_i[0]) + si_snr(mix_i, ref_i[1])) / 2
                output_score = best_pairwise_sisnr(est_i, ref_i)
                total_improvement += float((output_score - input_score).item())
                utterance_count += 1

    data_root_metric = ""
    if hasattr(dataset, "inferred_dataset_root"):
        data_root_metric = str(dataset.inferred_dataset_root)
    if not data_root_metric:
        dr = getattr(dataset, "root", None)
        data_root_metric = str(dr) if dr is not None else ""

    return {
        "data_root": data_root_metric,
        "split": str(getattr(dataset, "split", "")),
        "sample_rate": sample_rate or 8000,
        "num_examples": utterance_count,
        "avg_si_snri_db": total_improvement / max(utterance_count, 1),
    }
