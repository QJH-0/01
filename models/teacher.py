from __future__ import annotations

from pathlib import Path

import huggingface_hub
import torch
from data import MiniLibriMixDataset, collate_mini_librimix_batch, librimix_eval_json_dir
from models.convtasnet import ConvTasNet
from trainers.eval_utils import evaluate_model
from utils.helper import resolve_device
from utils.logger import append_experiment_record, build_eval_run_paths, make_run_timestamp, write_json


DEFAULT_MODEL_ID = "JorisCos/ConvTasNet_Libri2Mix_sepclean_8k"
DEFAULT_MODEL_ARGS = {
    "n_src": 2,
    "out_chan": None,
    "n_blocks": 8,
    "n_repeats": 3,
    "bn_chan": 128,
    "hid_chan": 512,
    "skip_chan": 128,
    "conv_kernel_size": 3,
    "norm_type": "gLN",
    "mask_act": "relu",
    "in_chan": None,
    "causal": False,
    "fb_name": "free",
    "kernel_size": 16,
    "n_filters": 512,
    "stride": 8,
    "encoder_activation": None,
    "sample_rate": 8000,
}


def get_model_args_from_config(config: dict[str, object] | None = None) -> dict[str, object]:
    model_args = dict(DEFAULT_MODEL_ARGS)
    if config is None:
        return model_args
    current = config
    for key in ("model", "model_args"):
        if not isinstance(current, dict) or key not in current:
            return model_args
        current = current[key]
    if isinstance(current, dict):
        model_args.update(current)
    return model_args


def load_pretrained_package_path(model_id: str = DEFAULT_MODEL_ID) -> Path:
    return Path(
        huggingface_hub.hf_hub_download(
            repo_id=model_id,
            filename=huggingface_hub.PYTORCH_WEIGHTS_NAME,
            library_name="conv_tasnet_binarization",
        )
    )


def load_teacher(model_id: str = DEFAULT_MODEL_ID, device: str | torch.device = "cpu") -> torch.nn.Module:
    package_path = load_pretrained_package_path(model_id)
    model = ConvTasNet.from_pretrained_path(package_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device)


def load_teacher_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    model_args: dict[str, object] | None = None,
) -> torch.nn.Module:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_args" in payload and "state_dict" in payload:
        model = ConvTasNet(**payload["model_args"])
        state_dict = payload["state_dict"]
    else:
        model = ConvTasNet(**(model_args or DEFAULT_MODEL_ARGS))
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device)


def evaluate_teacher_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path | None = None,
    json_dir: str | Path | None = None,
    index_root: str | Path | None = None,
    index_name: str = "default",
    split: str = "test",
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = "auto",
    max_examples: int | None = None,
    start_index: int = 0,
    output_root: str | Path = "Experiments",
    experiment_name: str = "e1_teacher_eval",
    pin_memory: bool | None = None,
    shuffle: bool = False,
    model_args: dict[str, object] | None = None,
    speakers: tuple[str, ...] | None = None,
) -> dict[str, float | int | str]:
    resolved_device = torch.device(resolve_device(device))
    model = load_teacher_checkpoint(checkpoint_path, device=resolved_device, model_args=model_args)
    resolved_json_dir = librimix_eval_json_dir(
        json_dir=json_dir,
        index_root=index_root,
        index_name=index_name,
        split=split,
    )
    dataset = MiniLibriMixDataset(
        json_dir=resolved_json_dir,
        dataset_root=data_root,
        max_examples=max_examples,
        start_index=start_index,
        **({"speakers": speakers} if speakers is not None else {}),
    )
    metrics = evaluate_model(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=resolved_device,
        collate_fn=collate_mini_librimix_batch,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    metrics["model_id"] = str(checkpoint_path)
    metrics["json_dir"] = str(resolved_json_dir)
    metrics["index_name"] = index_name
    timestamp = make_run_timestamp()
    paths = build_eval_run_paths(output_root=Path(str(output_root)), experiment_name=experiment_name, run_id=timestamp)
    write_json(paths["metrics"], metrics)
    write_json(paths["run_metrics"], metrics)

    record = {
        "run_id": timestamp,
        "experiment_name": experiment_name,
        "model_id": str(checkpoint_path),
        "data_root": metrics["data_root"],
        "split": metrics["split"],
        "index_name": index_name,
        "sample_rate": metrics["sample_rate"],
        "num_examples": metrics["num_examples"],
        "avg_si_snri_db": metrics["avg_si_snri_db"],
        "metrics_path": str(paths["metrics"]),
        "run_metrics_path": str(paths["run_metrics"]),
        "experiment_dir": str(paths["experiment_root"]),
    }
    append_experiment_record(paths["manifest"], record)
    metrics.update(record)
    return metrics
