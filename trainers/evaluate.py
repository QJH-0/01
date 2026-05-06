from __future__ import annotations

import argparse

from models.teacher import evaluate_teacher_checkpoint, get_model_args_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run E1 teacher evaluation from a local checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint or weights file to evaluate.")
    parser.add_argument(
        "--data_root",
        default=None,
        help="可选。仅当 json_dir 下缺少索引 JSON 且需从原始 wav 自动生成时，传入 MiniLibriMix 根目录。",
    )
    parser.add_argument(
        "--json_dir",
        default=None,
        help="TIGER 与 Libri2MixDataset 一致：含 mix_both.json、s1.json、s2.json 的目录（如 DataIndex/default/test）。",
    )
    parser.add_argument(
        "--index_root",
        default="DataIndex",
        help="未指定 --json_dir 时：legacy 路径 index_root/index_name/<split>。",
    )
    parser.add_argument("--index_name", default="default", help="legacy：index_root 下的集合名。")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Evaluation dataloader worker count.")
    parser.add_argument("--device", default="auto", help="Device string, e.g. auto/cpu/cuda.")
    parser.add_argument("--max_examples", type=int, default=None, help="Optional maximum number of examples.")
    parser.add_argument("--start_index", type=int, default=0, help="Start offset inside the selected index.")
    parser.add_argument("--output_root", default="Experiments", help="Experiment output root directory.")
    parser.add_argument("--experiment_name", default="e1_teacher_eval", help="Evaluation experiment name.")
    parser.add_argument("--pin_memory", choices=("true", "false"), default=None, help="Override DataLoader pin_memory.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle evaluation dataloader.")
    parser.add_argument(
        "--speakers",
        nargs=3,
        metavar=("MIX", "S1", "S2"),
        default=None,
        help="Index JSON stems. Default: mix_both s1 s2 (与 data.build_index 一致)。",
    )
    return parser.parse_args()


def _parse_pin_memory(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def main() -> None:
    args = parse_args()
    result = evaluate_teacher_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        json_dir=args.json_dir,
        index_root=args.index_root,
        index_name=args.index_name,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_examples=args.max_examples,
        start_index=args.start_index,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        pin_memory=_parse_pin_memory(args.pin_memory),
        shuffle=args.shuffle,
        model_args=get_model_args_from_config(None),
        speakers=tuple(args.speakers) if args.speakers else None,
    )
    print(f"Teacher SI-SNRi on {result['split']}: {result['avg_si_snri_db']:.2f} dB")


if __name__ == "__main__":
    main()
