from __future__ import annotations

import argparse

from data.libri_mix import DEFAULT_SPEAKERS, build_mini_librimix_index


def parse_args():
    parser = argparse.ArgumentParser("Build MiniLibriMix JSON index files (aligned with TIGER build_mini_librimix_index)")
    parser.add_argument("--in_dir", required=True, help="MiniLibriMix root directory (splits under here)")
    parser.add_argument("--out_dir", required=True, help="Output directory for JSON index files")
    parser.add_argument("--train_count", type=int, default=None, help="Optional number of train samples to index.")
    parser.add_argument("--val_count", type=int, default=None, help="Optional number of val samples to index.")
    parser.add_argument("--test_count", type=int, default=None, help="Optional number of test samples to index.")
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Subfolders per split. Default: mix_both s1 s2. 仅含 mix_clean 的数据集可传: mix_clean s1 s2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speakers = tuple(args.speakers) if args.speakers else DEFAULT_SPEAKERS
    build_mini_librimix_index(
        args.in_dir,
        args.out_dir,
        split_counts={
            "train": args.train_count,
            "val": args.val_count,
            "test": args.test_count,
        },
        speakers=speakers,
    )
    print(f"MiniLibriMix index written to: {args.out_dir}")


if __name__ == "__main__":
    main()
