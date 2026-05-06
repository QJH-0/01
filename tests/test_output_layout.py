import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import build_eval_run_paths, build_training_run_paths


def test_build_eval_run_paths_groups_outputs_under_experiment_and_run(tmp_path: Path) -> None:
    paths = build_eval_run_paths(tmp_path, "e1_teacher_eval", "20260506-120000")

    assert paths["run_dir"].as_posix().endswith("e1_teacher_eval")
    assert paths["metrics"].as_posix().endswith("e1_teacher_eval/final_metrics.json")
    assert "checkpoint" not in paths
    assert "conf_yaml" not in paths


def test_build_training_run_paths_groups_history_and_checkpoints_together(tmp_path: Path) -> None:
    paths = build_training_run_paths("e1_teacher_smoke_train", tmp_path, "20260506-120000")

    assert paths["run_dir"].as_posix().endswith("e1_teacher_smoke_train")
    assert paths["history_csv"].as_posix().endswith("e1_teacher_smoke_train/history.csv")
    assert paths["last_ckpt"].as_posix().endswith("e1_teacher_smoke_train/checkpoints/last.ckpt")
    assert paths["run_results_dir"].as_posix().endswith("results/20260506-120000")
    assert paths["run_history_csv"].parent == paths["run_results_dir"]
