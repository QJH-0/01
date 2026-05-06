import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import append_experiment_record, build_eval_run_paths


def test_build_eval_run_paths_creates_grouped_artifact_targets() -> None:
    paths = build_eval_run_paths(
        output_root=Path("artifacts"),
        experiment_name="e1_teacher_eval",
        run_id="20260506-120000",
    )

    assert paths["metrics"].as_posix().endswith("artifacts/e1_teacher_eval/final_metrics.json")
    assert paths["manifest"].as_posix().endswith("artifacts/e1_teacher_eval/history.jsonl")
    assert "checkpoint" not in paths
    assert "conf_yaml" not in paths


def test_append_experiment_record_preserves_multiple_runs(tmp_path: Path) -> None:
    manifest = tmp_path / "logs" / "e1_teacher_eval" / "history.jsonl"

    append_experiment_record(manifest, {"run_id": "run-a", "score": 15.2})
    append_experiment_record(manifest, {"run_id": "run-b", "score": 15.3})

    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "run-a"
    assert json.loads(lines[1])["run_id"] == "run-b"
