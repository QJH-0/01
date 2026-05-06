import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.teacher import DEFAULT_MODEL_ID, load_pretrained_package_path


def test_pretrained_package_loader_uses_huggingface_cache_without_asteroid_repo() -> None:
    path = load_pretrained_package_path(DEFAULT_MODEL_ID)
    assert path.name == "pytorch_model.bin"
    assert path.is_file()
