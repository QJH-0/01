import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.precision import resolve_amp_mode


def test_resolve_amp_mode_disables_mixed_precision_on_cpu() -> None:
    mode = resolve_amp_mode("16-mixed", device_type="cpu")

    assert mode.requested_precision == "16-mixed"
    assert mode.use_mixed_precision is False
    assert mode.autocast_dtype is None
    assert mode.scaler_enabled is False


def test_resolve_amp_mode_enables_mixed_precision_on_cuda() -> None:
    mode = resolve_amp_mode("16-mixed", device_type="cuda")

    assert mode.requested_precision == "16-mixed"
    assert mode.use_mixed_precision is True
    assert mode.autocast_dtype is not None
    assert mode.scaler_enabled is True


def test_resolve_amp_mode_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        resolve_amp_mode("bf16-mixed", device_type="cuda")
