import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import round_nested


def test_round_nested_rounds_float_values_recursively() -> None:
    payload = {
        "a": 1.234567,
        "b": [2.345678, {"c": 3.456789}],
        "d": "keep",
    }

    rounded = round_nested(payload, digits=4)

    assert rounded["a"] == 1.2346
    assert rounded["b"][0] == 2.3457
    assert rounded["b"][1]["c"] == 3.4568
    assert rounded["d"] == "keep"
