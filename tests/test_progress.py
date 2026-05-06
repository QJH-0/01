import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.progress import NullProgress, create_progress


class RecordingProgress:
    def __init__(self, total=None, desc=None, leave=None, dynamic_ncols=None):
        self.total = total
        self.desc = desc
        self.leave = leave
        self.dynamic_ncols = dynamic_ncols
        self.updates = []
        self.postfixes = []
        self.closed = False

    def update(self, value=1):
        self.updates.append(value)

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        payload = ordered_dict if ordered_dict is not None else kwargs
        self.postfixes.append(payload)

    def close(self):
        self.closed = True


def test_null_progress_accepts_updates_without_side_effects() -> None:
    progress = NullProgress(total=3, desc="train")

    progress.update()
    progress.set_postfix(loss=-1.23)
    progress.close()

    assert progress.n == 1
    assert progress.total == 3
    assert progress.closed is True


def test_create_progress_uses_factory_when_enabled() -> None:
    created = []

    def factory(**kwargs):
        bar = RecordingProgress(**kwargs)
        created.append(bar)
        return bar

    progress = create_progress(total=5, desc="train", enabled=True, factory=factory)

    progress.update(2)
    progress.set_postfix(loss=-0.5)
    progress.close()

    assert len(created) == 1
    assert created[0].total == 5
    assert created[0].desc == "train"
    assert created[0].updates == [2]
    assert created[0].postfixes == [{"loss": -0.5}]
    assert created[0].closed is True
