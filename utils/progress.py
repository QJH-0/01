from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from tqdm.auto import tqdm


class ProgressLike(Protocol):
    total: int | None
    desc: str | None

    def update(self, value: int = 1) -> None: ...

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs) -> None: ...

    def close(self) -> None: ...


@dataclass
class NullProgress:
    total: int | None = None
    desc: str | None = None
    leave: bool = False
    dynamic_ncols: bool = True
    n: int = 0
    closed: bool = False
    postfix: dict[str, Any] | None = None

    def update(self, value: int = 1) -> None:
        self.n += value

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs) -> None:
        self.postfix = ordered_dict if ordered_dict is not None else kwargs

    def close(self) -> None:
        self.closed = True


def create_progress(
    *,
    total: int | None,
    desc: str,
    enabled: bool = True,
    leave: bool = False,
    dynamic_ncols: bool = True,
    factory=None,
) -> ProgressLike:
    if not enabled:
        return NullProgress(total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)
    progress_factory = factory or tqdm
    return progress_factory(total=total, desc=desc, leave=leave, dynamic_ncols=dynamic_ncols)
