"""Pytest 会话级配置。"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _quiet_index_build_progress() -> None:
    """避免 data.libri_mix 建索引时的 tqdm 刷屏。"""
    os.environ["INDEX_BUILD_QUIET"] = "1"
    yield
