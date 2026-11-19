# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest


@pytest.fixture
def require_usd_python(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    helper = root / "demoapp/.usd_venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    usd_python = os.environ.get("USD_PYTHON", str(helper))
    if not Path(usd_python).is_file():
        pytest.skip("run demoapp setup or set USD_PYTHON to an isolated OpenUSD interpreter")
    monkeypatch.setenv("USD_PYTHON", usd_python)
