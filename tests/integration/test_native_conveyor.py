# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end proof that conveyor FMI execution does not import FMPy."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _env_from_demo_setup() -> dict[str, str]:
    env = os.environ.copy()
    env_file = ROOT / "demoapp" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="ascii").splitlines():
            if "=" not in line:
                continue
            name, value = line.removeprefix("export ").split("=", 1)
            if name == "PYTHONPATH":
                continue
            env[name] = value.strip().strip('"')
    return env


def test_public_host_runs_conveyor_without_fmpy() -> None:
    native_library = ROOT / "native" / "target" / "release" / (
        "ovfmi_native.dll" if sys.platform == "win32" else "libovfmi_native.so"
    )
    native_container = ROOT / "demoapp" / "usd" / "conveyor_demo.native.fmu"
    if not native_library.exists() or not native_container.exists():
        pytest.skip("native runtime and conveyor container have not been built")

    source_asset = ROOT / "demoapp" / "usd" / "conveyor" / "ConveyorFMI.usda"
    program = f"""
import json
import sys
from importlib.abc import MetaPathFinder

class BlockFmpy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fmpy' or fullname.startswith('fmpy.'):
            raise ImportError('FMPy is forbidden in the native runtime test')

sys.meta_path.insert(0, BlockFmpy())
from ovfmi import FmiHost

with FmiHost() as host:
    host.attach_ovstage(object(), source_asset={str(source_asset)!r})
    for _ in range(600):
        host.step_sync(1.0 / 60.0)
    with host.read(attribute_names=['drive:angular:physics:targetVelocity']) as result:
        values = result.groups[0].tensors[0].reshape(-1).tolist()

print(json.dumps({{
    'fmpy_loaded': any(name == 'fmpy' or name.startswith('fmpy.') for name in sys.modules),
    'values': values,
}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=_env_from_demo_setup(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    import json

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["fmpy_loaded"] is False
    assert payload["values"] == pytest.approx([16.0] * 5)
