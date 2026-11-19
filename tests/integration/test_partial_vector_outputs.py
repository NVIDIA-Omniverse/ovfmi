# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real FMI 2/3 models preserve caller vector writes through partial outputs."""

import subprocess
import sys

import pytest

from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend_and_models  # noqa: F401


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
def test_partial_outputs_preserve_caller_values(backend, fmi_version):
    command = [
        sys.executable, str(ROOT / "tests/integration/probe_partial_vector_outputs.py"),
        "--backend", backend, "--fmi-version", str(fmi_version), "--route", "write",
    ]
    result = subprocess.run(
        command, cwd=ROOT, env=_env_from_demo_setup(),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
