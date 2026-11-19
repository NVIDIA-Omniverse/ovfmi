# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real FMI 2/3 FMUs exercise first-step inputs and double-precision reads."""

import json
import subprocess
import sys

import pytest

from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend_and_models  # noqa: F401


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
@pytest.mark.parametrize("phase", ["initial", "reset", "running"])
@pytest.mark.parametrize("mapping_count", [0, 1])
@pytest.mark.parametrize("route", ["write", "stage"])
def test_current_inputs_and_float64_outputs(backend, fmi_version, phase, mapping_count, route):
    # Not exactly representable in Float32: equality must catch a downcast,
    # not merely agree to the relatively loose default pytest.approx tolerance.
    value = 0.12345678912345678
    command = [
        sys.executable, str(ROOT / "tests/integration/probe_first_step_inputs.py"),
        "--backend", backend, "--route", route, "--fmi-version", str(fmi_version),
        "--value", str(value), "--mapping-count", str(mapping_count),
    ]
    if phase == "reset":
        command.append("--after-reset")
    elif phase == "running":
        command.append("--warmup")
    run = subprocess.run(command, cwd=ROOT, env=_env_from_demo_setup(),
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout[-6000:] + run.stderr[-6000:]
    # Native USD diagnostics can flush after the Python result at shutdown.
    records = [json.loads(line) for line in run.stdout.splitlines()
               if line.startswith('{') and '"first_step_applied"' in line]
    assert len(records) == 1, run.stdout
    result = records[0]
    assert result["first_step_applied"], result
    for sample in result["public_samples"]:
        assert sample["dtype"] == "float64", result
        assert sample["value"] == -50 * value, result
    assert result["fmpy_loaded"] == (backend == "fmpy")
    assert result["output_after_reset_to_authored"] == 0.0
