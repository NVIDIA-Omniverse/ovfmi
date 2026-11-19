# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A host can attach and calculate after repeated FMPy extraction failures."""

import subprocess
import sys

import pytest

from probe_first_step_inputs import scene
from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend_and_models  # noqa: F401


@pytest.mark.parametrize("backend", ["fmpy"])
def test_public_host_can_attach_after_failure(backend, tmp_path):
    invalid_fmu = tmp_path / "invalid.fmu"
    invalid_fmu.write_bytes(b"inert-invalid-fmu-zip\n")
    invalid_source = tmp_path / "invalid.usda"
    invalid_source.write_text(scene(invalid_fmu), encoding="utf-8")
    valid_source = tmp_path / "valid.usda"
    valid_source.write_text(scene(ROOT / "demoapp/usd/PDController.fmu"), encoding="utf-8")
    extraction_root = tmp_path / "extractions"
    extraction_root.mkdir()
    program = f'''
from pathlib import Path
import tempfile
import zipfile
from ovfmi import AttributeWrite, FmiHost, FmiHostConfig

extraction_root = Path({str(extraction_root)!r})
tempfile.tempdir = str(extraction_root)
stage = object()
host = FmiHost(FmiHostConfig(backend={backend!r}))
try:
    for _ in range(3):
        try:
            host.attach_ovstage(stage, source_asset={str(invalid_source)!r})
        except zipfile.BadZipFile:
            pass
        else:
            raise AssertionError("Invalid FMU was accepted")
        assert host.instances == ()
        assert not list(extraction_root.iterdir())

    report = host.attach_ovstage(stage, source_asset={str(valid_source)!r})
    assert len(report.instances) == 1
    host.wait_op(host.write([AttributeWrite(("/World",), "probe:input", [[2.0]])]))
    host.step_sync(0.01)
    with host.read(attribute_names=["probe:output"]) as result:
        assert result.groups[0].tensors[0][0, 0] == -100.0
finally:
    host.release()
    host.release()
assert not list(extraction_root.iterdir())
'''
    run = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, env=_env_from_demo_setup(),
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stdout[-6000:] + run.stderr[-6000:]
