# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output-only attributes retain their width and unmapped values."""

import subprocess
import sys

import pytest

from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend_and_models  # noqa: F401
from probe_output_baselines import output_scene


def _run(program, *, expected_warning=None):
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT / "tests/integration",
        env=_env_from_demo_setup(), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    if expected_warning is not None:
        assert expected_warning in result.stderr, result.stderr
    return result.stdout


@pytest.fixture(autouse=True)
def require_controller(fmi_version):
    suffix = "fmu3" if fmi_version == 3 else "fmu"
    if not (ROOT / "demoapp/usd" / f"PDController.{suffix}").exists():
        pytest.skip("run demoapp setup to build PDController FMUs")


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
@pytest.mark.parametrize("initial,offset,count,declaration", [
    ([11.0, 0.0, 33.0], 0, 1, None),
    ([11.0, 0.0, 33.0], 1, 1, None),
    ([11.0, 0.0, 33.0], 2, 1, None),
    ([11.0, 0.0, 33.0], 1, 2, None),
    ([11.0, 0.0, 33.0], 0, 3, None),
    ([0.0], 0, 1, None),
    ([0.0], 0, 0, None),
    ([0.0, 0.0, 0.0], 1, 1, "custom double3 probe:output"),
    ([0.0, 0.0], 1, 1, ""),
    ([0.0, 0.0], 1, 1, "custom quatf probe:output = (1, 0, 0, 0)"),
    ([0.0, 0.0], 1, 1, "custom float3[] probe:output = [(11, 0, 33)]"),
])
def test_output_baselines(backend, fmi_version, initial, offset, count, declaration):
    _run(
        "from probe_output_baselines import check_case; "
        f"check_case({backend!r}, {fmi_version}, {initial!r}, {offset}, {count}, "
        f"declaration={declaration!r})"
    )


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
def test_mixed_output_baselines(backend, fmi_version):
    _run(
        "from probe_output_baselines import check_mixed; "
        f"check_mixed({backend!r}, {fmi_version})"
    )


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
def test_partial_vector_stage_refresh_and_publication(backend, fmi_version):
    _run("from probe_partial_vector_outputs import probe; "
         f"probe({backend!r}, {fmi_version}, 'stage')")


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
def test_output_default_preserves_missing_input_policy(backend, fmi_version, tmp_path):
    suffix = "fmu3" if fmi_version == 3 else "fmu"
    fmu = ROOT / "demoapp/usd" / f"PDController.{suffix}"
    source = tmp_path / "missing-input.usda"
    source.write_text(
        output_scene(fmu, "custom double3 probe:output", 1, 1).replace(
            'token fmi:usdAttribute = "probe:input"',
            'token fmi:usdAttribute = "probe:output"',
        ), encoding="utf-8",
    )
    _run(f'''
import pytest
from ovfmi import FmiHost, FmiHostConfig, MissingInputPolicy
with FmiHost(FmiHostConfig(backend={backend!r}, missing_input_policy=MissingInputPolicy.ERROR)) as host:
    with pytest.raises(ValueError, match="missing mapped FMI input /World.probe:output"):
        host.attach_ovstage(object(), source_asset={str(source)!r})
''')


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2])
@pytest.mark.parametrize("scope", ["disabled-instance", "disabled-connection", "excluded-root"])
def test_unsupported_inactive_output_keeps_control_running(backend, fmi_version, scope, tmp_path):
    fmu = ROOT / "demoapp/usd/PDController.fmu"
    control = output_scene(fmu, "custom double probe:output = 0", 0, 1).replace(
        "custom double probe:input = 0", "custom double probe:input = 0.125",
    )
    ignored = output_scene(fmu, "custom quatf probe:output = (1, 0, 0, 0)", 1, 1)
    ignored = ignored.replace('"World"', '"Other"').replace("/World", "/Other")
    if scope == "disabled-instance":
        ignored = ignored.replace("asset fmi:fmu", "bool fmi:enabled = false\n        asset fmi:fmu")
    elif scope == "disabled-connection":
        ignored = ignored.replace("rel fmi:targets", "bool fmi:enabled = false\n            rel fmi:targets")
    source = tmp_path / "inactive.usda"
    source.write_text(control + ignored.split("\n", 1)[1], encoding="utf-8")
    root = "/World" if scope == "excluded-root" else "/"
    _run(f'''
from ovfmi import FmiHost, FmiHostConfig
with FmiHost(FmiHostConfig(backend={backend!r}, root_prim={root!r})) as host:
    host.attach_ovstage(object(), source_asset={str(source)!r})
    host.step_sync(0.125)
    with host.read(prim_paths=["/World"], attribute_names=["probe:output"]) as read:
        assert len(read.groups) == 1, read.groups
        assert read.groups[0].tensors[0].tolist() == [[-6.25]], read.groups[0].tensors
''', expected_warning="no numeric output baseline for /Other.probe:output (quatf)")
