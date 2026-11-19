# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish fixed-width model outputs and read each original scene destination."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_native_conveyor import ROOT, _env_from_demo_setup
from stage_columns import stage_row
from test_runtime_backends import require_backend_and_models  # noqa: F401


def mapping(name, direction, variable, attribute, offset):
    return f'''def FmuMapping "{name}"
    {{
        token fmi:direction = "{direction}"
        token fmi:fmuAttribute = "{variable}"
        token fmi:usdAttribute = "{attribute}"
        int2 fmi:usdMapping = ({offset}, 1)
    }}'''


def scene(fmu, widths):
    lines = ['#usda 1.0', 'def Xform "World" {']
    baselines = {}
    for index, width in enumerate(widths):
        path = f"/World/State{index}"
        baseline = [float(101 + 10 * index + component) for component in range(width)]
        baselines[path] = baseline
        declaration = (f"double qa:result = {baseline[0]}" if width == 1 else
                       f"double3 qa:result = {tuple(baseline)}")
        lines.extend([
            f'def Xform "State{index}" {{',
            'custom double qa:input = 0', f'custom {declaration}', '}',
            f'def FmuInstance "Model{index}" {{', f'asset fmi:fmu = @{fmu.as_posix()}@',
            'def FmuConnection "Signals" {', f'rel fmi:targets = <{path}>',
            mapping("Input", "input", "pos_x", "qa:input", 0),
            mapping("Baseline", "input", "pos_y", "qa:result", 0),
            mapping("Output", "output", "force_x", "qa:result", 0 if width == 1 else 1),
            '}', '}',
        ])
    return '\n'.join(lines + ['}']), baselines


def publish(backend, fmi_version, widths, filtered, directory):
    from ovfmi import AttributeWrite, FmiHost, FmiHostConfig
    import ovstage
    from ovstage import population

    fmu = ROOT / "demoapp/usd" / ("PDController.fmu3" if fmi_version == 3 else "PDController.fmu")
    text, baselines = scene(fmu, widths)
    source = directory / "mixed.usda"
    source.write_text(text, encoding="utf-8")
    inputs = [-0.125, 0.25, -0.375, 0.5][:len(widths)]
    expected = {path: list(row) for path, row in baselines.items()}
    for (path, row), value in zip(expected.items(), inputs):
        row[0 if len(row) == 1 else 1] = -50 * value

    with ovstage.Stage("ovfmi.mixed-width-publication") as stage:
        population.open_usd(stage, str(source), ordinal=1)
        for path, row in baselines.items():
            stage_row(stage, path, 2, len(row), row)
        stage.advance_write_floor(2).wait()
        with FmiHost(FmiHostConfig(backend=backend)) as host:
            assert len(host.attach_ovstage(stage, source_asset=source).instances) == len(widths)
            host.wait_op(host.write([
                AttributeWrite((path,), "qa:input", [[value]])
                for path, value in zip(baselines, inputs)
            ]))
            host.step_sync(0.125)
            actual = {}
            with host.read() as read:
                for group in read.groups:
                    rows = group.tensors if group.is_array else group.tensors[0]
                    for path, row in zip(group.prim_paths, rows):
                        assert path not in actual
                        actual[path] = np.asarray(row).tolist()
            assert actual == expected
            if filtered:
                for path in baselines:
                    assert host.write_to_ovstage(3, prim_paths=[path]) == 1
            else:
                assert host.write_to_ovstage(3) == len(set(widths))
            stage.advance_write_floor(3).wait()
            actual = {path: stage_row(stage, path, 3, len(row))
                      for path, row in baselines.items()}
            assert actual == expected, actual


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
@pytest.mark.parametrize("widths,filtered", [
    ((1, 3, 1, 3), False),
    ((3, 1, 3, 1), False),
    ((1, 3, 1, 3), True),
    ((1, 1), False),
    ((3, 3), False),
])
def test_fixed_width_publication(backend, fmi_version, widths, filtered, tmp_path):
    if importlib.util.find_spec("ovstage") is None:
        pytest.skip("install ovstage to test scene publication")
    run = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), backend, str(fmi_version),
         ','.join(map(str, widths)), str(int(filtered)), str(tmp_path)],
        cwd=ROOT, env=_env_from_demo_setup(), capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stdout[-6000:] + run.stderr[-6000:]


if __name__ == "__main__":
    publish(sys.argv[1], int(sys.argv[2]), tuple(map(int, sys.argv[3].split(','))),
            bool(int(sys.argv[4])), Path(sys.argv[5]))
