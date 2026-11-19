# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate output ownership through both backends without ovstage."""

import subprocess
import sys

import pytest

from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend_and_models  # noqa: F401


def _scene(fmu, collision, second_variable):
    instances = []
    for index, value in enumerate((0.125, 0.5)):
        target = 0 if collision else index
        variable = second_variable if index else "force_x"
        instances.append(f'''
    def Xform "State{index}"
    {{
        custom double input = {value}
        custom double output = 0
    }}
    def FmuInstance "Model{index}"
    {{
        asset fmi:fmu = @{fmu.as_posix()}@
        def FmuConnection "Input"
        {{
            rel fmi:targets = </World/State{index}>
            def FmuMapping "Position"
            {{
                token fmi:direction = "input"
                token fmi:fmuAttribute = "pos_x"
                token fmi:usdAttribute = "input"
                int2 fmi:usdMapping = (0, 1)
            }}
        }}
        def FmuConnection "Output"
        {{
            rel fmi:targets = </World/State{target}>
            def FmuMapping "Force"
            {{
                token fmi:direction = "output"
                token fmi:fmuAttribute = "{variable}"
                token fmi:usdAttribute = "output"
                int2 fmi:usdMapping = (0, 1)
            }}
        }}
    }}
''')
    return '#usda 1.0\ndef Xform "World"\n{\n' + "".join(instances) + '}\n'


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
@pytest.mark.parametrize("strict,collision,second_variable", [
    pytest.param(True, True, "force_x", id="same-name-collision"),
    pytest.param(True, True, "force_y", id="different-name-collision"),
    pytest.param(True, False, "force_x", id="distinct-targets"),
    pytest.param(False, True, "force_x", id="nonstrict-last-owner"),
])
def test_output_ownership(backend, fmi_version, strict, collision, second_variable, tmp_path):
    fmu = ROOT / "demoapp/usd" / ("PDController.fmu" if fmi_version == 2 else "PDController.fmu3")
    source = tmp_path / "ownership.usda"
    source.write_text(_scene(fmu, collision, second_variable), encoding="utf-8")
    expected = {"/World/State0": [-25.0]} if collision else {
        "/World/State0": [-6.25], "/World/State1": [-25.0],
    }
    program = f'''
from ovfmi import FmiHost, FmiHostConfig

with FmiHost(FmiHostConfig(backend={backend!r}, strict_schema_validation={strict!r})) as host:
    if {strict and collision!r}:
        try:
            host.attach_ovstage(object(), source_asset={str(source)!r})
        except ValueError as error:
            assert 'overlapping output mapping' in str(error), str(error)
        else:
            raise AssertionError('strict attachment accepted conflicting output owners')
    else:
        report = host.attach_ovstage(object(), source_asset={str(source)!r})
        assert [item.prim_path for item in report.instances] == ['/World/Model0', '/World/Model1']
        for _ in range(2):
            host.step_sync(0.125)
            with host.read(attribute_names=['output']) as result:
                values = {{path: group.tensors[0][index].tolist()
                          for group in result.groups
                          for index, path in enumerate(group.prim_paths)}}
            assert values == {expected!r}, values
            assert host.time == 0.125, host.time
            host.reset()
            assert host.time == 0
'''
    run = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, env=_env_from_demo_setup(),
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stdout[-6000:] + run.stderr[-6000:]
