# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authored numeric inputs survive isolated USD parsing and FMI attachment."""

import subprocess
import sys

import pytest

from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend_and_models  # noqa: F401


INPUTS = [
    pytest.param("matrix2d", "((1, 2), (.125, 4))", 2, [1, 2, .125, 4], id="matrix2d"),
    pytest.param(
        "matrix3d", "((1, 2, 3), (4, 5, 6), (7, .125, 9))",
        7, [1, 2, 3, 4, 5, 6, 7, .125, 9], id="matrix3d",
    ),
    pytest.param(
        "matrix4d", "((2, .25, 0, 0), (0, 3, .5, 0), (0, 0, 4, 0), (11, .125, 33, 1))",
        13, [2, .25, 0, 0, 0, 3, .5, 0, 0, 0, 4, 0, 11, .125, 33, 1], id="matrix",
    ),
    pytest.param(
        "matrix4d", "((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, .125, 15, 16))",
        13, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, .125, 15, 16], id="dense-matrix",
    ),
    pytest.param("double3", "(11, .125, 33)", 1, [11, .125, 33], id="vector"),
    pytest.param("double", ".125", 0, [.125], id="scalar"),
]


def _scene(fmu, usd_type, value, offset):
    return f'''#usda 1.0
def Xform "World"
{{
    custom {usd_type} sim:input = {value}
    custom double sim:output = 0
    def FmuInstance "Controller"
    {{
        asset fmi:fmu = @{fmu.as_posix()}@
        def FmuConnection "Signals"
        {{
            rel fmi:targets = </World>
            def FmuMapping "Input"
            {{
                token fmi:direction = "input"
                token fmi:fmuAttribute = "pos_x"
                token fmi:usdAttribute = "sim:input"
                int2 fmi:usdMapping = ({offset}, 1)
            }}
            def FmuMapping "Output"
            {{
                token fmi:direction = "output"
                token fmi:fmuAttribute = "force_x"
                token fmi:usdAttribute = "sim:output"
                int2 fmi:usdMapping = (0, 1)
            }}
        }}
    }}
}}
'''


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
@pytest.mark.parametrize("usd_type,value,offset,expected", INPUTS)
def test_authored_input_components(backend, fmi_version, usd_type, value, offset, expected, tmp_path):
    suffix = "fmu3" if fmi_version == 3 else "fmu"
    fmu = ROOT / "demoapp/usd" / f"PDController.{suffix}"
    source = tmp_path / "input.usda"
    source.write_text(_scene(fmu, usd_type, value, offset), encoding="utf-8")
    program = f'''
from ovfmi import AttributeWrite, FmiHost, FmiHostConfig
from ovfmi.population import parse_source

parsed = parse_source({str(source)!r})
assert parsed['initial_values']['/World']['sim:input'] == {expected!r}
with FmiHost(FmiHostConfig(backend={backend!r})) as host:
    report = host.attach_ovstage(object(), source_asset={str(source)!r})
    assert len(report.instances) == 1
    for _ in range(2):
        host.step_sync(.125)
        with host.read(attribute_names=['sim:output']) as result:
            assert result.groups[0].tensors[0][0, 0] == -6.25
        values = {expected!r}
        values[{offset}] = .25
        host.wait_op(host.write([AttributeWrite(('/World',), 'sim:input', [values])]))
        host.step_sync(.125)
        with host.read(attribute_names=['sim:output']) as result:
            assert result.groups[0].tensors[0][0, 0] == -12.5
        host.reset()
'''
    run = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, env=_env_from_demo_setup(),
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stdout[-6000:] + run.stderr[-6000:]
