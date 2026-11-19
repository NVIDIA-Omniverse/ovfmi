# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run every behavior demo through the public API, with no GPU dependency."""

import json
import subprocess
import sys

import pytest

from test_native_conveyor import ROOT, _env_from_demo_setup


SCENES = [
    ("fmi_parser_test.usda", 1),
    ("pd_controller_test.usda", 1),
    ("two_fmu_orbit.usda", 2),
    ("ssp_orbit.usda", 1),
    ("conveyor/ConveyorFMI.usda", 1),
]


@pytest.fixture(autouse=True)
def require_backend_and_models(backend):
    if not (ROOT / "demoapp/usd/BouncingBall.fmu3").exists():
        pytest.skip("run demoapp setup to build the example FMUs")
    require_backend(backend)


def require_backend(backend):
    if backend == "native":
        from ovfmi._native_runtime import _library_path
        try:
            _library_path()
        except RuntimeError:
            pytest.skip("native runtime has not been built")
    else:
        import importlib.util
        if importlib.util.find_spec("fmpy") is None:
            pytest.skip("install ovfmi[fmpy] to test this backend")


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("scene,count", SCENES)
def test_demo_lifecycle(backend, scene, count):
    """Each backend steps, routes finite outputs, resets, and replays exactly."""
    program = f'''
import json
import sys
import numpy as np
from importlib.abc import MetaPathFinder

class BlockFmpy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fmpy' or fullname.startswith('fmpy.'):
            raise ImportError('Native execution must not import FMPy')

if {backend!r} == 'native':
    sys.meta_path.insert(0, BlockFmpy())
from ovfmi import FmiHost, FmiHostConfig

def snapshot(host):
    with host.read() as result:
        return {{g.attribute_name + ':' + ','.join(g.prim_paths):
                 [t.tolist() for t in g.tensors] for g in result.groups}}

with FmiHost(FmiHostConfig(backend={backend!r})) as host:
    report = host.attach_ovstage(object(), source_asset={str(ROOT / 'demoapp/usd' / scene)!r})
    assert len(report.instances) == {count}
    runs = []
    for run in range(2):
        samples = []
        for i in range(120):
            host.step_sync(1 / 60)
            samples.append(snapshot(host))
        runs.append(samples)
        host.reset()
        assert host.time == 0
        assert snapshot(host) == {{}}, 'reset must invalidate old output snapshots'
    assert runs[0] == runs[1], 'reset must restore the initial trajectory'
    assert runs[0][-1], 'no routed outputs'
    for sample in runs[0]:
        for tensors in sample.values():
            for tensor in tensors:
                assert np.isfinite(tensor).all()
    assert ({backend!r} == 'fmpy') == ('fmpy' in sys.modules)
    print(json.dumps({{'first': runs[0][0], 'last': runs[0][-1]}}))
'''
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, env=_env_from_demo_setup(),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if scene in ("fmi_parser_test.usda", "two_fmu_orbit.usda", "ssp_orbit.usda"):
        assert payload["first"] != payload["last"], "expected time-varying outputs"


@pytest.mark.parametrize("backend", ["native", "fmpy"])
def test_conveyor_controls(backend):
    """Dynamic public-API writes reach the packaged and Python SSP masters."""
    program = f'''
from ovfmi import AttributeWrite, FmiHost, FmiHostConfig
with FmiHost(FmiHostConfig(backend={backend!r})) as host:
    host.attach_ovstage(object(), source_asset={str(ROOT / 'demoapp/usd/conveyor/ConveyorFMI.usda')!r})
    def settle(expected):
        for i in range(600):
            host.step_sync(1 / 60)
        with host.read(attribute_names=['drive:angular:physics:targetVelocity']) as result:
            values = result.groups[0].tensors[0].reshape(-1)
            assert all(abs(x - expected) < 1e-4 for x in values), values
    settle(16)
    host.write([AttributeWrite(prim_paths=('/World/FMI/ConveyorControls',),
                              attribute_name='xformOp:translate', values=[[16, 16, 1]])])
    settle(0)
    host.write([AttributeWrite(prim_paths=('/World/FMI/ConveyorControls',),
                              attribute_name='xformOp:translate', values=[[8, 8, 0]])])
    settle(8)
'''
    result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                            env=_env_from_demo_setup(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


@pytest.mark.parametrize("backend", ["native", "fmpy"])
def test_authored_start_height(backend, tmp_path):
    source = ROOT / "demoapp/usd/fmi_parser_test.usda"
    text = source.read_text(encoding="utf-8")
    text = text.replace("(0, 1.0, 0)", "(0, 2.0, 0)")
    text = text.replace("@./BouncingBall.fmu3@", f"@{(source.parent / 'BouncingBall.fmu3').as_posix()}@")
    stage = tmp_path / "height.usda"
    stage.write_text(text, encoding="utf-8")
    program = f'''
from ovfmi import FmiHost, FmiHostConfig
with FmiHost(FmiHostConfig(backend={backend!r})) as host:
    host.attach_ovstage(object(), source_asset={str(stage)!r})
    for _ in range(2):
        host.step_sync(0.1)
        with host.read(attribute_names=['xformOp:translate']) as result:
            height = result.groups[0].tensors[0][0, 1]
            assert 1.9 < height < 2, height
        host.reset()
'''
    result = subprocess.run([sys.executable, "-c", program], cwd=ROOT,
                            env=_env_from_demo_setup(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("enable", [None, 1, 0], ids=["missing", "enabled", "disabled"])
def test_ssp_missing_input_keeps_fmu_start_value(backend, enable, tmp_path, require_usd_python):
    from ovfmi import AttributeWrite, FmiHost, FmiHostConfig, MissingInputPolicy

    inputs = {"operatorSpeed": 8, "rejectSpeed": 8, "enable": enable, "eStop": 0, "rawPresence": 0}
    declarations = "\n".join(f'    custom double {name}' + (f' = {value}' if value is not None else '')
                             for name, value in inputs.items())
    mappings = "\n".join(f'''
            def FmuMapping "{name}"
            {{
                token fmi:direction = "{direction}"
                token fmi:fmuAttribute = "{name}"
                token fmi:usdAttribute = "{name}"
                int2 fmi:usdMapping = (0, 1)
            }}''' for direction, name in [("input", name) for name in inputs] + [("output", "targetVelocity0")])
    source = tmp_path / "defaults.usda"
    source.write_text(f'''#usda 1.0
def Xform "World"
{{
{declarations}
    custom double targetVelocity0 = 0
    def SspInstance "Model"
    {{
        asset fmi:ssp = @{(ROOT / 'demoapp/usd/conveyor_demo.ssp').as_posix()}@
        def FmuConnection "Signals"
        {{
            rel fmi:targets = </World>
{mappings}
        }}
    }}
}}
''', encoding="utf-8")
    with FmiHost(FmiHostConfig(backend=backend, missing_input_policy=MissingInputPolicy.USE_FMU_START_VALUE)) as host:
        host.attach_ovstage(object(), source_asset=source)
        for _ in range(2):
            for _ in range(180):
                host.step_sync(1 / 60)
            with host.read(attribute_names=["targetVelocity0"]) as result:
                value = result.groups[0].tensors[0][0, 0]
                assert value == pytest.approx(0.0 if enable == 0 else 8.0, rel=0, abs=1e-5)
            assert host.time == pytest.approx(3.0)
            host.write([AttributeWrite(prim_paths=("/World",), attribute_name="enable", values=[[0.0]])])
            host.step_sync(1 / 60)
            host.reset()
    if enable is None:
        with FmiHost(FmiHostConfig(backend=backend, missing_input_policy=MissingInputPolicy.ERROR)) as host:
            with pytest.raises(ValueError, match="missing mapped FMI input /World.enable"):
                host.attach_ovstage(object(), source_asset=source)
