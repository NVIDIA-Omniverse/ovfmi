# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve authored models before loading either runtime."""

import shutil
import subprocess
import sys
import zipfile

import pytest

from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend_and_models  # noqa: F401


MODEL = ROOT / "demoapp/usd/PDController.fmu"
SSP = ROOT / "demoapp/usd/orbit_controller.ssp"


@pytest.fixture(autouse=True)
def require_assets():
    if not MODEL.is_file() or not SSP.is_file():
        pytest.skip("run demoapp setup to build the example models")


def _scene(path, asset, *, kind="fmu", enabled=True, root="World"):
    path.write_text(f'''#usda 1.0
def Xform "{root}" {{
    custom double test:input = 0.125
    custom double test:output = 0
    def {"SspInstance" if kind == "ssp" else "FmuInstance"} "Model" {{
        asset fmi:{kind} = @{asset}@
        bool fmi:enabled = {str(enabled).lower()}
        def FmuConnection "Values" {{
            rel fmi:targets = </{root}>
            def FmuMapping "Input" {{
                token fmi:fmuAttribute = "pos_x"
                token fmi:usdAttribute = "test:input"
                token fmi:direction = "input"
                int2 fmi:usdMapping = (0, 1)
            }}
            def FmuMapping "Output" {{
                token fmi:fmuAttribute = "force_x"
                token fmi:usdAttribute = "test:output"
                token fmi:direction = "output"
                int2 fmi:usdMapping = (0, 1)
            }}
        }}
    }}
}}
''', encoding="utf-8")
    return path


_PREAMBLE = '''
from pathlib import Path
import numpy as np
import pytest
from ovfmi import FmiHost, FmiHostConfig

def output(host, dt=0.125):
    host.step_sync(dt)
    assert host.time == dt
    with host.read(attribute_names=['test:output']) as result:
        assert len(result.groups) == 1
        values = result.groups[0].tensors[0]
        assert values.shape == (1, 1)
        assert values.dtype == np.float64
        return float(values[0, 0])

def assert_detached(host):
    assert host.instances == ()
    with pytest.raises(RuntimeError, match='not attached'):
        host.step_sync(0.125)

def assert_model_output(host, backend, kind):
    if kind == 'fmu':
        assert output(host) == -6.25
    else:
        dt = 1 / 60
        # Native SSPs transfer internal connections after stepping the components.
        expected = -6.25 if backend == 'native' else 75 * np.cos(dt) - 6.25
        assert output(host, dt) == pytest.approx(expected)
'''


def _run(program, cwd):
    result = subprocess.run(
        [sys.executable, "-c", _PREAMBLE + program],
        cwd=cwd, env=_env_from_demo_setup(), capture_output=True,
        text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("kind", ["fmu", "ssp"])
@pytest.mark.parametrize("occupied", [False, True])
@pytest.mark.parametrize("reference", ["absolute", "relative", "empty"])
def test_unresolved_model_rejects(backend, kind, occupied, reference, tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    missing = {
        "absolute": (tmp_path / f"missing.{kind}").as_posix(),
        "relative": f"./missing.{kind}",
        "empty": "",
    }[reference]
    source = _scene(tmp_path / "missing.usda", missing, kind=kind)
    valid = _scene(tmp_path / "valid.usda", MODEL.as_posix())
    if occupied:
        with zipfile.ZipFile(MODEL) as archive:
            archive.extractall(cwd)
        shutil.copy2(MODEL, cwd / "missing.fmu")
        shutil.copy2(SSP, cwd / "missing.ssp")
        shutil.copy2(SSP.with_suffix(".native.fmu"), cwd / "missing.native.fmu")
    _run(f'''
with FmiHost(FmiHostConfig(backend={backend!r})) as host:
    with pytest.raises(ValueError) as rejected:
        host.attach_ovstage(object(), source_asset={str(source)!r})
    message = str(rejected.value)
    assert '/World/Model' in message and {'fmi:' + kind!r} in message, message
    assert {repr(missing)!r} in message, message
    assert_detached(host)
    host.attach_ovstage(object(), source_asset={str(valid)!r})
    assert output(host) == -6.25
''', cwd)


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("storage", ["archive", "directory", "ssp"])
def test_explicit_model_identity_and_output(backend, storage, tmp_path):
    asset = SSP if storage == "ssp" else MODEL
    if storage == "directory":
        asset = tmp_path / "unpacked"
        with zipfile.ZipFile(MODEL) as archive:
            archive.extractall(asset)
    kind = "ssp" if storage == "ssp" else "fmu"
    source = _scene(tmp_path / "valid.usda", asset.as_posix(), kind=kind)
    _run(f'''
with FmiHost(FmiHostConfig(backend={backend!r})) as host:
    report = host.attach_ovstage(object(), source_asset={str(source)!r})
    assert len(report.instances) == 1
    instance = report.instances[0]
    assert Path(instance.source_asset) == Path({str(asset)!r})
    assert instance.kind == {kind!r} and instance.enabled
    assert_model_output(host, {backend!r}, {kind!r})
''', tmp_path)


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("kind", ["fmu", "ssp"])
@pytest.mark.parametrize("occupied", [False, True])
def test_bare_reference_preserves_usd_search_path(backend, kind, occupied, tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    asset = cwd / f"model.{kind}"
    source = _scene(tmp_path / "bare.usda", asset.name, kind=kind)
    if occupied:
        shutil.copy2(SSP if kind == "ssp" else MODEL, asset)
        if kind == "ssp":
            shutil.copy2(SSP.with_suffix(".native.fmu"), asset.with_suffix(".native.fmu"))
    _run(f'''
with FmiHost(FmiHostConfig(backend={backend!r})) as host:
    if {occupied!r}:
        report = host.attach_ovstage(object(), source_asset={str(source)!r})
        assert len(report.instances) == 1
        assert Path(report.instances[0].source_asset) == Path({str(asset)!r})
        assert report.instances[0].kind == {kind!r}
        assert_model_output(host, {backend!r}, {kind!r})
    else:
        with pytest.raises(ValueError, match={repr(asset.name)!r}):
            host.attach_ovstage(object(), source_asset={str(source)!r})
        assert_detached(host)
''', cwd)


@pytest.mark.parametrize("backend", ["native", "fmpy"])
def test_rejected_reattachment_detaches_and_host_is_reusable(backend, tmp_path):
    valid = _scene(tmp_path / "valid.usda", MODEL.as_posix())
    missing = _scene(tmp_path / "missing.usda", "./missing.fmu")
    _run(f'''
with FmiHost(FmiHostConfig(backend={backend!r})) as host:
    host.attach_ovstage(object(), source_asset={str(valid)!r})
    assert output(host) == -6.25
    with pytest.raises(ValueError):
        host.attach_ovstage(object(), source_asset={str(missing)!r})
    assert_detached(host)
    host.attach_ovstage(object(), source_asset={str(valid)!r})
    assert host.time == 0
    assert output(host) == -6.25
''', tmp_path)


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("kind,selection", [
    ("fmu", "disabled"), ("ssp", "disabled"),
    ("fmu", "excluded"), ("ssp", "excluded"), ("ssp", "ssp_disabled"),
])
def test_unresolved_unselected_model(backend, kind, selection, tmp_path):
    asset = f"./missing.{kind}"
    source = _scene(tmp_path / "filtered.usda", asset, kind=kind,
                    enabled=selection != "disabled",
                    root="WorldOther" if selection == "excluded" else "World")
    config = {"backend": backend}
    if selection == "excluded":
        config["root_prim"] = "/World"
    elif selection == "ssp_disabled":
        config["enable_ssp"] = False
    _run(f'''
with FmiHost(FmiHostConfig(**{config!r})) as host:
    report = host.attach_ovstage(object(), source_asset={str(source)!r})
    if {selection!r} == 'disabled':
        assert len(report.instances) == 1
        instance = report.instances[0]
        assert instance.kind == {kind!r} and not instance.enabled
        assert instance.source_asset == {asset!r}
    else:
        assert report.instances == ()
    host.step_sync(0.125)
    with host.read() as result:
        assert not result.groups
''', tmp_path)
