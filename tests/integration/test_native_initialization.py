# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile scalar FMUs that enforce the FMI initialization write constraints."""

import ctypes
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import pytest

from ovfmi import AttributeWrite, FmiHost, FmiHostConfig
from ovfmi._native_runtime import NativeInstance, NativeLibrary, _library_path


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", params=[2, 3], ids=["fmi2", "fmi3"])
def strict_fmu(request, tmp_path_factory):
    if platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip("the fixture requires x86-64")
    version = request.param
    build = tmp_path_factory.mktemp(f"strict-fmi{version}")
    source = FIXTURES / "strict_lifecycle.c"
    if sys.platform == "win32":
        compiler, platform_dir, extension = "cl", "win64" if version == 2 else "x86_64-windows", ".dll"
        library = build / f"StrictLifecycle{extension}"
        command = [compiler, "/nologo", "/LD", "/MT", "/O2", "/TC", "/std:c11",
                   f"/DFMI_VERSION={version}", str(source), f"/Fe:{library}"]
    elif sys.platform == "linux":
        compiler, platform_dir, extension = "cc", "linux64" if version == 2 else "x86_64-linux", ".so"
        library = build / f"StrictLifecycle{extension}"
        command = [compiler, "-std=c11", "-shared", "-fPIC", "-O2", f"-DFMI_VERSION={version}",
                   str(source), "-o", str(library)]
    else:
        pytest.skip(f"unsupported fixture platform: {sys.platform}")
    if shutil.which(compiler) is None:
        pytest.skip(f"the fixture requires {compiler} on PATH")
    result = subprocess.run(command, cwd=build, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    fmu = build / "StrictLifecycle.fmu"
    with zipfile.ZipFile(fmu, "w") as archive:
        archive.write(FIXTURES / f"strict_fmi{version}.xml", "modelDescription.xml")
        archive.write(library, f"binaries/{platform_dir}/{library.name}")
    return version, fmu, library


@pytest.fixture(scope="module")
def native():
    try:
        _library_path()
    except RuntimeError:
        pytest.skip("native runtime has not been built")
    return NativeLibrary()


@pytest.fixture
def handle(native, strict_fmu):
    handle = native._dll.ovfmi_native_create(os.fsencode(strict_fmu[1]))
    assert handle, str(native.error())
    try:
        yield handle
    finally:
        native._dll.ovfmi_native_destroy(handle)


def get_real(native, handle, name):
    value = ctypes.c_double()
    status = native._dll.ovfmi_native_get_real(handle, name.encode(), ctypes.byref(value))
    assert status == 0, str(native.error())
    return value.value


def test_fixture_initialization_contract(strict_fmu):
    version, _, library = strict_fmu
    dll = ctypes.CDLL(str(library))
    component = ctypes.c_void_p
    refs = ctypes.POINTER(ctypes.c_uint)
    values = ctypes.POINTER(ctypes.c_double)
    if version == 2:
        create = dll.fmi2Instantiate
        create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_char_p, component, ctypes.c_int, ctypes.c_int]
        create.restype = component
        handle = create(b"control", 1, b"strict-lifecycle", None, None, 0, 0)
        enter = dll.fmi2EnterInitializationMode
        enter.argtypes = [component]
        enter_args = [handle]
        setter, getter = dll.fmi2SetReal, dll.fmi2GetReal
        setter.argtypes = getter.argtypes = [component, refs, ctypes.c_size_t, values]
        extra = []
    else:
        create = dll.fmi3InstantiateCoSimulation
        create.argtypes = ([ctypes.c_char_p] * 3 + [ctypes.c_bool] * 4
                           + [refs, ctypes.c_size_t] + [component] * 3)
        create.restype = component
        handle = create(b"control", b"strict-lifecycle", None, False, False, False, False,
                        None, 0, None, None, None)
        enter = dll.fmi3EnterInitializationMode
        enter.argtypes = [component, ctypes.c_bool, ctypes.c_double,
                          ctypes.c_double, ctypes.c_bool, ctypes.c_double]
        enter_args = [handle, False, 0, 0, False, 0]
        setter, getter = dll.fmi3SetFloat64, dll.fmi3GetFloat64
        setter.argtypes = getter.argtypes = [component, refs, ctypes.c_size_t, values, ctypes.c_size_t]
        extra = [1]
    exit_initialization = getattr(dll, f"fmi{version}ExitInitializationMode")
    free = getattr(dll, f"fmi{version}FreeInstance")
    reset = getattr(dll, f"fmi{version}Reset")
    exit_initialization.argtypes = free.argtypes = reset.argtypes = [component]
    free.restype = None
    assert handle
    try:
        rate, value = ctypes.c_uint(0), ctypes.c_double(3)
        status = setter(handle, ctypes.byref(rate), 1, ctypes.byref(value), *extra)
        assert status == (3 if version == 2 else 0)
        assert reset(handle) == 0
        read_value = ctypes.c_double()
        status = getter(handle, ctypes.byref(rate), 1, ctypes.byref(read_value), *extra)
        assert status == (3 if version == 2 else 0)
        assert reset(handle) == 0
        guess = ctypes.c_uint(3)
        assert setter(handle, ctypes.byref(guess), 1, ctypes.byref(value), *extra) == 0
        assert enter(*enter_args) == 0
        assert setter(handle, ctypes.byref(rate), 1, ctypes.byref(value), *extra) == 0
        assert exit_initialization(handle) == 0
        initial_rate = ctypes.c_uint(4)
        assert getter(handle, ctypes.byref(initial_rate), 1, ctypes.byref(value), *extra) == 0
        assert value.value == 3
        assert reset(handle) == 0
        assert enter(*enter_args) == 0
        assert setter(handle, ctypes.byref(guess), 1, ctypes.byref(value), *extra) == 3
    finally:
        free(handle)


def test_abi_initial_inputs_and_later_start_values(native, handle, strict_fmu):
    dll = native._dll
    for name, value in [("rate", 2), ("rate", 3), ("gain", 2), ("integral", 4), ("guess", 6)]:
        assert dll.ovfmi_native_set_real(handle, name.encode(), value) == 0, str(native.error())
    if strict_fmu[0] == 3:
        assert get_real(native, handle, "rate") == 3
        assert get_real(native, handle, "integral") == 4
    assert dll.ovfmi_native_set_real(handle, b"missing", 1) == -1
    assert "unknown FMI variable" in str(native.error())
    assert dll.ovfmi_native_step(handle, 0.125) == 0, str(native.error())
    assert get_real(native, handle, "gain") == 2
    assert get_real(native, handle, "guess") == 6
    assert get_real(native, handle, "initial_rate") == 3
    assert get_real(native, handle, "integral") == 4.75
    assert dll.ovfmi_native_set_real(handle, b"rate", 4) == 0, str(native.error())
    assert dll.ovfmi_native_step(handle, 0.25) == 0, str(native.error())
    assert get_real(native, handle, "integral") == 5.75
    assert get_real(native, handle, "initial_rate") == 3


def test_abi_reset_discards_pending_and_applied_inputs(native, handle):
    dll = native._dll
    assert dll.ovfmi_native_set_real(handle, b"rate", 9) == 0, str(native.error())
    assert dll.ovfmi_native_reset(handle) == 0, str(native.error())
    for rate in [None, 3, None]:
        if rate is not None:
            assert dll.ovfmi_native_set_real(handle, b"rate", rate) == 0, str(native.error())
        assert dll.ovfmi_native_step(handle, 0.125) == 0, str(native.error())
        assert get_real(native, handle, "integral") == (rate or 1) * 0.125
        assert dll.ovfmi_native_reset(handle) == 0, str(native.error())


@pytest.mark.parametrize("strict_fmu", [2], indirect=True, ids=["fmi2"])
def test_rejected_pending_input_requires_reset(native, handle, strict_fmu):
    dll = native._dll
    assert dll.ovfmi_native_set_real(handle, b"rate", -1) == 0, str(native.error())
    assert dll.ovfmi_native_step(handle, 0.125) == -1
    assert "fmi2SetReal failed" in str(native.error())
    assert dll.ovfmi_native_set_real(handle, b"rate", 3) == -1
    assert "FMU failed" in str(native.error())
    assert dll.ovfmi_native_step(handle, 0.125) == -1
    assert "FMU failed" in str(native.error())
    assert dll.ovfmi_native_reset(handle) == 0, str(native.error())
    assert dll.ovfmi_native_step(handle, 0.125) == 0, str(native.error())
    assert get_real(native, handle, "integral") == 0.125


def test_native_instance_live_input_overrides_start_value(native, strict_fmu):
    instance = NativeInstance(native, None, strict_fmu[1])
    try:
        for _ in range(2):
            instance.set_start_values({"rate": 2, "gain": 2, "integral": 4, "guess": 6})
            result = instance.step(0.125, {"rate": 3}, ["integral", "initial_rate"])
            assert result["integral"] == 4.75
            assert result["initial_rate"] == 3
            instance.reset()
    finally:
        instance.close()


def test_public_host_restores_authored_inputs_and_accepts_live_override(native, strict_fmu, tmp_path):
    source = tmp_path / "strict.usda"
    source.write_text(f'''#usda 1.0
def Xform "World"
{{
    custom double probe:input = 2
    custom double probe:output = 0
    def FmuInstance "Integrator"
    {{
        asset fmi:fmu = @{strict_fmu[1].as_posix()}@
        def FmuConnection "Signals"
        {{
            rel fmi:targets = </World>
            def FmuMapping "Input"
            {{
                token fmi:direction = "input"
                token fmi:fmuAttribute = "rate"
                token fmi:usdAttribute = "probe:input"
                int2 fmi:usdMapping = (0, 1)
            }}
            def FmuMapping "Output"
            {{
                token fmi:direction = "output"
                token fmi:fmuAttribute = "integral"
                token fmi:usdAttribute = "probe:output"
                int2 fmi:usdMapping = (0, 1)
            }}
        }}
    }}
}}
''', encoding="utf-8")
    with FmiHost(FmiHostConfig(backend="native")) as host:
        host.attach_ovstage(object(), source_asset=source)
        for rate in [3, None, 4]:
            if rate is not None:
                host.wait_op(host.write([AttributeWrite(("/World",), "probe:input", [[rate]])]))
            host.step_sync(0.125)
            with host.read(attribute_names=["probe:output"]) as result:
                assert result.groups[0].tensors[0][0, 0] == (rate or 2) * 0.125
            host.reset()
