# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the native ABI's lifecycle/errors and both FMU storage formats."""

import ctypes
from pathlib import Path
import zipfile

import pytest

from ovfmi._native_runtime import NativeLibrary, _library_path


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native():
    try:
        _library_path()
    except RuntimeError:
        pytest.skip("native runtime has not been built")
    return NativeLibrary()


def test_invalid_handle_and_missing_archive(native, tmp_path):
    dll = native._dll
    assert not dll.ovfmi_native_create(str(tmp_path / "missing.fmu").encode())
    assert str(native.error())
    assert dll.ovfmi_native_step(None, 1) == -1
    assert "null" in str(native.error())
    assert dll.ovfmi_native_get_real(None, b"h", None) == -1
    dll.ovfmi_native_destroy(None)


@pytest.mark.parametrize("contents", [b"not a ZIP archive", b"PK\x03\x04\x00"])
def test_malformed_archive_reports_error(native, tmp_path, contents):
    path = tmp_path / "broken.fmu"
    path.write_bytes(contents)
    assert not native._dll.ovfmi_native_create(str(path).encode())
    assert str(native.error())


def test_archive_without_model_description_reports_error(native, tmp_path):
    path = tmp_path / "empty-model.fmu"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("resources/readme.txt", "Not an FMU")
    assert not native._dll.ovfmi_native_create(str(path).encode())
    assert str(native.error())


@pytest.mark.parametrize("archive", [False, True])
def test_fmi3_start_values_reset_and_errors(native, tmp_path, archive):
    source = ROOT / "demoapp/usd/BouncingBall.fmu3"
    if not source.exists():
        pytest.skip("build demo FMUs first")
    path = source
    if archive:
        path = tmp_path / "ball.fmu"
        with zipfile.ZipFile(path, "w") as fmu:
            for child in source.rglob("*"):
                if child.is_file():
                    fmu.write(child, child.relative_to(source))
    dll = native._dll
    handle = dll.ovfmi_native_create(str(path).encode())
    assert handle, str(native.error())
    try:
        value = ctypes.c_double()
        results = []
        for _ in range(2):
            assert dll.ovfmi_native_set_real(handle, b"h", 2) == 0
            assert dll.ovfmi_native_step(handle, 0.1) == 0, str(native.error())
            assert dll.ovfmi_native_get_real(handle, b"h", ctypes.byref(value)) == 0
            assert 1.9 < value.value < 2
            results.append(value.value)
            assert dll.ovfmi_native_step(handle, 0.1) == -1
            assert dll.ovfmi_native_step(handle, float("nan")) == -1
            assert dll.ovfmi_native_get_real(handle, b"missing", ctypes.byref(value)) == -1
            assert "unknown FMI variable" in str(native.error())
            assert dll.ovfmi_native_reset(handle) == 0
        assert results[0] == results[1]
    finally:
        dll.ovfmi_native_destroy(handle)


def test_unpacked_fmi2(native, tmp_path):
    source = ROOT / "demoapp/usd/MotorDrive.fmu"
    if not source.exists():
        pytest.skip("build demo FMUs first")
    with zipfile.ZipFile(source) as fmu:
        fmu.extractall(tmp_path)
    handle = native._dll.ovfmi_native_create(str(tmp_path).encode())
    assert handle, str(native.error())
    try:
        assert native._dll.ovfmi_native_step(handle, 0.01) == 0, str(native.error())
    finally:
        native._dll.ovfmi_native_destroy(handle)
