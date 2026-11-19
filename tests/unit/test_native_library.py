# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import ovfmi._native_runtime as native


@pytest.fixture(params=[("linux", "libovfmi_native.so"), ("win32", "ovfmi_native.dll")])
def library_layout(request, monkeypatch, tmp_path):
    system, name = request.param
    package = tmp_path / "python"
    package.mkdir()
    monkeypatch.setattr(native, "__file__", str(package / "_native_runtime.py"))
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform=system))
    monkeypatch.delenv("OVFMI_NATIVE_LIBRARY", raising=False)
    return tmp_path, package, name


def checkout_library(root, name, mode="release"):
    manifest = root / "native" / "Cargo.toml"
    manifest.parent.mkdir(exist_ok=True)
    manifest.touch()
    library = root / "native" / "target" / mode / name
    library.parent.mkdir(parents=True)
    library.touch()
    return library


def test_installed_library_precedes_checkout(library_layout):
    root, package, name = library_layout
    checkout_library(root, name)
    bundled = package / name
    bundled.touch()

    assert native._library_path() == bundled


def test_explicit_library_precedes_bundled(library_layout, monkeypatch):
    root, package, name = library_layout
    (package / name).touch()
    override = root / "deployment" / name
    monkeypatch.setenv("OVFMI_NATIVE_LIBRARY", str(override))

    assert native._library_path() == override


@pytest.mark.parametrize("mode", ["release", "debug"])
def test_editable_library_fallback(library_layout, mode):
    root, _, name = library_layout
    library = checkout_library(root, name, mode)

    assert native._library_path() == library


def test_missing_installed_library_rejects_unrelated_target(library_layout):
    root, package, name = library_layout
    unrelated = checkout_library(root, name)
    (root / "native" / "Cargo.toml").unlink()
    (package / name).mkdir()

    with pytest.raises(RuntimeError, match="native library is not built"):
        native._library_path()
    assert unrelated.is_file()


def test_installed_library_rejects_neighboring_checkout(library_layout, monkeypatch):
    root, package, name = library_layout
    installed = root / "ovfmi"
    package.rename(installed)
    monkeypatch.setattr(native, "__file__", str(installed / "_native_runtime.py"))
    checkout_library(root, name)

    with pytest.raises(RuntimeError, match="native library is not built"):
        native._library_path()
