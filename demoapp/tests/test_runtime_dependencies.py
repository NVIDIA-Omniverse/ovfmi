# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency diagnostics use metadata without loading native libraries."""

import pytest

import runtime_dependencies as dependencies


def test_exact_pins_are_reported(monkeypatch, capsys):
    pins = dependencies.requirements()
    assert set(pins) == {"ovstage", "ovrtx", "ovphysx", "physx-usd-schemas"}
    monkeypatch.setattr(dependencies, "version", pins.__getitem__)
    assert dependencies.describe_runtime(check=True)
    output = capsys.readouterr().out
    assert all(f"{name}=={value}" in output for name, value in pins.items())
    assert "WARNING" not in output


@pytest.mark.parametrize("package", dependencies.requirements())
@pytest.mark.parametrize("missing", [False, True])
def test_incorrect_packages_fail_setup_check(monkeypatch, capsys, package, missing):
    pins = dependencies.requirements()
    def lookup(name):
        if name == package:
            if missing:
                raise dependencies.PackageNotFoundError(name)
            return "0.0.0"
        return pins[name]
    monkeypatch.setattr(dependencies, "version", lookup)
    assert not dependencies.describe_runtime()
    assert "WARNING" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="do not match"):
        dependencies.describe_runtime(check=True)
