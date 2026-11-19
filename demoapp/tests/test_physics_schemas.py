# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from physics_schemas import register_physx_schemas


def test_registers_both_external_schema_families(monkeypatch):
    calls = []
    roots = [Path("provider/PhysxSchema/resources"),
             Path("provider/OmniUsdPhysicsDeformableSchema/resources")]
    monkeypatch.setitem(sys.modules, "ovphysx.samples.python_samples.schema_support",
                        SimpleNamespace(physx_schema_paths=lambda: roots))
    stage = SimpleNamespace(population=SimpleNamespace(register_usd_schemas=calls.append))
    register_physx_schemas(stage)
    assert calls == [[str(path) for path in roots]]


def test_missing_provider_is_not_silently_ignored(monkeypatch):
    def missing():
        raise FileNotFoundError("Install physx-usd-schemas")
    monkeypatch.setitem(sys.modules, "ovphysx.samples.python_samples.schema_support",
                        SimpleNamespace(physx_schema_paths=missing))
    with pytest.raises(FileNotFoundError, match="physx-usd-schemas"):
        register_physx_schemas(SimpleNamespace(population=SimpleNamespace(register_usd_schemas=lambda _: None)))


def test_registration_errors_propagate(monkeypatch):
    monkeypatch.setitem(sys.modules, "ovphysx.samples.python_samples.schema_support",
                        SimpleNamespace(physx_schema_paths=lambda: [Path("schemas")]))
    def fail(_paths):
        raise RuntimeError("registration was too late")
    with pytest.raises(RuntimeError, match="too late"):
        register_physx_schemas(SimpleNamespace(population=SimpleNamespace(register_usd_schemas=fail)))
