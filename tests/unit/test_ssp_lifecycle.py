# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSP lifecycle preserves initialization order and releases FMUs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("initialized,terminate_error", [(False, False), (True, False), (True, True)])
def test_destroy_always_frees_instances(tmp_path, initialized, terminate_error):
    pytest.importorskip("fmpy")
    from ovfmi._ssp_runtime import SspRuntimeInstance

    fmu = Mock()
    if terminate_error:
        fmu.terminate.side_effect = RuntimeError("FMU termination failed")
    fmu_dir = tmp_path / "component"
    ssp_dir = tmp_path / "ssp"
    fmu_dir.mkdir()
    ssp_dir.mkdir()
    instance = SspRuntimeInstance.__new__(SspRuntimeInstance)
    instance._components = [SimpleNamespace(fmu=fmu, initialized=initialized, unzipdir=str(fmu_dir))]
    instance._unzip_dir = str(ssp_dir)

    instance.destroy()

    fmu.freeInstance.assert_called_once_with()
    if initialized:
        fmu.terminate.assert_called_once_with()
    else:
        fmu.terminate.assert_not_called()
    assert not fmu_dir.exists()
    assert not ssp_dir.exists()


def test_cyclic_initialization_keeps_declaration_order():
    pytest.importorskip("fmpy")
    from ovfmi._ssp_runtime import SspRuntimeInstance

    from fmpy.ssp.ssd import Component, Connector

    components = [Component(name=name) for name in ("first", "second")]
    initialized = []
    for component in components:
        component.fmu = Mock()
        component.variables = {}
        component.connectors = [Connector(name=kind, kind=kind) for kind in ("input", "output")]
        for connector in component.connectors:
            connector.parent = component
            connector.value = 0.0
        component.fmu.exitInitializationMode.side_effect = lambda name=component.name: initialized.append(name)
    instance = SspRuntimeInstance.__new__(SspRuntimeInstance)
    instance._components = components
    instance._system_inputs = []
    instance._ready_sources = set()
    first, second = components
    instance._connections = [(first.connectors[1], second.connectors[0]),
                             (second.connectors[1], first.connectors[0])]

    instance._initialize_components()

    assert initialized == ["first", "second"]
