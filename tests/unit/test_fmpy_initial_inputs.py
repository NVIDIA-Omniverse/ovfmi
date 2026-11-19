# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialization combines live inputs with defaults without mutating them."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def test_live_initialization_values_preserve_defaults(monkeypatch):
    pytest.importorskip("fmpy")
    from ovfmi import _fmpy_runtime as runtime

    variables = [
        SimpleNamespace(name="label", causality="parameter", variability="fixed",
                        type="String", valueReference=1, dimensions=[]),
        SimpleNamespace(name="size", causality="structuralParameter", variability="fixed",
                        type="UInt32", valueReference=2, dimensions=[]),
    ]
    fmu = Mock()
    instance = runtime.FmuRuntimeInstance(None, SimpleNamespace(modelVariables=variables), fmu)
    defaults = {"label": "authored", "size": 1, "gain": 50.0, "pos_x": 0.0}
    instance.set_start_values(defaults)
    simulate = Mock(return_value=np.array([(0.01,)], dtype=[("time", np.float64)]))
    monkeypatch.setattr(runtime.fmpy, "simulate_fmu", simulate)
    monkeypatch.setattr(runtime, "fmpy_resolve_model_description_shapes", lambda *_: None)

    instance.step("unused.fmu", {"label": "live", "size": 2, "pos_x": 6.0}, [], 0.01)
    assert simulate.call_args.kwargs["start_values"] == {
        "label": "live", "size": 2, "gain": 50.0, "pos_x": 6.0}
    fmu.setString.assert_called_once_with([1], ["live"])
    fmu.setUInt32.assert_called_once_with([2], [2])
    assert defaults == {"label": "authored", "size": 1, "gain": 50.0, "pos_x": 0.0}

    instance.reset()
    instance.step("unused.fmu", {}, [], 0.01)
    assert simulate.call_args.kwargs["start_values"] == defaults


@pytest.fixture
def start_runtime(monkeypatch):
    pytest.importorskip("fmpy")
    from fmpy.model_description import ModelDescription
    from fmpy.simulation import apply_start_values
    from ovfmi import _fmpy_runtime as runtime

    def create(variable):
        fmu = Mock()
        instance = runtime.FmuRuntimeInstance(
            None, ModelDescription(fmiVersion="3.0", modelVariables=[variable]), fmu)

        def simulate(**kwargs):
            apply_start_values(fmu, kwargs["model_description"], kwargs["start_values"])
            return np.array([(0.01,)], dtype=[("time", np.float64)])

        monkeypatch.setattr(runtime.fmpy, "simulate_fmu", simulate)
        return instance, fmu

    return create


@pytest.mark.parametrize("variable_type,value", [
    ("Real", -0.125), ("Float32", -0.125), ("Float64", -0.125),
    ("Integer", -3), ("Int32", -3), ("UInt32", 3),
    ("Boolean", False), ("Boolean", True), ("String", "label"),
])
@pytest.mark.parametrize("source", ["authored", "live"])
def test_whole_scalar_start_values(start_runtime, variable_type, value, source):
    from fmpy.model_description import ScalarVariable

    variable = ScalarVariable(name="input", valueReference=1, type=variable_type,
                              python_type=type(value), causality="input")
    instance, fmu = start_runtime(variable)
    values = {"input": [value]}
    instance.set_start_values(values if source == "authored" else {})
    instance.step("unused.fmu", values if source == "live" else {}, [], 0.01)

    setter = getattr(fmu, "set" + variable_type)
    setter.assert_called_once_with([1], [value])
    assert type(setter.call_args.args[1][0]) is type(value)
    assert values == {"input": [value]}


@pytest.mark.parametrize("values", [[0.125], [0.125, 0.25]])
def test_array_start_values_keep_their_shape(start_runtime, values):
    from fmpy.model_description import Dimension, ScalarVariable

    variable = ScalarVariable(name="input", valueReference=1, type="Float64",
                              python_type=float, causality="input", shape=(len(values),),
                              dimensions=[Dimension(start=len(values), valueReference=None)])
    instance, fmu = start_runtime(variable)
    instance.step("unused.fmu", {"input": values}, [], 0.01)
    fmu.setFloat64.assert_called_once_with([1], values)


@pytest.mark.parametrize("values", [[], [False, True]])
def test_scalar_start_values_reject_invalid_lengths(start_runtime, values):
    from fmpy.model_description import ScalarVariable

    variable = ScalarVariable(name="input", valueReference=1, type="Boolean",
                              python_type=bool, causality="input")
    instance, fmu = start_runtime(variable)
    with pytest.raises(ValueError, match="input.*exactly one value"):
        instance.step("unused.fmu", {"input": values}, [], 0.01)
    fmu.setBoolean.assert_not_called()
