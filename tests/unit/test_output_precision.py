# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing preserves backend numeric types instead of forcing Float32."""

from types import SimpleNamespace

import numpy as np
import pytest

from ovfmi.routing import InputRouter, OutputRouter
from ovfmi import AttributeWrite, MissingInputPolicy


def router_with_targets(targets, baseline=None):
    inputs = InputRouter(baseline or {}, MissingInputPolicy.USE_FMU_START_VALUE)
    router = OutputRouter(baseline or {}, inputs, strict=True)
    router._maps["/Fmu"] = [("out", targets)]
    instance = SimpleNamespace(get_parser_instance=lambda: SimpleNamespace(path="/Fmu"))
    return router, instance


def emit(router, instance, values, dtype):
    result = np.zeros((), dtype=[("out", dtype, (len(values),))])
    result["out"] = values
    router.write_outputs(instance, ["out"], result)


@pytest.mark.parametrize("dtype,value", [
    (np.float64, -6.172839456172839), (np.float32, 0.125),
    (np.int32, 16777217), (np.bool_, True),
])
@pytest.mark.parametrize("count", [0, 1])
def test_scalar_output_preserves_type_and_value(dtype, value, count):
    router, instance = router_with_targets([("/A", "result", 0, count)])
    emit(router, instance, [value], dtype)
    tensor = router.snapshot()[0].tensors[0]
    assert tensor.dtype == np.dtype(dtype)
    assert tensor[0, 0] == value


def test_mixed_rows_promote_without_losing_double_precision():
    router, instance = router_with_targets([("/A", "result", 0, 0)])
    value = 0.12345678912345678
    emit(router, instance, [value], np.float64)
    router._maps["/Fmu"] = [("out", [("/B", "result", 0, 0)])]
    emit(router, instance, [0.25], np.float32)
    tensor = router.snapshot()[0].tensors[0]
    assert tensor.dtype == np.float64
    assert tensor[:, 0].tolist() == [value, 0.25]
    router.reset()
    assert not router.snapshot()
    emit(router, instance, [0.5], np.float32)
    assert router.snapshot()[0].tensors[0].dtype == np.float32


def test_partial_mapping_preserves_unmapped_baseline():
    value = 0.12345678912345678
    router, instance = router_with_targets(
        [("/A", "result", 1, 1)], {"/A": {"result": [value, 0.0, 2.0]}})
    emit(router, instance, [0.25], np.float32)
    tensor = router.snapshot()[0].tensors[0]
    assert tensor.dtype == np.float64
    assert tensor.tolist() == [[value, 0.25, 2.0]]


@pytest.mark.parametrize("prior_output", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_partial_output_preserves_current_inputs(prior_output, count):
    router, instance = router_with_targets(
        [("/A", "result", 1, count)], {"/A": {"result": [11.0, 0.0, 0.0, 33.0]}})
    inputs = router._input_router
    inputs._maps["/Fmu"] = {"in": ("/A", "result", 0, 0)}
    if prior_output:
        emit(router, instance, [-5.0] * count, np.float64)
    current = [0.12345678912345678, 100.0, 200.0, 44.0]
    inputs.write([AttributeWrite(("/A",), "result", [current])])
    produced = [-25.0, -50.0][:count]
    emit(router, instance, produced, np.float64)
    expected = list(current)
    expected[1:1 + count] = produced
    assert inputs.get_inputs_for(instance)["in"] == expected
    assert router.snapshot()[0].tensors[0].tolist() == [expected]


@pytest.mark.parametrize("dtype,values", [
    (np.float32, [0.125, 0.25]),
    (np.int64, [2**53 + 1, 2**53 + 3]),
])
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_disjoint_outputs_preserve_values_and_types(dtype, values, order):
    router, first = router_with_targets([("/A", "result", 0, 1)])
    second = SimpleNamespace(get_parser_instance=lambda: SimpleNamespace(path="/Other"))
    router._maps["/Other"] = [("out", [("/A", "result", 1, 1)])]
    for index in order:
        emit(router, (first, second)[index], [values[index]], dtype)
    tensor = router.snapshot()[0].tensors[0]
    assert tensor.dtype == np.dtype(dtype)
    assert tensor.tolist() == [values]


def test_whole_output_replaces_current_inputs():
    router, instance = router_with_targets(
        [("/A", "result", 0, 0)], {"/A": {"result": [11.0, 0.0, 33.0]}})
    inputs = router._input_router
    inputs._maps["/Fmu"] = {"in": ("/A", "result", 0, 0)}
    inputs.write([AttributeWrite(("/A",), "result", [[0.125, 100.0, 44.0]])])
    emit(router, instance, [0.25, 0.5], np.float32)
    assert inputs.get_inputs_for(instance)["in"] == [0.25, 0.5]
    tensor = router.snapshot()[0].tensors[0]
    assert tensor.dtype == np.float32
    assert tensor.tolist() == [[0.25, 0.5]]


def test_partial_output_reset_restores_baseline():
    baseline = [11.0, 0.0, 33.0]
    router, instance = router_with_targets(
        [("/A", "result", 1, 1)], {"/A": {"result": baseline}})
    inputs = router._input_router
    inputs.write([AttributeWrite(("/A",), "result", [[0.125, 100.0, 44.0]])])
    emit(router, instance, [-25.0], np.float64)
    previous = router.snapshot()[0].tensors[0]
    inputs.reset()
    router.reset()
    assert not router.snapshot()
    emit(router, instance, [-50.0], np.float64)
    assert router.snapshot()[0].tensors[0].tolist() == [[11.0, -50.0, 33.0]]
    assert previous.tolist() == [[0.125, -25.0, 44.0]]
    assert baseline == [11.0, 0.0, 33.0]


@pytest.mark.parametrize("prior_output", [False, True])
def test_partial_output_ignores_incomplete_inputs(prior_output):
    router, instance = router_with_targets(
        [("/A", "result", 2, 1)], {"/A": {"result": [11.0, 0.0, 33.0]}})
    if prior_output:
        emit(router, instance, [44.0], np.float64)
    router._maps["/Fmu"] = [("out", [("/A", "result", 1, 1)])]
    router._input_router.update_value("/A", "result", [0.125])
    emit(router, instance, [-25.0], np.float64)
    assert router.snapshot()[0].tensors[0].tolist() == [
        [11.0, -25.0, 44.0 if prior_output else 33.0]]


def test_caller_float32_values_preserve_double_snapshot():
    router, instance = router_with_targets(
        [("/A", "result", 1, 1)], {"/A": {"result": [11.0, 22.0, 33.0]}})
    router._input_router.write([
        AttributeWrite(("/A",), "result", np.asarray([[0.125, 100.0, 0.25]], dtype=np.float32))])
    emit(router, instance, [-25.0], np.float32)
    tensor = router.snapshot()[0].tensors[0]
    assert tensor.dtype == np.float64
    assert tensor.tolist() == [[0.125, -25.0, 0.25]]


@pytest.mark.parametrize("source", ["caller", "output"])
@pytest.mark.parametrize("dtype,values,scalar_type", [
    (np.bool_, [True, False], bool),
    (np.float32, [0.125, 0.25], float),
    (np.int64, [2**53 + 1, 2**53 + 3], int),
])
def test_backend_inputs_use_python_scalars(source, dtype, values, scalar_type):
    router, instance = router_with_targets([("/A", "result", 0, 0)])
    inputs = router._input_router
    inputs._maps["/Fmu"] = {
        "scalar": ("/A", "result", 0, 1),
        "range": ("/A", "result", 0, 2),
        "whole": ("/A", "result", 0, 0),
    }
    if source == "caller":
        inputs.write([AttributeWrite(("/A",), "result", np.asarray([values], dtype=dtype))])
    else:
        emit(router, instance, values, dtype)
    start = {}
    instance.set_start_values = start.update
    inputs.write_start_values(instance)
    for actual in (start, inputs.get_inputs_for(instance)):
        assert actual == {"scalar": values[0], "range": values, "whole": values}
        assert type(actual["scalar"]) is scalar_type
        assert all(type(value) is scalar_type for value in actual["range"] + actual["whole"])
    if source == "output":
        assert router.snapshot()[0].tensors[0].dtype == np.dtype(dtype)


def test_different_widths_preserve_each_type():
    router, instance = router_with_targets([("/A", "result", 0, 0)])
    emit(router, instance, [0.25], np.float32)
    router._maps["/Fmu"] = [("out", [("/B", "result", 0, 0)])]
    emit(router, instance, [0.12345678912345678, 2.0], np.float64)
    groups = router.snapshot()
    assert len(groups) == 2
    assert all(not group.is_array for group in groups)
    assert [group.prim_paths for group in groups] == [("/A",), ("/B",)]
    assert [group.tensors[0].dtype for group in groups] == [np.float32, np.float64]
    assert groups[0].tensors[0].tolist() == [[0.25]]
    assert groups[1].tensors[0].tolist() == [[0.12345678912345678, 2.0]]


def test_transform_stays_double_precision():
    router, instance = router_with_targets([("/A", "omni:xform", 0, 0)])
    emit(router, instance, np.eye(4).reshape(-1), np.float32)
    assert router.snapshot()[0].tensors[0].dtype == np.float64
