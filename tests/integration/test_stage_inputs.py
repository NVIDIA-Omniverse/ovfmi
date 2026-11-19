# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage columns retain row boundaries without requiring ovstage in CPU CI."""

from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from ovfmi import AttributeWrite, FmiHost, FmiHostConfig, MissingInputPolicy
from ovfmi.routing import InputRouter, _read_stage_attribute
from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend


VALUE = 0.12345678912345678


def column(values, shape, lanes, prim_indices, data_indices):
    tensor = SimpleNamespace(shape=shape, ndim=len(shape),
                             dtype=SimpleNamespace(lanes=lanes))
    return SimpleNamespace(
        is_delete=False, is_array=False, tensor_count=1, prim_list=1,
        prim_count=len(prim_indices), prim_index=prim_indices.__getitem__,
        data_row_index=data_indices.__getitem__, tensor=lambda _index: tensor,
        array=lambda _index: np.asarray(values).reshape(-1),
    )


@pytest.fixture
def stage_factory(monkeypatch):
    def make(groups, paths):
        groups = list(groups)
        # Prim and payload maps belong to distinct objects in ovstage 0.4.
        # Deliberately omit the old flattened fields from the group.
        for index, payload in enumerate(groups):
            groups[index] = SimpleNamespace(
                columns=(payload,), prim_list=payload.prim_list,
                prim_count=payload.prim_count, prim_index=payload.prim_index,
            )
        events = []
        dictionary = SimpleNamespace(
            create_path_list_from_strings=lambda names: tuple(names),
            intern_token=lambda name: name,
            get_path_strings=lambda _paths: paths,
            destroy_path_list=lambda _paths: events.append("paths"),
            destroy=lambda: events.append("dictionary"),
        )
        read = SimpleNamespace(wait=lambda: None, groups=lambda: groups)
        stage = SimpleNamespace(
            query_from_path_list=lambda paths: paths,
            read_attributes=lambda *_args: nullcontext(read),
            release_group=lambda group: events.append(
                group.columns[0]),
            release_query=lambda _query: SimpleNamespace(
                wait=lambda: events.append("query")),
        )
        monkeypatch.setitem(sys.modules, "ovstage", SimpleNamespace(
            PathDictionary=lambda _stage: dictionary,
            OrdinalRange=SimpleNamespace(latest=lambda ordinal: ordinal),
        ))
        return stage, events
    return make


@pytest.mark.parametrize("dtype,value", [
    (np.float64, VALUE), (np.float32, 0.125),
    (np.int64, 2**53 + 1), (np.uint64, 2**63 + 1),
])
@pytest.mark.parametrize("shape,lanes", [((3,), 3), ((3, 3), 1)])
def test_stage_rows_keep_precision_and_independent_indices(stage_factory, dtype, value, shape, lanes):
    values = np.array([[1, value, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
    group = column(values, shape, lanes, [2, 0], [0, 2])
    stage, events = stage_factory([group], ["/A", "/Unused", "/B"])
    rows = _read_stage_attribute(stage, ["/A", "/B"], "input", 2)
    assert rows == {"/B": values[0].tolist(), "/A": values[2].tolist()}
    assert type(rows["/B"][1]) is type(values[0, 1].item())
    assert events == [group, "query", "paths", "dictionary"]


@pytest.mark.parametrize("shape,lanes", [((2,), 16), ((2, 4, 4), 1), ((2, 4), 4)])
@pytest.mark.parametrize("offset,count", [(13, 1), (12, 3), (0, 0)])
def test_matrix_component_ranges_keep_row_order(stage_factory, shape, lanes, offset, count):
    values = np.arange(32, dtype=np.float64).reshape(2, 16)
    values[1, 13] = VALUE
    group = column(values, shape, lanes, [1, 2], [0, 1])
    stage, _events = stage_factory([group], ["/Unused", "/A", "/B"])
    router = InputRouter({}, MissingInputPolicy.ERROR)
    router._maps["/Model"] = {"input": ("/B", "matrix", offset, count)}
    router.update_from_stage(stage, 1, 2)
    instance = SimpleNamespace(get_parser_instance=lambda: SimpleNamespace(path="/Model"))
    expected = values[1].tolist()
    if count == 1:
        expected = expected[offset]
    elif count:
        expected = expected[offset:offset + count]
    assert router.get_inputs_for(instance) == {"input": expected}


def test_stage_groups_skip_unsupported_and_empty_data(stage_factory):
    groups = [column([], (0,), 3, [], []) for _ in range(4)]
    groups[0].is_delete = True
    groups[1].is_array = True
    groups[2].tensor_count = 2
    valid = column([VALUE], (1,), 1, [0], [0])
    stage, events = stage_factory([*groups, valid], ["/A"])
    assert _read_stage_attribute(stage, ["/A"], "input", 2) == {"/A": [VALUE]}
    assert events == [*groups, valid, "query", "paths", "dictionary"]


def test_stage_reader_releases_resources_on_invalid_column(stage_factory):
    group = column([1, 2], (1,), 3, [0], [0])
    stage, events = stage_factory([group], ["/A"])
    with pytest.raises(ValueError):
        _read_stage_attribute(stage, ["/A"], "input", 2)
    assert events == [group, "query", "paths", "dictionary"]


def scene(fmu, count, width, offset):
    prims = []
    declaration = "double input = 0" if width == 1 else "double3 input = (0, 0, 0)"
    for index in range(count):
        prims.append(f'''
def Xform "State{index}"
{{
    custom {declaration}
    custom double output = 0
}}
def FmuInstance "Model{index}"
{{
    asset fmi:fmu = @{fmu.as_posix()}@
    def FmuConnection "Signals"
    {{
        rel fmi:targets = </World/State{index}>
        def FmuMapping "Input"
        {{
            token fmi:direction = "input"
            token fmi:fmuAttribute = "pos_x"
            token fmi:usdAttribute = "input"
            int2 fmi:usdMapping = ({offset}, 1)
        }}
        def FmuMapping "Output"
        {{
            token fmi:direction = "output"
            token fmi:fmuAttribute = "force_x"
            token fmi:usdAttribute = "output"
            int2 fmi:usdMapping = (0, 1)
        }}
    }}
}}
''')
    return '#usda 1.0\ndef Xform "World"\n{\n' + "".join(prims) + "}\n"


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
@pytest.mark.parametrize("values,offset", [
    ([[1, VALUE, 3]], 1), ([[1, VALUE, 3], [7, 0.25, 9]], 0), ([[VALUE], [7]], 0),
])
def test_public_stage_input_and_write_control(stage_factory, tmp_path, monkeypatch, backend, fmi_version, values, offset):
    require_backend(backend)
    usd_python = _env_from_demo_setup().get("USD_PYTHON")
    if not usd_python:
        pytest.skip("USD_PYTHON is not configured")
    monkeypatch.setenv("USD_PYTHON", usd_python)
    fmu = ROOT / "demoapp/usd" / ("PDController.fmu" if fmi_version == 2 else "PDController.fmu3")
    if not fmu.exists():
        pytest.skip("run demoapp setup to build the example FMUs")
    values = np.array(values, dtype=np.float64)
    paths = tuple(f"/World/State{i}" for i in range(len(values)))
    order = list(reversed(range(len(paths))))
    group = column(values, (len(paths),), values.shape[1], order, order)
    stage, _events = stage_factory([group], paths)
    source = tmp_path / "inputs.usda"
    source.write_text(scene(fmu, len(paths), values.shape[1], offset), encoding="utf-8")

    def outputs(host):
        with host.read(attribute_names=["output"]) as read:
            result = {}
            for group in read.groups:
                assert group.tensors[0].dtype == np.float64
                result.update(zip(group.prim_paths, group.tensors[0][:, 0].tolist()))
            return result

    with FmiHost(FmiHostConfig(backend=backend)) as host:
        host.attach_ovstage(stage, source_asset=source)
        for _replay in range(2):
            for updated in (values, values + 0.5):
                group.array = lambda _index: updated.reshape(-1)
                host.update_from_ovstage(1, 2)
                host.step_sync(0.125)
                expected = {p: -50 * updated[i, offset] for i, p in enumerate(paths)}
                assert outputs(host) == expected
                written = updated + 0.25
                host.wait_op(host.write([AttributeWrite(paths, "input", written)]))
                host.step_sync(0.125)
                assert outputs(host) == {p: -50 * written[i, offset] for i, p in enumerate(paths)}
            host.reset()
            host.step_sync(0.125)
            assert outputs(host) == dict.fromkeys(paths, 0.0)
            host.reset()
