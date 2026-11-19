# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-width output rows remain associated with their destination prims."""

from types import SimpleNamespace

import numpy as np
import pytest

from ovfmi import MissingInputPolicy
from ovfmi.routing import InputRouter, OutputRouter


@pytest.fixture
def router():
    baseline = {
        "/Z": {"result": [0.0], "other": [0.0]},
        "/Y": {"result": [10.0, 0.0, 12.0]},
        "/A": {"result": [0.0]},
        "/B": {"result": [20.0, 0.0, 22.0]},
    }
    inputs = InputRouter(baseline, MissingInputPolicy.USE_FMU_START_VALUE)
    output = OutputRouter(baseline, inputs, strict=True)
    output._maps["/Fmu"] = [
        ("scalar_z", [("/Z", "result", 0, 1)]),
        ("vector_y", [("/Y", "result", 1, 1)]),
        ("scalar_a", [("/A", "result", 0, 1)]),
        ("vector_b", [("/B", "result", 1, 1)]),
        ("other", [("/Z", "other", 0, 1)]),
    ]
    instance = SimpleNamespace(get_parser_instance=lambda: SimpleNamespace(path="/Fmu"))
    names = output.get_outputs_for(instance)
    result = np.array([(1.0, 2.0, 3.0, 4.0, 5.0)],
                      dtype=[(name, np.float64) for name in names])[0]
    output.write_outputs(instance, names, result)
    return output


def test_mixed_widths_keep_complete_rows_and_prim_order(router):
    groups = router.snapshot()
    assert [(group.attribute_name, group.prim_paths) for group in groups] == [
        ("result", ("/Z", "/A")),
        ("result", ("/Y", "/B")),
        ("other", ("/Z",)),
    ]
    assert all(not group.is_array and len(group.tensors) == 1 for group in groups)
    assert [group.tensors[0].tolist() for group in groups] == [
        [[1.0], [3.0]],
        [[10.0, 2.0, 12.0], [20.0, 4.0, 22.0]],
        [[5.0]],
    ]


@pytest.mark.parametrize("paths,attributes,expected", [
    (["/B", "/Y"], ["result"], [("result", ("/Y", "/B"),
                                  [[10.0, 2.0, 12.0], [20.0, 4.0, 22.0]])]),
    (["/B", "/Z"], ["result"], [("result", ("/Z",), [[1.0]]),
                                  ("result", ("/B",), [[20.0, 4.0, 22.0]])]),
    (None, ["other"], [("other", ("/Z",), [[5.0]])]),
    (["/Z"], None, [("result", ("/Z",), [[1.0]]), ("other", ("/Z",), [[5.0]])]),
    ([], None, []),
    (None, [], []),
    (["/missing"], None, []),
    (None, ["missing"], []),
])
def test_filters_preserve_fixed_width_rows(router, paths, attributes, expected):
    groups = router.snapshot(paths, attributes)
    assert all(not group.is_array and len(group.tensors) == 1 for group in groups)
    assert [(group.attribute_name, group.prim_paths, group.tensors[0].tolist())
            for group in groups] == expected


def test_snapshot_values_are_owned(router):
    snapshot = router.snapshot()
    snapshot[0].tensors[0][0, 0] = 99.0
    assert router.snapshot()[0].tensors[0].tolist() == [[1.0], [3.0]]
    router.reset()
    assert not router.snapshot()
    assert snapshot[1].tensors[0].tolist() == [
        [10.0, 2.0, 12.0], [20.0, 4.0, 22.0],
    ]
