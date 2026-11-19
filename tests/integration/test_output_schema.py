# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output baseline metadata stays numeric and separate from missing inputs."""

import json

import pytest

from probe_output_baselines import output_scene
from test_output_baselines import _run


def _parse(tmp_path, declaration, direction="output"):
    source = tmp_path / "output.usda"
    source.write_text(
        output_scene(tmp_path / "unused.fmu", declaration, 1, 1).replace(
            'token fmi:direction = "output"', f'token fmi:direction = "{direction}"',
        ), encoding="utf-8",
    )
    return json.loads(_run(
        "import json; from ovfmi.population import parse_source; "
        f"print(json.dumps(parse_source({str(source)!r})))"
    ))


@pytest.mark.parametrize("declaration,defaults", [
    ("", {}),
    ("custom double3 probe:output", {"/World": {"probe:output": [0.0, 0.0, 0.0]}}),
])
def test_missing_output_value(tmp_path, declaration, defaults):
    parsed = _parse(tmp_path, declaration)
    assert parsed["initial_values"] == {"/World": {"probe:input": [0.0]}}
    assert parsed["output_defaults"] == defaults


@pytest.mark.parametrize("width", [2, 3, 4])
def test_output_matrix_baseline_is_flat(tmp_path, width):
    values = list(range(1, width * width + 1))
    rows = [str(tuple(values[start:start + width])) for start in range(0, len(values), width)]
    declaration = f"custom matrix{width}d probe:output = ({', '.join(rows)})"
    parsed = _parse(tmp_path, declaration)
    assert parsed["initial_values"]["/World"]["probe:output"] == values


@pytest.mark.parametrize("declaration", [
    "custom quatf probe:output = (1, 0, 0, 0)",
    "custom quatd probe:output = (1, 0, 0, 0)",
    "custom float3[] probe:output = [(11, 0, 33)]",
    "custom double[] probe:output = [11, 0, 33]",
    "custom token probe:output = \"value\"",
    "custom string probe:output = \"value\"",
    "custom asset probe:output = @unused@",
    "custom timecode probe:output = 2",
])
def test_unsupported_output_baseline_is_omitted(tmp_path, declaration):
    parsed = _parse(tmp_path, declaration)
    assert parsed["initial_values"] == {"/World": {"probe:input": [0.0]}}
    assert parsed["output_defaults"] == {}


def test_unsupported_input_still_fails(tmp_path):
    with pytest.raises(AssertionError, match="Quatf is not JSON serializable"):
        _parse(tmp_path, "custom quatf probe:output = (1, 0, 0, 0)", direction="input")
