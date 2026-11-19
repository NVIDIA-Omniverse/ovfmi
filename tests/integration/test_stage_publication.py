# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model outputs retain the numeric type of existing stage destinations."""

import subprocess
import sys

import pytest

from probe_first_step_inputs import scene
from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend_and_models  # noqa: F401


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
@pytest.mark.parametrize("attribute,usd_type,dtype", [
    ("inputs:intensity", "float", "float32"),
    ("probe:output", "float", "float32"),
    ("probe:output", "double", "float64"),
])
def test_existing_output_type(backend, fmi_version, attribute, usd_type, dtype, tmp_path):
    model = ROOT / "demoapp/usd" / ("PDController.fmu3" if fmi_version == 3 else "PDController.fmu")
    source = tmp_path / "publication.usda"
    text = scene(model).replace('def Xform "World"', 'def SphereLight "World"')
    text = text.replace("custom double probe:output = 0", f"custom {usd_type} probe:output = 101")
    text += scene(model).split("#usda 1.0\n", 1)[1].replace("World", "Other")
    source.write_text(text.replace("probe:output", attribute), encoding="utf-8")
    program = f'''
import numpy as np
import ovstage
from ovstage import population
from ovfmi import AttributeWrite, FmiHost, FmiHostConfig

with ovstage.Stage("ovfmi.publication") as stage:
    population.open_usd(stage, {str(source)!r}, ordinal=1)
    dictionary = ovstage.PathDictionary(stage)
    paths = dictionary.create_path_list_from_strings(["/World", "/Other"])
    query = stage.query_from_path_list(paths)
    attribute = dictionary.intern_token({attribute!r})

    def check_stage(ordinal, expected):
        with stage.read_attributes(query, [attribute], ovstage.OrdinalRange.latest(ordinal)) as read:
            read.wait()
            rows = {{}}
            for group in read.groups():
                try:
                    column = group.columns[0]
                    assert not column.is_array
                    array = np.asarray(column.array(0))
                    for i in range(group.prim_count):
                        path = dictionary.get_path_strings(group.prim_list)[group.prim_index(i)]
                        target_dtype = {dtype!r} if path == "/World" else "float64"
                        assert array.dtype == np.dtype(target_dtype), array.dtype
                        rows[path] = array.reshape(-1)[column.data_row_index(i)]
                finally:
                    stage.release_group(group)
            assert rows == {{"/World": np.asarray(expected, dtype={dtype!r}).item(),
                            "/Other": expected * 2}}, rows

    try:
        for path, target_dtype, value in [("/World", {dtype!r}, 101), ("/Other", "float64", 202)]:
            initial_paths = dictionary.create_path_list_from_strings([path])
            initial_query = stage.query_from_path_list(initial_paths)
            try:
                stage.write_attribute(initial_query, attribute, 2, np.array([value], dtype=target_dtype),
                                      is_array=False).wait()
            finally:
                stage.release_query(initial_query).wait()
                dictionary.destroy_path_list(initial_paths)
        stage.advance_write_floor(2).wait()
        check_stage(2, 101)
        with FmiHost(FmiHostConfig(backend={backend!r})) as host:
            host.attach_ovstage(stage, source_asset={str(source)!r})
            for ordinal, value in [(3, -0.125), (4, 0.12345678912345678)]:
                expected = -50 * value
                host.wait_op(host.write([AttributeWrite(("/World", "/Other"), "probe:input",
                                                       [[value], [value * 2]])]))
                host.step_sync(0.01)
                assert host.write_to_ovstage(ordinal) == 1
                stage.advance_write_floor(ordinal).wait()
                check_stage(ordinal, expected)
                with host.read() as read:
                    for group in read.groups:
                        tensor = group.tensors[0]
                        assert tensor.dtype == np.float64
                        assert {{path: tensor[i, 0] for i, path in enumerate(group.prim_paths)}} == {{
                            "/World": expected, "/Other": expected * 2}}
    finally:
        stage.release_query(query).wait()
        dictionary.destroy_path_list(paths)
        dictionary.destroy()
'''
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT, env=_env_from_demo_setup(),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
