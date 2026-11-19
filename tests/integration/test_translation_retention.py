# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translation publication preserves populated transforms through both backends."""

import subprocess
import sys
import tempfile

import pytest

from probe_first_step_inputs import scene
from test_native_conveyor import ROOT, _env_from_demo_setup
from test_runtime_backends import require_backend_and_models  # noqa: F401


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("fmi_version", [2, 3])
def test_public_translation_preserves_basis(backend, fmi_version, tmp_path):
    model = ROOT / "demoapp/usd" / ("PDController.fmu3" if fmi_version == 3 else "PDController.fmu")
    source = tmp_path / "translation.usda"
    text = scene(model).replace("custom double probe:output = 0", '''
    double3 xformOp:translate = (11, 13, 17)
    float3 xformOp:rotateXYZ = (0, 0, 90)
    float3 xformOp:scale = (2, 3, 4)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]''')
    text = text.replace('token fmi:usdAttribute = "probe:output"\n                int2 fmi:usdMapping = (0, 1)',
                        'token fmi:usdAttribute = "xformOp:translate"\n                int2 fmi:usdMapping = (1, 1)')
    text += text.split("#usda 1.0\n", 1)[1].replace("World", "Other")
    source.write_text(text, encoding="utf-8")
    program = f'''
import numpy as np
import ovstage
from ovstage import population
from ovfmi import AttributeWrite, FmiHost, FmiHostConfig
from stage_columns import stage_row

with ovstage.Stage("ovfmi.translation-retention") as stage:
    population.open_usd(stage, {str(source)!r}, ordinal=1)
    stage.advance_write_floor(1).wait()
    def matrix(path, ordinal):
        return np.array(stage_row(stage, path, ordinal, 16, attribute_name="omni:xform")).reshape(4, 4)
    initial = matrix("/World", 1)
    expected_basis = [[0, 2, 0], [-3, 0, 0], [0, 0, 4]]
    np.testing.assert_allclose(initial[:3, :3], expected_basis, atol=1e-12)
    with FmiHost(FmiHostConfig(backend={backend!r})) as host:
        host.attach_ovstage(stage, source_asset={str(source)!r})
        for ordinal, input_value in [(2, 0.125), (3, -0.25), (4, 0.125)]:
            if ordinal == 4:
                host.reset()
            host.wait_op(host.write([AttributeWrite(("/World", "/Other"), "probe:input",
                                                    [[input_value], [input_value]])]))
            host.step_sync(0.125)
            expected = initial.copy()
            expected[3, 1] = -50 * input_value
            with host.read(attribute_names=["xformOp:translate"]) as result:
                for group in result.groups:
                    for row in group.tensors[0]:
                        np.testing.assert_allclose(row, expected[3, :3], atol=1e-12)
            # Filtering must not modify the unrelated, identically authored object.
            host.write_to_ovstage(ordinal, prim_paths=["/World"])
            stage.advance_write_floor(ordinal).wait()
            np.testing.assert_allclose(matrix("/World", ordinal), expected, atol=1e-12, rtol=0)
            np.testing.assert_allclose(matrix("/Other", ordinal), initial, atol=1e-12, rtol=0)
'''
    # Avoid inherited native-service pipe handles holding subprocess.run open.
    with tempfile.TemporaryFile() as log:
        result = subprocess.run(
            [sys.executable, "-c", program], cwd=ROOT / "tests/integration",
            env=_env_from_demo_setup(), stdout=log, stderr=subprocess.STDOUT, timeout=90,
        )
        log.seek(0)
        output = log.read().decode("utf-8", errors="replace")
    assert result.returncode == 0, output[-12000:]
