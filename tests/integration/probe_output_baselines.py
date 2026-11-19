# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check output baselines; --stage also verifies real ovstage publication."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ovfmi import AttributeWrite, FmiHost, FmiHostConfig
from probe_first_step_inputs import ROOT, scene
from stage_columns import stage_row


def output_scene(fmu, declaration, offset, count):
    return scene(fmu).replace(
        "custom double probe:output = 0", declaration
    ).replace(
        'token fmi:usdAttribute = "probe:output"\n                int2 fmi:usdMapping = (0, 1)',
        f'token fmi:usdAttribute = "probe:output"\n                int2 fmi:usdMapping = ({offset}, {count})',
    )


def check_case(backend, fmi_version, initial, offset, count, use_stage=False, declaration=None):
    fmu = ROOT / "demoapp/usd" / ("PDController.fmu3" if fmi_version == 3 else "PDController.fmu")
    if not fmu.exists():
        raise FileNotFoundError(f"Run demo setup to generate {fmu}")
    if declaration is None:
        declaration = "custom double probe:output = 0"
        if len(initial) == 3:
            declaration = "custom double3 probe:output = (11, 0, 33)"
    samples = []
    with tempfile.TemporaryDirectory(prefix="ovfmi-output-baselines-") as temporary, ExitStack() as stack:
        source = Path(temporary) / "output.usda"
        source.write_text(output_scene(fmu, declaration, offset, count), encoding="utf-8")
        stage = object()
        if use_stage:
            import ovstage
            from ovstage import population

            stage = stack.enter_context(ovstage.Stage("ovfmi.output-baselines"))
            population.open_usd(stage, str(source), ordinal=1)
            # Population omits custom columns in the validated ovstage profile.
            stage_row(stage, "/World", 2, len(initial), initial, "probe:output")
            stage.advance_write_floor(2).wait()
            np.testing.assert_array_equal(
                stage_row(stage, "/World", 2, len(initial), attribute_name="probe:output"), initial,
            )

        host = stack.enter_context(FmiHost(FmiHostConfig(backend=backend)))
        report = host.attach_ovstage(stage, source_asset=source)
        assert len(report.instances) == 1, report.instances
        ordinal = 3

        def step(value):
            nonlocal ordinal
            if value is not None:
                host.wait_op(host.write([AttributeWrite(("/World",), "probe:input", [[value]])]))
            host.step_sync(0.125)
            force = -50 * (value or 0)
            expected = list(initial) if count else [force]
            if count:
                expected[offset:offset + count] = [force] * count
            with host.read(attribute_names=["probe:output"]) as read:
                assert len(read.groups) == 1, read.groups
                group = read.groups[0]
                assert group.prim_paths == ("/World",), group.prim_paths
                assert not group.is_array, group
                tensor = group.tensors[0]
                assert tensor.dtype == np.float64, tensor.dtype
                assert tensor.shape == (1, len(expected)), tensor
                np.testing.assert_allclose(tensor[0], expected, rtol=1e-12, atol=1e-12)
                observed = {"host": tensor[0].tolist()}
            if use_stage:
                host.write_to_ovstage(ordinal, attribute_names=["probe:output"])
                stage.advance_write_floor(ordinal).wait()
                observed["stage"] = stage_row(
                    stage, "/World", ordinal, len(expected), attribute_name="probe:output",
                )
                np.testing.assert_allclose(observed["stage"], expected, rtol=1e-12, atol=1e-12)
                ordinal += 1
            samples.append(observed)

        step(0.125)
        assert host.time == 0.125, host.time
        step(0.25)
        host.reset()
        assert host.time == 0, host.time
        with host.read() as read:
            assert not read.groups, read.groups
        step(None)
        step(0.125)
        assert samples[0] == samples[-1], samples
    return samples


def check_mixed(backend, fmi_version, use_stage=False):
    fmu = ROOT / "demoapp/usd" / ("PDController.fmu3" if fmi_version == 3 else "PDController.fmu")
    targets = [
        ("Authored", "custom double3 qa:result = (11, 0, 33)", [11.0, 0.0, 33.0]),
        ("Unset", "custom double3 qa:result", [0.0, 0.0, 0.0]),
        ("Absent", "", [0.0, 0.0]),
        ("Pair", "custom double2 qa:result = (44, 0)", [44.0, 0.0]),
    ]
    declarations = "\n".join(
        f'def Xform "{name}" {{ custom double probe:input = 0.125\n{declaration}\n}}'
        for name, declaration, _row in targets
    )
    paths = ", ".join(f"</World/{name}>" for name, _declaration, _row in targets)
    text = output_scene(fmu, declarations, 1, 1).replace(
        'rel fmi:targets = </World>', f'rel fmi:targets = [{paths}]',
    ).replace('token fmi:usdAttribute = "probe:output"', 'token fmi:usdAttribute = "qa:result"')
    expected = {f"/World/{name}": row[:1] + [-6.25] + row[2:] for name, _declaration, row in targets}
    samples = []
    with tempfile.TemporaryDirectory(prefix="ovfmi-mixed-baselines-") as temporary, ExitStack() as stack:
        source = Path(temporary) / "mixed.usda"
        source.write_text(text, encoding="utf-8")
        stage = object()
        if use_stage:
            import ovstage
            from ovstage import population

            stage = stack.enter_context(ovstage.Stage("ovfmi.mixed-baselines"))
            population.open_usd(stage, str(source), ordinal=1)
            for name, _declaration, row in targets:
                stage_row(stage, f"/World/{name}", 2, len(row), row)
            stage.advance_write_floor(2).wait()
        host = stack.enter_context(FmiHost(FmiHostConfig(backend=backend)))
        host.attach_ovstage(stage, source_asset=source)
        for ordinal in (3, 4):
            host.step_sync(0.125)
            actual = {}
            with host.read(attribute_names=["qa:result"]) as read:
                for group in read.groups:
                    assert not group.is_array, group
                    for path, row in zip(group.prim_paths, group.tensors[0]):
                        assert path not in actual, path
                        actual[path] = row.tolist()
            assert actual == expected, actual
            observed = {"host": actual}
            if use_stage:
                assert host.write_to_ovstage(ordinal, attribute_names=["qa:result"]) == 2
                stage.advance_write_floor(ordinal).wait()
                observed["stage"] = {
                    path: stage_row(stage, path, ordinal, len(row)) for path, row in expected.items()
                }
                assert observed["stage"] == expected, observed
            samples.append(observed)
            host.reset()
            with host.read() as read:
                assert not read.groups, read.groups
        assert samples[0] == samples[-1], samples
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "fmpy"], required=True)
    parser.add_argument("--fmi-version", choices=[2, 3], type=int, default=2)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault(
        "USD_PYTHON", str(ROOT / "demoapp/.usd_venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )),
    )
    results = {}
    for name, initial, offset, count in [
        ("vector", [11.0, 0.0, 33.0], 1, 1),
        ("scalar", [0.0], 0, 1),
    ]:
        results[name] = check_case(args.backend, args.fmi_version, initial, offset, count, args.stage)
    results["unset"] = check_case(
        args.backend, args.fmi_version, [0.0, 0.0, 0.0], 1, 1, args.stage,
        declaration="custom double3 probe:output",
    )
    results["mixed"] = check_mixed(args.backend, args.fmi_version, args.stage)
    print(json.dumps(results))
