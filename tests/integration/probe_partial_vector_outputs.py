# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check partial output snapshots, with optional stage publication and refresh.

Use --route write for caller writes without ovstage. Use --route stage with
an installed ovstage runtime to check unchanged two-prim stage inputs,
caller-vector publication, and independent scalar force readback.
"""

import argparse
import contextlib
from importlib.abc import MetaPathFinder
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class BlockImports(MetaPathFinder):
    def __init__(self, names):
        self.names = names

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.names:
            raise ImportError(f"This probe route must not import {fullname}")


def _scene(fmu):
    def connection(direction, variable, attribute, offset, targets):
        paths = ", ".join(f"<{path}>" for path in targets)
        return f'''def FmuConnection "{direction}"
        {{
            rel fmi:targets = [{paths}]
            def FmuMapping "Signal"
            {{
                token fmi:direction = "{direction}"
                token fmi:fmuAttribute = "{variable}"
                token fmi:usdAttribute = "{attribute}"
                int2 fmi:usdMapping = ({offset}, 1)
            }}
        }}'''

    def instance(name, connections):
        signals = "\n".join(connections)
        return f'''def FmuInstance "{name}"
    {{
        asset fmi:fmu = @{fmu.as_posix()}@
        {signals}
    }}'''

    vectors = {"/World/A": (11, 22, 33), "/World/B": (44, 55, 66)}
    body = ['custom double sim:producer = 0.5']
    body.append(instance("Producer", [
        connection("input", "pos_x", "sim:producer", 0, ["/World"]),
        connection("output", "force_x", "sim:vector", 1, vectors),
    ]))
    for prim, values in vectors.items():
        suffix = prim.rsplit("/", 1)[1]
        attribute = f"custom double3 sim:vector = {values}"
        body.append(f'def Xform "{suffix}" {{\n{attribute}\n}}')
        body.append(f"custom double sim:force{suffix} = 0")
        body.append(instance(f"Consumer{suffix}", [
            connection("input", "pos_x", "sim:vector", 0, [prim]),
            connection("output", "force_x", f"sim:force{suffix}", 0, ["/World"]),
        ]))
    return '#usda 1.0\ndef Xform "World" {\n' + "\n".join(body) + '\n}\n'


def write_authored_vectors(stage, ordinal):
    import ovstage

    dictionary = ovstage.PathDictionary(stage)
    paths = dictionary.create_path_list_from_strings(["/World/A", "/World/B"])
    query = stage.query_from_path_list(paths)
    try:
        values = np.asarray([[11, 22, 33], [44, 55, 66]], dtype=np.float64)
        column = ovstage.make_dltensor(
            values, dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 64, 3),
            shape=[2], ndim=1,
        )
        stage.write_attribute(query, "sim:vector", ordinal, column, is_array=False).wait()
        stage.advance_write_floor(ordinal).wait()
    finally:
        stage.release_query(query).wait()
        dictionary.destroy_path_list(paths)
        dictionary.destroy()


def publish_and_check_outputs(host, stage, ordinal, outputs):
    import ovstage

    assert host.write_to_ovstage(ordinal) == len(outputs)
    stage.advance_write_floor(ordinal).wait()
    dictionary = ovstage.PathDictionary(stage)
    try:
        for attribute, expected in outputs.items():
            paths = dictionary.create_path_list_from_strings(list(expected))
            query = stage.query_from_path_list(paths)
            try:
                token = dictionary.intern_token(attribute)
                width = len(next(iter(expected.values())))
                with stage.read_attributes(query, [token], ovstage.OrdinalRange.latest(ordinal)) as read:
                    read.wait()
                    observed = {}
                    for group in read.groups():
                        try:
                            column = group.columns[0]
                            names = dictionary.get_path_strings(group.prim_list)
                            values = np.asarray(column.array(0)).reshape(-1, width)
                            for index in range(group.prim_count):
                                observed[names[group.prim_index(index)]] = values[column.data_row_index(index)].copy()
                        finally:
                            stage.release_group(group)
                    assert observed.keys() == expected.keys()
                    for prim, value in expected.items():
                        np.testing.assert_array_equal(observed[prim], value)
            finally:
                stage.release_query(query).wait()
                dictionary.destroy_path_list(paths)
    finally:
        dictionary.destroy()


def check_outputs(host, vectors, stage=None, ordinal=None):
    with host.read() as result:
        outputs = {group.attribute_name: dict(zip(group.prim_paths, group.tensors[0]))
                   for group in result.groups}
    expected_outputs = {"sim:vector": vectors}
    for prim, expected in vectors.items():
        actual = outputs["sim:vector"][prim]
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64
        suffix = prim.rsplit("/", 1)[1]
        force = "sim:force" + suffix
        expected_outputs[force] = {"/World": [-50 * expected[0]]}
        np.testing.assert_array_equal(outputs[force]["/World"], expected_outputs[force]["/World"])
    if stage is not None:
        publish_and_check_outputs(host, stage, ordinal, expected_outputs)


def probe(backend, fmi_version, route):
    forbidden = {"fmpy"} if backend == "native" else set()
    if route == "write":
        forbidden.add("ovstage")
    sys.meta_path.insert(0, BlockImports(forbidden))
    from ovfmi import AttributeWrite, FmiHost, FmiHostConfig

    stage_refresh = route == "stage"
    model = ROOT / "demoapp/usd" / ("PDController.fmu3" if fmi_version == 3 else "PDController.fmu")
    with tempfile.TemporaryDirectory(prefix="ovfmi-partial-vector-") as directory:
        source = Path(directory) / "partial-vector.usda"
        source.write_text(_scene(model), encoding="utf-8")
        with contextlib.ExitStack() as stack:
            stage = None
            if stage_refresh:
                import ovstage
                from ovstage import population

                stage = stack.enter_context(ovstage.Stage("ovfmi.partial-vector"))
                population.open_usd(stage, str(source), ordinal=1)
            host = stack.enter_context(FmiHost(FmiHostConfig(backend=backend)))
            report = host.attach_ovstage(stage if stage is not None else object(), source_asset=source)
            assert [instance.prim_path for instance in report.instances] == [
                "/World/" + name for name in ["Producer", "ConsumerA", "ConsumerB"]
            ]
            authored_outputs = {"/World/A": [11, -25, 33], "/World/B": [44, -25, 66]}
            ordinal = 2
            for _ in range(2):
                if stage_refresh:
                    write_authored_vectors(stage, ordinal)
                    for _ in range(3):
                        host.update_from_ovstage(ordinal, ordinal)
                        host.step_sync(0.01)
                        ordinal += 1
                        check_outputs(host, authored_outputs, stage, ordinal)
                    ordinal += 1
                for x, producer, z in [(0.125, 0.5, 0.25),
                                       (0.12345678912345678, 0.75, 0.9876543219876543)]:
                    current = {prim: [x + index, -101, z + index]
                               for index, prim in enumerate(authored_outputs)}
                    host.wait_op(host.write([
                        AttributeWrite(tuple(current), "sim:vector", list(current.values())),
                        AttributeWrite(("/World",), "sim:producer", [[producer]]),
                    ]))
                    expected = {prim: [row[0], -50 * producer, row[2]]
                                for prim, row in current.items()}
                    for _ in range(2):
                        host.step_sync(0.01)
                        check_outputs(host, expected, stage, ordinal)
                        ordinal += 1
                host.reset()
                with host.read() as result:
                    assert not result.groups
            host.step_sync(0.01)
            check_outputs(host, authored_outputs, stage, ordinal)
    assert (backend == "fmpy") == ("fmpy" in sys.modules)
    assert not forbidden.intersection(name.split(".", 1)[0] for name in sys.modules)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "fmpy"], required=True)
    parser.add_argument("--fmi-version", type=int, choices=[2, 3], required=True)
    parser.add_argument("--route", choices=["write", "stage"], required=True)
    arguments = parser.parse_args()
    helper = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    os.environ.setdefault("USD_PYTHON", str(ROOT / "demoapp/.usd_venv" / helper))
    probe(arguments.backend, arguments.fmi_version, arguments.route)
    print(json.dumps({
        "backend": arguments.backend, "fmi_version": arguments.fmi_version, "route": arguments.route,
    }))
