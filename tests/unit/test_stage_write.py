# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage publication preserves destination types and caller-owned ordinals."""

from contextlib import contextmanager
from math import prod
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import ovstage
import pytest

from ovfmi import ReadGroup
from ovfmi.routing import write_stage_group


@contextmanager
def query(stage, paths):
    dictionary = ovstage.PathDictionary(stage)
    path_list = dictionary.create_path_list_from_strings(paths)
    handle = stage.query_from_path_list(path_list)
    try:
        yield dictionary, handle
    finally:
        stage.release_query(handle).wait()
        dictionary.destroy_path_list(path_list)
        dictionary.destroy()


def author(stage, paths, attribute, ordinal, values, semantic=0):
    code = ovstage.DLDataTypeCode.kDLFloat if values.dtype.kind == "f" else ovstage.DLDataTypeCode.kDLInt
    tensor = ovstage.make_dltensor(
        values, dtype=ovstage.DLDataType(code, values.dtype.itemsize * 8, values.shape[1]),
        shape=[len(values)], ndim=1,
    )
    with query(stage, paths) as (_, handle):
        stage.write_attribute(handle, attribute, ordinal, tensor,
                              is_array=False, semantic=semantic).wait()


def read_rows(stage, paths, attribute, ordinal):
    rows = {}
    with query(stage, paths) as (dictionary, handle):
        token = dictionary.intern_token(attribute)
        with stage.read_attributes(handle, [token], ovstage.OrdinalRange.latest(ordinal)) as read:
            read.wait()
            for group in read.groups():
                try:
                    column = group.columns[0]
                    assert not column.is_array
                    tensor = column.tensor(0)
                    width = tensor.dtype.lanes * prod(
                        tensor.shape[axis] for axis in range(1, tensor.ndim)
                    )
                    values = np.asarray(column.array(0)).reshape(-1, width)
                    names = dictionary.get_path_strings(group.prim_list)
                    for index in range(group.prim_count):
                        path = names[group.prim_index(index)]
                        rows[path] = values[column.data_row_index(index)].copy()
                finally:
                    stage.release_group(group)
    assert set(rows) == set(paths)
    return rows


def publish(stage, paths, attribute, ordinal, values):
    write_stage_group(stage, ReadGroup(tuple(paths), attribute, (values,), False), ordinal)


def check_unsealed_transform():
    with ovstage.Stage("ovfmi.unsealed-transform") as stage:
        identity = np.eye(4).reshape(1, 16)
        author(stage, ("/Camera", "/Model"), "omni:xform", 1,
               np.repeat(identity, 2, axis=0), ovstage.AttributeSemantic.MATRIX)
        stage.advance_write_floor(1).wait()
        camera = identity.copy()
        camera[0, 12:15] = [-5, 7, 11]
        publish(stage, ("/Model",), "xformOp:translate", 2, np.array([[1.5, -2.25, 3.125]]))
        # Publication must leave ordinal 2 open for same-frame camera writes.
        author(stage, ("/Camera",), "omni:xform", 2, camera, ovstage.AttributeSemantic.MATRIX)
        stage.advance_write_floor(2).wait()
        rows = read_rows(stage, ("/Camera", "/Model"), "omni:xform", 2)
        model = identity.copy()
        model[0, 12:15] = [1.5, -2.25, 3.125]
        for path, expected in (("/Camera", camera), ("/Model", model)):
            assert rows[path].dtype == np.float64
            np.testing.assert_array_equal(rows[path], expected[0])


def check_unsealed_destination():
    with ovstage.Stage("ovfmi.unsealed-destination") as stage:
        matrix = np.diag([2., 3., 4., 1.]).reshape(1, 16)
        author(stage, ("/Model",), "omni:xform", 1, matrix, ovstage.AttributeSemantic.MATRIX)
        # Do not guess identity or seal someone else's pending transform.
        with pytest.raises(ValueError, match="requires sealed omni:xform"):
            publish(stage, ("/Model",), "xformOp:translate", 2, np.array([[1., 2., 3.]]))
        author(stage, ("/Camera",), "omni:xform", 1, matrix, ovstage.AttributeSemantic.MATRIX)
        stage.advance_write_floor(2).wait()
        rows = read_rows(stage, ("/Model", "/Camera"), "omni:xform", 2)
        for row in rows.values():
            np.testing.assert_array_equal(row, matrix[0])


def check_pending_transform():
    with ovstage.Stage("ovfmi.pending-transform") as stage:
        matrix = np.diag([2., 3., 4., 1.]).reshape(1, 16)
        author(stage, ("/Model",), "omni:xform", 1, matrix, ovstage.AttributeSemantic.MATRIX)
        stage.advance_write_floor(1).wait()
        matrix[0, 0] = 7
        author(stage, ("/Model",), "omni:xform", 2, matrix, ovstage.AttributeSemantic.MATRIX)
        with pytest.raises(ValueError, match="before other transform writes"):
            publish(stage, ("/Model",), "xformOp:translate", 2, np.array([[1., 2., 3.]]))
        # Failed publication must leave the pending matrix and ordinal intact.
        author(stage, ("/Camera",), "omni:xform", 2, matrix, ovstage.AttributeSemantic.MATRIX)
        stage.advance_write_floor(2).wait()
        for row in read_rows(stage, ("/Model", "/Camera"), "omni:xform", 2).values():
            np.testing.assert_array_equal(row, matrix[0])


def check_transform_retention():
    with ovstage.Stage("ovfmi.transform-retention") as stage:
        # Rotation, nonuniform scale and shear must survive, not just position.
        basis = np.array([[0, 2, 0, 0], [-3, 0, 0, 0], [0.5, 0, 4, 0],
                          [11, 13, 17, 1]], dtype=np.float64)
        author(stage, ("/A",), "omni:xform", 1, basis.reshape(1, 16),
               ovstage.AttributeSemantic.MATRIX)
        author(stage, ("/B",), "omni:xform", 1, np.eye(4).reshape(1, 16),
               ovstage.AttributeSemantic.MATRIX)
        stage.advance_write_floor(1).wait()
        paths = ("/B", "/Missing", "/A")
        for ordinal in (2, 4):
            if ordinal == 4:
                # Use the caller's latest sealed basis, not an attachment-time cache.
                basis[2, 2] = 8
                author(stage, ("/A",), "omni:xform", 3, basis.reshape(1, 16),
                       ovstage.AttributeSemantic.MATRIX)
                stage.advance_write_floor(3).wait()
            values = np.array([[1, ordinal, 3], [4, ordinal, 6], [11, ordinal, 17]])
            publish(stage, paths, "xformOp:translate", ordinal, values)
            stage.advance_write_floor(ordinal).wait()
            rows = read_rows(stage, paths, "omni:xform", ordinal)
            for index, path in enumerate(paths):
                expected = basis.copy() if path == "/A" else np.eye(4)
                expected[3, :3] = values[index]
                np.testing.assert_array_equal(rows[path], expected.reshape(-1))
            assert rows["/B"].dtype == np.float64


def check_first_transform():
    with ovstage.Stage("ovfmi.first-transform") as stage:
        stage.advance_write_floor(1).wait()
        publish(stage, ("/New",), "xformOp:translate", 2, np.array([[1, 2, 3]]))
        stage.advance_write_floor(2).wait()
        expected = np.eye(4)
        expected[3, :3] = [1, 2, 3]
        np.testing.assert_array_equal(
            read_rows(stage, ("/New",), "omni:xform", 2)["/New"], expected.reshape(-1))


def check_mixed_destinations(source_dtype):
    with ovstage.Stage("ovfmi.mixed-destinations") as stage:
        attribute = "inputs:intensity"
        author(stage, ("/A",), attribute, 1, np.array([[101]], dtype=np.float32))
        author(stage, ("/B",), attribute, 1, np.array([[202]], dtype=np.float64))
        stage.advance_write_floor(1).wait()
        paths = ("/B", "/C", "/A")
        values = np.array([[0.12345678912345678], [2.12345678912345678],
                           [-3.9876543219876543]], dtype=source_dtype)
        publish(stage, paths, attribute, 2, values)
        stage.advance_write_floor(2).wait()
        expected = {path: row.astype(dtype) for path, row, dtype in zip(
            paths, values, (np.float64, source_dtype, np.float32))}
        for ordinal in (2, 3):
            if ordinal == 3:
                for path, value in (("/A", 7.12345678912345678), ("/B", -8.12345678912345678)):
                    update = np.array([[value]], dtype=source_dtype)
                    publish(stage, (path,), attribute, ordinal, update)
                    expected[path] = update[0].astype(expected[path].dtype)
                stage.advance_write_floor(ordinal).wait()
            rows = read_rows(stage, paths, attribute, ordinal)
            for path in paths:
                assert rows[path].dtype == expected[path].dtype
                np.testing.assert_array_equal(rows[path], expected[path])


def check_rejected_type(conflict):
    initial = (np.array([[101]], dtype=np.int32) if conflict == "integer"
               else np.array([[11, 22, 33]], dtype=np.float32))
    with ovstage.Stage("ovfmi.rejected-type") as stage:
        author(stage, ("/A",), "result", 1, initial)
        stage.advance_write_floor(1).wait()
        with pytest.raises(ovstage.OvstageError) as caught:
            publish(stage, ("/A",), "result", 2, np.array([[0.12345678912345678]]))
        assert caught.value.code == ovstage.ErrorCode.INVALID_ARGUMENT
        stage.advance_write_floor(2).wait()
        row = read_rows(stage, ("/A",), "result", 2)["/A"]
        assert row.dtype == initial.dtype
        np.testing.assert_array_equal(row, initial[0])


def run_case(name, *args):
    # Native libraries may launch a persistent service that inherits stdout.
    # A PIPE can then wait for EOF even after the child exits or times out.
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), name, *args],
            stdout=output, stderr=subprocess.STDOUT, timeout=30,
        )
        output.seek(0)
        log = output.read().decode('utf-8', errors='replace')
    assert result.returncode == 0, log[-12000:]


def test_unsealed_transform_publication():
    run_case("transform")


def test_translation_retains_existing_transform():
    run_case("retention")


def test_translation_creates_missing_transform():
    run_case("first")


def test_unsealed_destination_is_not_silently_replaced_or_sealed():
    run_case("unsealed")


def test_pending_transform_is_not_overwritten_or_sealed():
    run_case("pending")


@pytest.mark.parametrize("source_dtype", ["float64", "float32"])
def test_mixed_destinations_and_repeated_filtered_writes(source_dtype):
    run_case("mixed", source_dtype)


@pytest.mark.parametrize("conflict", ["integer", "width"])
def test_rejected_type_preserves_destination(conflict):
    run_case("rejected", conflict)


if __name__ == "__main__":
    {"transform": check_unsealed_transform, "mixed": check_mixed_destinations,
     "rejected": check_rejected_type, "retention": check_transform_retention,
     "first": check_first_transform, "unsealed": check_unsealed_destination,
     "pending": check_pending_transform}[sys.argv[1]](*sys.argv[2:])
