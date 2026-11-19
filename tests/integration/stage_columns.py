# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and seed fixed-width columns for publication probes."""

import numpy as np


def stage_row(stage, path, ordinal, width, values=None, attribute_name="qa:result"):
    import ovstage

    dictionary = ovstage.PathDictionary(stage)
    paths = dictionary.create_path_list_from_strings([path])
    query = stage.query_from_path_list(paths)
    try:
        attribute = dictionary.intern_token(attribute_name)
        if values is not None:
            data = np.asarray(values, dtype=np.float64)
            tensor = ovstage.make_dltensor(
                data, dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 64, len(data)),
                shape=[1], ndim=1,
            )
            stage.write_attribute(query, attribute, ordinal, tensor, is_array=False).wait()
            return
        rows = []
        with stage.read_attributes(query, [attribute], ovstage.OrdinalRange.latest(ordinal)) as read:
            read.wait()
            for group in read.groups():
                try:
                    column = group.columns[0]
                    assert not column.is_array and not column.is_delete
                    assert column.tensor_count == 1 and group.prim_count == 1
                    assert dictionary.get_path_strings(group.prim_list)[group.prim_index(0)] == path
                    data = np.asarray(column.array(0))
                    assert data.dtype == np.float64
                    rows.append(data.reshape(-1, width)[column.data_row_index(0)].tolist())
                finally:
                    stage.release_group(group)
        assert len(rows) == 1
        return rows[0]
    finally:
        stage.release_query(query).wait()
        dictionary.destroy_path_list(paths)
        dictionary.destroy()
