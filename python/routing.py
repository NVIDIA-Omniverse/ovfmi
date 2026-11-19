# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD-space input and output routing for the public ovfmi data plane."""

from __future__ import annotations

from collections import OrderedDict
from itertools import chain
from math import prod

import numpy as np

from .types import AttributeWrite, MissingInputPolicy, ReadGroup


def _component(values: list, offset: int, count: int):
    selected = values[offset : offset + count] if count else values
    selected = [item.item() if isinstance(item, np.generic) else item for item in selected]
    return selected[0] if count == 1 else selected


def _as_rows(values) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim == 0:
        rows = rows.reshape(1, 1)
    elif rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    else:
        rows = rows.reshape(rows.shape[0], -1)
    return rows


def _ovstage_column(ovstage, values):
    data = np.ascontiguousarray(values)
    if data.dtype.kind not in "fiu":
        data = data.astype(np.float64)
    leading = data.shape[0]
    lanes = 1 if data.ndim == 1 else int(np.prod(data.shape[1:]))
    codes = {
        "f": ovstage.DLDataTypeCode.kDLFloat,
        "i": ovstage.DLDataTypeCode.kDLInt,
        "u": ovstage.DLDataTypeCode.kDLUInt,
    }
    return ovstage.make_dltensor(
        data,
        dtype=ovstage.DLDataType(codes[data.dtype.kind], data.dtype.itemsize * 8, lanes),
        shape=[leading],
        ndim=1,
    )


def _read_stage_attribute(stage, prim_paths, attribute_name, ordinal):
    import ovstage

    if not prim_paths:
        return {}
    dictionary = ovstage.PathDictionary(stage)
    path_list = dictionary.create_path_list_from_strings(prim_paths)
    query = stage.query_from_path_list(path_list)
    rows = {}
    try:
        attribute = dictionary.intern_token(attribute_name)
        with stage.read_attributes(
            query,
            [attribute],
            ovstage.OrdinalRange.latest(ordinal),
        ) as read:
            read.wait()
            for group in read.groups():
                try:
                    # ovstage 0.4 separates column payloads from shared prim rows.
                    # This query requests exactly one attribute; keep the owning
                    # group alive until its column's values have been copied.
                    column = group.columns[0]
                    if column.is_delete or column.is_array or column.tensor_count != 1:
                        continue
                    tensor = column.tensor(0)
                    width = max(1, tensor.dtype.lanes) * prod(
                        tensor.shape[axis] for axis in range(1, tensor.ndim)
                    )
                    # ovstage exposes flat storage, including vector lanes.
                    values = np.asarray(column.array(0)).reshape(-1, width)
                    paths = dictionary.get_path_strings(group.prim_list)
                    for local_index in range(group.prim_count):
                        prim_index = group.prim_index(local_index)
                        data_index = column.data_row_index(local_index)
                        if prim_index < len(paths) and data_index < len(values):
                            rows[paths[prim_index]] = values[data_index].tolist()
                finally:
                    stage.release_group(group)
        return rows
    finally:
        stage.release_query(query).wait()
        dictionary.destroy_path_list(path_list)
        dictionary.destroy()


def prepare_stage_group(stage, group: ReadGroup) -> ReadGroup:
    """Resolve translation-only outputs against the latest sealed transforms.

    Prepare every group before writing any of them: older ovstage versions
    reject reads of a column with any outstanding unsealed write, even on an
    unrelated prim. Never advance the caller's write floor as a workaround.
    """
    import ovstage

    if not group.prim_paths or not group.tensors:
        return group
    attribute_name = group.attribute_name
    values = group.tensors[0]
    semantic = group.semantic
    # ovrtx consumes the canonical local-matrix column.  Preserve USD-space
    # xformOp identity in read(), but translate the convenience stage write.
    if attribute_name == "xformOp:translate":
        translations = _as_rows(values)
        # Translation owns only the last row, not the existing rotation/scale
        # basis. Read at the caller's current sealed floor, never at the output
        # ordinal (which is deliberately still open), and never seal it here.
        with stage.get_attribute_write_floor("omni:xform") as floor_query:
            floor = floor_query.fetch()
        if floor == 0:
            # With no sealed snapshot, old ovstage queries can hide newly
            # written prims. Do not mistake an unsealed matrix for absence.
            raise ValueError(
                "translation publication requires sealed omni:xform data; "
                "seal initial stage population first"
            )
        try:
            existing = _read_stage_attribute(stage, group.prim_paths, "omni:xform", floor)
        except ovstage.OvstageError as error:
            # Newer latest-only storage reports OUT_OF_RANGE when a pending
            # write has already replaced the requested sealed snapshot.
            if error.code in (
                ovstage.ErrorCode.WRITE_FLOOR_VIOLATION,
                getattr(ovstage.ErrorCode, "OUT_OF_RANGE", None),
            ):
                raise ValueError(
                    "translation publication requires sealed omni:xform data; "
                    "call write_to_ovstage before other transform writes in this ordinal"
                ) from error
            raise
        matrices = np.repeat(np.eye(4)[None, :, :], len(translations), axis=0)
        for index, path in enumerate(group.prim_paths):
            if path in existing:
                matrices[index] = np.asarray(existing[path]).reshape(4, 4)
        matrices[:, 3, : min(3, translations.shape[1])] = translations[:, :3]
        attribute_name = "omni:xform"
        values = matrices
        semantic = int(ovstage.AttributeSemantic.MATRIX)
    return ReadGroup(group.prim_paths, attribute_name, (values,), group.is_array, semantic)


def write_stage_group(stage, group: ReadGroup, ordinal: int) -> None:
    group = prepare_stage_group(stage, group)
    if not group.prim_paths or not group.tensors:
        return
    _write_stage_attribute(
        stage, group.prim_paths, group.attribute_name, ordinal, group.tensors[0],
        group.is_array, group.semantic
    )


def _write_stage_attribute(stage, paths, attribute_name, ordinal, values, is_array, semantic):
    import ovstage

    dtypes = [values.dtype]
    if not is_array and values.dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        dtypes.append(np.dtype(np.float32 if values.dtype == np.float64 else np.float64))
    dictionary = ovstage.PathDictionary(stage)
    path_list = dictionary.create_path_list_from_strings(paths)
    query = stage.query_from_path_list(path_list)
    try:
        for dtype in dtypes:
            try:
                stage.write_attribute(
                    query,
                    attribute_name,
                    ordinal,
                    _ovstage_column(ovstage, values.astype(dtype, copy=False)),
                    is_array=is_array,
                    semantic=semantic,
                ).wait()
                return
            except ovstage.OvstageError as error:
                conflict = f"existing attribute '{attribute_name}' has a different type"
                if (dtype == dtypes[-1] or error.code != ovstage.ErrorCode.INVALID_ARGUMENT
                        or conflict not in error.message):
                    raise
                # ovstage has no type lookup for unsealed data. Retry floating
                # widths only after it rejects a type change.
                if len(paths) > 1:
                    for index, path in enumerate(paths):
                        _write_stage_attribute(
                            stage, (path,), attribute_name, ordinal, values[index:index + 1],
                            is_array, semantic,
                        )
                    return
    finally:
        stage.release_query(query).wait()
        dictionary.destroy_path_list(path_list)
        dictionary.destroy()


class InputRouter:
    def __init__(self, initial_values: dict, policy: MissingInputPolicy):
        self._initial_values = initial_values
        self.reset()
        self._policy = policy
        self._maps: dict[str, dict[str, tuple[str, str, int, int]]] = {}

    def reset(self) -> None:
        """Restore authored inputs, discarding previous steps and caller writes."""
        self._values = {
            prim: {attribute: list(values) for attribute, values in attributes.items()}
            for prim, attributes in self._initial_values.items()
        }

    def cache_connections(self, instance) -> None:
        from ._parser import FmuDirection

        mapped = {}
        for connection in instance.get_parser_instance().connections:
            if not connection.enabled:
                continue
            for target in connection.targets:
                for mapping in connection.mappings:
                    if mapping.direction == FmuDirection.INPUT:
                        offset, count = mapping.usdMapping
                        mapped[mapping.fmiAttributeName] = (
                            target,
                            mapping.usdAttributeName,
                            offset,
                            count,
                        )
        self._maps[instance.get_parser_instance().path] = mapped

    def empty_cache_for(self, instance) -> None:
        self._maps.pop(instance.get_parser_instance().path, None)

    def write_start_values(self, instance) -> None:
        values = {}
        for name, (prim, attribute, offset, count) in self._maps.get(
            instance.get_parser_instance().path, {}
        ).items():
            row = self._values.get(prim, {}).get(attribute)
            if row is None:
                if self._policy == MissingInputPolicy.ERROR:
                    raise ValueError(f"missing mapped FMI input {prim}.{attribute}")
                if self._policy == MissingInputPolicy.ZERO:
                    values[name] = 0.0 if count in (0, 1) else [0.0] * count
                continue
            values[name] = _component(row, offset, count)
        instance.set_start_values(values)

    def get_inputs_for(self, instance) -> dict:
        inputs = {}
        for name, (prim, attribute, offset, count) in self._maps.get(
            instance.get_parser_instance().path, {}
        ).items():
            row = self._values.get(prim, {}).get(attribute)
            if row is not None:
                inputs[name] = _component(row, offset, count)
        return inputs

    def get_value(self, prim: str, attribute: str) -> list:
        return list(self._values.get(prim, {}).get(attribute, []))

    def update_value(self, prim: str, attribute: str, values) -> None:
        # Snapshots need NumPy scalar types; input reads convert to Python scalars.
        self._values.setdefault(prim, {})[attribute] = list(values)

    def write(self, writes: list[AttributeWrite]) -> None:
        for write in writes:
            if write.is_array:
                raise NotImplementedError("ragged array FMI inputs are not implemented")
            rows = _as_rows(write.values)
            if len(rows) != len(write.prim_paths):
                raise ValueError(
                    f"{write.attribute_name}: expected {len(write.prim_paths)} rows, "
                    f"received {len(rows)}"
                )
            for index, prim in enumerate(write.prim_paths):
                self.update_value(prim, write.attribute_name, rows[index].tolist())

    def update_from_stage(self, stage, from_ordinal: int, to_ordinal: int) -> None:
        if from_ordinal > to_ordinal:
            raise ValueError("from_ordinal must be <= to_ordinal")
        by_attribute: OrderedDict[str, list[str]] = OrderedDict()
        for mappings in self._maps.values():
            for prim, attribute, _offset, _count in mappings.values():
                paths = by_attribute.setdefault(attribute, [])
                if prim not in paths:
                    paths.append(prim)
        for attribute, paths in by_attribute.items():
            for prim, values in _read_stage_attribute(
                stage, paths, attribute, to_ordinal
            ).items():
                self.update_value(prim, attribute, values)

    def mapped_prim_paths(self, attribute_name: str) -> set[str]:
        return {
            prim
            for mappings in self._maps.values()
            for prim, attribute, _offset, _count in mappings.values()
            if attribute == attribute_name
        }


class OutputRouter:
    def __init__(self, initial_values: dict, input_router: InputRouter, strict: bool):
        self._baseline = {
            prim: {attribute: list(values) for attribute, values in attributes.items()}
            for prim, attributes in initial_values.items()
        }
        self._input_router = input_router
        self._strict = strict
        self._maps: dict[str, list] = {}
        self._latest: OrderedDict[tuple[str, str], list] = OrderedDict()

    def reset(self) -> None:
        """Discard output snapshots; mappings remain attached."""
        self._latest.clear()

    def cache_connections(self, instance) -> None:
        from ._parser import FmuDirection

        mappings = []
        for connection in instance.get_parser_instance().connections:
            if not connection.enabled:
                continue
            for mapping in connection.mappings:
                if mapping.direction != FmuDirection.OUTPUT:
                    continue
                offset, count = mapping.usdMapping
                targets = []
                for target in connection.targets:
                    targets.append((target, mapping.usdAttributeName, offset, count))
                mappings.append((mapping.fmiAttributeName, targets))
        path = instance.get_parser_instance().path
        if self._strict:
            self._validate_ownership(path, mappings)
        self._maps[path] = mappings

    def _validate_ownership(self, path: str, mappings: list) -> None:
        owners: dict[tuple[str, str], list] = {}
        existing = ((p, m) for p, m in self._maps.items() if p != path)
        for owner_path, owner_mappings in chain(existing, [(path, mappings)]):
            for name, targets in owner_mappings:
                owner = (owner_path, name)
                for prim, attribute, offset, count in targets:
                    # A whole-attribute write replaces every component.
                    start, end = (offset, offset + count) if count else (0, float("inf"))
                    previous = owners.setdefault((prim, attribute), [])
                    if owner_path == path:
                        for other_owner, other_start, other_end in previous:
                            if owner != other_owner and start < other_end and other_start < end:
                                raise ValueError(
                                    "overlapping output mapping for "
                                    f"{prim}.{attribute}[{max(start, other_start)}]"
                                )
                    previous.append((owner, start, end))

    def empty_cache_for(self, instance) -> None:
        self._maps.pop(instance.get_parser_instance().path, None)

    def get_outputs_for(self, instance) -> list[str]:
        return [name for name, _targets in self._maps.get(instance.get_parser_instance().path, [])]

    def write_outputs(self, instance, _outputs, result) -> None:
        names = getattr(result.dtype, "names", ()) or ()
        for name, targets in self._maps.get(instance.get_parser_instance().path, []):
            if name not in names:
                continue
            # Retain NumPy scalar types so snapshot can infer the backend's
            # dtype; tolist() would turn even Float32 values into Python floats.
            produced = list(np.asarray(result[name]).reshape(-1))
            for prim, attribute, offset, count in targets:
                key = (prim, attribute)
                if count == 0:
                    row = list(produced)
                else:
                    width = max(offset + count, len(self._baseline.get(prim, {}).get(attribute, [])))
                    row = list(
                        self._latest.get(
                            key,
                            self._baseline.get(prim, {}).get(attribute, [0.0] * width),
                        )
                    )
                    current = self._input_router.get_value(prim, attribute)
                    # Incomplete stage rows can contain components from another prim.
                    if len(current) >= max(width, len(row)):
                        row = current
                    if len(row) < width:
                        row.extend([0.0] * (width - len(row)))
                    payload = produced if len(produced) > 1 else produced * count
                    row[offset : offset + count] = payload[:count]
                self._latest[key] = row
                self._input_router.update_value(prim, attribute, row)

    def snapshot(self, prim_paths=None, attribute_names=None) -> list[ReadGroup]:
        path_filter = set(prim_paths) if prim_paths is not None else None
        attribute_filter = set(attribute_names) if attribute_names is not None else None
        grouped: OrderedDict[tuple[str, int], list[tuple[str, list]]] = OrderedDict()
        for (prim, attribute), values in self._latest.items():
            if path_filter is not None and prim not in path_filter:
                continue
            if attribute_filter is not None and attribute not in attribute_filter:
                continue
            grouped.setdefault((attribute, len(values)), []).append((prim, list(values)))

        groups = []
        for (attribute, _width), rows in grouped.items():
            # Infer/promote numeric types across mapped components and
            # rows instead of silently narrowing FMI Real/Float64 to Float32.
            dtype = np.float64 if attribute == "omni:xform" else None
            tensors = (np.asarray([values for _prim, values in rows], dtype=dtype),)
            semantic = 0
            if attribute == "omni:xform":
                try:
                    import ovstage

                    semantic = int(ovstage.AttributeSemantic.MATRIX)
                except ImportError:
                    pass
            groups.append(
                ReadGroup(
                    prim_paths=tuple(prim for prim, _values in rows),
                    attribute_name=attribute,
                    tensors=tensors,
                    is_array=False,
                    semantic=semantic,
                )
            )
        return groups
