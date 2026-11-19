# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes bridge to the Python-free ovfmi native runtime."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np


def _library_path() -> Path:
    override = os.environ.get("OVFMI_NATIVE_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    package = Path(__file__).resolve().parent
    root = package.parent
    name = "ovfmi_native.dll" if sys.platform == "win32" else "libovfmi_native.so"
    candidates = [package / name]
    if package.name == "python" and (root / "native" / "Cargo.toml").is_file():
        candidates.extend(root / "native" / "target" / mode / name for mode in ("release", "debug"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "ovfmi native library is not built; run `cargo build --manifest-path native/Cargo.toml --release --locked` "
        "or select FmiHostConfig(backend='fmpy') with the ovfmi[fmpy] extra installed"
    )


class NativeLibrary:
    def __init__(self) -> None:
        self._dll = ctypes.CDLL(str(_library_path()))
        self._dll.ovfmi_native_last_error.restype = ctypes.c_char_p
        self._dll.ovfmi_native_create.argtypes = [ctypes.c_char_p]
        self._dll.ovfmi_native_create.restype = ctypes.c_void_p
        self._dll.ovfmi_native_set_real.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
        self._dll.ovfmi_native_set_real.restype = ctypes.c_int
        self._dll.ovfmi_native_get_real.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
        self._dll.ovfmi_native_get_real.restype = ctypes.c_int
        self._dll.ovfmi_native_step.argtypes = [ctypes.c_void_p, ctypes.c_double]
        self._dll.ovfmi_native_step.restype = ctypes.c_int
        self._dll.ovfmi_native_reset.argtypes = [ctypes.c_void_p]
        self._dll.ovfmi_native_reset.restype = ctypes.c_int
        self._dll.ovfmi_native_destroy.argtypes = [ctypes.c_void_p]
        self._dll.ovfmi_native_destroy.restype = None

    def error(self) -> RuntimeError:
        raw = self._dll.ovfmi_native_last_error()
        return RuntimeError(raw.decode("utf-8", errors="replace") if raw else "native ovfmi error")


class NativeInstance:
    def __init__(self, library: NativeLibrary, parser_instance, fmu_path: Path) -> None:
        self._library = library
        self._parser_instance = parser_instance
        self._handle = library._dll.ovfmi_native_create(os.fsencode(fmu_path))
        if not self._handle:
            raise library.error()
        self._start_values = {}

    def get_parser_instance(self):
        return self._parser_instance

    def set_start_values(self, values: dict) -> None:
        self._start_values = dict(values)
        self.set_inputs(values)

    def set_inputs(self, values: dict) -> None:
        for name, value in values.items():
            array = np.asarray(value).reshape(-1)
            if len(array) != 1:
                raise NotImplementedError("native routing supports scalar Real/Float64 variables only")
            status = self._library._dll.ovfmi_native_set_real(
                self._handle, name.encode(), float(array[0])
            )
            if status != 0:
                raise self._library.error()

    def step(self, target_time: float, inputs: dict, outputs: list[str]):
        self.set_inputs(inputs)
        if self._library._dll.ovfmi_native_step(self._handle, target_time) != 0:
            raise self._library.error()
        dtype = [("time", np.float64), *((name, np.float64) for name in outputs)]
        values = [target_time]
        for name in outputs:
            value = ctypes.c_double()
            status = self._library._dll.ovfmi_native_get_real(
                self._handle, name.encode(), ctypes.byref(value)
            )
            if status != 0:
                raise self._library.error()
            values.append(value.value)
        return np.asarray([tuple(values)], dtype=np.dtype(dtype))[0]

    def reset(self) -> None:
        if self._library._dll.ovfmi_native_reset(self._handle) != 0:
            raise self._library.error()

    def close(self) -> None:
        if self._handle:
            self._library._dll.ovfmi_native_destroy(self._handle)
            self._handle = None


class NativeBackend:
    """Native fmi-rs backend; Python only marshals public API values."""

    def __init__(self, input_router, output_router) -> None:
        self._input_router = input_router
        self._output_router = output_router
        self._library = NativeLibrary()
        self._instances: list[NativeInstance] = []

    @staticmethod
    def _prepared_path(parser_instance) -> Path:
        source = Path(parser_instance.fmu)
        if getattr(parser_instance, "_is_ssp", False):
            prepared = source.with_suffix(".native.fmu")
            if not prepared.exists():
                raise RuntimeError(
                    f"native SSP container is missing: {prepared}; run demoapp/build_fmu.py"
                )
            return prepared
        return source

    def populate(self, instances: dict) -> None:
        try:
            for parser_instance in instances.values():
                if not parser_instance.enabled:
                    continue
                instance = NativeInstance(
                    self._library, parser_instance, self._prepared_path(parser_instance)
                )
                self._instances.append(instance)
                self._input_router.cache_connections(instance)
                self._input_router.write_start_values(instance)
                self._output_router.cache_connections(instance)
        except Exception:
            self.release()
            raise

    def step(self, target_time: float) -> None:
        for instance in self._instances:
            result = instance.step(
                target_time,
                self._input_router.get_inputs_for(instance),
                self._output_router.get_outputs_for(instance),
            )
            self._output_router.write_outputs(instance, [], result)

    def reset(self) -> None:
        for instance in self._instances:
            instance.reset()
            self._input_router.write_start_values(instance)

    def release(self) -> None:
        for instance in reversed(self._instances):
            self._input_router.empty_cache_for(instance)
            self._output_router.empty_cache_for(instance)
            instance.close()
        self._instances.clear()
