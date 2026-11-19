# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Master-backend boundary and backend selection."""

from __future__ import annotations

from typing import Protocol

from ._native_runtime import NativeBackend


class MasterBackend(Protocol):
    """Private contract implemented by the FMI master backend."""

    def populate(self, instances: dict) -> None: ...
    def step(self, target_time: float) -> None: ...
    def reset(self) -> None: ...
    def release(self) -> None: ...


class FmpyBackend:
    """Adapter around the existing fmpy FMI/SSP runtime."""

    def __init__(self, input_router, output_router):
        try:
            from ._fmpy_runtime import FMIRuntime
        except ModuleNotFoundError as error:
            if error.name == "fmpy":
                raise RuntimeError(
                    "The fmpy backend requires FMPy; install the ovfmi[fmpy] extra"
                ) from error
            raise

        self._runtime = FMIRuntime(input_router, output_router)
        self._populated = False

    def populate(self, instances: dict) -> None:
        self._populated = True
        try:
            self._runtime.init(instances)
            self._runtime.resume()
        except Exception:
            self.release()
            raise

    def step(self, target_time: float) -> None:
        if not self._populated:
            raise RuntimeError("backend is not populated")
        self._runtime.step(target_time)

    def reset(self) -> None:
        if self._populated:
            self._runtime.reset()
            self._runtime.resume()

    def release(self) -> None:
        if self._populated:
            self._runtime.destroy()
            self._populated = False
