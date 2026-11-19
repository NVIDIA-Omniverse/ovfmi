# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ovstage-backed renderer using ovrtx's public attached-stage API."""

from __future__ import annotations

import os
from pathlib import Path

from ovrtx import Renderer, RendererConfig
from ovrtx._src import bindings
from ovrtx._src.renderer import Operation
from startup_progress import startup_phase


class OvstageRenderer(Renderer):
    """An ovrtx renderer populated from a caller-owned shared ovstage."""

    _ORDINAL = 1

    @property
    def ovstage(self):
        """Stage shared with ovrtx and ovphysx."""
        return self._ovstage

    @property
    def ordinal(self) -> int:
        """Latest sealed ovstage ordinal visible to rendering."""
        return self._render_ordinal

    def __init__(self, *, usd_file=None, usda=None, cuda_device_id=None):
        if (usd_file is None) == (usda is None):
            raise ValueError("Pass exactly one of usd_file or usda")
        self._handle = None
        self._ovstage = None
        self._render_ordinal = self._ORDINAL
        self._dll_dir_cookies = []
        config = RendererConfig()
        config._attach_mode = bindings._AttachMode.BORROW
        if cuda_device_id is not None:
            config.active_cuda_gpus = str(cuda_device_id)
        self._prepare_windows_dll_search()
        ovstage = self._configure_ovstage_runtime()
        from physics_schemas import register_physx_schemas

        # Register schemas before any shared USD schema-registry access.
        register_physx_schemas(ovstage)
        # Renderer owns shared Carbonite before stage population begins.
        with startup_phase(
            "Creating ovrtx renderer",
            note="Initializes native plugins, cache services and the GPU. "
                 "A pause here is not necessarily shader compilation.",
        ):
            super().__init__(config=config)
        try:
            with startup_phase("Populating ovstage from USD"):
                self._create_and_populate_stage(usd_file=usd_file, usda=usda)
            with startup_phase(
                "Attaching ovstage to ovrtx",
                note="ovrtx can compile ray-tracing pipelines during attachment, "
                     "before the first frame. A cold cache can take several minutes.",
            ):
                self.attach_ovstage(self._ovstage)
                self.update_from_stage(self._ORDINAL)
        except Exception:
            self.destroy()
            raise

    @staticmethod
    def _ovrtx_bin_path() -> Path:
        hint = bindings.OVRTX_LIBRARY_PATH_HINT or os.environ.get(
            "OVRTX_LIBRARY_PATH_HINT"
        )
        if hint:
            return Path(hint)
        import ovrtx  # noqa: PLC0415

        return Path(ovrtx.__file__).resolve().parent / "bin"

    def _prepare_windows_dll_search(self) -> None:
        """Expose ovrtx's nested flat-runtime DLL closure on Windows."""
        if os.name != "nt":
            return
        ovrtx_bin = self._ovrtx_bin_path()
        dll_dirs = sorted({path.parent for path in ovrtx_bin.rglob("*.dll")})
        for directory in dll_dirs:
            try:
                self._dll_dir_cookies.append(os.add_dll_directory(str(directory)))
            except OSError:
                pass
        os.environ["PATH"] = os.pathsep.join(
            [*(str(path) for path in dll_dirs), os.environ.get("PATH", "")]
        )

    def _configure_ovstage_runtime(self):
        """Point ovstage at the runtime bundled with this ovrtx package."""
        ovrtx_bin = self._ovrtx_bin_path()
        os.environ["OVSTAGE_LIBRARY_PATH_HINT"] = str(ovrtx_bin)

        # Import order is part of shared-runtime setup.
        import ovstage  # noqa: PLC0415

        ovstage._src.bindings.OVSTAGE_LIBRARY_PATH_HINT = str(ovrtx_bin)
        return ovstage

    def _create_and_populate_stage(self, *, usd_file, usda) -> None:
        ovstage = self._configure_ovstage_runtime()
        from ovphysx.population import physics_population_descs

        stage = ovstage.Stage("ov-fmi")
        try:
            populate = (ovstage.population.open_usd_with_desc if usd_file is not None
                        else ovstage.population.open_usd_from_string_with_desc)
            source = str(Path(usd_file).resolve()) if usd_file is not None else usda
            populate(stage, source, ordinal=self._ORDINAL,
                     time=ovstage.population.DEFAULT_TIME,
                     descs=physics_population_descs(rendering=True))
            stage.advance_write_floor(self._ORDINAL).wait()
        except Exception:
            stage.destroy()
            raise
        self._ovstage = stage

    def update_from_ovstage(self, ordinal: int) -> None:
        """Select a sealed ovstage ordinal for the next frame."""
        if ordinal < self._render_ordinal:
            raise ValueError(
                f"ovstage ordinal cannot move backwards: {ordinal} < {self._render_ordinal}"
            )
        self.update_from_stage(ordinal)
        self._render_ordinal = ordinal

    def step_async(self, render_products: set[str], delta_time: float, *,
                   ordinal: int | None = None) -> Operation:
        """Render the selected sealed ordinal of the attached stage."""
        if ordinal is None:
            ordinal = self._render_ordinal
        else:
            ordinal = self._normalize_ordinal(ordinal)
            if ordinal < self._render_ordinal:
                raise ValueError(
                    f"ovstage ordinal cannot move backwards: {ordinal} < {self._render_ordinal}"
                )
        operation = super().step_async(render_products, delta_time, ordinal=ordinal)
        self._render_ordinal = ordinal
        return operation

    def destroy(self):
        """Destroy the renderer before releasing its borrowed stage."""
        stage = getattr(self, "_ovstage", None)
        super().destroy()
        self._ovstage = None
        if stage is not None:
            stage.destroy()
