# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real adapter's initialization order without loading GPU plugins."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    def load(*, registration_error=False):
        events = []
        runtime = tmp_path / "ovrtx-bin"
        bindings = SimpleNamespace(
            OVRTX_LIBRARY_PATH_HINT=str(runtime),
            _AttachMode=SimpleNamespace(BORROW=1),
        )
        stage_bindings = SimpleNamespace(OVSTAGE_LIBRARY_PATH_HINT=None)
        # Ensure mutations by the real adapter are restored after each test.
        monkeypatch.setenv("OVSTAGE_LIBRARY_PATH_HINT", "previous-runtime")

        def register(_paths=None):
            assert stage_bindings.OVSTAGE_LIBRARY_PATH_HINT == str(runtime)
            assert os.environ["OVSTAGE_LIBRARY_PATH_HINT"] == str(runtime)
            events.append("register")
            if registration_error:
                raise RuntimeError("schema registration failed")

        class Renderer:
            def __init__(self, *, config):
                assert config._attach_mode == bindings._AttachMode.BORROW
                self.captured_config = config
                events.append("renderer")

        class Stage:
            def __init__(self, name):
                events.append("stage")

            def advance_write_floor(self, ordinal):
                assert ordinal == 1
                return SimpleNamespace(wait=lambda: None)

            def destroy(self):
                events.append("stage-destroy")

        def populate_desc(stage, source, *, ordinal, time, descs):
            assert descs == ["physics", "rendering"] and time == -1
            assert ordinal == 1
            events.append("populate-desc")
        population = SimpleNamespace(
            open_usd_with_desc=populate_desc,
            open_usd_from_string_with_desc=populate_desc,
            DEFAULT_TIME=-1, register_usd_schemas=register,
        )
        Renderer.attach_ovstage = lambda self, stage: events.append("public-attach")
        Renderer.update_from_stage = lambda self, ordinal: events.append(("public-update", ordinal))
        Renderer.step_async = lambda self, products, dt, *, ordinal: (products, dt, ordinal)
        Renderer._normalize_ordinal = staticmethod(int)
        Renderer.destroy = lambda self: events.append("renderer-destroy")
        monkeypatch.setitem(sys.modules, "ovphysx.population", SimpleNamespace(
            physics_population_descs=lambda *, rendering: ["physics", "rendering"] if rendering else [],
        ))
        monkeypatch.setitem(sys.modules, "ovphysx.samples.python_samples.schema_support",
                            SimpleNamespace(physx_schema_paths=lambda: [tmp_path / "schemas"]))
        modules = {
            "ovrtx": dict(Renderer=Renderer, RendererConfig=SimpleNamespace),
            "ovrtx._src": dict(bindings=bindings),
            "ovrtx._src.renderer": dict(Operation=object, RenderProductSetOutputs=object),
            "ovstage": dict(
                _src=SimpleNamespace(bindings=stage_bindings), population=population,
                Stage=Stage, PopulationDomain=SimpleNamespace(RENDERING=1, PHYSICS=2),
            ),
            "ovphysx": dict(),
        }
        for name, attributes in modules.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            monkeypatch.setitem(sys.modules, name, module)

        source = Path(__file__).resolve().parents[1] / "ovstage_renderer.py"
        spec = importlib.util.spec_from_file_location("_schema_order_adapter", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = module.OvstageRenderer
        monkeypatch.setattr(cls, "_prepare_windows_dll_search", lambda self: None)
        return cls, events

    return load


@pytest.mark.parametrize("device_id", [None, 0, 1])
def test_renderer_uses_selected_display_device(adapter, device_id):
    cls, _events = adapter()
    renderer = cls(usda="#usda 1.0", cuda_device_id=device_id)
    assert getattr(renderer.captured_config, "active_cuda_gpus", None) == (
        str(device_id) if device_id is not None else None
    )


def test_registration_failure_prevents_renderer_and_population(adapter):
    cls, events = adapter(registration_error=True)
    with pytest.raises(RuntimeError, match="schema registration failed"):
        cls(usda="#usda 1.0")
    assert events[0] == "register"
    assert "renderer" not in events and "stage" not in events


@pytest.mark.parametrize("source", ["usd_file", "usda"])
def test_new_runtime_uses_descriptions_and_public_renderer_api(adapter, source):
    cls, events = adapter()
    renderer = cls(**{source: "scene.usda" if source == "usd_file" else "#usda 1.0"})
    assert events == ["register", "renderer", "stage", "populate-desc",
                      "public-attach", ("public-update", 1)]
    renderer.update_from_ovstage(7)
    assert renderer.step_async({"/Render"}, 0.1) == ({"/Render"}, 0.1, 7)
    assert events[-1] == ("public-update", 7)
    with pytest.raises(ValueError, match="backwards"):
        renderer.update_from_ovstage(6)


def test_renderer_releases_borrowed_stage_after_renderer(adapter):
    cls, events = adapter()
    renderer = cls(usda="#usda 1.0")
    renderer.destroy()
    assert events[-2:] == ["renderer-destroy", "stage-destroy"]
    renderer.destroy()
    assert events.count("stage-destroy") == 1
