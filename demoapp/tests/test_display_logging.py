# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
import io
from itertools import count
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as ov_fmi_main
import frame_display


@pytest.fixture
def run_demo(monkeypatch, tmp_path):
    source = tmp_path / "scene.usda"
    source.write_text("#usda 1.0\n")
    pixels = np.arange(16, dtype=np.uint8).reshape(2, 2, 4)
    cuda_device = object()
    mapping = SimpleNamespace(wait_event=5678)
    map_color = Mock(side_effect=lambda **kwargs: nullcontext(mapping if kwargs else pixels))
    color = SimpleNamespace(map=map_color)
    product = SimpleNamespace(frames=[SimpleNamespace(render_vars={"LdrColor": color})])
    renderer = SimpleNamespace(ovstage=object(), step=lambda *_args: {"/Render/Camera": product})
    create_renderer = Mock(return_value=renderer)
    fmi = Mock()
    fmi.attach_ovstage.return_value = SimpleNamespace(instances=[])
    display = Mock()
    display_device = PropertyMock(return_value=1)
    monkeypatch.setattr(type(display), "cuda_device_id", display_device, raising=False)
    display.should_close.return_value = False
    create_display = Mock(return_value=display)
    cuda_available = Mock(return_value=True)
    monkeypatch.setattr(frame_display, "cuda_frame", lambda _mapping: nullcontext(
        frame_display.CudaFrame(1234, 2, 2, 8, 0)))

    monkeypatch.setattr(ov_fmi_main, "time", SimpleNamespace(monotonic=count().__next__))
    monkeypatch.delenv("OVRTX_LIBRARY_PATH_HINT", raising=False)
    monkeypatch.setattr(ov_fmi_main, "_parse_fmi_schema", lambda _path: {
        "render_products": ["/Render/Camera"],
    })
    monkeypatch.setattr(ov_fmi_main, "_read_camera_matrix", lambda *_args: np.eye(4))
    monkeypatch.setattr(ov_fmi_main, "FirstPersonCameraController", lambda *_args, **_kwargs:
                        SimpleNamespace(update=lambda _dt: None))
    for name, module in {
        "gpu_display": SimpleNamespace(GPUDisplay=create_display,
                                       is_cuda_interop_available=cuda_available),
        "ovrtx": SimpleNamespace(Device=SimpleNamespace(CUDA=cuda_device)),
        "ovstage_renderer": SimpleNamespace(OvstageRenderer=create_renderer),
        "runtime_dependencies": SimpleNamespace(describe_runtime=lambda: None),
        "ovfmi": SimpleNamespace(AttributeWrite=object, FmiHost=lambda _config: fmi,
                                FmiHostConfig=SimpleNamespace),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def run(*, encoding="cp1252", cuda=True, display_error=None, headless=False,
            cuda_error=None, device_error=None):
        cuda_available.return_value = cuda
        create_display.side_effect = display_error
        display.present_frame_cuda.side_effect = cuda_error
        display_device.side_effect = device_error
        argv = ["main.py", str(source), "--no-physics", "--duration", "0.01"]
        if headless:
            argv.append("--headless")
        monkeypatch.setattr(sys, "argv", argv)
        with (
            io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict") as stdout,
            io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict") as stderr,
        ):
            with monkeypatch.context() as streams:
                streams.setattr(sys, "stdout", stdout)
                streams.setattr(sys, "stderr", stderr)
                ov_fmi_main.main()
                stdout.flush()
                stderr.flush()
                output = stdout.buffer.getvalue().decode(encoding)
                errors = stderr.buffer.getvalue().decode(encoding)
        fmi.release.assert_called_once_with()
        return SimpleNamespace(display=display, create_display=create_display,
                               pixels=pixels, output=output, errors=errors,
                               cuda_available=cuda_available, map_color=map_color,
                               cuda_device=cuda_device, create_renderer=create_renderer)

    return run


@pytest.mark.parametrize("encoding", ["ascii", "cp1252", "utf-8"])
@pytest.mark.parametrize("cuda", [True, False])
def test_redirected_logs_preserve_live_display(run_demo, encoding, cuda):
    result = run_demo(encoding=encoding, cuda=cuda)
    result.display.poll_events.assert_called_once_with()
    if cuda:
        result.display.present_frame_cuda.assert_called_once_with(
            1234, 2, 2, pitch=8, device_id=0, wait_event=5678)
        result.map_color.assert_called_once_with(device=result.cuda_device, sync_stream=0)
        result.display.present_frame_cpu.assert_not_called()
        assert "GPU display: CUDA/OpenGL GPU copy (no CPU readback)" in result.output
    else:
        result.display.present_frame_cpu.assert_called_once()
        np.testing.assert_array_equal(result.display.present_frame_cpu.call_args.args[0], result.pixels)
        result.map_color.assert_called_once_with()
        assert "GPU display: CPU readback and OpenGL upload" in result.output
        result.display.present_frame_cuda.assert_not_called()
    result.display.destroy.assert_called_once_with()
    assert "live display window active" in result.output
    assert "Color frames: 1 read, 1 presented." in result.output
    assert "zero-copy" not in result.output
    assert "GPU-resident" not in result.output
    result.cuda_available.assert_called_once_with()
    assert result.create_renderer.call_args.kwargs["cuda_device_id"] == (1 if cuda else None)
    assert result.errors == ""


def test_cuda_failure_reports_actual_cpu_route(run_demo):
    result = run_demo(encoding="ascii", cuda_error=RuntimeError("interop failed \u2014 device mismatch"))
    result.display.present_frame_cuda.assert_called_once()
    result.display.present_frame_cpu.assert_called_once()
    np.testing.assert_array_equal(result.display.present_frame_cpu.call_args.args[0], result.pixels)
    assert "GPU display: CPU readback and OpenGL upload" in result.output
    assert "device mismatch" in result.output
    assert "no CPU readback" not in result.output
    assert result.errors == ""


def test_cuda_device_selection_failure_preserves_cpu_display(run_demo):
    result = run_demo(encoding="ascii", device_error=RuntimeError("GL device query failed"))
    result.display.present_frame_cuda.assert_not_called()
    result.display.present_frame_cpu.assert_called_once()
    assert result.create_renderer.call_args.kwargs["cuda_device_id"] is None
    assert "GPU display: CPU readback and OpenGL upload (GL device query failed)" in result.output
    assert "live display window active" in result.output
    assert result.errors == ""


@pytest.mark.parametrize("error", [ImportError("glfw unavailable"), RuntimeError("GLFW init failed")])
def test_display_failure_still_falls_back_to_headless(run_demo, error):
    result = run_demo(display_error=error)
    assert "Could not open display window" in result.errors
    assert str(error) in result.errors
    assert "Falling back to headless mode" in result.errors
    assert "Color frames: 1 read, 0 presented." in result.output
    result.display.present_frame_cpu.assert_not_called()
    result.display.destroy.assert_not_called()
    assert "GPU display:" not in result.output


def test_headless_does_not_create_display(run_demo):
    result = run_demo(headless=True)
    result.create_display.assert_not_called()
    assert "GPU display:" not in result.output
    assert "Color frames: 1 read, 0 presented." in result.output
    assert result.errors == ""
