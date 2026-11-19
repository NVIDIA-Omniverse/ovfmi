# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import ctypes
import gc
from pathlib import Path
import sys
from unittest.mock import Mock
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_display import FramePresenter, cpu_color_frame, cuda_frame


class Device(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("ordinal", ctypes.c_int32)]


class DataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p), ("device", Device), ("rank", ctypes.c_int32),
        ("dtype", DataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)), ("offset", ctypes.c_uint64),
    ]


class ManagedTensor(ctypes.Structure):
    _fields_ = [("tensor", Tensor), ("context", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


CapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_new_capsule = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, CapsuleDestructor,
)(("PyCapsule_New", ctypes.pythonapi))
_capsule_owners = {}


@CapsuleDestructor
def _release_capsule(capsule_address):
    owner = _capsule_owners.pop(capsule_address)
    owner.events.append("capsule released")


@pytest.fixture(autouse=True)
def check_capsule_cleanup():
    yield
    gc.collect()
    assert not _capsule_owners


class CudaMapping:
    def __init__(self, *, shape=(2, 3, 4), strides=None, dtype=(1, 8, 1),
                 device=(2, 1), pointer=0x1000, offset=0, events=None, wait_event=0xBEEF):
        self.events = events if events is not None else []
        self._wait_event = wait_event
        self.event_error = None
        self.shape = (ctypes.c_int64 * len(shape))(*shape)
        self.strides = None if strides is None else (ctypes.c_int64 * len(strides))(*strides)
        self.header = ManagedTensor(Tensor(
            pointer, Device(*device), len(shape), DataType(*dtype),
            self.shape, self.strides, offset,
        ), None, None)

    def __dlpack__(self):
        self.events.append("capsule created")
        capsule = _new_capsule(ctypes.addressof(self.header), b"dltensor", _release_capsule)
        _capsule_owners[id(capsule)] = self
        return capsule

    def __dlpack_device__(self):
        return self.header.tensor.device.kind, self.header.tensor.device.ordinal

    @property
    def wait_event(self):
        if self.event_error is not None:
            raise RuntimeError(self.event_error)
        return self._wait_event


@pytest.mark.parametrize("strides,pitch", [(None, 12), ((12, 4, 1), 12), ((32, 4, 1), 32)])
def test_cuda_descriptor_preserves_offset_and_row_pitch(strides, pitch):
    mapping = CudaMapping(strides=strides, offset=7)
    with cuda_frame(mapping) as frame:
        assert (frame.pointer, frame.width, frame.height, frame.pitch, frame.device_id) == (
            0x1007, 3, 2, pitch, 1,
        )
        assert mapping.events == ["capsule created"]
        gc.collect()
        assert "capsule released" not in mapping.events
    gc.collect()
    assert mapping.events == ["capsule created", "capsule released"]


@pytest.mark.parametrize("arguments", [
    {"shape": (2, 3, 3)},
    {"shape": (2, 3)},
    {"shape": (0, 3, 4)},
    {"shape": (2, -1, 4)},
    {"dtype": (2, 8, 1)},
    {"dtype": (1, 16, 1)},
    {"dtype": (1, 8, 4)},
    {"strides": (8, 4, 1)},
    {"strides": (-12, 4, 1)},
    {"strides": (24, 8, 1)},
    {"strides": (24, 4, 2)},
    {"device": (1, 0)},
    {"device": (2, -1)},
    {"pointer": None},
])
def test_cuda_descriptor_rejects_unsupported_buffers(arguments):
    mapping = CudaMapping(**arguments)
    with pytest.raises((ValueError, RuntimeError)):
        with cuda_frame(mapping):
            pytest.fail("Invalid frame was accepted")
    gc.collect()
    assert mapping.events.count("capsule released") == mapping.events.count("capsule created")


def test_cuda_descriptor_releases_capsule_when_consumer_fails():
    mapping = CudaMapping()
    with pytest.raises(RuntimeError, match="copy failed"):
        with cuda_frame(mapping):
            raise RuntimeError("copy failed")
    gc.collect()
    assert mapping.events == ["capsule created", "capsule released"]


def test_cuda_descriptor_retains_mapping_until_consumer_finishes():
    events = []
    mapping = CudaMapping(events=events)
    owner = weakref.ref(mapping)
    with cuda_frame(mapping) as frame:
        del mapping
        gc.collect()
        assert owner() is not None
        assert frame.pointer == 0x1000
        assert events == ["capsule created"]
    gc.collect()
    assert owner() is None
    assert events == ["capsule created", "capsule released"]


def test_cuda_descriptor_rejects_missing_shape():
    mapping = CudaMapping()
    mapping.header.tensor.shape = None
    with pytest.raises((ValueError, RuntimeError)):
        with cuda_frame(mapping):
            pytest.fail("Missing shape was accepted")


def test_cuda_descriptor_rejects_a_real_cpu_dlpack_producer():
    with pytest.raises((ValueError, RuntimeError)):
        with cuda_frame(np.zeros((2, 3, 4), dtype=np.uint8)):
            pytest.fail("CPU memory was accepted as a CUDA pointer")


class RenderVar:
    def __init__(self, pixels=None, *, cuda_error=None, events=None):
        self.events = events if events is not None else []
        self.pixels = np.arange(24, dtype=np.uint8).reshape(2, 3, 4) if pixels is None else pixels
        self.cuda_mapping = CudaMapping(strides=(32, 4, 1), offset=7, events=self.events)
        self.cuda_error = cuda_error
        self.maps = []

    @contextmanager
    def map(self, **kwargs):
        self.maps.append(kwargs)
        mode = "CUDA" if "device" in kwargs else "CPU"
        self.events.append(f"{mode} map")
        if mode == "CUDA" and self.cuda_error is not None:
            raise RuntimeError(self.cuda_error)
        try:
            yield self.cuda_mapping if mode == "CUDA" else self.pixels
        finally:
            self.events.append(f"{mode} unmap")


@pytest.mark.parametrize("channels", [3, 4])
def test_cpu_mapping_returns_original_color_pixels(channels):
    pixels = np.arange(6 * channels, dtype=np.uint8).reshape(2, 3, channels)
    render_var = RenderVar(pixels)
    with cpu_color_frame(render_var) as actual:
        np.testing.assert_array_equal(actual, pixels)
        assert np.shares_memory(actual, pixels)
        assert render_var.events == ["CPU map"]
    assert render_var.events == ["CPU map", "CPU unmap"]


@pytest.mark.parametrize("pixels", [
    np.zeros((2, 3, 4), dtype=np.float32),
    np.zeros((2, 3, 1), dtype=np.uint8),
    np.zeros((2, 3), dtype=np.uint8),
    np.zeros((0, 3, 4), dtype=np.uint8),
])
def test_cpu_mapping_releases_invalid_image(pixels):
    render_var = RenderVar(pixels)
    with pytest.raises((ValueError, RuntimeError)):
        with cpu_color_frame(render_var):
            pytest.fail("Invalid CPU image was accepted")
    assert render_var.events == ["CPU map", "CPU unmap"]


@pytest.mark.parametrize("wait_event", [None, 0xBEEF])
def test_presenter_forwards_producer_event_without_cpu_readback(capsys, wait_event):
    render_var = RenderVar()
    render_var.cuda_mapping._wait_event = wait_event
    display = Mock()

    def present(*_args, **kwargs):
        assert capsys.readouterr().out == ""
        assert render_var.events == ["CUDA map", "capsule created"]
        assert kwargs["wait_event"] == wait_event
        render_var.events.append("CUDA presented")

    display.present_frame_cuda.side_effect = present
    presenter = FramePresenter(display, "cuda device", cuda_available=True)
    presenter.present(render_var)
    display.present_frame_cuda.assert_called_once_with(
        0x1007, 3, 2, pitch=32, device_id=1, wait_event=wait_event,
    )
    display.present_frame_cpu.assert_not_called()
    assert render_var.maps == [{"device": "cuda device", "sync_stream": 0}]
    assert render_var.events == [
        "CUDA map", "capsule created", "CUDA presented",
        "capsule released", "CUDA unmap",
    ]
    assert "CUDA" in capsys.readouterr().out
    display.present_frame_cuda.side_effect = None
    presenter.present(render_var)
    assert display.present_frame_cuda.call_count == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure_stage", ["map", "event", "metadata", "copy"])
def test_cuda_failure_releases_mapping_and_falls_back_once(failure_stage, capsys):
    render_var = RenderVar(cuda_error="injected failure" if failure_stage == "map" else None)
    if failure_stage == "event":
        render_var.cuda_mapping.event_error = "injected failure"
    elif failure_stage == "metadata":
        render_var.cuda_mapping.header.tensor.dtype.bits = 16
    display = Mock()
    if failure_stage == "copy":
        display.present_frame_cuda.side_effect = fail_cuda_presentation

    def present_cpu(pixels):
        assert "CPU readback" not in capsys.readouterr().out
        if failure_stage != "map":
            assert render_var.events.index("CUDA unmap") < render_var.events.index("CPU map")
        np.testing.assert_array_equal(pixels, render_var.pixels)

    display.present_frame_cpu.side_effect = present_cpu
    presenter = FramePresenter(display, "cuda device", cuda_available=True)
    presenter.present(render_var)
    first_log = capsys.readouterr()
    assert "CPU" in first_log.out
    assert "injected failure" in first_log.out + first_log.err or failure_stage == "metadata"
    display.present_frame_cpu.side_effect = None
    presenter.present(render_var)
    second_log = capsys.readouterr()
    assert second_log.out + second_log.err == ""
    assert render_var.maps == [{"device": "cuda device", "sync_stream": 0}, {}, {}]
    assert display.present_frame_cpu.call_count == 2


def fail_cuda_presentation(*_args, **_kwargs):
    raise RuntimeError("injected failure")


def test_fallback_does_not_retain_failed_render_output():
    render_var = RenderVar()
    output_ref = weakref.ref(render_var)
    mapping_ref = weakref.ref(render_var.cuda_mapping)
    display = Mock()
    display.present_frame_cuda.side_effect = fail_cuda_presentation
    presenter = FramePresenter(display, "cuda device", cuda_available=True)
    presenter.present(render_var)
    del render_var
    gc.collect()
    assert output_ref() is None
    assert mapping_ref() is None
    assert not _capsule_owners
    presenter.present(RenderVar())
    assert display.present_frame_cuda.call_count == 1
    assert display.present_frame_cpu.call_count == 2


def test_cuda_failure_after_success_changes_route_once(capsys):
    render_var = RenderVar()
    display = Mock()
    presenter = FramePresenter(display, "cuda device", cuda_available=True)
    presenter.present(render_var)
    assert "CUDA" in capsys.readouterr().out
    display.present_frame_cuda.side_effect = fail_cuda_presentation
    presenter.present(render_var)
    fallback_log = capsys.readouterr().out
    assert "CPU readback" in fallback_log
    assert "injected failure" in fallback_log
    presenter.present(render_var)
    assert capsys.readouterr().out == ""
    assert display.present_frame_cuda.call_count == 2
    assert display.present_frame_cpu.call_count == 2
    assert render_var.maps == [
        {"device": "cuda device", "sync_stream": 0},
        {"device": "cuda device", "sync_stream": 0}, {}, {},
    ]


def test_missing_cuda_bindings_use_cpu_without_cuda_mapping(capsys):
    render_var = RenderVar()
    display = Mock()
    presenter = FramePresenter(display, "cuda device", cuda_available=False)
    presenter.present(render_var)
    presenter.present(render_var)
    assert render_var.maps == [{}, {}]
    assert display.present_frame_cpu.call_count == 2
    display.present_frame_cuda.assert_not_called()
    assert capsys.readouterr().out.count("CPU readback") == 1


def test_failed_cpu_presentation_is_not_reported_as_success(capsys):
    display = Mock()
    display.present_frame_cpu.side_effect = RuntimeError("OpenGL upload failed")
    presenter = FramePresenter(display, "cuda device", cuda_available=False)
    render_var = RenderVar()
    with pytest.raises(RuntimeError, match="OpenGL upload failed"):
        presenter.present(render_var)
    assert "CPU readback" not in capsys.readouterr().out
    assert render_var.events == ["CPU map", "CPU unmap"]
