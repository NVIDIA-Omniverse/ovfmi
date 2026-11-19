# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Present ovrtx color frames through CUDA, with a CPU fallback."""

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass

import numpy as np


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    # DLPack's unversioned managed tensor starts with this header.
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_capsule_pointer = ctypes.PYFUNCTYPE(
    ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p,
)(("PyCapsule_GetPointer", ctypes.pythonapi))


@dataclass(frozen=True)
class CudaFrame:
    pointer: int
    width: int
    height: int
    pitch: int
    device_id: int


@contextmanager
def cuda_frame(mapping):
    """Borrow a CUDA image; retain its DLPack capsule until the copy finishes."""
    # Omit max_version to request the unversioned ABI on both ovrtx profiles.
    capsule = mapping.__dlpack__()
    address = _capsule_pointer(capsule, b"dltensor")
    tensor = ctypes.cast(address, ctypes.POINTER(_DLTensor)).contents
    if tensor.device.device_type != 2 or tensor.device.device_id < 0:
        raise ValueError("LdrColor mapping is not CUDA device memory")
    if (tensor.ndim != 3 or not tensor.shape or not tensor.data
            or (tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes) != (1, 8, 1)):
        raise ValueError("Expected a nonempty uint8 CUDA RGBA image")
    height, width, channels = tensor.shape[:3]
    if height <= 0 or width <= 0 or channels != 4:
        raise ValueError("Expected a nonempty uint8 CUDA RGBA image")
    strides = tuple(tensor.strides[:3]) if tensor.strides else (width * 4, 4, 1)
    if strides[0] < width * 4 or strides[1:] != (4, 1):
        raise ValueError("Unsupported CUDA RGBA image strides")
    # Keep the capsule unconsumed. Its destructor releases the borrowed view.
    yield CudaFrame(tensor.data + tensor.byte_offset, width, height,
                    strides[0], tensor.device.device_id)


@contextmanager
def cpu_color_frame(render_var):
    with render_var.map() as mapping:
        pixels = np.from_dlpack(mapping)
        if (pixels.ndim != 3 or pixels.shape[2] not in (3, 4)
                or pixels.size == 0 or pixels.dtype != np.uint8):
            raise RuntimeError(
                "Expected a nonempty uint8 RGB/RGBA LdrColor image; "
                f"received shape={pixels.shape}, dtype={pixels.dtype}"
            )
        yield pixels


class FramePresenter:
    def __init__(self, display, cuda_device, *, cuda_available, cuda_unavailable_reason=None):
        self._display = display
        self._cuda_device = cuda_device
        self._try_cuda = cuda_available
        self._cpu_reason = str(cuda_unavailable_reason or "CUDA Python bindings are unavailable")
        self._reported_route = None

    def present(self, render_var):
        if self._try_cuda:
            try:
                with render_var.map(device=self._cuda_device, sync_stream=0) as mapping:
                    with cuda_frame(mapping) as frame:
                        self._display.present_frame_cuda(
                            frame.pointer, frame.width, frame.height,
                            pitch=frame.pitch, device_id=frame.device_id,
                            wait_event=mapping.wait_event,
                        )
            except Exception as error:
                self._try_cuda = False
                self._cpu_reason = str(error)
            else:
                if self._reported_route != "cuda":
                    print("GPU display: CUDA/OpenGL GPU copy (no CPU readback)", flush=True)
                    self._reported_route = "cuda"
                return
        with cpu_color_frame(render_var) as pixels:
            self._display.present_frame_cpu(pixels)
        if self._reported_route != "cpu":
            reason = str(self._cpu_reason).encode("ascii", "backslashreplace").decode("ascii")
            print(f"GPU display: CPU readback and OpenGL upload ({reason})", flush=True)
            self._reported_route = "cpu"
