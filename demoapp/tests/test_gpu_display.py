# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from enum import IntEnum
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


class _CudaError(IntEnum):
    success = 0
    failure = 1


@pytest.fixture
def display_env(monkeypatch):
    def make(binding_namespace="modern"):
        events = []
        resources = set()
        resource_devices = {}
        mapped = set()
        state = SimpleNamespace(device=7, pointer_device=0, gl_devices=[0],
                                pointer_type=2, alignment=4, copy_pending=False,
                                producer_pending=False)
        runtime = ModuleType("cuda.bindings.runtime")
        runtime.cudaGraphicsRegisterFlags = SimpleNamespace(
            cudaGraphicsRegisterFlagsWriteDiscard=2)
        runtime.cudaMemcpyKind = SimpleNamespace(cudaMemcpyDeviceToDevice=3)
        runtime.cudaGLDeviceList = SimpleNamespace(cudaGLDeviceListAll=1)
        runtime.cudaMemoryType = SimpleNamespace(cudaMemoryTypeDevice=2,
                                                cudaMemoryTypeManaged=3)

        def cuda_call(name, result=(), action=None):
            def call(*args):
                events.append((name, args))
                values = action(*args) if action else result
                return (_CudaError.success, *values)
            method = Mock(side_effect=call)
            setattr(runtime, name, method)
            return method

        def register(*_args):
            assert state.device == state.pointer_device
            resource = object()
            resources.add(resource)
            resource_devices[resource] = state.device
            return (resource,)

        def unregister(resource):
            assert resource not in mapped
            assert state.device == resource_devices[resource]
            resources.remove(resource)
            del resource_devices[resource]
            return ()

        def map_resource(_count, resource, _stream):
            assert resource in resources, "CUDA registration must precede mapping"
            assert resource not in mapped
            assert state.device == resource_devices[resource]
            mapped.add(resource)
            return ()

        def unmap_resource(_count, resource, _stream):
            assert state.device == resource_devices[resource]
            mapped.remove(resource)
            return ()

        def set_device(device):
            state.device = device
            return ()

        def copy(*_args):
            assert not state.producer_pending
            state.copy_pending = True
            return ()

        def synchronize(*_args):
            state.copy_pending = False
            return ()

        def wait_event(_event):
            assert state.device == state.pointer_device
            state.producer_pending = False
            return ()

        cuda_call("cudaGetDevice", action=lambda: (state.device,))
        cuda_call("cudaSetDevice", action=set_device)
        cuda_call("cudaPointerGetAttributes", action=lambda _ptr: (
            SimpleNamespace(device=state.pointer_device, type=state.pointer_type),))
        cuda_call("cudaGLGetDevices", action=lambda *_args: (
            len(state.gl_devices), state.gl_devices))
        cuda_call("cudaGraphicsGLRegisterImage", action=register)
        cuda_call("cudaGraphicsUnregisterResource", action=unregister)
        cuda_call("cudaGraphicsMapResources", action=map_resource)
        cuda_call("cudaGraphicsUnmapResources", action=unmap_resource)
        cuda_call("cudaGraphicsSubResourceGetMappedArray", result=("mapped-array",))
        cuda_call("cudaMemcpy2DToArray", action=copy)
        cuda_call("cudaStreamSynchronize", action=synchronize)
        cuda_call("cudaEventSynchronize", action=wait_event)

        gl = Mock()
        for name in ("GL_TRUE", "GL_TEXTURE_2D", "GL_RGBA", "GL_RGB", "GL_RGBA8",
                     "GL_UNSIGNED_BYTE", "GL_UNPACK_ALIGNMENT"):
            setattr(gl, name, name)
        gl.glGenTextures.return_value = 5
        gl.glGetIntegerv.side_effect = lambda _name: state.alignment

        def upload(name):
            def call(*args):
                events.append((name, args))
                if name == "glTexImage2D":
                    assert not resources, "Unregister CUDA before replacing GL storage"
            return call

        gl.glTexImage2D.side_effect = upload("glTexImage2D")
        gl.glTexSubImage2D.side_effect = upload("glTexSubImage2D")
        gl.glDeleteTextures.side_effect = upload("glDeleteTextures")

        def pixel_store(name, value):
            events.append(("glPixelStorei", (name, value)))
            state.alignment = value

        gl.glPixelStorei.side_effect = pixel_store
        glfw = Mock()
        glfw.init.return_value = True
        glfw.create_window.return_value = "window"
        monkeypatch.setitem(sys.modules, "glfw", glfw)
        monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(GL=gl))
        cuda = ModuleType("cuda")
        bindings = ModuleType("cuda.bindings")
        if binding_namespace in ("modern", "both"):
            bindings.runtime = runtime
        if binding_namespace in ("legacy", "both"):
            cuda.cudart = runtime
            cuda.cuda = ModuleType("cuda.cuda")
        monkeypatch.setitem(sys.modules, "cuda", cuda)
        monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
        monkeypatch.delitem(sys.modules, "cuda.bindings.runtime", raising=False)
        monkeypatch.delitem(sys.modules, "cuda.cudart", raising=False)
        monkeypatch.delitem(sys.modules, "cuda.cuda", raising=False)

        name = "_gpu_display_under_test"
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[1] / "gpu_display.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module.GPUDisplay, "_create_shader", lambda _self: 10)
        monkeypatch.setattr(module.GPUDisplay, "_create_fullscreen_quad", lambda _self: 11)
        display = module.GPUDisplay()

        def draw():
            assert not state.copy_pending
            assert not mapped
            events.append(("draw", ()))

        display._draw_quad = Mock(side_effect=draw)
        events.clear()
        return SimpleNamespace(display=display, module=module, cuda=runtime, gl=gl,
                               glfw=glfw, events=events, resources=resources,
                               mapped=mapped, state=state)

    return make


@pytest.mark.parametrize("binding_namespace", ["modern", "legacy"])
def test_supported_cuda_bindings_import(display_env, binding_namespace):
    env = display_env(binding_namespace)
    assert env.module.is_cuda_interop_available()


def test_cpu_display_works_without_cuda_bindings(display_env):
    env = display_env("none")
    assert not env.module.is_cuda_interop_available()
    with pytest.raises(RuntimeError, match="bindings are unavailable"):
        _ = env.display.cuda_device_id
    env.display.present_frame_cpu(np.zeros((2, 2, 4), dtype=np.uint8))
    env.display._draw_quad.assert_called_once()
    with pytest.raises(RuntimeError):
        env.display.present_frame_cuda(4096, 2, 2)


def test_cuda_device_selection_uses_first_opengl_device(display_env):
    env = display_env()
    env.state.gl_devices = [2, 1]
    env.state.pointer_device = 2
    env.glfw.make_context_current.reset_mock()
    assert env.display.cuda_device_id == 2
    env.glfw.make_context_current.assert_called_once_with("window")
    env.cuda.cudaGraphicsGLRegisterImage.assert_not_called()
    env.display.present_frame_cuda(4096, 3, 2, device_id=2)
    env.cuda.cudaGLGetDevices.assert_called_once()
    env.display._draw_quad.assert_called_once()
    assert env.state.device == 7


def test_cuda_device_selection_fails_without_opengl_device(display_env):
    env = display_env()
    env.state.gl_devices = []
    with pytest.raises(RuntimeError, match="No CUDA device"):
        _ = env.display.cuda_device_id
    env.cuda.cudaSetDevice.assert_not_called()
    env.cuda.cudaGraphicsGLRegisterImage.assert_not_called()


def test_cuda_device_selection_retries_failed_lookup(display_env):
    env = display_env()
    lookup = env.cuda.cudaGLGetDevices.side_effect
    env.cuda.cudaGLGetDevices.side_effect = lambda *_args: (_CudaError.failure,)
    with pytest.raises(RuntimeError):
        _ = env.display.cuda_device_id
    env.cuda.cudaGLGetDevices.side_effect = lookup
    env.state.gl_devices = [2]
    assert env.display.cuda_device_id == 2
    assert env.cuda.cudaGLGetDevices.call_count == 2


def test_cpu_to_cuda_at_same_dimensions_registers_texture(display_env):
    env = display_env("both")
    env.display.present_frame_cpu(np.zeros((2, 3, 4), dtype=np.uint8))
    env.display.present_frame_cuda(4096, 3, 2)
    env.cuda.cudaGraphicsGLRegisterImage.assert_called_once()
    env.cuda.cudaMemcpy2DToArray.assert_called_once()
    assert not env.mapped


@pytest.mark.parametrize("next_route", ["cpu", "cuda"])
def test_resize_unregisters_before_replacing_texture(display_env, next_route):
    env = display_env("both")
    env.display.present_frame_cuda(4096, 3, 2)
    env.events.clear()
    if next_route == "cpu":
        env.display.present_frame_cpu(np.zeros((4, 5, 4), dtype=np.uint8))
    else:
        env.display.present_frame_cuda(8192, 5, 4)
    names = [name for name, _args in env.events]
    assert names.index("cudaGraphicsUnregisterResource") < names.index("glTexImage2D")
    assert not env.mapped


def test_registration_can_retry_without_resizing(display_env):
    env = display_env("both")
    register = env.cuda.cudaGraphicsGLRegisterImage.side_effect
    env.cuda.cudaGraphicsGLRegisterImage.side_effect = lambda *_args: (_CudaError.failure,)
    with pytest.raises(RuntimeError):
        env.display.present_frame_cuda(4096, 3, 2)
    assert env.state.device == 7
    env.cuda.cudaGraphicsGLRegisterImage.side_effect = register
    env.display.present_frame_cuda(4096, 3, 2)
    env.cuda.cudaMemcpy2DToArray.assert_called_once()
    assert not env.mapped


def test_cuda_copy_respects_source_row_pitch(display_env):
    env = display_env()
    env.display.present_frame_cuda(4096, 3, 2, pitch=16, device_id=0)
    args = env.cuda.cudaMemcpy2DToArray.call_args.args
    assert args[3:7] == (4096, 16, 12, 2)
    assert not env.mapped
    assert env.state.device == 7
    env.display._draw_quad.assert_called_once()
    env.cuda.cudaEventSynchronize.assert_not_called()


@pytest.mark.parametrize("binding_namespace", ["modern", "legacy"])
def test_cuda_waits_for_producer_on_frame_device(display_env, binding_namespace):
    env = display_env(binding_namespace)
    env.state.producer_pending = True
    env.display.present_frame_cuda(4096, 3, 2, device_id=0, wait_event=32768)
    env.cuda.cudaEventSynchronize.assert_called_once_with(32768)
    names = [name for name, _args in env.events]
    assert names.index("cudaEventSynchronize") < names.index("cudaMemcpy2DToArray")
    assert names.index("cudaStreamSynchronize") < names.index("draw")
    assert not env.state.producer_pending
    assert env.state.device == 7
    env.display._draw_quad.assert_called_once()


def test_failed_producer_wait_does_not_copy_or_draw(display_env):
    env = display_env()
    env.state.producer_pending = True
    env.cuda.cudaEventSynchronize.side_effect = lambda _event: (_CudaError.failure,)
    with pytest.raises(RuntimeError):
        env.display.present_frame_cuda(4096, 3, 2, device_id=0, wait_event=32768)
    env.cuda.cudaMemcpy2DToArray.assert_not_called()
    env.cuda.cudaGraphicsMapResources.assert_not_called()
    env.display._draw_quad.assert_not_called()
    assert not env.resources
    assert env.state.producer_pending
    assert env.state.device == 7


@pytest.mark.parametrize("kwargs", [
    {"channels": 3}, {"pitch": 11}, {"pitch": -1}, {"width": 0},
    {"height": -1}, {"cuda_ptr": 0},
])
def test_invalid_cuda_frames_do_not_allocate_or_copy(display_env, kwargs):
    env = display_env()
    arguments = {"cuda_ptr": 4096, "width": 3, "height": 2, **kwargs}
    with pytest.raises((ValueError, RuntimeError)):
        env.display.present_frame_cuda(**arguments)
    env.gl.glTexImage2D.assert_not_called()
    env.cuda.cudaMemcpy2DToArray.assert_not_called()


def test_cuda_rejects_pointer_on_another_device(display_env):
    env = display_env()
    env.state.pointer_device = 1
    with pytest.raises(ValueError):
        env.display.present_frame_cuda(4096, 3, 2, device_id=0)
    env.cuda.cudaGraphicsGLRegisterImage.assert_not_called()
    env.cuda.cudaMemcpy2DToArray.assert_not_called()


def test_cuda_rejects_opengl_on_another_device(display_env):
    env = display_env()
    env.state.gl_devices = [1]
    with pytest.raises(RuntimeError):
        env.display.present_frame_cuda(4096, 3, 2, device_id=0)
    env.cuda.cudaGraphicsGLRegisterImage.assert_not_called()
    env.cuda.cudaMemcpy2DToArray.assert_not_called()


def test_cuda_rejects_host_pointer(display_env):
    env = display_env()
    env.state.pointer_type = 1
    with pytest.raises(ValueError):
        env.display.present_frame_cuda(4096, 3, 2)
    env.cuda.cudaMemcpy2DToArray.assert_not_called()


def test_cuda_copy_failure_unmaps_and_does_not_draw(display_env):
    env = display_env()
    env.cuda.cudaMemcpy2DToArray.side_effect = lambda *_args: (_CudaError.failure,)
    with pytest.raises(RuntimeError):
        env.display.present_frame_cuda(4096, 3, 2)
    assert not env.mapped
    env.cuda.cudaGraphicsUnmapResources.assert_called_once()
    env.display._draw_quad.assert_not_called()
    assert env.state.device == 7
    env.display.present_frame_cpu(np.zeros((2, 3, 4), dtype=np.uint8))
    env.display._draw_quad.assert_called_once()


def test_failed_cuda_unmap_is_retried_before_cpu_upload(display_env):
    env = display_env()
    unmap = env.cuda.cudaGraphicsUnmapResources.side_effect
    env.cuda.cudaGraphicsUnmapResources.side_effect = lambda *_args: (_CudaError.failure,)
    with pytest.raises(RuntimeError):
        env.display.present_frame_cuda(4096, 3, 2)
    assert env.mapped
    env.display._draw_quad.assert_not_called()
    env.cuda.cudaGraphicsUnmapResources.side_effect = unmap
    env.events.clear()
    env.display.present_frame_cpu(np.zeros((2, 3, 4), dtype=np.uint8))
    names = [name for name, _args in env.events]
    assert names.index("cudaGraphicsUnmapResources") < names.index("glTexSubImage2D")
    assert not env.mapped
    assert not env.resources
    assert env.state.device == 7
    env.display._draw_quad.assert_called_once()


def test_destroy_unregisters_once_before_deleting_texture(display_env):
    env = display_env()
    env.display.present_frame_cuda(4096, 3, 2)
    env.events.clear()
    env.display.destroy()
    env.display.destroy()
    names = [name for name, _args in env.events]
    assert names.index("cudaGraphicsUnregisterResource") < names.index("glDeleteTextures")
    env.cuda.cudaGraphicsUnregisterResource.assert_called_once()
    env.gl.glDeleteTextures.assert_called_once()
    env.glfw.destroy_window.assert_called_once()
    assert not env.resources
    assert env.state.device == 7


def test_successful_unregister_is_recorded_when_device_restore_fails(display_env):
    env = display_env()
    env.display.present_frame_cuda(4096, 3, 2)
    set_device = env.cuda.cudaSetDevice.side_effect

    def fail_restore(device):
        return (_CudaError.failure,) if device == 7 else set_device(device)

    env.cuda.cudaSetDevice.side_effect = fail_restore
    with pytest.raises(RuntimeError, match="CUDA error"):
        env.display.present_frame_cpu(np.zeros((2, 3, 4), dtype=np.uint8))
    assert not env.resources
    assert env.display._cuda_resource is None
    assert env.display._cuda_device_id is None
    assert not env.display._cuda_interop_active

    env.cuda.cudaSetDevice.side_effect = set_device
    env.display.present_frame_cpu(np.zeros((2, 3, 4), dtype=np.uint8))
    env.display.destroy()
    env.cuda.cudaGraphicsUnregisterResource.assert_called_once()
    env.gl.glTexSubImage2D.assert_called_once()
    assert env.display._draw_quad.call_count == 2


@pytest.mark.parametrize("resize", [True, False])
def test_cpu_rgb_upload_uses_tight_row_alignment(display_env, resize):
    env = display_env("none")
    if not resize:
        env.display.present_frame_cpu(np.zeros((2, 3, 4), dtype=np.uint8))
        env.events.clear()
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    env.display.present_frame_cpu(pixels)
    names = [name for name, _args in env.events]
    upload_name = "glTexSubImage2D"
    upload_index = names.index(upload_name)
    before = [args[1] for name, args in env.events[:upload_index]
              if name == "glPixelStorei"]
    assert before[-1] == 1
    assert env.state.alignment == 4
    np.testing.assert_array_equal(getattr(env.gl, upload_name).call_args.args[-1], pixels)


def test_cpu_upload_preserves_noncontiguous_pixels(display_env):
    env = display_env("none")
    pixels = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)[:, ::-1]
    assert not pixels.flags.c_contiguous
    env.display.present_frame_cpu(pixels)
    uploaded = env.gl.glTexSubImage2D.call_args.args[-1]
    assert uploaded.flags.c_contiguous
    np.testing.assert_array_equal(uploaded, pixels)


def test_failed_cpu_upload_restores_unpack_alignment(display_env):
    env = display_env("none")
    env.gl.glTexSubImage2D.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        env.display.present_frame_cpu(np.zeros((2, 3, 3), dtype=np.uint8))
    assert env.state.alignment == 4
    env.display._draw_quad.assert_not_called()


def test_cuda_integer_error_code_is_not_ignored(display_env):
    env = display_env()
    with pytest.raises(RuntimeError):
        env.module._check_cuda((1,))
