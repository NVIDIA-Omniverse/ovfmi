# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed FMU construction releases owned files and preserves borrowed files."""

from pathlib import Path
import shutil
import zipfile

import pytest

pytest.importorskip("fmpy")
from fmpy.model_description import ValidationError
from lxml.etree import XMLSyntaxError

from ovfmi import _fmpy_runtime as runtime


ROOT = Path(__file__).resolve().parents[2]
MODEL_DESCRIPTION = ROOT / "tests/packaging/modelDescription.xml"


@pytest.fixture
def extraction_root(tmp_path, monkeypatch):
    directory = tmp_path / "extractions"
    directory.mkdir()
    monkeypatch.setattr(runtime.tempfile, "tempdir", str(directory))
    return directory


def _model(tmp_path, *, borrowed=False, metadata=None):
    if metadata is None:
        metadata = MODEL_DESCRIPTION.read_bytes()
    contents = {"modelDescription.xml": metadata, "resources/sentinel.txt": b"caller data"}
    if borrowed:
        source = tmp_path / "model"
        for name, data in contents.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    else:
        source = tmp_path / "model.fmu"
        with zipfile.ZipFile(source, "w") as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
    return source


@pytest.mark.parametrize("kind,error_type", [
    ("missing", FileNotFoundError),
    ("invalid_zip", zipfile.BadZipFile),
    ("invalid_xml", XMLSyntaxError),
    ("invalid_schema", ValidationError),
])
def test_failed_construction_releases_extraction(kind, error_type, tmp_path, extraction_root):
    source = tmp_path / "missing.fmu"
    if kind == "invalid_zip":
        source.write_bytes(b"inert-invalid-fmu-zip\n")
    elif kind == "invalid_xml":
        source = _model(tmp_path, metadata=b"<broken")
    elif kind == "invalid_schema":
        source = _model(tmp_path, metadata=b'<fmiModelDescription fmiVersion="2.0"/>')

    for _ in range(3):
        with pytest.raises(error_type):
            runtime.FmuRuntimeExtractedFMU(str(source))
        assert list(extraction_root.iterdir()) == []


@pytest.mark.parametrize("kind,error_type", [
    ("invalid_zip", zipfile.BadZipFile),
    ("invalid_xml", XMLSyntaxError),
])
def test_cleanup_failure_preserves_original_error(
    kind, error_type, tmp_path, extraction_root, monkeypatch,
):
    if kind == "invalid_xml":
        source = _model(tmp_path, metadata=b"<broken")
    else:
        source = tmp_path / "invalid.fmu"
        source.write_bytes(b"inert-invalid-fmu-zip\n")
    cleanup_attempts = []

    def fail_rmtree(path, ignore_errors=False):
        cleanup_attempts.append(Path(path))
        if not ignore_errors:
            raise PermissionError("injected locked file")

    with monkeypatch.context() as patch:
        patch.setattr(runtime.shutil, "rmtree", fail_rmtree)
        with pytest.raises(error_type):
            runtime.FmuRuntimeExtractedFMU(str(source))

    assert len(cleanup_attempts) == 1
    assert cleanup_attempts[0].parent == extraction_root


def test_failed_borrowed_model_preserves_files(tmp_path, extraction_root):
    source = _model(tmp_path, borrowed=True, metadata=b"<broken")

    with pytest.raises(XMLSyntaxError):
        runtime.FmuRuntimeExtractedFMU(str(source))

    assert (source / "modelDescription.xml").read_bytes() == b"<broken"
    assert (source / "resources/sentinel.txt").read_bytes() == b"caller data"
    assert list(extraction_root.iterdir()) == []


@pytest.mark.parametrize("borrowed", [False, True])
def test_destroy_is_idempotent_and_preserves_source(borrowed, tmp_path, extraction_root):
    source = _model(tmp_path, borrowed=borrowed)
    model = runtime.FmuRuntimeExtractedFMU(str(source))
    extracted = Path(model.get_unzip_dir())
    assert (extracted / "modelDescription.xml").is_file()

    model.destroy()
    model.destroy()

    assert source.exists()
    assert extracted.exists() == borrowed
    if borrowed:
        assert (source / "resources/sentinel.txt").read_bytes() == b"caller data"
    assert list(extraction_root.iterdir()) == []


@pytest.fixture
def staged_windows_binary(tmp_path, monkeypatch):
    binary = tmp_path / "Integrator.dll"
    binary.write_bytes(b"inert staged binary")
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.FmuRuntimeExtractedFMU, "_find_staged_windows_fmu_binary",
                        lambda _self, _identifier: binary)
    return binary


@pytest.mark.parametrize("borrowed,operation", [
    (True, "mkdtemp"),
    (True, "copytree"),
    (False, "mkdir"),
    (True, "mkdir"),
    (False, "copy2"),
    (True, "copy2"),
])
def test_windows_augmentation_failure_releases_owned_files(
    borrowed, operation, tmp_path, extraction_root, staged_windows_binary, monkeypatch,
):
    source = _model(tmp_path, borrowed=borrowed)
    failure = OSError("injected augmentation failure")
    def fail(*_args, **_kwargs):
        raise failure

    if operation == "mkdtemp":
        monkeypatch.setattr(runtime.tempfile, "mkdtemp", fail)
    elif operation == "copytree":
        copytree = shutil.copytree

        def fail_copytree(source_path, *args, **kwargs):
            result = copytree(source_path, *args, **kwargs)
            if Path(source_path) == source:
                raise failure
            return result

        monkeypatch.setattr(runtime.shutil, "copytree", fail_copytree)
    elif operation == "mkdir":
        mkdir = Path.mkdir

        def fail_mkdir(path, *args, **kwargs):
            if path.name == "win64":
                raise failure
            return mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    else:
        monkeypatch.setattr(runtime.shutil, "copy2", fail)

    with pytest.raises(OSError) as raised:
        runtime.FmuRuntimeExtractedFMU(str(source))

    assert raised.value is failure
    assert list(extraction_root.iterdir()) == []
    assert source.exists()
    if borrowed:
        assert (source / "resources/sentinel.txt").read_bytes() == b"caller data"
        assert not (source / "binaries").exists()


@pytest.mark.parametrize("borrowed", [False, True])
def test_windows_augmentation_cleanup(
    borrowed, tmp_path, extraction_root, staged_windows_binary,
):
    source = _model(tmp_path, borrowed=borrowed)
    model = runtime.FmuRuntimeExtractedFMU(str(source))
    extracted = Path(model.get_unzip_dir())
    assert (extracted / "binaries/win64/Integrator.dll").read_bytes() == b"inert staged binary"

    model.destroy()
    model.destroy()

    assert list(extraction_root.iterdir()) == []
    assert source.exists()
    if borrowed:
        assert (source / "resources/sentinel.txt").read_bytes() == b"caller data"
        assert not (source / "binaries").exists()
