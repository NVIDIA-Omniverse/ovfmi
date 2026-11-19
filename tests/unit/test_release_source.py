# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
from pathlib import Path
import shutil
import tarfile

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "collect_distributions",
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "collect-distributions.py",
)
collector = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collector)

_BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_wheel", Path(__file__).resolve().parents[2] / ".github" / "scripts" / "build-wheel.py",
)
builder = importlib.util.module_from_spec(_BUILD_SPEC)
_BUILD_SPEC.loader.exec_module(builder)


@pytest.fixture
def release_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "VERSION.md").write_text("0.2.0", encoding="utf-8")
    monkeypatch.setenv("ARTIFACT_NAME", "release")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    for platform, tag in (("linux-x86_64", "manylinux_2_28_x86_64"), ("windows-amd64", "win_amd64")):
        directory = tmp_path / "platform-dist" / f"release-{platform}"
        directory.mkdir(parents=True)
        (directory / f"ovfmi-0.2.0-py3-none-{tag}.whl").touch()
        (directory / "ovfmi-0.2.0.tar.gz").write_bytes(b"platform-local source")
    source = tmp_path / "platform-dist" / "release-source" / "ovfmi-0.2.0.tar.gz"
    source.parent.mkdir()
    source.write_bytes(b"shared source used by both builds")
    monkeypatch.setenv("SDIST_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    return source


def test_collects_the_shared_source_instead_of_platform_copies(release_artifacts, tmp_path):
    collector.main()

    assert (tmp_path / "dist" / release_artifacts.name).read_bytes() == release_artifacts.read_bytes()


def test_rejects_a_replaced_source_archive(release_artifacts):
    release_artifacts.write_bytes(b"different source")

    with pytest.raises(ValueError, match="Source archive checksum mismatch"):
        collector.main()


@pytest.mark.parametrize("platform, replacement", [
    ("windows-amd64", "ovfmi-0.2.0-py3-none-any.whl"),
    ("linux-x86_64", "ovfmi-0.2.0-py3-none-linux_x86_64.whl"),
    ("windows-amd64", "ovfmi-0.3.0-py3-none-win_amd64.whl"),
    ("windows-amd64", None),
])
def test_rejects_missing_or_invalid_platform_wheels(release_artifacts, tmp_path, platform, replacement):
    directory = tmp_path / "platform-dist" / f"release-{platform}"
    if replacement:
        next(directory.glob("*.whl")).unlink()
        (directory / replacement).touch()
    else:
        shutil.rmtree(directory)

    with pytest.raises(ValueError):
        collector.main()


def test_rejects_duplicate_platform_wheels(release_artifacts, tmp_path):
    directory = tmp_path / "platform-dist" / "release-windows-amd64"
    (directory / "ovfmi-0.2.0-1-py3-none-win_amd64.whl").touch()

    with pytest.raises(ValueError, match="Expected one windows-amd64 wheel"):
        collector.main()


@pytest.mark.parametrize("invalid", ["checksum", "archive-layout"])
def test_invalid_source_never_reaches_the_wheel_build(tmp_path, monkeypatch, invalid):
    sdist = tmp_path / "source.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name in ("one/pyproject.toml", "two/pyproject.toml"):
            member = tarfile.TarInfo(name)
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid source must fail before extraction or wheel building")

    monkeypatch.setattr(builder.subprocess, "run", forbidden)
    checksum = hashlib.sha256(sdist.read_bytes()).hexdigest()
    if invalid == "checksum":
        checksum = "0" * 64
        monkeypatch.setattr(builder.tarfile, "open", forbidden)

    with pytest.raises(ValueError, match="checksum mismatch|must contain one directory"):
        builder.build(sdist, checksum, tmp_path / "dist")
