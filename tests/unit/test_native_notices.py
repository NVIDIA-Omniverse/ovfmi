# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "generate_native_notices",
    Path(__file__).resolve().parents[2] / "tools" / "generate_native_notices.py",
)
notices = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(notices)


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "crate"
    root.mkdir()
    return {
        "name": "test-crate",
        "version": "1.0.0",
        "source": "registry+https://github.com/rust-lang/crates.io-index",
        "license": "MIT",
        "manifest_path": str(root / "Cargo.toml"),
        "license_file": None,
    }


def test_missing_license_text_fails(package, tmp_path):
    with pytest.raises(RuntimeError, match="No license text found for test-crate 1.0.0"):
        notices.package_licenses(package, tmp_path / "fallbacks")


def test_collects_nested_vendored_notices(package, tmp_path):
    root = Path(package["manifest_path"]).parent
    (root / "LICENSE").write_text("upstream license", encoding="utf-8")
    (root / "vendor").mkdir()
    (root / "vendor" / "NOTICE.txt").write_text("vendored notice", encoding="utf-8")

    files, provenance = notices.package_licenses(package, tmp_path / "fallbacks")

    assert [name for name, _ in files] == ["LICENSE", "vendor/NOTICE.txt"]
    assert provenance is None


def test_fallback_requires_matching_source_revision(package, tmp_path):
    root = Path(package["manifest_path"]).parent
    (root / ".cargo_vcs_info.json").write_text(json.dumps({"git": {"sha1": "actual"}}))
    fallback = tmp_path / "fallbacks" / "test-crate-1.0.0"
    fallback.mkdir(parents=True)
    provenance = {
        "package": package["name"], "version": package["version"],
        "source": package["source"], "git_revision": "other", "files": ["LICENSE"],
    }
    (fallback / "provenance.json").write_text(json.dumps(provenance))
    (fallback / "LICENSE").write_text("upstream license", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No license text found"):
        notices.package_licenses(package, tmp_path / "fallbacks")

    provenance["git_revision"] = "actual"
    (fallback / "provenance.json").write_text(json.dumps(provenance))
    files, recorded = notices.package_licenses(package, tmp_path / "fallbacks")
    assert files == [("LICENSE", fallback / "LICENSE")]
    assert recorded == provenance


@pytest.fixture
def compiler_notices(tmp_path, monkeypatch):
    manifest = tmp_path / "Cargo.toml"
    manifest.touch()
    manifest.with_name("Cargo.lock").write_text("locked dependencies")
    sysroot = tmp_path / "toolchain"
    documentation = sysroot / "share" / "doc" / "rust"
    (documentation / "licenses").mkdir(parents=True)
    for name in ("COPYRIGHT-library.html", "licenses/Apache-2.0.txt", "licenses/MIT.txt", "licenses/Unicode-3.0.txt"):
        (documentation / name).write_text(f"upstream {name}", encoding="utf-8")

    def cargo_output(command, **kwargs):
        assert command[0] == "cargo"
        if command[1] == "metadata":
            return json.dumps({"packages": [], "resolve": {"nodes": []}})
        assert command[1] == "rustc"
        assert "--locked" in command and "--release" in command
        assert command[command.index("--target") + 1] == "x86_64-pc-windows-msvc"
        if command[-1] == "--print=sysroot":
            return str(sysroot) + "\n"
        assert command[-1] == "-vV"
        return "rustc 1.97.1\nrelease: 1.97.1\ncommit-hash: abc123\nhost: x86_64-pc-windows-msvc\n"

    monkeypatch.setattr(notices.subprocess, "check_output", cargo_output)
    return manifest, documentation


def test_bundles_the_build_compilers_standard_library_notices(compiler_notices, tmp_path):
    manifest, documentation = compiler_notices
    output = tmp_path / "notices"

    notices.generate(manifest, "x86_64-pc-windows-msvc", output)

    inventory = json.loads((output / "packages.json").read_text())
    runtime = inventory["rust_standard_library"]
    assert runtime["version"] == "1.97.1"
    assert runtime["commit"] == "abc123"
    assert runtime["host"] == "x86_64-pc-windows-msvc"
    copied = {Path(item["path"]).relative_to("rust-standard-library").as_posix(): item for item in runtime["files"]}
    assert set(copied) == {"COPYRIGHT-library.html", "licenses/Apache-2.0.txt", "licenses/MIT.txt", "licenses/Unicode-3.0.txt"}
    for name, item in copied.items():
        assert (output / item["path"]).read_bytes() == (documentation / name).read_bytes()
        assert item["sha256"] == hashlib.sha256((documentation / name).read_bytes()).hexdigest()


@pytest.mark.parametrize("missing", ["COPYRIGHT-library.html", "licenses/MIT.txt", "licenses/Apache-2.0.txt"])
def test_missing_compiler_notices_fail_the_build(compiler_notices, tmp_path, missing):
    manifest, documentation = compiler_notices
    (documentation / missing).unlink()

    with pytest.raises(RuntimeError, match="Rust standard-library notice"):
        notices.generate(manifest, "x86_64-pc-windows-msvc", tmp_path / "notices")
