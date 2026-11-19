# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real Linux setup preflight without pip, downloads or builds."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


_SETUP = Path(__file__).resolve().parents[2] / "demoapp" / "setup.sh"


class TestLinuxCargoSetup:
    # These are POSIX shell/permission tests, not skipped backend coverage on
    # Windows (CI rejects skipped tests). Linux CI collects and runs this class.
    __test__ = os.name == "posix"

    @pytest.fixture
    def sandbox(self, tmp_path):
        home = tmp_path / "user home"
        home.mkdir()
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        # No system Cargo/Python is reachable. Exit at the first venv command,
        # proving the preflight ran before any package/environment changes.
        for name in ("dirname", "mkdir", "mktemp", "rmdir"):
            (bin_dir / name).symlink_to(shutil.which(name))
        cargo = bin_dir / "cargo"
        cargo.write_text("#!/bin/sh\nexit 99\n")
        cargo.chmod(0o755)
        python = bin_dir / "python3"
        python.write_text(
            "#!/bin/sh\n"
            'printf "CHILD_CARGO_HOME=%s\\n" "${CARGO_HOME-UNSET}"\n'
            'printf "CHILD_RUSTUP_HOME=%s\\n" "$RUSTUP_HOME"\n'
            "exit 73\n"
        )
        python.chmod(0o755)
        env = dict(os.environ, HOME=str(home), PATH=str(bin_dir),
                   RUSTUP_HOME=str(tmp_path / "rustup"))
        for key in ("CARGO_HOME", "XDG_CACHE_HOME", "SKIP_NATIVE_BUILD", "BASH_ENV"):
            env.pop(key, None)
        return home, bin_dir, env

    def run_setup(self, sandbox, **overrides):
        _, _, env = sandbox
        return subprocess.run(
            [shutil.which("bash"), str(_SETUP)],
            env=dict(env, **overrides), capture_output=True, text=True, timeout=30,
        )

    def assert_reached_venv(self, result, cargo_home, sandbox):
        assert result.returncode == 73, result.stdout + result.stderr
        assert f"CHILD_CARGO_HOME={cargo_home}\n" in result.stdout
        assert f"CHILD_RUSTUP_HOME={sandbox[2]['RUSTUP_HOME']}\n" in result.stdout

    @pytest.fixture
    def readonly(self):
        if os.geteuid() == 0:
            pytest.skip("Permission-denial tests must run as a non-root user")
        paths = []

        def protect(path):
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o555)
            paths.append(path)
            return path

        yield protect
        for path in reversed(paths):
            path.chmod(0o755)

    def test_default_writable_home_is_unchanged(self, sandbox):
        result = self.run_setup(sandbox)
        self.assert_reached_venv(result, "UNSET", sandbox)
        assert not result.stderr
        assert not (sandbox[0] / ".cache").exists()
        assert not list(sandbox[0].rglob(".ovfmi-write-test.*"))

    def test_explicit_writable_home_keeps_settings(self, sandbox, tmp_path):
        cargo = tmp_path / "custom cargo"
        cargo.mkdir()
        (cargo / "config.toml").write_text('[net]\noffline = true\n')
        result = self.run_setup(sandbox, CARGO_HOME=str(cargo))
        self.assert_reached_venv(result, cargo, sandbox)
        assert not result.stderr
        assert (cargo / "config.toml").read_text() == '[net]\noffline = true\n'

    @pytest.mark.parametrize("nested", [False, True])
    def test_readonly_cache_falls_back(self, sandbox, tmp_path, readonly, nested):
        original = tmp_path / "system cargo"
        protected = readonly(original / "git" / "db" if nested else original)
        result = self.run_setup(sandbox, CARGO_HOME=str(original))
        fallback = sandbox[0] / ".cache" / "ovfmi" / "cargo"
        self.assert_reached_venv(result, fallback, sandbox)
        assert "not writable" in result.stderr and "this setup run only" in result.stderr
        assert (protected.stat().st_mode & 0o777) == 0o555
        assert not list(tmp_path.rglob(".ovfmi-write-test.*"))

    def test_default_readonly_home_uses_xdg_cache(self, sandbox, tmp_path, readonly):
        readonly(sandbox[0] / ".cargo")
        xdg = tmp_path / "xdg cache"
        result = self.run_setup(sandbox, XDG_CACHE_HOME=str(xdg))
        self.assert_reached_venv(result, xdg / "ovfmi" / "cargo", sandbox)
        assert "CARGO_HOME" not in sandbox[2]  # Parent environment is unchanged.

    def test_unusable_fallback_fails_before_installation(self, sandbox, tmp_path, readonly):
        original = readonly(tmp_path / "system cargo")
        readonly(sandbox[0] / ".cache")
        result = self.run_setup(sandbox, CARGO_HOME=str(original))
        assert result.returncode == 1
        assert "Fallback Cargo cache is unavailable" in result.stderr
        assert "Set CARGO_HOME" in result.stderr
        assert "Creating app venv" not in result.stdout

    @pytest.mark.parametrize("name", ["config", "config.toml", "credentials", "credentials.toml"])
    def test_custom_settings_are_not_silently_lost(self, sandbox, tmp_path, readonly, name):
        original = tmp_path / "configured cargo"
        original.mkdir()
        content = "sentinel configuration or credential content\n"
        (original / name).write_text(content)
        readonly(original)
        result = self.run_setup(sandbox, CARGO_HOME=str(original))
        assert result.returncode == 1
        assert "automatic fallback would bypass" in result.stderr
        assert name in result.stderr
        assert content.strip() not in result.stdout + result.stderr
        assert (original / name).read_text() == content
        assert not (sandbox[0] / ".cache").exists()
        assert "Creating app venv" not in result.stdout

    def test_missing_cargo_fails_before_installation(self, sandbox):
        (sandbox[1] / "cargo").unlink()
        result = self.run_setup(sandbox)
        assert result.returncode == 1
        assert "Rust/Cargo is required" in result.stderr
        assert "Creating app venv" not in result.stdout
        assert not (sandbox[0] / ".cargo").exists()

    def test_fmpy_only_setup_does_not_touch_cargo(self, sandbox, tmp_path, readonly):
        (sandbox[1] / "cargo").unlink()
        original = readonly(tmp_path / "system cargo")
        result = self.run_setup(sandbox, CARGO_HOME=str(original), SKIP_NATIVE_BUILD="1")
        self.assert_reached_venv(result, original, sandbox)
        assert not result.stderr
        assert not (sandbox[0] / ".cache").exists()
