# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


def native_target():
    if struct.calcsize("P") != 8 or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("ovfmi native wheels require x86_64 Python on Linux or Windows")
    if sys.platform == "linux":
        return "x86_64-unknown-linux-gnu", "libovfmi_native.so"
    if sys.platform == "win32":
        return "x86_64-pc-windows-msvc", "ovfmi_native.dll"
    raise RuntimeError("ovfmi native wheels support Linux and Windows only")


class NativeBuild(build_py):
    def run(self):
        if self.editable_mode:
            return
        target, name = native_target()
        super().run()
        root = Path(__file__).resolve().parent
        target_dir = Path(self.get_finalized_command("build").build_temp).resolve() / "cargo"
        subprocess.run(
            [
                "cargo", "build",
                "--manifest-path", str(root / "native" / "Cargo.toml"),
                "--release", "--locked",
                "--target", target,
                "--target-dir", str(target_dir),
            ],
            check=True,
            cwd=root,
        )
        destination = Path(self.build_lib) / "ovfmi" / name
        shutil.copy2(target_dir / target / "release" / name, destination)
        subprocess.run(
            [
                sys.executable, str(root / "tools" / "generate_native_notices.py"),
                "--manifest", str(root / "native" / "Cargo.toml"),
                "--target", target,
                "--output", str(destination.parent / "_native_licenses"),
                "--target-dir", str(target_dir),
            ],
            check=True,
            cwd=root,
        )


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class NativeWheel(bdist_wheel):
    def get_tag(self):
        native_target()
        _, _, platform_tag = super().get_tag()
        # The ctypes library uses no Python ABI.
        return "py3", "none", platform_tag


setup(
    distclass=NativeDistribution,
    cmdclass={"build_py": NativeBuild, "bdist_wheel": NativeWheel},
)
