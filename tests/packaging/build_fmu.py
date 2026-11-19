# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a platform FMU without FMPy or FMI header dependencies."""

import argparse
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile


def build_fmu(output: Path, initialize_from_input: bool = False) -> None:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("The packaging fixture requires x86-64")
    source = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="ovfmi-fixture-") as directory:
        build = Path(directory)
        if sys.platform == "win32":
            platform_dir = "win64"
            library = build / "Integrator.dll"
            command = [
                "cl", "/nologo", "/LD", "/MT", "/O2", "/TC",
                str(source / "integrator.c"), f"/Fe:{library}",
            ]
        elif sys.platform == "linux":
            platform_dir = "linux64"
            library = build / "Integrator.so"
            command = [
                "cc", "-shared", "-fPIC", "-O2", str(source / "integrator.c"),
                "-o", str(library),
            ]
        else:
            raise RuntimeError(f"Unsupported fixture platform: {sys.platform}")
        if initialize_from_input:
            command.append("/DINITIALIZE_FROM_INPUT" if sys.platform == "win32"
                           else "-DINITIALIZE_FROM_INPUT")
        subprocess.run(command, cwd=build, check=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(source / "modelDescription.xml", "modelDescription.xml")
            archive.write(library, f"binaries/{platform_dir}/{library.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize-from-input", action="store_true")
    args = parser.parse_args()
    build_fmu(args.output.resolve(), args.initialize_from_input)
