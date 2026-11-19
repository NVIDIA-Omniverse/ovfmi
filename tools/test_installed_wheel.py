#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install a wheel in isolation and execute an FMU through its native runtime."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--version", help="expected installed distribution version")
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    fixtures = Path(__file__).resolve().parents[1] / "tests" / "packaging"
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "OVFMI_NATIVE_LIBRARY", "USD_PYTHON"):
        env.pop(key, None)

    with tempfile.TemporaryDirectory(prefix="ovfmi-wheel-test-") as directory:
        root = Path(directory)

        def run(*command: str) -> None:
            subprocess.run(command, cwd=root, env=env, check=True)

        def environment(name: str) -> Path:
            target = root / name
            venv.EnvBuilder(with_pip=True).create(target)
            return target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

        runtime_python = environment("runtime")
        usd_python = environment("usd")
        run(str(runtime_python), "-I", "-m", "pip", "install", str(wheel))
        run(str(usd_python), "-I", "-m", "pip", "install", "usd-core")
        if args.version:
            run(
                str(runtime_python), "-I", "-c",
                "import sys; from importlib.metadata import version; "
                "assert version('ovfmi') == sys.argv[1], version('ovfmi')",
                args.version,
            )
        fmu = root / "Integrator.fmu"
        run(sys.executable, "-I", str(fixtures / "build_fmu.py"), "--output", str(fmu))
        smoke = root / "smoke_native.py"
        shutil.copy2(fixtures / "smoke_native.py", smoke)
        env["USD_PYTHON"] = str(usd_python)
        run(str(runtime_python), "-I", str(smoke), str(fmu))


if __name__ == "__main__":
    main()
