# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report NVIDIA package versions without importing their native runtimes."""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def requirements():
    """Read exact package pins from the installation source of truth."""
    path = Path(__file__).with_name("requirements.txt")
    return dict(line.strip().split("==") for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#"))


def describe_runtime(check=False):
    """Print installed versions and optionally enforce setup's pinned packages."""
    pins = requirements()
    installed = {}
    for name in pins:
        try:
            installed[name] = version(name)
        except PackageNotFoundError:
            installed[name] = "not installed"
    print("NVIDIA runtime packages: " + ", ".join(
        f"{name}=={value}" for name, value in installed.items()), flush=True)
    matches = installed == pins
    if not matches:
        message = "Installed NVIDIA packages do not match requirements.txt. Rerun demo setup; see README.md."
        if check:
            raise RuntimeError(message)
        print("WARNING: " + message, flush=True)
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    describe_runtime(parser.parse_args().check)
