# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect the tested platform wheels and their shared source archive."""

import hashlib
import os
from pathlib import Path
import shutil

from packaging.tags import Tag
from packaging.utils import parse_wheel_filename


def main():
    version = Path("VERSION.md").read_text().strip()
    artifact_name = os.environ["ARTIFACT_NAME"]
    source = Path("platform-dist")
    destination = Path("dist")
    destination.mkdir()
    for platform, tag in (
        ("linux-x86_64", "manylinux_2_28_x86_64"),
        ("windows-amd64", "win_amd64"),
    ):
        directory = source / f"{artifact_name}-{platform}"
        wheels = list(directory.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"Expected one {platform} wheel, found {wheels}")
        wheel = wheels[0]
        name, wheel_version, _, tags = parse_wheel_filename(wheel.name)
        if name != "ovfmi" or str(wheel_version) != version:
            raise ValueError(f"Unexpected distribution: {wheel.name}")
        if Tag("py3", "none", tag) not in tags or any(t.platform == "any" for t in tags):
            raise ValueError(f"Expected a native {tag} wheel: {wheel.name}")
        shutil.copy2(wheel, destination / wheel.name)

    sdist = source / f"{artifact_name}-source" / f"ovfmi-{version}.tar.gz"
    digest = hashlib.sha256(sdist.read_bytes()).hexdigest()
    if digest != os.environ["SDIST_SHA256"]:
        raise ValueError(f"Source archive checksum mismatch: {sdist.name}")
    shutil.copy2(sdist, destination / sdist.name)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"version={version}\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
        summary.write(f"## Tested ovfmi {version} distributions\n\n")
        summary.write(f"Source archive SHA-256: `{digest}`\n\n")
        for path in sorted(destination.iterdir()):
            summary.write(f"- `{path.name}`\n")


if __name__ == "__main__":
    main()
