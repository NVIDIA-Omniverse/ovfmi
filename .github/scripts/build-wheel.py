# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a wheel from the verified release source archive."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def build(sdist, expected_sha256, outdir):
    digest = hashlib.sha256(sdist.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"Source archive checksum mismatch: {sdist.name}")
    print(f"Building from {sdist.name}; SHA-256: {digest}", flush=True)
    with tempfile.TemporaryDirectory(prefix="ovfmi-wheel-") as temporary:
        with tarfile.open(sdist, "r:gz") as archive:
            archive.extractall(temporary, filter="data")
        sources = list(Path(temporary).iterdir())
        if len(sources) != 1 or not sources[0].is_dir():
            raise ValueError("Source archive must contain one directory")
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir.resolve()), str(sources[0])],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    build(args.sdist, args.sha256, args.outdir)
