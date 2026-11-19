#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

curl --proto '=https' --tlsv1.2 --fail --silent --show-error https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain 1.97.1
export PATH="$HOME/.cargo/bin:/opt/python/cp312-cp312/bin:$PATH"
python -m pip install build

python .github/scripts/build-wheel.py "$1" --sha256 "$2" --outdir /tmp/ovfmi-wheels
auditwheel repair --plat manylinux_2_28_x86_64 --wheel-dir dist /tmp/ovfmi-wheels/*.whl
