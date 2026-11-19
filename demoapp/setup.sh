#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# setup.sh - install the Python runtime for ov-fmi and build the demo FMUs.
#
# Defaults:
#   - create/use isolated demoapp/.venv
#   - install the pinned ovrtx, ovstage, and ovphysx packages into that venv
#   - install ovfmi, its Python dependencies, and build-time FMPy tooling
#   - create/use demoapp/.usd_venv for usd-core parsing
#   - build generated FMU and SSP archives into demoapp/usd/
#
# Optional environment variables:
#   CARGO_HOME=/path/to/writable/cargo-home (otherwise checked automatically)
#   SKIP_NATIVE_BUILD=1
#   SKIP_FMU_BUILD=1
#   INSTALL_CUDA_PYTHON=1
#
# Optional fmi_usd_helper build:
#   OVPHYSX_SDK_DIR=/path/to/ovphysx-sdk
#   OPENUSD_INCLUDE_DIR=/path/to/openusd_src

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OVFMI_DIR="$REPO_ROOT"
APP_VENV="$SCRIPT_DIR/.venv"
APP_PYTHON="$APP_VENV/bin/python"
USD_VENV="$SCRIPT_DIR/.usd_venv"
USD_PYTHON="$USD_VENV/bin/python"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# Check before pip or FMU preparation can invoke a native build. Some container
# images set CARGO_HOME to a system directory that is not writable by the user.
cargo_cache_writable() {
    local directory probe
    for directory in "$1" "$1/git" "$1/git/db" "$1/git/checkouts" \
                     "$1/registry" "$1/registry/index" "$1/registry/cache" "$1/registry/src"; do
        mkdir -p -- "$directory" 2>/dev/null || return 1
        probe="$(mktemp -d "$directory/.ovfmi-write-test.XXXXXX" 2>/dev/null)" || return 1
        rmdir -- "$probe" || return 1
    done
}

prepare_cargo_home() {
    local original="${CARGO_HOME:-$HOME/.cargo}"
    local fallback="${XDG_CACHE_HOME:-$HOME/.cache}/ovfmi/cargo"
    local config
    if cargo_cache_writable "$original"; then
        return 0
    fi

    # Relocating config can change relative paths; do not silently discard it
    # or copy credentials into a second location. Let customized installs choose
    # a writable home themselves, preserving their settings and access policy.
    for config in config config.toml credentials credentials.toml; do
        if [ -e "$original/$config" ] || [ -L "$original/$config" ]; then
            echo "ERROR: Cargo cache is not writable: $original" >&2
            echo "It contains $config; automatic fallback would bypass Cargo settings or credentials." >&2
            echo "Set CARGO_HOME to a writable location with your required Cargo configuration, then rerun setup." >&2
            return 1
        fi
    done

    if [[ "$fallback" != /* ]] || ! cargo_cache_writable "$fallback"; then
        echo "ERROR: Cargo cache is not writable: $original" >&2
        echo "Fallback Cargo cache is unavailable: $fallback" >&2
        echo "Set CARGO_HOME to a writable directory and rerun setup; do not run setup with sudo." >&2
        return 1
    fi
    export CARGO_HOME="$fallback"
    echo "WARNING: Cargo cache is not writable: $original" >&2
    echo "Using CARGO_HOME=$CARGO_HOME for this setup run only; dependencies may be downloaded again." >&2
}

if [ "${SKIP_NATIVE_BUILD:-0}" != "1" ]; then
    if ! command -v cargo >/dev/null 2>&1; then
        echo "ERROR: Rust/Cargo is required to build the native ovfmi runtime." >&2
        echo "Install rustup from https://rustup.rs/ and rerun setup." >&2
        exit 1
    fi
    prepare_cargo_home
fi

echo "Creating app venv at $APP_VENV ..."
# Refresh without --clear: retain local packages, but stop inheriting globals.
python3 -m venv "$APP_VENV"

echo "Installing Python packages..."
"$APP_PYTHON" -m pip install --upgrade pip
echo "Installing NVIDIA runtime packages into $APP_VENV ..."
"$APP_PYTHON" -m pip install -r "$REQUIREMENTS"
"$APP_PYTHON" -m pip install -e "$OVFMI_DIR[build]"
"$APP_PYTHON" -m pip install -e "$SCRIPT_DIR"
"$APP_PYTHON" "$SCRIPT_DIR/runtime_dependencies.py" --check

OVRTX_BIN="$("$APP_PYTHON" - <<'EOF'
from pathlib import Path
import ovrtx

print(Path(ovrtx.__file__).resolve().parent / "bin")
EOF
)"
if [ ! -f "$OVRTX_BIN/libovrtx-dynamic.so" ]; then
    echo ""
    echo "ERROR: Installed ovrtx package does not contain libovrtx-dynamic.so at $OVRTX_BIN."
    echo ""
    exit 1
fi
echo "Using ovrtx native library from: $OVRTX_BIN"

if [ "${INSTALL_CUDA_PYTHON:-0}" = "1" ]; then
    "$APP_PYTHON" -m pip install cuda-python
else
    echo "Skipping optional cuda-python bindings. CUDA display requires compatible bindings and runtime libraries."
fi

# --- optionally build fmi_usd_helper ----------------------------------------
HELPER_DIR="$SCRIPT_DIR/fmi_usd_helper"
HELPER_BUILD="$HELPER_DIR/build"

if [ -n "${OVPHYSX_SDK_DIR:-}" ] && [ -n "${OPENUSD_INCLUDE_DIR:-}" ]; then
    echo "Building fmi_usd_helper..."
    cmake -B "$HELPER_BUILD" \
          -S "$HELPER_DIR" \
          -DOVPHYSX_SDK_DIR="$OVPHYSX_SDK_DIR" \
          -DOPENUSD_INCLUDE_DIR="$OPENUSD_INCLUDE_DIR" \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HELPER_BUILD" --config Release
    echo "fmi_usd_helper built at: $HELPER_BUILD/fmi_usd_helper"
    echo "  -> usd-core subprocess fallback will not be used."
else
    echo "Skipping fmi_usd_helper; usd-core subprocess fallback will be used for USD parsing."
fi

# --- isolated usd-core venv (ovrtx refuses to load when pxr is installed) ----
echo "Creating isolated usd-core venv at $USD_VENV ..."
if [ ! -x "$USD_PYTHON" ]; then
    python3 -m venv "$USD_VENV"
fi
echo "Installing usd-core into isolated venv..."
"$USD_PYTHON" -m pip install --upgrade pip
"$USD_PYTHON" -m pip install --quiet usd-core

# --- emit a small env file --------------------------------------------------
ENV_FILE="$SCRIPT_DIR/.env"
cat > "$ENV_FILE" <<ENVEOF
# Source this file before running ov-fmi with a different Python executable:
#   source demoapp/.env
export OVRTX_LIBRARY_PATH_HINT="$OVRTX_BIN"
export OVSTAGE_LIBRARY_PATH_HINT="$OVRTX_BIN"
export PYTHONPATH="$SCRIPT_DIR:\${PYTHONPATH:-}"
export USD_PYTHON="$USD_PYTHON"
ENVEOF

# --- build generated FMU and SSP archives ----------------------------------
if [ "${SKIP_FMU_BUILD:-0}" = "1" ]; then
    echo "Skipping FMU/SSP build because SKIP_FMU_BUILD=1."
else
    echo "Building demo FMUs and SSPs..."
    if [ -n "${CXX:-}" ] || command -v g++ >/dev/null 2>&1; then
        "$APP_PYTHON" "$SCRIPT_DIR/build_fmu.py"
    elif command -v clang++ >/dev/null 2>&1; then
        CXX=clang++ "$APP_PYTHON" "$SCRIPT_DIR/build_fmu.py"
    else
        echo ""
        echo "ERROR: Cannot build FMUs because neither g++ nor clang++ was found."
        echo "       Install a C++17 compiler, or rerun with SKIP_FMU_BUILD=1."
        echo ""
        exit 1
    fi
fi

# --- build the Python-free FMI runtime --------------------------------------
if [ "${SKIP_NATIVE_BUILD:-0}" = "1" ]; then
    echo "Skipping native runtime build; use --backend fmpy."
else
    echo "Building native ovfmi runtime..."
    cargo build --manifest-path "$REPO_ROOT/native/Cargo.toml" --release --locked
fi

echo ""
echo "Setup complete."
echo "Wrote $ENV_FILE"
echo ""
echo "Run:"
echo "  $APP_PYTHON $SCRIPT_DIR/main.py $SCRIPT_DIR/usd/fmi_parser_test.usda"
