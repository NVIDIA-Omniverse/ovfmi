<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Library-Only Setup

Use this guide to install ovfmi from a source checkout without the graphical
application. To run the supplied rendered examples, follow [Getting Started in the README](../README.md#getting-started)
instead; it creates the required environments and builds the models for you.

## Prerequisites

For a source installation, provide the following tools:

- Python 3.10–3.13 on Windows x64 or Linux x86-64.
- [Rust through rustup](https://rustup.rs/) and Cargo.
- A platform C linker: MSVC x64 on Windows, or a C toolchain on Linux.
- Network access to retrieve the locked Cargo and Python dependencies.
- A writable Cargo cache. If the system's `CARGO_HOME` is unwritable, select a
  user-owned cache before building; the demo script's automatic fallback does
  not apply to a manual Cargo invocation.

Stage integration additionally requires a compatible ovstage package and a
separate Python environment containing `usd-core` for OpenUSD schema discovery.
The library does not require ovrtx, ovphysx, or an NVIDIA GPU.
Refer to the [demo dependencies](../README.md#nvidia-dependencies) for
the supported NVIDIA package combination.

## Install From Source

Start in the repository root. Use your checkout of the intended release;
this guide does not assume that the prerelease is already available on PyPI.

### Create an Environment on Windows

This example uses Python 3.12. Install that version or replace `-3.12` with
another installed, supported version. From PowerShell, create and activate
a Python virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell policy prevents activation, run the commands in this guide with
`.\.venv\Scripts\python.exe` instead of `python`. Activation is not required.

### Create an Environment on Linux

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install and Build

In that environment, install the editable package and build the native library:

```bash
python -m pip install -e .
cargo build --manifest-path native/Cargo.toml --release --locked
```

Cargo must finish successfully and place the shared library in
`native/target/release`. Both Windows and Linux use the same Cargo command.
If it fails, resolve the compiler, cache, or dependency error before attaching
a model; [troubleshooting](troubleshooting.md) covers common failures.

To use the Python FMPy backend as well, install its optional dependency:

```bash
python -m pip install -e ".[fmpy]"
```

A platform wheel includes the native shared library and does not require Rust.
When installing a supplied wheel, install that artifact with pip instead of
using the source-build commands. Linux wheels require glibc 2.28 or newer.
An editable install uses the local Cargo output; `OVFMI_NATIVE_LIBRARY` can
override that location for development.

## Configure Stage Integration

Install the ovstage version compatible with your application's other NVIDIA
libraries. The current supported integration uses:

```bash
python -m pip install ovstage==0.4.0.1794520352
python -m venv .usd_venv
```

Install OpenUSD in the helper environment, not the application's environment.
On Windows, run:

```powershell
.\.usd_venv\Scripts\python.exe -m pip install usd-core
$env:USD_PYTHON = (Resolve-Path .\.usd_venv\Scripts\python.exe).Path
```

On Linux, run:

```bash
.usd_venv/bin/python -m pip install usd-core
export USD_PYTHON="$PWD/.usd_venv/bin/python"
```

Keep the original USD source and referenced model assets available at runtime.
The [public API](api.md) requires `source_asset` during attachment because
ovstage does not expose the custom USD-FMI prims.

## Verify the Python Installation

In the application environment, run:

```bash
python -c "from importlib.metadata import version; from ovfmi import FmiHost, FmiHostConfig; print('ovfmi', version('ovfmi'), 'backend', FmiHostConfig().backend)"
```

For this source version, the result is `ovfmi 0.3.3 backend native`.
This verifies the installed Python API, not native loading, model execution,
or stage compatibility. The [demo verification](../README.md#headless-verification)
tests an end-to-end graphical application, and the [CPU integration tests](development.md#testing)
test model execution without a GPU.

Continue with the [public API lifecycle](api.md#lifecycle) for your own application.
