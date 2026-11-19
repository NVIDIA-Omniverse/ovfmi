<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Use these checks if [demo setup](../README.md#getting-started) or a demo run fails.
For current limitations, refer to [known issues](../README.md#known-issues).

## Blank Window During Renderer Startup

The display window is created before the renderer finishes initialization and
may be blank or unresponsive while a native call blocks. The demo prints flushed
`[startup]` messages before and after renderer creation, USD population, stage
attachment, and the first render step, including elapsed time on completion.

Stage attachment and the first frame can take several minutes while ovrtx
compiles and caches shaders and ray-tracing pipelines.
These messages identify the pending operation, not shader-compilation progress:
an extended pause should not automatically be attributed to shaders without
native-log evidence. If startup does not finish, capture the
last `[startup]` message and subsequent native logs when reporting the problem.
`--headless --duration 1` can help distinguish a display-window issue from a
renderer startup issue; `--duration` limits simulation time, not startup time.

## Compiler Not Found on Windows

Install Visual Studio Build Tools with **Desktop development with C++**. The
setup script can usually find the Visual Studio compiler environment from a
normal PowerShell. Only use `-SkipFmuBuild` if you already have generated model
archives for this platform; it cannot make a fresh checkout runnable without
a compiler.

## Conveyor Physics Is Disabled

Confirm that all three NVIDIA runtime packages are installed in the demo
virtual environment:

Windows:

```powershell
.\demoapp\.venv\Scripts\python.exe -m pip show ovrtx ovstage ovphysx
```

Linux:

```bash
demoapp/.venv/bin/python -m pip show ovrtx ovstage ovphysx
```

If a package is missing or the combination is wrong, rerun the normal setup
command. No deletion or base-environment installation is needed.
Setup must also install `physx-usd-schemas`.
An error about missing PhysX schema families or registration after population
requires restarting the demo after setup; adding schemas to an already
populated process cannot repair its schema registry.
The demo registers schemas before initializing the shared runtime.
Refer to [runtime initialization](development.md#runtime-initialization);
changing import order alone is not a general recovery step.

## Generated FMU Archives Are Missing

Rerun setup, or rebuild the generated archives directly.

Windows:

```powershell
.\demoapp\.venv\Scripts\python.exe demoapp\build_fmu.py
```

Linux:

```bash
demoapp/.venv/bin/python demoapp/build_fmu.py
```

## Wrong Windows FMU Architecture

The error `WinError 193` usually means the FMU DLL was built for the wrong Windows architecture,
for example by an x86 Visual Studio compiler environment while running 64-bit
Python. Rerun setup from a normal PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File demoapp\setup.ps1
```

Rerunning setup should print a line like:

```text
Using Visual Studio environment: ...\vcvars64.bat
```

If that line does not appear, setup did not find the x64 compiler environment.
Reopen the Visual Studio Build Tools installer and confirm **Desktop
development with C++** and **MSVC v143 x64/x86 build tools** are installed.
`build_fmu.py` checks generated DLL architecture before packaging FMUs so this
failure is caught during setup rather than later during simulation.

## Live Display Does Not Open on Linux

Check that the machine has a display server and OpenGL runtime libraries. On
headless systems, use `--headless` or `--png`; an RTX GPU is still required.

## Duplicate Carbonite Plugin Warnings

Warnings can appear when ovrtx and ovphysx are loaded in the same Python
process. They are expected with the currently tested package combination as
long as physics initializes and the simulation runs.

## Missing Native Runtime or FMPy

For a native library load failure, repeat the [native source build](library-only-setup.md#install-from-source)
in the environment used to run your application. A successful Python import
alone does not verify that the shared library or a model can load.

If the error is `The fmpy backend requires FMPy; install the ovfmi[fmpy] extra`,
install the FMPy extra as described in [library setup](library-only-setup.md#install-from-source).
Demo setup installs that extra automatically. No backend silently falls back
to the other one.

## Unwritable Cargo Cache on Linux

If setup asks you to select `CARGO_HOME`, choose a directory writable by your
account and preserve any custom Cargo configuration and credentials there.
Rerun setup with that environment variable. Do not run the entire setup as
root. The [Linux setup instructions](../README.md#linux)
explain the automatic fallback when no custom configuration needs preservation.
