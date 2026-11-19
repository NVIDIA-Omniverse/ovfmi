<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Development and Testing

Run these commands from the repository root after [demo setup](../README.md#getting-started).
The generated models and isolated USD helper are required for integration tests.

## Testing

The regular setup installs pytest into the demo environment.

Windows:

```powershell
.\demoapp\.venv\Scripts\python.exe -m pytest tests
.\demoapp\.venv\Scripts\python.exe -m pytest demoapp\tests
```

Linux:

```bash
demoapp/.venv/bin/python -m pytest tests
demoapp/.venv/bin/python -m pytest demoapp/tests
```

The library integration suite exercises all five behavior demos with both
backends without a GPU, verifies reset/replay and conveyor controls, and blocks
FMPy imports in native runs. Run setup first to generate the test models.

The partial-vector CPU regressions run without ovstage. After routing changes or
ovstage dependency updates, run the standalone probe in the prepared demo environment.
It checks unchanged stage vectors through refresh and reset, plus publication of
caller-written vectors. It needs ovstage but no GPU.

```powershell
.\demoapp\.venv\Scripts\python.exe tests\integration\probe_partial_vector_outputs.py --backend native --fmi-version 3 --route stage
```

On Linux:

```bash
demoapp/.venv/bin/python tests/integration/probe_partial_vector_outputs.py --backend native --fmi-version 3 --route stage
```

Run all four combinations: each backend (`native`, `fmpy`) with each FMI version
(`2`, `3`). Success exits with code zero and prints a JSON result. This probe does
not verify externally changed stage-vector components.

For the opt-in graphical matrix on an RTX workstation (five scenes per backend):

```powershell
$env:OVFMI_TEST_RENDER = "headful"
.\demoapp\.venv\Scripts\python.exe -m pytest demoapp/tests/test_backend_rendering.py
Remove-Item Env:OVFMI_TEST_RENDER
```

On Linux:

```bash
OVFMI_TEST_RENDER=headful demoapp/.venv/bin/python -m pytest demoapp/tests/test_backend_rendering.py
```

Use `headless` instead of `headful`
to test rendering without a display server; an RTX GPU is still required.

## Versioning and Releases

When bumping the release version, update `VERSION.md`, `[project].version` in
both `pyproject.toml` and `demoapp/pyproject.toml`, `[package].version` in
`native/Cargo.toml`, and the `ovfmi-native` package entry in `native/Cargo.lock`.
The version-consistency unit test checks the manifests against `VERSION.md`;
Cargo's locked builds also require an up-to-date lockfile. Update documentation
that describes the current source version, but preserve historical versions and
dependency pins. Versions follow Python's PEP 440 format. The internal release workflow
tags the commit used to build its wheel and source distribution as `v<version>`.
Public source publication creates the same version tag on the reviewed,
filtered snapshot in the public repository. For example, version `0.3.0` uses
tag `v0.3.0` in each repository.

The demo application has separate package metadata but uses the same release
version as the library. Its ovfmi dependency accepts compatible patch releases
within that minor series; a minor-version bump also needs that dependency range
updated.


## Runtime Initialization

The sample applies these integration rules for both FMI backends:

- Demo setup pins and installs the supported NVIDIA combination in
  `demoapp/.venv`. Pip uses its configured package index; `PIP_FIND_LINKS`
  can point at matching local wheels. No global/user packages are changed.
- Stage population now goes through ovstage, which ovrtx renders directly in
  zero-copy BORROW mode. There is no private replicated rendering stage or
  legacy compatibility population pass.
- ovphysx attaches to the same caller-owned ovstage as ovrtx. Physics
  outputs are read with `PhysX.read()` and published to output-only ovstage
  ordinals that are never drained back into physics. Renderer-consumed local
  transforms are authored to the same ordinal before it is sealed and rendered
  directly from the shared Fabric.
- Register both schema families from the separate `physx-usd-schemas`
  distribution before renderer construction or any shared USD schema-registry
  access. Discover paths through the ovphysx sample helper; do not import
  `physx_usd_schemas`, which loads stock USD rather than the shared runtime.
- Initialize the renderer before populating the single shared stage, then
  create the physics engine. Populate with `physics_population_descs(rendering=True)`.
  ovrtx uses its public attached-stage API; no private ctypes bridge is needed.
- Read ovstage attribute payloads from `group.columns` while retaining the
  owning group until values are copied.
- PhysX pose reads use explicit query, read-session, and group-fetch operations.
  [physics_compat.py](../demoapp/physics_compat.py) keeps their lifetimes around
  pose publication without changing control/output ordinal separation.
- The implementation is in [physics_schemas.py](../demoapp/physics_schemas.py)
  and [ovstage_renderer.py](../demoapp/ovstage_renderer.py).
  FMI backend selection does not change this sequence.
- On Windows, the app exits directly after a physics run to avoid native DLL
  unload crashes during process shutdown. The simulation has already completed
  at that point.

## Optional Developer Workflows

### Optional USD Helper

Do not build this for normal demo use. The default USD parser path is enough.

`fmi_usd_helper` is a C++ USD parser intended for experiments where the
`usd-core` Python fallback is not desired. It links against USD libraries from
an ovphysx SDK and OpenUSD headers, so it is version-sensitive.

## Project Layout

```text
ovfmi/
  python/                       Shipping ovfmi Python package
  native/                       Rust/fmi-rs runtime and private C ABI
  tools/                        Build-time SSP-to-container conversion
  tests/                        Library unit and integration tests
  demoapp/                      Rendering/physics sample application
    fmu/                        Demo FMU source folders
    ssp/                        Demo SSP source folders
    usd/                        Demo stages and generated model archives
  docs/                         Architecture, schema, and API design documents
```

## Documentation Checks

Keep the complete first-run path in [the root README](../README.md#getting-started).
Keep optional `cuda-python` setup and the other demo options there too;
reference pages must not become mandatory detours or duplicate setup instructions.

Before submitting documentation changes, perform these checks:

- Run `git diff --check` from the repository root; success produces no output.
- Check relative file links and heading anchors, including links from moved pages.
- Check Markdown headings, code fences, tables, and descriptive image text.
- Compare commands, package pins, and public symbols against their source definitions.
- Identify which examples are runnable commands and which are illustrative fragments.
- State what was actually tested; do not equate a documentation check with a fresh
  installation or graphical validation.

Do not add Sphinx-only directives to these GitHub Markdown files.
Keep internal review reports and private infrastructure links out of public
product documentation.
