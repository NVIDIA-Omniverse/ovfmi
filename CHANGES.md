<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changes

## 0.3.3

- Updated stock fmi-rs to v0.3.5 and the locked Rust ZIP dependency to 9.6.1.
- Adopted ovstage 0.4, ovrtx 0.7, and ovphysx 0.8 as the sole supported demo
  combination. Setup pins the exact builds and their separate PhysX schemas.
- Removed the validated/experimental profile split, legacy API branches, and
  the ovrtx 0.5 child-transform workaround. Rendering uses the public
  attached-stage API; both native and FMPy backends remain supported.
- CI tests the current combination on every supported Python version.

## 0.3.0 — changes from 0.2.0

The following records the original 0.3.0 release. Its dependency profiles and
limitations are historical; see the README for current setup and support.

### Native FMI execution alternative

- Added a Rust/fmi-rs native runtime alongside the existing Python FMPy master.
  Both use the public Python `FmiHost` API; choose with
  `FmiHostConfig(backend="native" | "fmpy")` or the demo's `--backend` option.
  Native is now the default. Missing backends fail explicitly without fallback.
- Native execution loads, initializes, steps, resets, and releases FMI 2 and
  FMI 3 Co-Simulation models, with scalar Real/Float64 routing. FMU archives,
  unpacked models, multiple instances, authored start values, and USD component
  mappings are supported.
- Added build-time conversion of the supported SSP subset to native container
  FMUs using FMPy. Conveyor and SSP orbit produce both `.ssp` and `.native.fmu`
  artifacts. FMPy is needed for preparation, not native FMI execution.
- Native stepping does not embed Python. The host API, USD routing, schema
  discovery, and demo application remain Python. The native runtime exposes an
  internal C ABI, not a public C equivalent of the full USD-facing host API.
- Uses stock upstream fmi-rs and FMPy 0.3.32 container runtimes, without local
  runtime forks. The converter normalizes container metadata as documented in
  [runtime details](docs/native-runtime-poc.md).

### Examples, dependencies, and setup

- Both backends execute bouncing ball, PD controller, two-FMU orbit, SSP orbit,
  and conveyor through the same ovstage/ovrtx/ovphysx demo integration.
- Setup builds the native library, demo FMUs, and native SSP containers. A
  FMPy-only setup remains available through `-SkipNativeBuild` (Windows) or
  `SKIP_NATIVE_BUILD=1` (Linux).
- Retained the public 0.2 NVIDIA runtime baseline: ovrtx `0.4.0.346409`,
  ovstage `0.1.0.346039`, and ovphysx `0.5.9`. Native FMI does not require a
  dependency upgrade. Newer-runtime compatibility code remains available,
  including PhysX schema registration and the updated physics read API/Warp
  arrays; legacy schema registration remains supported.
- Setup now installs a selectable, exactly pinned NVIDIA package profile into
  the isolated demo environment: `validated` by default, or `experimental`
  (ovrtx `0.5.0.377615`, ovstage `0.2.0.377349`, ovphysx `0.6.3`). Global Python
  packages are untouched. Startup reports actual versions and warns about
  experimental or custom combinations; profile selection is independent of
  the FMI backend and does not change simulation ordering or solver settings.
- Added Pillow to normal demo dependencies for image output and GPU tests.
- Aligned Python, demo, native crate, and `VERSION.md` metadata to `0.3.0`.

### Rendering compatibility and diagnostics

- Accepts both legacy `LdrColor` keys and ovrtx 0.5 full-path render-output keys,
  fixing a blank display even when the renderer has produced color frames.
- Validates color outputs and reports missing, ambiguous, empty, or unsupported
  images instead of silently discarding frames. Logs read/presented frame counts.
- Adds named startup phases, elapsed timings, and cold-cache shader-compilation
  warnings. These identify blocking calls, not shader-compilation percentages.
- Adds an opt-in `--refresh-rigid-descendants`, ovrtx 0.5.0-scoped workaround
  for stale meshes below moving rigid bodies: republish retained child-local transforms while preserving FMI-driven
  child updates and keeping rendering output out of physics control ordinals.

### Testing and CI

- Added both-backend scene integration, reset/replay, controller, and native
  no-FMPy-import coverage, plus native Rust tests that execute without Python.
- Internal CI builds the native library and demo models, tests both backends
  on Linux Python 3.10–3.13 and Windows Python 3.12, and checks distribution
  metadata. GPU rendering remains an opt-in test, not a GitHub-hosted CI claim.
- GPU tests cover all five scenes with both backends, verify delivered frames,
  check conveyor motion with a stationary camera, and measure physical box
  height over 15 seconds to catch jumps after reversal. The restored runtime
  baseline also passed 60-second conveyor runs with both backends on Windows
  and Linux; bounded tests do not imply universal long-run stability.

### Compatibility boundaries and known issues

- Native SSP containers use a different coupling schedule from the Python
  sequential master, introducing a communication-step delay. Transient
  trajectories are not expected to be bit-identical.
- The native bridge supports scalar real Co-Simulation, not the complete FMI
  or SSP standards. See [supported features](docs/native-runtime-poc.md).
- Platform wheels bundle the native shared library for Windows x64 and
  Linux x86-64. Source builds require Rust/Cargo and a platform C linker.
- The experimental ovrtx/ovstage combination still renders conveyor box materials
  incorrectly when both rendering and physics domains are populated.
- Fixed dropped `physx*` attributes on the experimental stack by registering
  ovphysx's codeless schemas before shared USD schema-registry access, including
  renderer construction. The validated stack retains its legacy initialization
  order; the demo simulation loop is unchanged.
- Large vertical box motion after conveyor reversal remains reproducible with
  the native backend on Windows and Linux using the experimental stack, even
  after correcting schema registration. Its remaining cause is not established;
  use the validated profile for reliable operation.
- The OpenUSD pre-parser remains necessary for USD-FMI schema discovery.

See [Known Issues](README.md#known-issues) for the current qualifications.
