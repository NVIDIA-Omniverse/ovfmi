<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ovfmi — FMI/SSP co-simulation for OpenUSD digital twins

Load Functional Mock-up Interface (FMI) behavior models from an OpenUSD stage, step them with a native or FMPy runtime, and
synchronize simulation data through a reusable Python API.

*Pre-release. The API and the USD-FMI schema may change between minor versions.*

## What ovfmi Does

Universal Scene Description (USD) describes world state; FMI and System
Structure and Parameterization (SSP) supply simulation behavior. `ovfmi` joins
them through a custom USD schema, so a single USD layer can describe both a scene
and its simulation model in a form that stays readable and composable by USD
tools.

A Functional Mock-up Unit (FMU) is a packaged FMI model.
Applications use the Python API to discover `FmuInstance` and `SspInstance`
prims, instantiate their referenced models through the selected backend, route values between
model variables and USD attributes, and advance the models.

The library supports these operations:

- Declaratively embed individual FMUs and packaged SSP systems in USD.
- Run scalar FMI 2 Real / FMI 3 Float64 Co-Simulation models through Rust/fmi-rs,
  or select the Python FMPy backend through the same public API.
- Use supported SSP systems through prebuilt native container FMUs or the
  Python sequential SSP master. FMPy prepares native containers at build time.
- Map FMI variables to USD attributes, including selected vector components and
  ranges.
- Step deterministically and synchronously, with explicit data routing.

It is aimed at digital-twin, controls and robotics engineers bringing
vendor-neutral FMUs and SSPs into USD-based workflows, and at integrators
assembling standards-based simulation applications.

## Requirements

The library has the following requirements:

- Python 3.10 to 3.13, on Linux x86-64 (glibc 2.28 or newer) or Windows x64.
- NumPy, which installs with this package.
- FMPy for the `fmpy` backend, installed through the optional extra below.
- The ovfmi native library, included in the platform wheel.
- A compatible ovstage environment for the caller-owned stage, and an isolated
  OpenUSD Python environment for schema discovery.

The NVIDIA Omniverse libraries are deliberately not automatic pip dependencies.
Install ovstage separately for stage integration. ovrtx and ovphysx are used by
the sample application, which ships with the source rather than in this package.

## Installation

When the matching release is available from your package index, install it with:

```bash
python -m pip install ovfmi
```

The platform wheel includes the default native backend and requires no Rust
toolchain. Building from a source archive requires Rust/Cargo and a platform C
linker; the build fetches the dependencies pinned in `native/Cargo.lock`.
For the optional Python backend, install `ovfmi[fmpy]` and select
`FmiHostConfig(backend="fmpy")`. Native FMI execution does not embed Python,
but the public API and USD routing remain Python.

## Public Python API

Import supported symbols directly from `ovfmi`. The package's `__all__` is the
compatibility boundary; submodules and names beginning with an underscore are
implementation details.

After installation, display the API documentation with this command:

```bash
python -c "from ovfmi import FmiHost; help(FmiHost)"
```

The command prints the host's public methods. Actual model execution requires
a caller-owned stage, the original USD source, and its model assets.
This package description is not the graphical demo setup guide; use the
root README in the source checkout for that complete workflow.

The public symbols are:

| Symbol | Purpose |
|---|---|
| `FmiHost` | Owns FMI/SSP instances, routing state and simulation lifecycle for one caller-owned ovstage. |
| `FmiHostConfig` | Selects the backend and configures discovery, schema validation, SSP support and missing-input behavior. |
| `MissingInputPolicy` | Selects how unavailable mapped inputs are initialized. |
| `PopulationReport`, `InstanceInfo` | Describe instances created during attachment. |
| `AttributeWrite` | Supplies input values identified by USD prim path and attribute name. |
| `ReadResult`, `ReadGroup` | Provide an owned snapshot of output values in USD space. |

Every public class and method carries detailed docstrings covering ownership,
filtering, operation tokens, ordinals and backend semantics. Both backends are
synchronous and use the same completion-token lifecycle for `step()` and `write()`.

## Known Limitations

The native runtime supports scalar real Co-Simulation variables, not all FMI
types or interfaces. Native SSP conversion supports flat FMI 2 Real systems;
its container propagation schedule differs from the Python sequential master,
so trajectories need not be identical. Refer to the source repository's runtime
documentation for the full support boundary.

Schema discovery needs the original USD file. The currently supported ovstage
release does not expose data authored with unregistered custom schemas, which
includes the USD-FMI `FmuInstance`, `SspInstance`, `FmuConnection` and
`FmuMapping` prims, so instances cannot be discovered from a populated ovstage
alone. `FmiHost.attach_ovstage()` therefore accepts the source file through its
`source_asset` argument and pre-parses it with OpenUSD in an isolated helper
process. Simulation values still flow through the caller-owned ovstage; the
auxiliary parse supplies only the schema description and initial authored
values.

## Standards and References

The implementation builds on these standards and tools:

- [FMI standard](https://fmi-standard.org/)
- [SSP standard](https://ssp-standard.org/)
- [OpenUSD](https://openusd.org/)
- [FMPy](https://github.com/CATIA-Systems/FMPy)

## License

Apache-2.0. Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
