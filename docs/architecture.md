<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ovfmi Architecture

`ovfmi` is the repository's primary FMI/SSP simulation product. It owns model
discovery, FMI lifecycle management, and
the routing between USD attribute identities and FMI variables. Rendering,
windowing, camera control, and ovphysx stepping remain application concerns.

Applications select `FmiHostConfig(backend="native")` (the default) or
`FmiHostConfig(backend="fmpy")` and use only the public types exported from
`ovfmi`. The native runtime is a Rust library built on fmi-rs and reached
through a private ctypes bridge. For this backend, FMPy is build tooling:
it translates supported SSP archives into native container FMUs before the
application starts. The FMPy backend instead uses the Python FMI/SSP master.

## Lifecycle

For attachment, input routing, stepping, output consumption, and cleanup,
follow the [public API lifecycle](api.md#lifecycle). The application owns the
populated stage and passes its original USD path to the host.

The public boundary names data in USD space: prim paths and USD attribute
names. FMI variable names, component mapping, FMU extraction, and SSP
coordination remain private.

The ovstage population used by the demo does not expose custom FMI schema
columns. `source_asset` supplies the USD source, which ovfmi parses in an
isolated USD process to discover those columns.

## Backend

The package is intentionally layered as follows:

![The sample application calls FmiHost and its routing and lifecycle layer, then selects NativeBackend through a private C ABI to fmi-rs, or FmpyBackend with the Python FMI/SSP master. A separate build-time path converts SSP source with build_ssp_container.py and FMPy into a native container FMU.](images/architecture.svg)

*Both runtimes share the public Python API. Native SSP containers are prepared
with FMPy before the application starts.*

`NativeBackend` implements the private backend protocol. The Python layer
marshals values and ovstage identities; FMU loading, lifecycle calls, and
stepping execute in the native library. FMPy is not imported by the runtime
path when `native` is selected. Both implement the same private backend
protocol; the demo never selects or accesses private implementation classes.

## Development

Use [development and testing](development.md) for the platform-specific test
commands and runtime initialization rules. For a first demo run, start with
[Getting Started](../README.md#getting-started).
