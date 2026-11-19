<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ovfmi Documentation

Use this index for the ovfmi Python library and its optional graphical demo.
The [project overview](../README.md) explains the product and intended audience.
FMI is the Functional Mock-up Interface; SSP is System Structure and
Parameterization. An FMU is a packaged FMI model.

## Get Started

Start with **[Getting Started in the README](../README.md#getting-started)**.
That single page takes you from prerequisites and cloning to a running demo
on Windows or Linux. You do not need a separate library install first.

After the demo works, use the [example catalog](../examples/README.md) to find
scene sources, or [setup options in the README](../README.md#build-options) to change setup.
For an existing application that does not need the demo, use
[library-only setup](library-only-setup.md).

## Concepts and Reference

The following pages describe integration boundaries and supported behavior:

- [Architecture](architecture.md): library ownership, application responsibilities, and the runtime layers.
- [Public Python API](api.md): supported symbols, lifecycle, and data routing.
- [USD-FMI schema](USD-FMI-SCHEMA.md): instance prims, connections, and variable mappings.
- [Native and FMPy backends](native-runtime-poc.md): supported models, SSP preparation, and numerical differences.
- [Demo command reference](../README.md#command-reference): viewer and simulation options.

## Diagnose Problems

Use the [troubleshooting guide](troubleshooting.md) for installation and startup
failures. Check [known issues](../README.md#known-issues) for current limitations. Follow the [security policy](../SECURITY.md)
for vulnerability reports.

## Develop and Release

Contributor and maintainer material is separate from application setup:

- [Development and testing](development.md): CPU tests, optional GPU tests, and initialization order.
- [Changes from 0.2 to 0.3](../CHANGES.md): release changes and compatibility.

The [package description](public/README.md) supplies the wheel's metadata page.
It is not a separate documentation site. Historical investigation files in
this directory are not current setup instructions.
