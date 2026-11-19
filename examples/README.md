<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Example Catalog

The examples use the graphical application in `demoapp/`; this directory is
an index, not a second copy of the sample code. Complete [demo setup](../README.md#getting-started)
before running them. Setup builds the Functional Mock-up Units (FMUs) and
System Structure and Parameterization (SSP) archives from their source.

## Choose a Scene

All five behavior examples support the native and FMPy backends.
The source targets ovfmi 0.3.3 and the
[documented NVIDIA dependencies](../README.md#nvidia-dependencies).

The scenes demonstrate the following behavior:

| Scene | Expected Behavior |
|---|---|
| [Bouncing ball](../demoapp/usd/fmi_parser_test.usda) | One FMU drives a sphere's vertical position without ovphysx. |
| [Proportional–derivative controller](../demoapp/usd/pd_controller_test.usda) | An FMU reads body state and applies a restoring force through ovphysx. |
| [Two-FMU orbit](../demoapp/usd/two_fmu_orbit.usda) | A trajectory generator and controller exchange values through USD attributes. |
| [SSP orbit](../demoapp/usd/ssp_orbit.usda) | An SSP hides the generator/controller wiring behind system-level connectors. |
| [Conveyor](../demoapp/usd/conveyor/ConveyorFMI.usda) | A sensor and SSP controller drive five roller zones and reverse package travel. |
| [Conveyor asset preview](../demoapp/usd/conveyor/Conveyor.usda) | The base scene omits the FMI control overlay; use it to inspect the asset. |

## Run an Example

Use [Getting Started in the README](../README.md#getting-started) for the
conveyor's complete setup and launch instructions. The
[additional demo commands](../README.md#try-more-examples) cover the other scenes
on Windows and Linux. There is no separate examples environment to install.

Each command defaults to the native backend; append `--backend fmpy` to compare
the Python master. Native and FMPy trajectories need not be identical because
their [SSP coupling schedules differ](../docs/native-runtime-poc.md#coupling-and-numerical-differences).

After setup, [headless verification](../README.md#headless-verification)
provides a bounded run with expected console output. It still requires an RTX GPU.
For failures, refer to [troubleshooting](../docs/troubleshooting.md).

## Source and Assets

The examples keep their code and assets in these locations:

- [Application loop](../demoapp/main.py): integration through the public ovfmi API.
- [Model builder](../demoapp/build_fmu.py): FMU compilation and SSP preparation.
- `demoapp/fmu/`: FMU source models.
- `demoapp/ssp/`: SSP source descriptions.
- `demoapp/usd/`: scene assets and generated runtime archives.

Use [development and testing](../docs/development.md) to verify changes.
Unit and CPU integration tests do not replace a graphical check of these scenes.
