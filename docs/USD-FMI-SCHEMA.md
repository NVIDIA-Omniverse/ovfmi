<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# USD-FMI Schema Reference

This document specifies the USD prim types and attributes used to embed
[FMI](https://fmi-standard.org/) co-simulation models (FMUs) and
[SSP](https://ssp-standard.org/) archives into OpenUSD scenes.

## Overview

The schema enables declarative binding between FMI variables and USD scene
attributes.  At runtime an FMI host reads these prims, instantiates the
referenced models, and synchronises data each simulation step.

Two top-level instance prim types are defined:

| Prim type | Purpose |
|-----------|---------|
| `FmuInstance` | References a single `.fmu` archive (FMI 2.0 or 3.0) |
| `SspInstance` | References an `.ssp` archive containing multiple internally-wired FMUs |

Both share the same child prim hierarchy (`FmuConnection` → `FmuMapping`) for
describing how model variables map to USD attributes.

---

## Prim Types

### FmuInstance

A single FMU co-simulation component.

```usda
def FmuInstance "MyController"
{
    asset fmi:fmu = @./path/to/model.fmu@
    bool  fmi:enabled = true
}
```

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `fmi:fmu` | `asset` | Yes | Asset path to the `.fmu` file (resolved relative to the layer) |
| `fmi:enabled` | `bool` | No | Enable/disable this instance (default: `true`) |

---

### SspInstance

An SSP (System Structure and Parameterization) archive encapsulating multiple
FMUs with pre-defined internal wiring.  From the USD perspective the SSP is a
black box — only **system-level connectors** (the SSP's external inputs and
outputs) are exposed as mappings.  Internal FMU variables and inter-FMU
connections remain hidden.

```usda
def SspInstance "OrbitController"
{
    asset fmi:ssp = @./orbit_controller.ssp@
    bool  fmi:enabled = true
}
```

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `fmi:ssp` | `asset` | Yes | Asset path to the `.ssp` archive |
| `fmi:enabled` | `bool` | No | Enable/disable this instance (default: `true`) |

**Design rationale:**  SSP support enables embedding pre-packaged simulation
components without deconstructing them into individual FMU prims.  This is
useful when components are authored and validated as SSPs by external
toolchains, and exposing internal wiring would break encapsulation.

---

### FmuConnection

A child prim of `FmuInstance` or `SspInstance` that groups mappings targeting
the same USD prim(s).

```usda
def FmuConnection "PhysicsIO"
{
    rel fmi:targets = </World/Sphere>
}
```

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `fmi:targets` | `rel` (relationship) | Yes | One or more target prim paths that the mappings read from / write to |
| `fmi:enabled` | `bool` | No | Enable/disable this connection (default: `true`) |

A single `FmuInstance` or `SspInstance` may have multiple `FmuConnection`
children, each targeting different prims.

---

### FmuMapping

A child prim of `FmuConnection` defining a single variable binding between the
FMI model and a USD attribute on the connection's target prim(s).

```usda
def FmuMapping "HeightWrite"
{
    token fmi:fmuAttribute = "h"
    token fmi:usdAttribute = "xformOp:translate"
    token fmi:direction    = "output"
    int2  fmi:usdMapping   = (1, 1)
}
```

**Attributes:**

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `fmi:fmuAttribute` | `token` | Yes | FMI variable name (as declared in `modelDescription.xml`) |
| `fmi:usdAttribute` | `token` | Yes | USD attribute name on the target prim, or a `physx:` routing directive |
| `fmi:direction` | `token` | Yes | `"input"` (USD → FMU) or `"output"` (FMU → USD) |
| `fmi:usdMapping` | `int2` | Yes | Component selector `(offset, count)` — refer to [component selection](#component-selection) |

---

## Attribute Details

### fmi:direction

| Value | Data flow | Description |
|-------|-----------|-------------|
| `"input"` | USD → FMU | The USD attribute value is read each step and set as an FMU input variable |
| `"output"` | FMU → USD | The FMU output variable is read each step and written to the USD attribute |

### Component Selection

The `int2` value `(offset, count)` selects which component(s) of a
multi-component USD attribute participate in the mapping.

| Pattern | Meaning |
|---------|---------|
| `(0, 0)` | Scalar — the entire attribute is a single float |
| `(i, 1)` | Single component at index `i` of a vector or matrix |
| `(i, n)` | Range of `n` components starting at index `i` |

**For `xformOp:translate` (double3):**

- `(0, 1)` = X component
- `(1, 1)` = Y component
- `(2, 1)` = Z component

**For `omni:xform` (4×4 column-major matrix, 16 floats):**

- `(12, 1)` = translation X
- `(13, 1)` = translation Y
- `(14, 1)` = translation Z
- `(0, 3)` = first column (X axis basis)

### fmi:usdAttribute — Standard USD Attributes

Mappings identify attributes on target prims. The current runtime routes
real-valued data; an authored name alone does not guarantee that its type is
supported. Common numeric targets are listed below:

| Attribute | Type | Notes |
|-----------|------|-------|
| `xformOp:translate` | `double3` | Local translation |
| `omni:xform` | `matrix4d` | Full 4×4 transform (ovrtx 0.2.0+ column-major) |
| `inputs:intensity` | `float` | Light intensity |
| Custom attributes | Numeric types | Must be compatible with scalar Real/Float64 routing |

### fmi:usdAttribute — Physics Routing Directives

Special `physx:` prefixed names route data to/from ovphysx tensor bindings
rather than visual USD attributes.  These enable closed-loop physics
co-simulation.

| Directive | Direction | Tensor | Description |
|-----------|-----------|--------|-------------|
| `physx:position` | input | `RIGID_BODY_POSE [N,7]` | Body position (xyz from the 7-component pose quaternion+position tensor) |
| `physx:velocity` | input | `RIGID_BODY_VELOCITY [N,6]` | Body linear velocity (xyz from the 6-component lin+ang velocity tensor) |
| `physx:force` | output | `RIGID_BODY_FORCE [N,3]` | External force to apply to body (xyz) |

For physics directives, `usdMapping` selects the vector component:

- `(0, 1)` = X
- `(1, 1)` = Y
- `(2, 1)` = Z

### Stage-Routed Physics Values

The conveyor sample also exchanges real USD attributes through ovstage:

| Attribute | Direction | Meaning |
|---|---|---|
| `sensor:presence` | input | Runtime attribute populated from an ovphysx overlap query and read by FMI on the next frame. |
| `drive:angular:physics:targetVelocity` | output | Drive attribute written to an ovstage control ordinal and consumed through `PhysX.update_from_ovstage()`. |

The conveyor FMUs use the legacy articulation tensor convention of radians
per second. At the ovstage boundary, the demo converts their drive outputs
to the USD angular-drive unit of degrees per second.

For overlap sensors, a prim whose name starts with `Sensor` identifies the
sensor, and a child `UsdGeomSphere` supplies the preferred shape query.
After each physics step, the demo performs an ovphysx `SHAPE`/`ANY` overlap
and publishes its Boolean result as `sensor:presence` on the sensor prim.
FMI consumes that attribute on the next step. If no child shape is available,
the demo uses the sensor's world position and radius for a sphere query.

---

## SSP Specifics

### Archive Structure

An `.ssp` file is a ZIP archive following the
[SSP 1.0 standard](https://ssp-standard.org/publications/SSP10/SystemStructureAndParameterization10.pdf):

```text
orbit_controller.ssp
├── SystemStructure.ssd      # XML: system topology, connections, connectors
└── resources/
    ├── TrajectoryGenerator.fmu
    └── PDController.fmu
```

### System-Level Connectors

The `SystemStructure.ssd` defines which variables are exposed at the system
boundary.  Only these appear as valid `fmi:fmuAttribute` names in USD mappings.
Internal FMU-to-FMU connections and intermediate variables are not accessible
from USD.

Example: An SSP that internally wires a trajectory generator to a PD controller
might expose only:

| System connector | Direction | Description |
|-----------------|-----------|-------------|
| `pos_x`, `pos_y`, `pos_z` | input | Body position (fed to internal PD controller) |
| `vel_x`, `vel_y`, `vel_z` | input | Body velocity (fed to internal PD controller) |
| `force_x`, `force_y`, `force_z` | output | Computed force (from internal PD controller) |

The internal orbit target generation is entirely hidden from USD.

### FmuInstance vs SspInstance: When to Use Which

| Scenario | Use |
|----------|-----|
| Single model, all variables visible to USD | `FmuInstance` |
| Multiple models with explicit USD-level wiring (relay prims) | Multiple `FmuInstance` prims |
| Pre-packaged component from external toolchain | `SspInstance` |
| Internal wiring should not be exposed/modified | `SspInstance` |
| Encapsulate internal connections behind system-level connectors | `SspInstance` |

---

## Execution Semantics

### Stepping Order

- Multiple `FmuInstance` prims are stepped in **USD scene traversal order**
  (depth-first pre-order).  This is significant when one FMU writes a USD
  attribute that another reads — the authoring order determines causality.

- An `SspInstance` exposes system-level connectors. Its internal coupling
  schedule depends on the selected backend, not just the component order in
  `SystemStructure.ssd`. Refer to [coupling and numerical differences](native-runtime-poc.md#coupling-and-numerical-differences).

### Physics Integration

When the scene contains prims with `PhysicsRigidBodyAPI`, the demo application
orchestrates the following sequence. The ovfmi library does not step physics:

1. Reads pose and velocity tensors from the physics engine
2. Feeds values into FMU/SSP inputs mapped to `physx:position` / `physx:velocity`
3. Steps all FMU/SSP instances
4. Writes FMU/SSP outputs mapped to `physx:force` back to the force tensor
5. Advances the physics simulation

### Auto-Detection

The demo detects rigid-body prims (`PhysicsRigidBodyAPI`) from the USD scene
and initializes ovphysx unless `--no-physics` is selected. Library-only
applications must provide their own physics integration.

---

## Example Schema Fragments

These fragments illustrate mappings within a larger USD scene; they are not
standalone files. Use the [example catalog](../examples/README.md) for complete
stages, target prims, and the models generated during setup.

### Bouncing Ball Visualization

An FMU computes ball height internally and writes it to the sphere's Y
translation each frame.  No physics engine involved.

```usda
def FmuInstance "FmuInstance"
{
    bool fmi:enabled = 1
    asset fmi:fmu = @./BouncingBall.fmu3@

    def FmuConnection "HeightConnection"
    {
        rel fmi:targets = </World/Sphere>

        def FmuMapping "HeightWrite"
        {
            token fmi:direction = "output"
            token fmi:fmuAttribute = "h"
            int2 fmi:usdMapping = (1, 1)
            token fmi:usdAttribute = "xformOp:translate"
        }

        def FmuMapping "HeightRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "h"
            int2 fmi:usdMapping = (1, 1)
            token fmi:usdAttribute = "xformOp:translate"
        }
    }
}
```

### PD Controller With Physics Feedback

A PD controller FMU reads the body's physics state and outputs a restoring
force with gravity compensation.

```usda
def FmuInstance "PDControllerFMU"
{
    bool fmi:enabled = 1
    asset fmi:fmu = @./PDController.fmu3@

    def FmuConnection "ForceConnection"
    {
        rel fmi:targets = </World/Sphere>

        # Position inputs
        def FmuMapping "PosXRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "pos_x"
            int2 fmi:usdMapping = (0, 1)
            token fmi:usdAttribute = "physx:position"
        }
        def FmuMapping "PosYRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "pos_y"
            int2 fmi:usdMapping = (1, 1)
            token fmi:usdAttribute = "physx:position"
        }
        def FmuMapping "PosZRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "pos_z"
            int2 fmi:usdMapping = (2, 1)
            token fmi:usdAttribute = "physx:position"
        }

        # Velocity inputs
        def FmuMapping "VelXRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "vel_x"
            int2 fmi:usdMapping = (0, 1)
            token fmi:usdAttribute = "physx:velocity"
        }
        def FmuMapping "VelYRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "vel_y"
            int2 fmi:usdMapping = (1, 1)
            token fmi:usdAttribute = "physx:velocity"
        }
        def FmuMapping "VelZRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "vel_z"
            int2 fmi:usdMapping = (2, 1)
            token fmi:usdAttribute = "physx:velocity"
        }

        # Force outputs
        def FmuMapping "ForceXWrite"
        {
            token fmi:direction = "output"
            token fmi:fmuAttribute = "force_x"
            int2 fmi:usdMapping = (0, 1)
            token fmi:usdAttribute = "physx:force"
        }
        def FmuMapping "ForceYWrite"
        {
            token fmi:direction = "output"
            token fmi:fmuAttribute = "force_y"
            int2 fmi:usdMapping = (1, 1)
            token fmi:usdAttribute = "physx:force"
        }
        def FmuMapping "ForceZWrite"
        {
            token fmi:direction = "output"
            token fmi:fmuAttribute = "force_z"
            int2 fmi:usdMapping = (2, 1)
            token fmi:usdAttribute = "physx:force"
        }
    }
}
```

### SSP Orbit Controller

A single `SspInstance` wrapping a trajectory generator and PD controller.
Internal orbit computation is hidden; only aggregate physics I/O is exposed.

```usda
def SspInstance "OrbitController"
{
    bool fmi:enabled = 1
    asset fmi:ssp = @./orbit_controller.ssp@

    def FmuConnection "PhysicsIO"
    {
        rel fmi:targets = </World/Sphere>

        # Position inputs
        def FmuMapping "PosXRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "pos_x"
            int2 fmi:usdMapping = (0, 1)
            token fmi:usdAttribute = "physx:position"
        }
        def FmuMapping "PosYRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "pos_y"
            int2 fmi:usdMapping = (1, 1)
            token fmi:usdAttribute = "physx:position"
        }
        def FmuMapping "PosZRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "pos_z"
            int2 fmi:usdMapping = (2, 1)
            token fmi:usdAttribute = "physx:position"
        }

        # Velocity inputs
        def FmuMapping "VelXRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "vel_x"
            int2 fmi:usdMapping = (0, 1)
            token fmi:usdAttribute = "physx:velocity"
        }
        def FmuMapping "VelYRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "vel_y"
            int2 fmi:usdMapping = (1, 1)
            token fmi:usdAttribute = "physx:velocity"
        }
        def FmuMapping "VelZRead"
        {
            token fmi:direction = "input"
            token fmi:fmuAttribute = "vel_z"
            int2 fmi:usdMapping = (2, 1)
            token fmi:usdAttribute = "physx:velocity"
        }

        # Force outputs
        def FmuMapping "ForceXWrite"
        {
            token fmi:direction = "output"
            token fmi:fmuAttribute = "force_x"
            int2 fmi:usdMapping = (0, 1)
            token fmi:usdAttribute = "physx:force"
        }
        def FmuMapping "ForceYWrite"
        {
            token fmi:direction = "output"
            token fmi:fmuAttribute = "force_y"
            int2 fmi:usdMapping = (1, 1)
            token fmi:usdAttribute = "physx:force"
        }
        def FmuMapping "ForceZWrite"
        {
            token fmi:direction = "output"
            token fmi:fmuAttribute = "force_z"
            int2 fmi:usdMapping = (2, 1)
            token fmi:usdAttribute = "physx:force"
        }
    }
}
```

---

## Compatibility

The current runtime supports the following subset of the schema:

| Feature | FMI 2.0 | FMI 3.0 | SSP 1.0 |
|---------|---------|---------|---------|
| `FmuInstance` | Scalar Real | Scalar Float64 | Not applicable |
| `SspInstance` | Not applicable | Not applicable | Flat FMI 2 Real systems |
| Co-Simulation | Supported | Supported | Supported subset |
| Model Exchange | Unsupported | Unsupported | Unsupported |

The native backend executes FMI 2.0 Co-Simulation with scalar `Real` variables
and FMI 3.0 Co-Simulation with scalar `Float64` variables. Its build-time SSP
converter supports the flat subset used by the bundled examples and emits
an FMI 2 container FMU. Runtime limits are separate from the version-agnostic
USD schema. Refer to [runtime support and numerical differences](native-runtime-poc.md)
for the complete boundary.

---

## References

- [FMI Standard](https://fmi-standard.org/) — Functional Mock-up Interface
- [SSP Standard](https://ssp-standard.org/) — System Structure and Parameterization
- [OpenUSD](https://openusd.org/) — Universal Scene Description
- [fmi-rs](https://github.com/CATIA-Systems/fmi-rs) — native FMI runtime
- [FMPy](https://github.com/CATIA-Systems/FMPy) — build-time SSP/container tooling
