<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Public Python API

Import supported symbols directly from `ovfmi`. The
[package exports](../python/__init__.py) define the compatibility boundary.
Submodules, names beginning with an underscore, and the native C ABI are
implementation details.

## Preconditions

Before attaching a host, complete [library setup](library-only-setup.md) and
provide a populated, caller-owned ovstage, its original USD source, and the
referenced FMUs. FMU means Functional Mock-up Unit, a packaged FMI model.
SSP systems need a prepared container for native execution; refer to
[backend support](native-runtime-poc.md#ssp-buildruntime-split).

Treat FMUs as executable native code. Only load models from sources you trust;
ovfmi does not sandbox them.

## Public Symbols

The supported public types have the following roles:

| Symbol | Purpose |
|---|---|
| `FmiHost` | Owns FMI and SSP instances, routing state, and simulation lifecycle for a borrowed ovstage. |
| `FmiHostConfig` | Selects the backend and configures discovery, schema validation, SSP support, and missing-input behavior. |
| `MissingInputPolicy` | Selects initialization behavior when a mapped input is unavailable. |
| `PopulationReport`, `InstanceInfo` | Describe the instances created during attachment. |
| `AttributeWrite` | Supplies values identified by USD prim path and attribute name. |
| `ReadResult`, `ReadGroup` | Provide an owned snapshot of output values in USD space. |

The [host method docstrings](../python/api.py) and
[value-type docstrings](../python/types.py) are the detailed API reference,
including parameter types, defaults, filtering, and ownership.
After installation, display the host reference in your terminal:

```bash
python -c "from ovfmi import FmiHost; help(FmiHost)"
```

The command prints the class documentation and its public methods.

## Lifecycle

An application performs the following lifecycle. This is an integration
sequence, not a standalone runnable program; renderer and physics ownership
remain with the application.

1. Construct `FmiHost` with `FmiHostConfig(backend="native")` or `backend="fmpy"`.
   The default is native. There is no automatic backend fallback.
2. Call `attach_ovstage(stage, source_asset=usd_path)` with the populated stage
   and source path. Check `PopulationReport.instances` against the instances
   you expect.
3. Supply current mapped inputs with `update_from_ovstage(from_ordinal, to_ordinal)`
   or `write(writes)`. An ordinal identifies a stage change boundary; the
   current binding samples current values rather than replaying historical
   snapshots.
4. Advance the model with `step_sync(dt)`, where `dt` is a positive finite
   interval in seconds. A successful step advances `host.time` by that interval.
5. Consume outputs through `read()` or publish them with
   `write_to_ovstage(output_ordinal)`. Translation publication preserves the
   rotation and scale of the latest sealed local matrix. Seal initial stage
   population first, and call this method before other `omni:xform` writes
   (including camera updates) in the output ordinal. The caller still owns
   sealing; ovfmi never advances the write floor. Use a `ReadResult` context
   manager, or release the result explicitly after use.
6. Release the host with `release()`, or use `FmiHost` as a context manager.
   The caller still owns and must release its ovstage.

`step()` and `write()` return completion tokens even though both current
backends execute synchronously. Refer to their docstrings for waiting and
token release. Do not infer asynchronous execution from those return values.

`reset()` resets the host's FMUs and routing state, not external physics or
the caller's stage. Restore those separately before replaying a coupled system.

## Failure Handling

An enabled model whose authored `fmi:fmu` or `fmi:ssp` asset path cannot be
resolved by OpenUSD raises `ValueError` before any model is loaded. Disabled
instances, instances outside `root_prim`, and SSPs excluded by `enable_ssp=False`
are not checked. Resolution follows the configured OpenUSD resolver. With the
default resolver, a bare filename can resolve through search paths, including
the working directory; a `./` path is relative to its authoring layer.

Attachment propagates parsing and model-instantiation failures after releasing
partially constructed backend state. A missing `source_asset` is an error.
For missing libraries or dependencies, use [troubleshooting](troubleshooting.md#missing-native-runtime-or-fmpy).

A failed step does not imply that an FMU can roll back; do not assume that
continuing after an exception preserves a valid coupled simulation.
Validate backend-specific numerical behavior for your model and step size.

For a complete integration, inspect [the demo application](../demoapp/main.py).
Its renderer, physics, window, and camera code are sample application concerns,
not capabilities owned by `FmiHost`.
