<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# fmi-rs issue reports

These issues were confirmed against CATIA-Systems/fmi-rs `main` at commit
`e144c629a7fac87959bc52a0b2862a148fc4b02c`.

## Resolution

All four reports have been fixed upstream and closed:

- [Issue #4](https://github.com/CATIA-Systems/fmi-rs/issues/4): FMI 2 memory callbacks
- [Issue #5](https://github.com/CATIA-Systems/fmi-rs/issues/5): null FMI 2 instances
- [Issue #6](https://github.com/CATIA-Systems/fmi-rs/issues/6): default feature build
- [Issue #7](https://github.com/CATIA-Systems/fmi-rs/issues/7): Windows SUNDIALS generator

ovfmi was verified against unmodified upstream commit
`6a831ad2a4b106acf60a845733c6d39b4b6c344b` on 2026-08-20. The local fmi-rs
source snapshot and compatibility patch were then retired.

Test environment:

- Windows 11, x86_64 MSVC
- rustc 1.97.1
- cargo 1.97.1
- CMake 3.30.3
- Visual Studio 2022

The FMPy-generated container FMU failure observed during ovfmi testing is not
included here. That container also fails when loaded through FMPy itself, so it
cannot currently be attributed to fmi-rs.

## Issue summaries

### FMI-RS-1: Correct unsafe FMI 2 memory callbacks

fmi-rs allocates FMI 2 callback memory with a size-dependent Rust allocation
layout but releases every allocation using a hard-coded one-byte layout. It
also returns uninitialized memory even though FMI requires `calloc` semantics.
This is undefined behavior and can cause heap corruption or access violations.
Use matched, zero-initializing allocation and release callbacks, such as
`calloc` and `free`, and add a regression test.

### FMI-RS-2: Reject null FMI 2 instances

When an FMU returns `NULL` from `fmi2Instantiate`, fmi-rs logs an error but
still returns `Ok(FMU2)` with a null component pointer. Subsequent FMI calls may
then crash. Instantiation must return an error immediately when the component
pointer is null, with coverage for both Co-Simulation and Model Exchange.

### FMI-RS-3: Fix the default feature build

The current crate does not compile with `cargo check` because the default
features omit SUNDIALS while simulation modules import SUNDIALS symbols
unconditionally. Gate solver-dependent modules behind the `sundials` feature
so the low-level FMI APIs compile with the declared defaults.

### FMI-RS-4: Make the Windows SUNDIALS CMake generator portable

The SUNDIALS build selects `Visual Studio 18 2026` for every MSVC Rust target,
causing `cargo check --features sundials` to fail on Visual Studio 2022 systems.
Do not infer a Visual Studio version from the Rust ABI target; let CMake choose
an installed generator or provide an explicit configuration override.

---

## Detailed GitHub issue 1

### Title

FMI 2 `allocateMemory` and `freeMemory` callbacks invoke undefined behavior

### Description

The FMI 2 host callbacks in `fmi-rs/src/fmi2/mod.rs` currently allocate with a
dynamically sized Rust `Layout`, but always deallocate using
`Layout::from_size_align_unchecked(1, 1)`:

```rust
unsafe extern "C" fn allocateMemory(nobj: usize, size: usize) -> *mut c_void {
    unsafe {
        let layout = std::alloc::Layout::from_size_align_unchecked(nobj * size, 1);
        std::alloc::alloc(layout) as *mut c_void
    }
}

unsafe extern "C" fn freeMemory(obj: *mut c_void) {
    if !obj.is_null() {
        unsafe {
            std::alloc::dealloc(
                obj as *mut u8,
                std::alloc::Layout::from_size_align_unchecked(1, 1),
            );
        }
    }
}
```

Rust requires the layout passed to `dealloc` to match the original allocation.
The hard-coded one-byte layout therefore invokes undefined behavior for almost
every allocation.

Additionally, FMI 2 defines `allocateMemory(nobj, size)` with `calloc`
semantics: the returned memory must be initialized to zero.
`std::alloc::alloc()` returns uninitialized memory.

### Minimal reproducer

A minimal FMI 2 FMU can allocate its instance through the supplied callbacks
and release it from `fmi2FreeInstance`:

```c
typedef struct {
    fmi2CallbackFreeMemory freeMemory;
    double state[16];
} Instance;

fmi2Component fmi2Instantiate(
    fmi2String instanceName,
    fmi2Type fmuType,
    fmi2String guid,
    fmi2String resourceLocation,
    const fmi2CallbackFunctions* callbacks,
    fmi2Boolean visible,
    fmi2Boolean loggingOn)
{
    Instance* instance =
        callbacks->allocateMemory(1, sizeof(Instance));

    if (!instance) {
        return NULL;
    }

    /* FMI requires the allocation to be zero-filled at this point. */
    instance->freeMemory = callbacks->freeMemory;
    return instance;
}

void fmi2FreeInstance(fmi2Component component)
{
    Instance* instance = component;
    instance->freeMemory(instance);
}
```

Load and drop the FMU with fmi-rs:

```rust
let builder = FMU2Builder::new(&"CallbackAllocator.fmu")?;
let fmu = builder.instantiate_cs("instance")?;
drop(fmu);
```

### Expected behavior

- Allocation is zero-initialized.
- `fmi2FreeInstance` safely releases the allocation.

### Actual behavior

- Memory is uninitialized.
- The allocation is returned to the Rust allocator with a different layout.
- Depending on allocator and platform, this can cause heap corruption or an
  access violation.

### Suggested correction

One straightforward implementation is paired `calloc` and `free` callbacks:

```rust
unsafe extern "C" fn allocate_memory(
    nobj: usize,
    size: usize,
) -> *mut c_void {
    unsafe { libc::calloc(nobj, size) }
}

unsafe extern "C" fn free_memory(ptr: *mut c_void) {
    unsafe { libc::free(ptr) }
}
```

It would also be prudent to handle multiplication overflow and zero-sized
requests explicitly.

---

## Detailed GitHub issue 2

### Title

`FMU2::instantiate()` returns `Ok` when `fmi2Instantiate()` returns `NULL`

### Description

In `fmi-rs/src/fmi2/mod.rs`, the result of `fmi2Instantiate` is checked only
for call logging:

```rust
if component.is_null() {
    self.log_call(fmi2Status::fmi2Error, &message);
} else {
    self.log_call(fmi2Status::fmi2OK, &message);
}

self.component = component;

Ok(())
```

Consequently, `instantiate_cs()` and `instantiate_me()` can return `Ok(FMU2)`
containing a null component pointer.

### Minimal reproducer

Use a structurally valid FMI 2 FMU whose instantiation function returns null:

```c
fmi2Component fmi2Instantiate(/* standard FMI 2 arguments */)
{
    return NULL;
}
```

Then load it:

```rust
let builder = FMU2Builder::new(&"InstantiateFails.fmu")?;
let result = builder.instantiate_cs("instance");

assert!(result.is_err()); // Currently fails because result is Ok(...).
```

### Expected behavior

Instantiation returns `Err`, preferably with a specific error indicating that
the FMU returned a null component.

### Actual behavior

Instantiation returns an `Ok(FMU2)` containing a null component pointer. A
subsequent operation such as `setupExperiment()` passes that pointer to the
FMU and may crash if the FMU does not defensively reject it.

### Suggested correction

Return before storing the component:

```rust
if component.is_null() {
    return Err(SimulationError::FMICall);
}

self.component = component;
Ok(())
```

This affects both Model Exchange and Co-Simulation because they share the
internal instantiation path.

---

## Detailed GitHub issue 3

### Title

`cargo check` fails because default features exclude SUNDIALS while `sim`
imports it unconditionally

### Description

The crate does not compile with its declared default feature set.

### Reproducer

```console
git clone https://github.com/CATIA-Systems/fmi-rs.git
cd fmi-rs
git checkout e144c629a7fac87959bc52a0b2862a148fc4b02c
cargo check
```

### Actual result

```text
error[E0433]: cannot find `sundials` in `crate`
 --> fmi-rs/src/sim/fmi3/me.rs:7:12
  |
7 | use crate::sundials::solver::ida::Dae3;
  |            ^^^^^^^^ could not find `sundials` in the crate root

error[E0433]: cannot find `sundials` in `crate`
 --> fmi-rs/src/sim/solver.rs:3:5
  |
3 |     sundials::solver::ida::Dae3,
  |     ^^^^^^^^ could not find `sundials` in the crate root
```

`Cargo.toml` declares:

```toml
[features]
default = ["zip"]
sundials = []
```

`src/lib.rs` correctly gates the SUNDIALS module:

```rust
#[cfg(feature = "sundials")]
pub mod sundials;
```

The solver-dependent simulation modules nevertheless import it without
equivalent feature gating.

### Expected behavior

`cargo check` should compile the low-level FMI APIs without requiring a native
solver.

### Suggested correction

- Gate solver-dependent modules and exports behind `feature = "sundials"`.
- Keep low-level FMI 2/FMI 3 loading and Co-Simulation APIs available without
  SUNDIALS.
- Alternatively, expose the high-level simulation solver as a separate
  feature.

Adding SUNDIALS to the default features would make the normal build
substantially heavier and currently encounters the separate Windows generator
issue below.

---

## Detailed GitHub issue 4

### Title

SUNDIALS build hard-codes `Visual Studio 18 2026` for every MSVC target

### Description

Enabling the SUNDIALS feature fails on a normal Visual Studio 2022
installation.

### Reproducer

```console
git clone https://github.com/CATIA-Systems/fmi-rs.git
cd fmi-rs
git checkout e144c629a7fac87959bc52a0b2862a148fc4b02c
cargo check --features sundials
```

### Actual result

```text
Sundials not found. Downloading and building SUNDIALS v7.8.0...

Failed to configure SUNDIALS with generator Visual Studio 18 2026:

CMake Error: Could not create named generator Visual Studio 18 2026

Generators
* Visual Studio 17 2022
  Visual Studio 16 2019
  ...
```

The cause is in `fmi-rs/build/sundials.rs`:

```rust
let generator = if target.contains("msvc") {
    "Visual Studio 18 2026"
} else if target.contains("windows-gnu") {
    "MinGW Makefiles"
} else {
    "Unix Makefiles"
};
```

The Rust target identifies an ABI, not the installed Visual Studio version.
Every `*-pc-windows-msvc` target therefore selects Visual Studio 2026 even
when only Visual Studio 2022 is installed.

### Expected behavior

The build should use an installed generator and should not infer a Visual
Studio release from the Rust ABI target.

### Suggested correction

- Prefer allowing CMake to select its normal default generator.
- Alternatively, use the `cmake` crate or a broadly available generator such
  as Ninja.
- If an explicit generator is necessary, expose it through an environment or
  Cargo configuration override.

Removing the explicit `-G` argument for MSVC may be the simplest portable
behavior.
