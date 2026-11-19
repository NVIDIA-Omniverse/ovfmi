<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Third-Party Notices

This file records the third-party software and content used by `ovfmi`. It
covers direct Python dependencies, packages handled by the setup scripts,
optional runtime components, external prerequisites, bundled demo assets, and
standards used by the sample code.

The ovfmi Apache-2.0 license does not replace the upstream licenses of
third-party components. Packages resolved at installation time carry their
authoritative license texts in their wheels or source archives. Preserve those
texts when redistributing a resolved environment, generated FMU/SSP archive,
SDK bundle, or native binary package.

This inventory is not a lockfile-generated SBOM. Several dependencies are
unpinned, and package installers resolve transitive dependencies. A distributor
must generate notices from the exact environment and artifacts it distributes.

## 1. Direct Python dependencies

The following packages are declared by `pyproject.toml` and
`demoapp/pyproject.toml`. They are installed from a configured Python package
index and are not vendored in this repository.

| Package | Project constraint | License | Use | Upstream |
|---|---|---|---|---|
| `FMPy` | optional `build` and `fmpy` extras, `==0.3.32` | BSD-2-Clause, plus bundled component notices | Build-time SSP/container-FMU preparation and the optional Python FMI/SSP runtime backend | <https://github.com/CATIA-Systems/FMPy> |
| `NumPy` | `>=2.2` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Array and tensor handling | <https://github.com/numpy/numpy> |
| `glfw` / pyGLFW | `>=2.8` | MIT | OpenGL window and input handling | <https://github.com/FlorianRhiem/pyGLFW> |
| `PyOpenGL` | `>=3.1` | BSD-style, with bundled component notices | OpenGL Python bindings | <https://github.com/mcfletch/pyopengl> |
| `Pillow` | `>=10.4` | MIT-CMU, with bundled component notices | Direct demo dependency for image/PNG output | <https://github.com/python-pillow/Pillow> |
| `pytest` | `>=8` | MIT | Test runner installed with the demo application | <https://github.com/pytest-dev/pytest> |

Important license locations in installed distributions include:

| Package | Files to preserve |
|---|---|
| `FMPy` | `fmpy-*.dist-info/licenses/LICENSE.txt` and `fmpy/remoting/license.txt` |
| `NumPy` | `numpy-*.dist-info/licenses/` and license files shipped inside `numpy/` |
| `glfw` | `glfw-*.dist-info/licenses/LICENSE.txt` |
| `PyOpenGL` | The upstream `license.txt` and bundled `OpenGL/DLLS/*COPYING*` files |
| `Pillow` | `pillow-*.dist-info/licenses/` and any additional license files shipped in the resolved distribution |
| `pytest` | `pytest-*.dist-info/licenses/LICENSE` |

FMPy, NumPy, and Pillow ship additional component notices. Preserve their complete
license directories instead of copying only the primary license file.
The demo setup installs the FMPy extra for the Python backend and model
preparation. Pillow is installed with the demo even when PNG output is not used.

## 2. NVIDIA runtime packages

Demo setup installs the supported pinned combination into the isolated
`demoapp/.venv` environment, using pip's configured package index or local wheel
sources. It does not modify base or user-installed Python packages.
The pins are recorded in [requirements.txt](demoapp/requirements.txt).

| Package | Version |
|---|---|
| ovstage | `0.4.0.1794520352` |
| ovrtx | `0.7.0.1794596436` |
| ovphysx | `0.8.0.43443703471` |
| physx-usd-schemas | `25.11.2.72696984` |

These packages are installed separately; they are not bundled in the ovfmi
wheel or vendored in this source tree. Library-only stage integration requires
a compatible ovstage installation, not the demo's renderer or physics packages.
These pins describe a dependency selection, not a new license grant.

| Package | License reported by reviewed package metadata | Use |
|---|---|---|
| `ovrtx` | NVIDIA Proprietary Software | RTX rendering |
| `ovstage` | NVIDIA Proprietary Software | Shared USD data-plane stage |
| `ovphysx` | `LicenseRef-NVIDIA-Omniverse` | Optional rigid-body physics |
| `physx-usd-schemas` | `LicenseRef-NVIDIA-Omniverse` | PhysX schema resources registered by the demo |

Preserve each resolved package's own license and notice files when
redistributing it. Reviewed packages contain these locations:

| Package | Files to preserve |
|---|---|
| `ovrtx` | `ovrtx/THIRD-PARTY-NOTICES.txt`, package license files, and component `PACKAGE-LICENSES/` directories |
| `ovstage` | `ovstage/THIRD-PARTY-NOTICES.txt` and package license files |
| `ovphysx` | `ovphysx-*.dist-info/licenses/LICENSE.txt`, `ovphysx-*.dist-info/licenses/ovphysx-LICENSES.zip`, and notices shipped with its documentation |

Any dependencies resolved by these packages, such as `packaging`, must also be
included in a distribution-specific license inventory.

## 3. Setup, build, and optional Python packages

| Package | Project use | License |
|---|---|---|
| `setuptools>=77` | PEP 517 build backend declared by the root project | MIT |
| `usd-core` | Unpinned package installed in `demoapp/.usd_venv` for isolated USD parsing | `LicenseRef-TOST-1.0` |
| `cuda-python` | Optional CUDA/OpenGL transfer for the demo display | NVIDIA Software License |

Preserve the license directory from the resolved `setuptools`, `usd-core`, and
`cuda-python` distributions when those packages are redistributed. Pillow is a
direct demo dependency listed in section 1; its full license file includes
notices for bundled image-codec components.

## 4. Native runtime dependencies

The native runtime resolves the unmodified `fmi-rs` 0.3.5
source from upstream commit
`99b9a6011abae49a6e191918cde9fe49e100562d`. It is BSD-2-Clause licensed and
is fetched by Cargo rather than vendored in this repository. `native/Cargo.lock`
records the exact Rust dependency graph. A binary distribution must preserve
the upstream fmi-rs license and generate notices for every crate resolved for
the wheel's target.

`tempfile` is also a direct Rust dependency, declared as `tempfile = "3"` in
`native/Cargo.toml` and resolved to 3.27.0 in `native/Cargo.lock`. It provides
temporary-file and directory handling under `MIT OR Apache-2.0`; its upstream
is [tempfile](https://github.com/Stebalien/tempfile). Preserve its upstream
license texts and the notices for its resolved target dependencies in binary
distributions. Neither a lockfile entry nor this description alone establishes
which transitive components are included in a final artifact.

Platform wheels include `ovfmi/_native_licenses/packages.json` with the resolved
target dependency inventory, Cargo lockfile hash, and hashes of the accompanying
upstream license and notice files. The wheel build generates this bundle from
Cargo metadata and refuses dependencies with missing license texts.

The native library also links the Rust standard library. The wheel includes
the build compiler's `COPYRIGHT-library.html` and accompanying license texts
from its sysroot. `packages.json` records the compiler version, commit, host,
and notice hashes separately from the Cargo package inventory.

Generated Container FMUs embed the stock native runtime distributed with FMPy
0.3.32. That runtime incorporates fmi-rs and its Rust dependency graph, so a
distributor must include the applicable FMPy, fmi-rs, and resolved Rust-crate
notices with generated FMUs.

## 5. Transitive Python dependencies

FMPy 0.3.32 declares dependencies including `attrs`, `cmake`, `jinja2`, `lark`, `lxml`,
`msgpack`, `nbformat`, and `numpy`. Pytest and the NVIDIA runtime packages also
resolve their own dependencies. These packages are not declared directly by
all ovfmi manifests and may vary by platform and installer state. Include every
resolved distribution and its license files in a distribution-specific SBOM
and notice bundle.

## 6. External build and runtime prerequisites

The following components are used from the host system and are not vendored by
this repository:

| Prerequisite | Use | License family |
|---|---|---|
| Python 3.10-3.13 | Application runtime | Python Software Foundation License |
| Git and Git LFS | Source and asset checkout | GPL-2.0-only and distribution-specific notices |
| Visual Studio 2022 Build Tools, GCC, or Clang | C++17 demo FMU compilation | Toolchain-specific |
| OpenGL and X11 runtime libraries | Live display | Vendor/distribution-specific |
| NVIDIA display driver | RTX rendering | NVIDIA driver license |
| CUDA Toolkit | Optional CUDA runtime support for the demo display | NVIDIA CUDA Toolkit EULA |
| CMake | Optional `demoapp/fmi_usd_helper` build | Apache-2.0 and bundled component notices |
| Rust compiler and Cargo | Native ovfmi runtime compilation; linked standard-library notices are bundled as described above | Apache-2.0 OR MIT, plus component notices |
| OpenUSD 25.11 headers | Optional `fmi_usd_helper` build | `LicenseRef-TOST-1.0` |
| ovphysx SDK | Optional libraries for `fmi_usd_helper` | NVIDIA Omniverse/package-specific terms |

Generated FMU and SSP archives can contain binaries produced by the local C++
toolchain. Include applicable compiler runtime notices when distributing those
archives.

## 7. Bundled USD, MDL, mesh, image, and texture assets

The repository includes demo USD stages, MDL materials, meshes, screenshots,
and textures:

| Asset group | Paths | Notice |
|---|---|---|
| ovfmi-authored demo stages and screenshots | `demoapp/usd/*.usda`, `fmu-ball-test.png`, `physx-sim-test.png` | NVIDIA project/demo material unless a file states otherwise |
| Conveyor and box assets | `demoapp/usd/conveyor/` | SimReady-derived USD, MDL, mesh, and texture content under CC BY 4.0; see [attribution](demoapp/usd/conveyor/ATTRIBUTION.md) |
| Conveyor screenshot | `docs/images/conveyor-demo.png` | Shows the SimReady-derived scene described below |
| Demo model sources | `demoapp/fmu/`, `demoapp/ssp/` | NVIDIA project/demo material unless a file states otherwise |

The conveyor and box assets and their included USD, MDL, mesh, and texture
dependencies derive from NVIDIA SimReady assets. Public source and package
license links are listed in the [asset attribution](demoapp/usd/conveyor/ATTRIBUTION.md).
Their original creator is NVIDIA Corporation. These assets are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
not the source code's Apache-2.0 license. The
[full legal terms](https://creativecommons.org/licenses/by/4.0/legalcode.en)
govern use of the assets. They have been modified from their
original states to support custom physics engine, FMI backend workflows, and
demo simulation scripts. Preserve this attribution and the assets'
[attribution notice](demoapp/usd/conveyor/ATTRIBUTION.md), together with existing provenance,
when redistributing them, and identify any further modifications. No additional
restrictions are imposed on the rights granted by CC BY 4.0.

The asset license does not relicense the demo's FMI model source code or
generated FMUs. This notice document remains under Apache-2.0.

USD layers also reference runtime-provided MDL materials that are not bundled
with these assets; those remain subject to the licenses of the packages that
provide them.

## 8. Standards and specification materials

| Standard | Project use | Notice |
|---|---|---|
| FMI (Functional Mock-up Interface) | Demo FMU metadata, runtime behavior, and USD-FMI mappings | FMPy carries notices for FMI headers and schemas from the MODELISAR consortium and Modelica Association Project FMI. Preserve those notices when redistributing FMPy or copied specification materials. |
| SSP (System Structure and Parameterization) | Demo `.ssd` files and SSP runtime behavior | Preserve notices associated with any SSP specification files or tooling included in a distribution. |

The checked-in `demoapp/fmu/fmi2/fmi2_minimal.h` is an NVIDIA-authored minimal
set of declarations and carries its own NVIDIA SPDX header; it is not a copied
FMI SDK header.

## 9. Redistribution checklist

Before distributing `ovfmi` or a package containing its dependencies:

1. Include this file or an updated equivalent.
2. Generate an SBOM and dependency notice from the exact resolved environment.
3. Include complete license files from redistributed wheels, source archives,
   native SDKs, generated model archives, and binary bundles.
4. Include the NVIDIA runtime packages' own third-party notices when those
   packages or their contents are redistributed.
5. Preserve the CC BY 4.0 attribution and modification notice for the
   SimReady-derived `demoapp/usd/conveyor/` assets.
6. Do not classify NVIDIA proprietary package licenses as open-source licenses.

## 10. Sources reviewed

The dependency roles were checked against the
source manifests and setup scripts on 2026-10-06. This documentation update
does not verify final release artifacts or grant asset redistribution rights.

- `pyproject.toml`
- `demoapp/pyproject.toml`
- `demoapp/setup.ps1`
- `demoapp/setup.sh`
- `demoapp/requirements.txt`
- `native/Cargo.toml` and `native/Cargo.lock`
- `demoapp/fmi_usd_helper/CMakeLists.txt`
- `README.md`
- `docs/USD-FMI-SCHEMA.md`
- checked-in asset provenance strings and SPDX headers
- installed package metadata and license inventories reviewed on 2026-07-22,
  with the FMPy and fmi-rs entries updated on 2026-09-28, for FMPy 0.3.32,
  fmi-rs commit `6a831ad2`, NumPy 2.4.3, glfw 2.10.2, PyOpenGL 3.1.10, pytest 9.1.0,
  Pillow 12.2.0, usd-core 26.5, ovrtx 0.4.0.346409, ovstage 0.1.0.346039,
  and ovphysx 0.5.9
- native dependency entries updated on 2026-11-17 for stock fmi-rs 0.3.5
  (commit `99b9a601`) and the resolved graph in `native/Cargo.lock`, including
  zip 9.6.1
