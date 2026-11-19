# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

param(
    [switch]$SkipFmuBuild,
    [switch]$SkipNativeBuild,
    [switch]$InstallCudaPython
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$OvFmiDir = $RepoRoot
$AppVenv = Join-Path $ScriptDir ".venv"
$UsdVenv = Join-Path $ScriptDir ".usd_venv"

function Invoke-AppPip {
    & $AppPython -m pip @args
    if ($LASTEXITCODE -ne 0) {
        throw "App dependency installation failed. See pip output above."
    }
}

function Get-InstalledOvrtxBin {
    param([string]$PythonExe)

    $code = @'
from pathlib import Path
import ovrtx

bin_dir = Path(ovrtx.__file__).resolve().parent.joinpath('bin')
print(bin_dir)
'@
    $bin = & $PythonExe -c $code
    if ($LASTEXITCODE -ne 0) {
        throw "Could not locate installed ovrtx package."
    }
    $dll = Join-Path $bin "ovrtx-dynamic.dll"
    if (-not (Test-Path -LiteralPath $dll)) {
        throw "Installed ovrtx package does not contain ovrtx-dynamic.dll at $dll"
    }
    return (Resolve-Path $bin).Path
}

function Find-VcVars64 {
    $vswhere = $null
    if (${env:ProgramFiles(x86)}) {
        $candidate = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path -LiteralPath $candidate) {
            $vswhere = $candidate
        }
    }

    if ($vswhere) {
        $installRoots = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        foreach ($root in $installRoots) {
            if (-not $root) {
                continue
            }
            $candidate = Join-Path $root "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path -LiteralPath $candidate) {
                return (Resolve-Path $candidate).Path
            }
        }
    }

    $roots = @()
    if ($env:ProgramFiles) {
        $roots += Join-Path $env:ProgramFiles "Microsoft Visual Studio\2022"
    }
    if (${env:ProgramFiles(x86)}) {
        $roots += Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022"
    }

    $editions = @("BuildTools", "Community", "Professional", "Enterprise")
    foreach ($root in $roots) {
        foreach ($edition in $editions) {
            $candidate = Join-Path $root "$edition\VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path -LiteralPath $candidate) {
                return (Resolve-Path $candidate).Path
            }
        }
    }

    foreach ($root in $roots) {
        if (Test-Path -LiteralPath $root) {
            $candidate = Get-ChildItem -LiteralPath $root -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($candidate) {
                return $candidate.FullName
            }
        }
    }

    return $null
}

function Invoke-FmuBuild {
    param([string]$PythonExe)

    if ($SkipFmuBuild) {
        Write-Host "Skipping FMU/SSP build because -SkipFmuBuild was specified."
        return
    }

    $BuildScript = Join-Path $ScriptDir "build_fmu.py"
    Write-Host "Building demo FMUs and SSPs..."

    if ($env:CXX) {
        Write-Host "Using compiler from CXX: $env:CXX"
        & $PythonExe $BuildScript
    } else {
        $VcVars = Find-VcVars64
        if (-not $VcVars) {
            throw @"
Cannot build FMUs because setup.ps1 could not find vcvars64.bat.

Install Visual Studio 2022 Build Tools with:
  - Desktop development with C++
  - MSVC v143 x64/x86 build tools

Then rerun from a normal PowerShell:
  powershell -ExecutionPolicy Bypass -File demoapp\setup.ps1

Do not rely on the generic Visual Studio "Developer Command Prompt" button for
this setup path; it may not target x64. If you intentionally want to use a
custom compiler, set CXX to an x64 compiler command. To skip FMU compilation,
rerun setup.ps1 with -SkipFmuBuild.
"@
        }

        Write-Host "Using Visual Studio environment: $VcVars"
        $cmd = "call `"$VcVars`" && `"$PythonExe`" `"$BuildScript`""
        & cmd.exe /d /s /c $cmd
    }

    if ($LASTEXITCODE -ne 0) {
        throw "FMU/SSP build failed."
    }
}

# Refresh without --clear: retain local packages, but stop inheriting globals.
py -m venv $AppVenv
if ($LASTEXITCODE -ne 0) {
    throw "Could not create or refresh the isolated demo virtual environment."
}

function Get-Cargo {
    $Cargo = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
    if (-not (Test-Path -LiteralPath $Cargo)) {
        $Cargo = "cargo"
    }
    return $Cargo
}

function Invoke-NativeBuild {
    if ($SkipNativeBuild) {
        Write-Host "Skipping native runtime build; use --backend fmpy."
        return
    }
    $Cargo = Get-Cargo
    Write-Host "Building native ovfmi runtime..."
    & $Cargo build --manifest-path (Join-Path $RepoRoot "native\Cargo.toml") --release --locked
    if ($LASTEXITCODE -ne 0) {
        throw "Native ovfmi runtime build failed. Install Rust with rustup and retry."
    }
}
if (-not (Test-Path -LiteralPath $UsdVenv)) {
    python -m venv $UsdVenv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the isolated USD virtual environment."
    }
}

$AppPython = Join-Path $AppVenv "Scripts\python.exe"
$UsdPython = Join-Path $UsdVenv "Scripts\python.exe"

Invoke-AppPip install --upgrade pip
$Requirements = Join-Path $ScriptDir "requirements.txt"
Write-Host "Installing NVIDIA runtime packages into $AppVenv ..."
Invoke-AppPip install -r $Requirements
Invoke-AppPip install -e "${OvFmiDir}[build]"
Invoke-AppPip install -e $ScriptDir
& $AppPython (Join-Path $ScriptDir "runtime_dependencies.py") --check
if ($LASTEXITCODE -ne 0) {
    throw "NVIDIA runtime dependency validation failed."
}
$OvrtxBin = Get-InstalledOvrtxBin $AppPython
Write-Host "Using ovrtx native library from: $OvrtxBin"

if ($InstallCudaPython) {
    Invoke-AppPip install cuda-python
} else {
    Write-Host "Skipping optional cuda-python bindings. CUDA display requires compatible bindings and runtime libraries."
}

& $UsdPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "USD environment pip update failed."
}
& $UsdPython -m pip install usd-core
if ($LASTEXITCODE -ne 0) {
    throw "USD parser installation failed."
}

$EnvFile = Join-Path $ScriptDir ".env"
@"
OVRTX_LIBRARY_PATH_HINT="$OvrtxBin"
OVSTAGE_LIBRARY_PATH_HINT="$OvrtxBin"
USD_PYTHON="$UsdPython"
PYTHONPATH="$ScriptDir;$env:PYTHONPATH"
"@ | Set-Content -LiteralPath $EnvFile -Encoding ASCII

Invoke-FmuBuild $AppPython
Invoke-NativeBuild

Write-Host "Wrote $EnvFile"
Write-Host "Run:"
Write-Host "  $AppPython demoapp\main.py demoapp\usd\fmi_parser_test.usda"
