# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect license files for the native dependencies and Rust standard library."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def license_files(root):
    prefixes = ("license", "licence", "copying", "copyright", "notice", "unlicense")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name.lower().startswith(prefixes))


def package_licenses(package, fallbacks):
    root = Path(package["manifest_path"]).resolve().parent
    files = license_files(root)
    if package.get("license_file"):
        declared = (root / package["license_file"]).resolve()
        if not declared.is_file():
            raise RuntimeError(f"Missing declared license file: {declared}")
        files = sorted(set(files) | {declared})
    if files:
        return [(path.name if not path.is_relative_to(root) else path.relative_to(root).as_posix(), path) for path in files], None
    if package["source"].startswith("git+"):
        for parent in root.parents:
            if (parent / ".git").exists():
                files = [path for path in license_files(parent) if path.parent == parent]
                if files:
                    return [(path.name, path) for path in files], None
                break
    fallback = fallbacks / f"{package['name']}-{package['version']}"
    provenance_file = fallback / "provenance.json"
    if provenance_file.is_file():
        provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        vcs = json.loads((root / ".cargo_vcs_info.json").read_text(encoding="utf-8"))
        if (
            provenance["package"] == package["name"]
            and provenance["version"] == package["version"]
            and provenance["source"] == package["source"]
            and provenance["git_revision"] == vcs["git"]["sha1"]
        ):
            return [(name, fallback / name) for name in provenance["files"]], provenance
    raise RuntimeError(f"No license text found for {package['name']} {package['version']}")


def copy_license_files(files, output, directory):
    entries = []
    for name, source in files:
        destination = output / directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entries.append({
            "path": destination.relative_to(output).as_posix(),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
    return entries


def standard_library_notices(manifest, target, output, target_dir):
    # Cargo selects the compiler, including environment and config overrides.
    command = ["cargo", "rustc", "--lib", "--release", "--locked",
               "--manifest-path", str(manifest), "--target", target]
    if target_dir is not None:
        command.extend(["--target-dir", str(target_dir)])
    sysroot = Path(subprocess.check_output(command + ["--", "--print=sysroot"], text=True).strip())
    version = subprocess.check_output(command + ["--", "-vV"], text=True)
    compiler = dict(line.split(": ", 1) for line in version.splitlines() if ": " in line)
    documentation = sysroot / "share" / "doc" / "rust"
    required = ("COPYRIGHT-library.html", "licenses/Apache-2.0.txt", "licenses/MIT.txt")
    for name in required:
        if not (documentation / name).is_file():
            raise RuntimeError(f"Missing Rust standard-library notice: {documentation / name}")
    paths = [documentation / "COPYRIGHT-library.html"]
    paths.extend(path for path in sorted((documentation / "licenses").rglob("*")) if path.is_file())
    files = [(path.relative_to(documentation).as_posix(), path) for path in paths]
    return {
        "version": compiler["release"],
        "commit": compiler["commit-hash"],
        "host": compiler["host"],
        "files": copy_license_files(files, output, "rust-standard-library"),
    }


def generate(manifest, target, output, target_dir=None):
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--format-version", "1",
         "--manifest-path", str(manifest), "--filter-platform", target],
        text=True,
    ))
    resolved = {node["id"] for node in metadata["resolve"]["nodes"]}
    packages = sorted(
        (package for package in metadata["packages"] if package["id"] in resolved and package["source"]),
        key=lambda package: (package["name"], package["version"]),
    )
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    inventory = []
    for package in packages:
        files, provenance = package_licenses(package, manifest.parent / "licenses")
        entry = {key: package[key] for key in ("name", "version", "source", "license")}
        entry["files"] = copy_license_files(files, output, f"{package['name']}-{package['version']}")
        if provenance:
            entry["provenance"] = provenance
        inventory.append(entry)
    (output / "packages.json").write_text(json.dumps({
        "target": target,
        "cargo_lock_sha256": hashlib.sha256(manifest.with_name("Cargo.lock").read_bytes()).hexdigest(),
        "packages": inventory,
        "rust_standard_library": standard_library_notices(manifest, target, output, target_dir),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path)
    args = parser.parse_args()
    generate(args.manifest.resolve(), args.target, args.output.resolve(), args.target_dir)
