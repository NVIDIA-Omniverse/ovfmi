#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check wheels and source distributions against ovfmi's release policy.

Validate metadata, wheel tags, and source archive layout before release.
The platform allowlist and build-tag restriction are repository policy.
This script does not resolve dependencies or run abi3audit.

Needs packaging and trove-classifiers, which both workflows install alongside
build and twine. A missing trove-classifiers is an error rather than a skipped
check, so no run can report a pass while classifier validity went unchecked.

    python3 tools/check_distributions.py dist/*
    python3 tools/check_distributions.py --self-test
"""

from __future__ import annotations

import argparse
import email
import io
import re
import sys
import tarfile
import zipfile
from functools import lru_cache
from pathlib import Path

try:  # tomllib is 3.11+, and ovfmi still supports 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - depends on the interpreter
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        raise SystemExit(
            "this script needs Python 3.11 or newer, or tomli installed on 3.10"
        ) from None

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

# Platform tags permitted by the repository's release policy.
VALID_PLATFORM_TAGS = frozenset(
    {
        "any",
        "win32",
        "win_arm64",
        "win_amd64",
        "win_ia64",
        "manylinux1_x86_64",
        "manylinux1_i686",
        "manylinux2010_x86_64",
        "manylinux2010_i686",
        "manylinux2014_x86_64",
        "manylinux2014_i686",
        "manylinux2014_aarch64",
        "manylinux2014_armv7l",
        "manylinux2014_ppc64",
        "manylinux2014_ppc64le",
        "manylinux2014_s390x",
        "linux_armv6l",
        "linux_armv7l",
    }
)

VALID_ARCH_TAGS = ("x86_64", "i686", "aarch64", "armv7l", "ppc64le", "s390x")

_GLIBC_MUSL_TAG = re.compile(
    r"^(many|musl)linux_\d+_\d+_(" + "|".join(VALID_ARCH_TAGS) + r")$"
)

_INTERPRETER_TAG = re.compile(r"^(py|cp|pp|jy|ip)(\d)(\d+)?$")
_ABI_VERSION_TAG = re.compile(r"^(cp|pypy)(\d)(\d+)\w*$")
_PYTHON_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (\d+)\.(\d+)$")


def _normalize(name: str) -> str:
    """Escape a distribution name the way a wheel filename does (PEP 427)."""
    return re.sub(r"[-_.]+", "_", name).lower()


@lru_cache(maxsize=1)
def _load_trove_classifiers() -> frozenset[str]:
    """Return the canonical classifier list, or refuse to run without it.

    Skipping this check on a missing import would let a run print a pass while
    one of the checks was quietly switched off.
    """
    try:
        from trove_classifiers import classifiers
    except ImportError:
        raise SystemExit(
            "classifier validity cannot be checked without trove-classifiers: "
            "python -m pip install trove-classifiers"
        ) from None
    return frozenset(classifiers)


def _check_metadata(meta: email.message.Message) -> list[str]:
    """Check the core metadata in wheel METADATA and sdist PKG-INFO files."""
    problems: list[str] = []

    for field in ("Metadata-Version", "Name", "Version"):
        if not meta.get(field):
            problems.append(f"metadata is missing the {field} field")

    version = meta.get("Version") or ""
    if "+" in version:
        problems.append(
            f"version {version!r} carries a local version identifier, which "
            "public index servers do not accept"
        )

    classifiers = meta.get_all("Classifier") or []
    known = _load_trove_classifiers()
    for classifier in classifiers:
        if classifier.startswith("Private ::"):
            problems.append(f"classifier {classifier!r} starts with 'Private ::'")
        elif classifier not in known:
            problems.append(f"classifier {classifier!r} is not a valid Trove classifier")

    requires_python = meta.get("Requires-Python")
    if requires_python:
        try:
            specifier = SpecifierSet(requires_python)
        except InvalidSpecifier:
            problems.append(f"Requires-Python {requires_python!r} is not a valid specifier")
        else:
            for classifier in classifiers:
                match = _PYTHON_CLASSIFIER.match(classifier)
                if match and not specifier.contains(Version(".".join(match.groups()))):
                    problems.append(
                        f"classifier {classifier!r} claims a version that "
                        f"Requires-Python {requires_python!r} excludes"
                    )

    return problems


def check_wheel(path: Path) -> list[str]:
    problems: list[str] = []

    stem = path.name[: -len(".whl")]
    fields = stem.split("-")
    if len(fields) == 6:
        problems.append(
            f"wheel filename carries build tag {fields[2]!r}; release policy rejects build tags"
        )
        del fields[2]
    if len(fields) != 5:
        return problems + [f"wheel filename {path.name!r} is not a valid wheel name"]

    dist, version, python_tags, abi_tags, platform_tags = fields

    for platform_tag in platform_tags.split("."):
        if platform_tag not in VALID_PLATFORM_TAGS and not _GLIBC_MUSL_TAG.match(platform_tag):
            problems.append(
                f"platform tag {platform_tag!r} is not permitted by the release policy"
            )

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_names = [n for n in names if re.fullmatch(r"[^/]+\.dist-info/METADATA", n)]
        if not metadata_names:
            return problems + [f"{path.name} has no .dist-info/METADATA"]
        meta = email.message_from_bytes(archive.read(metadata_names[0]))
        has_py_typed = any(n.endswith("/py.typed") or n == "py.typed" for n in names)

    problems += _check_metadata(meta)

    classifiers = meta.get_all("Classifier") or []
    if has_py_typed and "Typing :: Typed" not in classifiers:
        problems.append("wheel ships py.typed but does not declare the 'Typing :: Typed' classifier")

    name = meta.get("Name") or ""
    if name and _normalize(name) != _normalize(dist):
        problems.append(
            f"wheel filename name tag {dist!r} does not match metadata Name {name!r}"
        )

    metadata_version = meta.get("Version") or ""
    if metadata_version:
        try:
            filename_matches = Version(version) == Version(metadata_version)
        except InvalidVersion:
            filename_matches = version == metadata_version
        if not filename_matches:
            problems.append(
                f"wheel filename version tag {version!r} does not match "
                f"metadata Version {metadata_version!r}"
            )

    requires_python = meta.get("Requires-Python")
    if requires_python:
        try:
            specifier = SpecifierSet(requires_python)
        except InvalidSpecifier:
            specifier = None
        if specifier is not None:
            problems += _check_tag_versions(python_tags, specifier, requires_python, "python")
            problems += _check_tag_versions(abi_tags, specifier, requires_python, "ABI")

    return problems


def _check_tag_versions(
    tags: str, specifier: SpecifierSet, requires_python: str, kind: str
) -> list[str]:
    """Reject a tag that names an exact interpreter Requires-Python excludes."""
    problems: list[str] = []
    pattern = _INTERPRETER_TAG if kind == "python" else _ABI_VERSION_TAG
    for tag in tags.split("."):
        match = pattern.match(tag)
        if not match:
            continue
        _, major, minor = match.groups()
        if not minor:
            continue
        if not specifier.contains(Version(f"{major}.{minor}")):
            problems.append(
                f"{kind} tag {tag!r} names an interpreter that "
                f"Requires-Python {requires_python!r} excludes"
            )
    return problems


def check_sdist(path: Path) -> list[str]:
    problems: list[str] = []

    with tarfile.open(path) as archive:
        names = archive.getnames()
        roots = {name.split("/")[0] for name in names if name and not name.startswith("/")}
        if len(roots) != 1:
            problems.append(
                f"source distribution has {len(roots)} top-level entries; the "
                "format requires exactly one"
            )
        root = min(roots) if roots else ""

        expected_root = path.name[: -len(".tar.gz")]
        if root and root != expected_root:
            problems.append(
                f"source distribution top-level directory {root!r} does not "
                f"match the expected {expected_root!r}"
            )

        # PKG-INFO carries the same core metadata a wheel declares, so the
        # required fields and the local version identifier are checked here too
        # rather than only on the wheel built from the same version.
        pkg_info_name = f"{root}/PKG-INFO"
        if pkg_info_name in names:
            member = archive.extractfile(pkg_info_name)
            problems += _check_metadata(
                email.message_from_bytes(member.read() if member is not None else b"")
            )
        else:
            problems.append(f"source distribution has no {pkg_info_name}")

        pyproject_name = f"{root}/pyproject.toml"
        if pyproject_name not in names:
            return problems + [f"source distribution has no {pyproject_name}"]

        member = archive.extractfile(pyproject_name)
        raw = member.read() if member is not None else b""

    try:
        pyproject = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return problems + [f"source distribution pyproject.toml does not parse: {exc}"]

    build_system = pyproject.get("build-system")
    if not isinstance(build_system, dict):
        problems.append("source distribution pyproject.toml has no [build-system] table")
    elif "requires" not in build_system:
        problems.append("source distribution [build-system] table has no requires key")

    return problems


def check(path: Path) -> list[str]:
    if path.name.endswith(".whl"):
        return check_wheel(path)
    if path.name.endswith(".tar.gz"):
        return check_sdist(path)
    return [f"{path.name} is neither a wheel nor a source distribution"]


def _annotate(path: Path, problem: str) -> None:
    escaped = problem.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error file={path.name}::{escaped}")


def run(paths: list[Path]) -> int:
    # Load the classifier list up front so a missing dependency stops the run
    # before any "OK:" line is printed, whatever kind of distribution is first.
    _load_trove_classifiers()

    failed = False
    for path in paths:
        problems = check(path)
        if problems:
            failed = True
            for problem in problems:
                _annotate(path, problem)
        else:
            print(f"OK: {path.name} passes the distribution release checks")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Self-tests. Kept in the script so CI proves the checker still catches what it
# claims to before it is trusted to gate a distribution.
# ---------------------------------------------------------------------------

_BASE_METADATA = {
    "Metadata-Version": "2.4",
    "Name": "ovfmi",
    "Version": "0.2.0",
    "Requires-Python": ">=3.10,<3.14",
}


def _make_wheel(
    tmp: Path,
    *,
    filename: str = "ovfmi-0.2.0-py3-none-any.whl",
    metadata: dict[str, str] | None = None,
    classifiers: tuple[str, ...] = ("Programming Language :: Python :: 3.10",),
    py_typed: bool = False,
) -> Path:
    fields = dict(_BASE_METADATA if metadata is None else metadata)
    lines = [f"{key}: {value}" for key, value in fields.items()]
    lines += [f"Classifier: {value}" for value in classifiers]
    dist_info = f"{fields.get('Name', 'ovfmi')}-{fields.get('Version', '0')}.dist-info"

    path = tmp / filename
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", "\n".join(lines) + "\n")
        archive.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        if py_typed:
            archive.writestr("ovfmi/py.typed", "")
    return path


def _make_sdist(
    tmp: Path,
    *,
    filename: str = "ovfmi-0.2.0.tar.gz",
    root: str = "ovfmi-0.2.0",
    pyproject: str | None = '[build-system]\nrequires = ["setuptools>=77"]\n',
    version: str | None = "0.2.0",
    classifiers: tuple[str, ...] = (),
) -> Path:
    path = tmp / filename
    with tarfile.open(path, "w:gz") as archive:
        if version is not None:
            lines = ["Metadata-Version: 2.4", "Name: ovfmi", f"Version: {version}"]
            lines += [f"Classifier: {value}" for value in classifiers]
            pkg_info = ("\n".join(lines) + "\n").encode()
            info = tarfile.TarInfo(f"{root}/PKG-INFO")
            info.size = len(pkg_info)
            archive.addfile(info, io.BytesIO(pkg_info))
        if pyproject is not None:
            raw = pyproject.encode("utf-8")
            info = tarfile.TarInfo(f"{root}/pyproject.toml")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        else:
            info = tarfile.TarInfo(f"{root}/setup.py")
            info.size = 0
            archive.addfile(info, io.BytesIO(b""))
    return path


def self_test() -> int:
    import tempfile

    _load_trove_classifiers()

    failures: list[str] = []

    def expect(label: str, problems: list[str], substring: str | None) -> None:
        if substring is None:
            if problems:
                failures.append(f"{label}: expected no problems, got {problems}")
        elif not any(substring in problem for problem in problems):
            failures.append(f"{label}: expected {substring!r} in {problems}")

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)

        expect("clean wheel", check(_make_wheel(tmp)), None)

        (tmp / "build-tag").mkdir()
        expect(
            "build tag",
            check(_make_wheel(tmp / "build-tag", filename="ovfmi-0.2.0-1-py3-none-any.whl")),
            "build tag",
        )

        (tmp / "platform").mkdir()
        expect(
            "bad platform tag",
            check(_make_wheel(tmp / "platform", filename="ovfmi-0.2.0-py3-none-linux_x86_64.whl")),
            "platform tag",
        )

        (tmp / "manylinux").mkdir()
        expect(
            "manylinux platform tag",
            check(
                _make_wheel(
                    tmp / "manylinux",
                    filename="ovfmi-0.2.0-py3-none-manylinux_2_34_x86_64.whl",
                )
            ),
            None,
        )

        (tmp / "local").mkdir()
        expect(
            "local version",
            check(
                _make_wheel(
                    tmp / "local",
                    filename="ovfmi-0.2.0+nv1-py3-none-any.whl",
                    metadata={**_BASE_METADATA, "Version": "0.2.0+nv1"},
                )
            ),
            "local version identifier",
        )

        (tmp / "classifier").mkdir()
        expect(
            "private classifier",
            check(_make_wheel(tmp / "classifier", classifiers=("Private :: Do Not Upload",))),
            "Private ::",
        )

        # This case is also what makes a missing trove-classifiers install
        # visible: without it the checker exits rather than passing silently.
        (tmp / "invalid-classifier").mkdir()
        expect(
            "invalid classifier",
            check(
                _make_wheel(
                    tmp / "invalid-classifier",
                    classifiers=("Development Status :: 99 - Imaginary",),
                )
            ),
            "is not a valid Trove classifier",
        )

        (tmp / "requires").mkdir()
        expect(
            "classifier outside Requires-Python",
            check(
                _make_wheel(
                    tmp / "requires",
                    classifiers=("Programming Language :: Python :: 3.9",),
                )
            ),
            "Requires-Python",
        )

        (tmp / "abi").mkdir()
        expect(
            "interpreter tag outside Requires-Python",
            check(
                _make_wheel(
                    tmp / "abi",
                    filename="ovfmi-0.2.0-cp39-cp39-manylinux_2_34_x86_64.whl",
                )
            ),
            "Requires-Python",
        )

        (tmp / "typed").mkdir()
        expect(
            "py.typed without classifier",
            check(_make_wheel(tmp / "typed", py_typed=True)),
            "Typing :: Typed",
        )

        (tmp / "typed-ok").mkdir()
        expect(
            "py.typed with classifier",
            check(
                _make_wheel(
                    tmp / "typed-ok",
                    classifiers=("Programming Language :: Python :: 3.10", "Typing :: Typed"),
                    py_typed=True,
                )
            ),
            None,
        )

        (tmp / "name").mkdir()
        expect(
            "name mismatch",
            check(_make_wheel(tmp / "name", metadata={**_BASE_METADATA, "Name": "other"})),
            "does not match metadata Name",
        )

        (tmp / "missing").mkdir()
        expect(
            "missing metadata field",
            check(
                _make_wheel(
                    tmp / "missing",
                    metadata={"Name": "ovfmi", "Version": "0.2.0"},
                    classifiers=(),
                )
            ),
            "missing the Metadata-Version field",
        )

        expect("clean sdist", check(_make_sdist(tmp)), None)

        (tmp / "sdist-root").mkdir()
        expect(
            "sdist root mismatch",
            check(_make_sdist(tmp / "sdist-root", root="ovfmi")),
            "top-level directory",
        )

        (tmp / "sdist-none").mkdir()
        expect(
            "sdist without pyproject",
            check(_make_sdist(tmp / "sdist-none", pyproject=None)),
            "pyproject.toml",
        )

        (tmp / "sdist-build").mkdir()
        expect(
            "sdist without build-system",
            check(_make_sdist(tmp / "sdist-build", pyproject='[project]\nname = "ovfmi"\n')),
            "[build-system]",
        )

        (tmp / "sdist-local").mkdir()
        expect(
            "sdist with a local version identifier",
            check(
                _make_sdist(
                    tmp / "sdist-local",
                    filename="ovfmi-0.2.0+nv1.tar.gz",
                    root="ovfmi-0.2.0+nv1",
                    version="0.2.0+nv1",
                )
            ),
            "local version identifier",
        )

        (tmp / "sdist-pkginfo").mkdir()
        expect(
            "sdist without PKG-INFO",
            check(_make_sdist(tmp / "sdist-pkginfo", version=None)),
            "PKG-INFO",
        )

        (tmp / "sdist-classifier").mkdir()
        expect(
            "sdist with an invalid classifier",
            check(
                _make_sdist(
                    tmp / "sdist-classifier",
                    classifiers=("Development Status :: 99 - Imaginary",),
                )
            ),
            "is not a valid Trove classifier",
        )

    for failure in failures:
        print(f"::error::self-test {failure}")
    if failures:
        return 1
    print("OK: check_distributions self-tests pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dists", nargs="*", type=Path, help="wheels and sdists to check")
    parser.add_argument(
        "--self-test", action="store_true", help="verify the checker against crafted distributions"
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.dists:
        parser.error("pass at least one distribution, or --self-test")
    return run(args.dists)


if __name__ == "__main__":
    sys.exit(main())
