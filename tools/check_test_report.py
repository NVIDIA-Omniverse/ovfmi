# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject incomplete CPU CI runs, especially silently skipped native tests.

Local pytest remains permissive for FMPy-only development. CI must build both
backends and all models; only the separately opt-in RTX tests may be skipped.
"""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET


REQUIRED_MODULES = {
    "test_native_conveyor",
    "test_native_lifecycle",
    "test_runtime_backends",
}


def validate_report(path: Path) -> list[str]:
    """Return actionable errors for a pytest JUnit XML report."""
    cases = ET.parse(path).getroot().findall(".//testcase")
    errors = []
    covered = set()
    if not cases:
        errors.append("No tests were reported")
    for case in cases:
        module = case.get("classname", "").split(".")[-1]
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("failure") is not None or case.find("error") is not None:
            errors.append(f"Failed test: {name}")
        elif case.find("skipped") is not None:
            if module != "test_backend_rendering":
                errors.append(f"Unexpected skipped test: {name}")
        else:
            covered.add(module)
    for module in sorted(REQUIRED_MODULES - covered):
        errors.append(f"Required runtime coverage missing: {module}")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python tools/check_test_report.py <pytest-junit.xml>", file=sys.stderr)
        return 2
    try:
        errors = validate_report(Path(sys.argv[1]))
    except (OSError, ET.ParseError) as error:
        errors = [f"Cannot read test report: {error}"]
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("Both runtime suites executed; no unexpected skips or failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
