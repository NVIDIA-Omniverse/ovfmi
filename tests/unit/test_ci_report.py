# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CI coverage gate must fail closed when runtimes are unavailable."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "check_test_report", Path(__file__).resolve().parents[2] / "tools/check_test_report.py"
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def report(tmp_path, extra="", omit=None):
    cases = "".join(
        f'<testcase classname="tests.integration.{name}" name="test_runtime"/>'
        for name in CHECKER.REQUIRED_MODULES if name != omit
    )
    path = tmp_path / "report.xml"
    path.write_text(f"<testsuites><testsuite>{cases}{extra}</testsuite></testsuites>")
    return path


def test_both_backends_and_opt_in_gpu_skips_are_accepted(tmp_path):
    path = report(tmp_path, '<testcase classname="demoapp.tests.test_backend_rendering"><skipped/></testcase>')
    assert CHECKER.validate_report(path) == []


@pytest.mark.parametrize("status", ["skipped", "failure", "error"])
def test_incomplete_cpu_tests_are_rejected(tmp_path, status):
    path = report(tmp_path, f'<testcase classname="tests.integration.test_runtime_backends"><{status}/></testcase>')
    assert CHECKER.validate_report(path)


@pytest.mark.parametrize("module", sorted(CHECKER.REQUIRED_MODULES))
def test_uncollected_runtime_suite_is_rejected(tmp_path, module):
    assert CHECKER.validate_report(report(tmp_path, omit=module))


def test_empty_report_is_rejected(tmp_path):
    path = tmp_path / "empty.xml"
    path.write_text("<testsuites/>")
    assert CHECKER.validate_report(path)
