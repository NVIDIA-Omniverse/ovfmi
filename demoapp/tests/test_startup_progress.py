# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import startup_progress


def test_phase_reports_before_blocking_work_and_times_completion(monkeypatch, capsys):
    times = iter([10.0, 12.5])
    monkeypatch.setattr(startup_progress.time, "monotonic", lambda: next(times))
    with startup_progress.startup_phase("Creating renderer", note="Cache may be cold."):
        assert capsys.readouterr().out == (
            "[startup] Creating renderer...\n[startup] Cache may be cold.\n"
        )
    assert capsys.readouterr().out == "[startup] Creating renderer completed in 2.5s.\n"


def test_phase_preserves_failure(monkeypatch, capsys):
    monkeypatch.setattr(startup_progress.time, "monotonic", lambda: 10.0)
    error = RuntimeError("native initialization failed")
    with pytest.raises(RuntimeError) as exc:
        with startup_progress.startup_phase("Creating renderer"):
            raise error
    assert exc.value is error
    output = capsys.readouterr().out
    assert "failed after 0.0s" in output
    assert "completed" not in output


def test_phase_flushes_redirected_output(monkeypatch):
    calls = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: calls.append(kwargs))
    with startup_progress.startup_phase("Creating renderer", note="Please wait."):
        pass
    assert len(calls) == 3
    assert all(call.get("flush") is True for call in calls)
