# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flushed, timed startup diagnostics around blocking native operations."""

from contextlib import contextmanager
import time


@contextmanager
def startup_phase(label: str, *, note: str = ""):
    """Report call boundaries, not inferred native progress or shader activity."""
    print(f"[startup] {label}...", flush=True)
    if note:
        print(f"[startup] {note}", flush=True)
    started = time.monotonic()
    try:
        yield
    except BaseException:
        print(f"[startup] {label} failed after {time.monotonic() - started:.1f}s.",
              flush=True)
        raise
    else:
        print(f"[startup] {label} completed in {time.monotonic() - started:.1f}s.",
              flush=True)
