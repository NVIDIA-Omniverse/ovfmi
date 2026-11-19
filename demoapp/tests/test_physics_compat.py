# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from physics_compat import pose_groups


@pytest.mark.parametrize("fail_consumer", [False, True])
def test_query_session_fetch_and_cleanup(fail_consumer):
    events = []
    values = iter(["position-group", "orientation-group", None])

    def fetch():
        events.append("fetch")
        return SimpleNamespace(wait=lambda: next(values))

    @contextmanager
    def query():
        events.append("query-open")
        try:
            yield "query"
        finally:
            events.append("query-close")

    @contextmanager
    def session():
        events.append("read-open")
        try:
            yield SimpleNamespace(fetch_next=fetch)
        finally:
            events.append("read-close")

    def select(object_type, *, scope):
        assert (object_type, scope) == ("rigid", "all")
        return SimpleNamespace(wait=query)

    def read(selection, attributes):
        assert selection == "query"
        assert attributes == ["position", "orientation"]
        return SimpleNamespace(wait=session)

    physx = SimpleNamespace(query=select, read=read)
    try:
        with pose_groups(physx, "rigid", "all") as groups:
            assert next(groups) == "position-group"
            assert "read-close" not in events
            if fail_consumer:
                raise RuntimeError("stage write failed")
            assert list(groups) == ["orientation-group"]
    except RuntimeError:
        assert fail_consumer
    assert events[:2] == ["query-open", "read-open"]
    assert events[-2:] == ["read-close", "query-close"]
    assert events.count("fetch") == (1 if fail_consumer else 3)
