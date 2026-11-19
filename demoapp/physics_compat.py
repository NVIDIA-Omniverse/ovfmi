# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep asynchronous PhysX pose reads alive through stage publication."""

from contextlib import contextmanager


@contextmanager
def pose_groups(physx, object_type, scope):
    """Keep read groups and their path dictionaries alive through stage writes."""
    with physx.query(object_type, scope=scope).wait() as query:
        with physx.read(query, ["position", "orientation"]).wait() as result:
            def groups():
                while (group := result.fetch_next().wait()) is not None:
                    yield group
            yield groups()
