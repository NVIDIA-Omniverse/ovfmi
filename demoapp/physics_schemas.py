# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register PhysX schemas before accessing the shared USD schema registry."""


def register_physx_schemas(ovstage):
    """Register both schema families before renderer construction or population.

    The discovery helper reads distribution metadata only. Importing
    physx_usd_schemas itself would load a separate stock USD runtime.
    """
    from ovphysx.samples.python_samples.schema_support import physx_schema_paths

    ovstage.population.register_usd_schemas(
        [str(path) for path in physx_schema_paths()]
    )
