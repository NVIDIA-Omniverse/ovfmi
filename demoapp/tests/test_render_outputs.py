# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import _color_render_var


@pytest.mark.parametrize("key", ["LdrColor", "/Render/Camera/LdrColor", "/Render/Vars/LdrColor"])
def test_color_output_accepts_legacy_name_and_full_paths(key):
    color = object()
    assert _color_render_var({key: color, "/Render/Camera/Depth": object()}) is color


@pytest.mark.parametrize("outputs", [
    {},
    {"Depth": object()},
    {"NotLdrColor": object()},
    {"/A/LdrColor": object(), "/B/LdrColor": object()},
])
def test_missing_or_ambiguous_color_is_not_silently_discarded(outputs):
    with pytest.raises(RuntimeError, match="Expected exactly one LdrColor"):
        _color_render_var(outputs)
