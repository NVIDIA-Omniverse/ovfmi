# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMOAPP = REPO_ROOT / "demoapp"
sys.path.insert(0, str(DEMOAPP))

import main as ov_fmi_main  # noqa: E402
from fmi_parser import (  # noqa: E402
    FmuDirection,
    FmuParserConnection,
    FmuParserInstance,
    FmuParserMapping,
)


@pytest.mark.parametrize("warp_columns", [False, True])
def test_physics_read_uses_keyword_scope_and_host_columns(monkeypatch, warp_columns):
    writes = []
    converted = []
    class WarpColumn:
        def __init__(self, values):
            self.values = values
        def numpy(self):
            converted.append(self)
            return self.values

    def group(name, values):
        tensor = np.asarray(values, dtype=np.float32)
        if warp_columns:
            tensor = WarpColumn(tensor)
        return SimpleNamespace(attribute=name, tensors=[tensor], prim_list=7,
                               is_delete=False, is_array=False, semantic=0)

    class PhysX:
        def query(self, object_type, *, scope):
            assert scope == "all"
            return SimpleNamespace(wait=lambda: nullcontext(object_type))

        def read(self, object_type, names):
            assert names == ["position", "orientation"]
            groups = [group("position", [[1, 2, 3]]),
                      group("orientation", [[0, 0, 0, 1]])] if object_type == "rigid" else []
            values = iter(groups)
            session = SimpleNamespace(fetch_next=lambda: SimpleNamespace(
                wait=lambda: next(values, None)))
            return SimpleNamespace(wait=lambda: nullcontext(session))

    def write(query, name, ordinal, values, **_kwargs):
        assert query == 7 and ordinal == 3
        writes.append((name, np.asarray(values).copy()))
        return SimpleNamespace(wait=lambda: None)

    stage = SimpleNamespace(query_from_path_list=lambda paths: paths,
                            write_attribute=write,
                            release_query=lambda _q: SimpleNamespace(wait=lambda: None))
    paths = SimpleNamespace(token_to_string=lambda token: token,
                            get_path_strings=lambda _paths: ["/World/body"],
                            destroy=lambda: None)
    monkeypatch.setitem(sys.modules, "ovstage", SimpleNamespace(
        PathDictionary=lambda _stage: paths, AttributeSemantic=SimpleNamespace(MATRIX=1)))
    monkeypatch.setitem(sys.modules, "ovphysx.types", SimpleNamespace(
        ObjectScope=SimpleNamespace(ALL="all"),
        SimObjectType=SimpleNamespace(RIGID_BODY="rigid", ARTICULATION_LINK="link")))
    monkeypatch.setattr(ov_fmi_main, "_ovstage_column", lambda _stage, values, **_kwargs: values)
    result = ov_fmi_main._write_physics_poses_to_ovstage(PhysX(), SimpleNamespace(ovstage=stage), 3)
    assert list(result) == ["/World/body"]
    assert result["/World/body"].shape == (4, 4)
    assert [name for name, _ in writes] == ["sim:position", "sim:orientation", "omni:fabric:worldMatrix"]
    np.testing.assert_array_equal(writes[0][1], [[1, 2, 3]])
    assert len(converted) == (2 if warp_columns else 0)


class _RuntimeInstance:
    def __init__(self, parser_instance):
        self._parser_instance = parser_instance

    def get_parser_instance(self):
        return self._parser_instance


def test_drive_target_is_written_only_to_control_ordinal(monkeypatch):
    parser_instance = FmuParserInstance(
        enabled=True,
        fmu="unused.fmu",
        path="/World/FMI",
        connections=[
            FmuParserConnection(
                enabled=True,
                targets=["/World/Joint"],
                mappings=[
                    FmuParserMapping(
                        fmiAttributeName="targetVelocity",
                        usdAttributeName="drive:angular:physics:targetVelocity",
                        direction=FmuDirection.OUTPUT,
                        usdMapping=(0, 0),
                    )
                ],
            )
        ],
    )
    instance = _RuntimeInstance(parser_instance)
    input_head = ov_fmi_main.OvrtxInputHead({})
    output_tail = ov_fmi_main.OvstageOutputTail(object(), input_head)
    output_tail.cache_connections(instance)

    result = np.array([(math.pi,)], dtype=[("targetVelocity", np.float32)])[0]
    output_tail.write_outputs(instance, ["targetVelocity"], result)

    writes = []
    monkeypatch.setattr(
        ov_fmi_main,
        "_write_ovstage_attribute",
        lambda _renderer, paths, attr, values, ordinal, **_kwargs: writes.append(
            (paths, attr, values.copy(), ordinal)
        ),
    )

    assert output_tail.flush_to_ovstage(3) is False
    assert output_tail.flush_controls_to_ovstage(2) is True
    assert len(writes) == 1
    assert writes[0][0] == ["/World/Joint"]
    assert writes[0][1] == "drive:angular:physics:targetVelocity"
    np.testing.assert_allclose(writes[0][2], [[180.0]], rtol=1.0e-6)
    assert writes[0][3] == 2


def test_overlap_presence_is_published_as_stage_attribute(monkeypatch):
    class _PhysX:
        def __init__(self):
            self.calls = []

        def overlap(self, geometry_type, **kwargs):
            self.calls.append((geometry_type, kwargs))
            return [{"rigid_body": 1}]

    physx = _PhysX()
    writes = []
    monkeypatch.setattr(
        ov_fmi_main,
        "_write_ovstage_attribute",
        lambda _renderer, paths, attr, values, ordinal, **_kwargs: writes.append(
            (paths, attr, values.copy(), ordinal)
        ),
    )

    wrote = ov_fmi_main._publish_overlap_sensor_results(
        physx,
        object(),
        {
            "/World/Sensor": {
                "query_shape": "/World/Sensor/Sphere",
                "position": [0.0, 0.0, 0.0],
                "radius": 0.1,
            }
        },
        ordinal=3,
        shape_geometry_type="shape",
        sphere_geometry_type="sphere",
        query_mode="any",
    )

    assert wrote is True
    assert physx.calls == [
        ("shape", {"mode": "any", "prim_path": "/World/Sensor/Sphere"})
    ]
    assert writes[0][0] == ["/World/Sensor"]
    assert writes[0][1] == "sensor:presence"
    np.testing.assert_array_equal(writes[0][2], [[1.0]])
    assert writes[0][3] == 3


def test_stage_input_refresh_can_be_limited_to_sensor_presence(monkeypatch):
    head = ov_fmi_main.OvrtxInputHead(
        {
            "/World/Sensor": {"sensor:presence": [0.0]},
            "/World/Control": {"xformOp:translate": [16.0, 16.0, 0.0]},
        }
    )
    head._input_map = {
        "/World/FMI": {
            "presence": ("/World/Sensor", "sensor:presence", 0, 1),
            "speed": ("/World/Control", "xformOp:translate", 0, 1),
        }
    }
    reads = []

    def _read(_renderer, paths, attr, ordinal):
        reads.append((paths, attr, ordinal))
        return {"/World/Sensor": [1.0]}

    monkeypatch.setattr(ov_fmi_main, "_read_ovstage_attribute_values", _read)
    head.update_from_ovstage(object(), 3, {"sensor:presence"})

    assert reads == [(["/World/Sensor"], "sensor:presence", 3)]
    assert head.get_inputs_for(_RuntimeInstance(FmuParserInstance(
        enabled=True,
        fmu="unused.fmu",
        path="/World/FMI",
        connections=[],
    ))) == {"presence": 1.0, "speed": 16.0}
