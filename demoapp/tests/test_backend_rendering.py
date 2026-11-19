# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in GPU smoke matrix. Set OVFMI_TEST_RENDER=headless or headful."""

import os
import json
import re
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODE = os.environ.get("OVFMI_TEST_RENDER")
pytestmark = pytest.mark.skipif(MODE not in ("headless", "headful"),
                                reason="set OVFMI_TEST_RENDER=headless or headful on an RTX machine")


@pytest.mark.parametrize("backend", ["native", "fmpy"])
@pytest.mark.parametrize("scene", ["fmi_parser_test.usda", "pd_controller_test.usda",
                                  "two_fmu_orbit.usda", "ssp_orbit.usda",
                                  "conveyor/ConveyorFMI.usda"])
def test_rendered_demo(backend, scene, tmp_path):
    args = [sys.executable, str(ROOT / "demoapp/main.py"), str(ROOT / "demoapp/usd" / scene),
            "--backend", backend, "--duration", "1"]
    if MODE == "headless":
        args.append("--headless")
    if "conveyor" in scene:
        args += ["--up-axis", "Z"]
    # A renderer may launch a persistent cache service that inherits stdout.
    # A PIPE reader can then wait indefinitely for EOF after the demo exits,
    # including inside subprocess.run's timeout cleanup. A file also preserves
    # diagnostics if initialization fails or times out.
    log_path = tmp_path / "demo.log"
    with log_path.open("wb") as output:
        try:
            result = subprocess.run(args, cwd=ROOT, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=180)
        except subprocess.TimeoutExpired:
            pytest.fail("Demo exceeded 180 seconds:\n" +
                        log_path.read_text(encoding="utf-8", errors="replace")[-12000:])
    log = log_path.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, log[-12000:]
    assert f"FMI backend: {backend}" in log
    assert "Done: 60 frames" in log, log[-12000:]
    assert "fmi=on" in log
    counts = re.search(r"Color frames: (\d+) read, (\d+) presented\.", log)
    assert counts is not None, log[-12000:]
    assert int(counts[1]) > 0, log[-12000:]
    if MODE == "headful":
        assert "Renderer ready (live display window active)." in log, log[-12000:]
        assert "Could not open display window" not in log
        assert int(counts[2]) == int(counts[1]), log[-12000:]
        assert "First color frame presented to display." in log, log[-12000:]
    if scene != "fmi_parser_test.usda":
        assert "physics=on" in log, log[-12000:]
    assert "Traceback" not in log


@pytest.mark.parametrize("backend", ["native", "fmpy"])
def test_conveyor_moves_with_stationary_camera(backend, tmp_path):
    """Counting frames alone cannot catch frozen geometry or temporal noise."""
    # CPU CI collects but skips GPU tests without installing demo dependencies.
    from PIL import Image

    args = [sys.executable, str(ROOT / "demoapp/tests/capture_render_frames.py"),
            str(tmp_path), str(ROOT / "demoapp/usd/conveyor/ConveyorFMI.usda"),
            "--backend", backend, "--duration", "2", "--up-axis", "Z"]
    if MODE == "headless":
        args.append("--headless")
    log_path = tmp_path / "motion.log"
    with log_path.open("wb") as output:
        result = subprocess.run(args, cwd=ROOT, stdout=output,
                                stderr=subprocess.STDOUT, timeout=360)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, log[-12000:]
    assert "physics=on" in log and "fmi=on" in log, log[-12000:]
    with Image.open(tmp_path / "frame-30.png") as image:
        before = np.asarray(image, dtype=np.int16)[:, :, :3]
    with Image.open(tmp_path / "frame-119.png") as image:
        after = np.asarray(image, dtype=np.int16)[:, :, :3]
    # A fixed camera and settled shaders make this a motion check. The broken
    # renderer changed <0.01% of pixels by >30 levels; moving boxes changed ~8%.
    changed_fraction = np.mean(np.max(np.abs(after - before), axis=2) > 30)
    assert changed_fraction > 0.01, f"Geometry appears frozen: {changed_fraction=:.5f}"


@pytest.mark.parametrize("backend", ["native", "fmpy"])
def test_conveyor_stays_level_after_reversal(backend, tmp_path):
    """Run past the sensor reversal; short rendering tests miss contact jumps."""
    args = [sys.executable, str(ROOT / "demoapp/tests/capture_render_frames.py"),
            str(tmp_path), str(ROOT / "demoapp/usd/conveyor/ConveyorFMI.usda"),
            "--backend", backend, "--duration", "15", "--up-axis", "Z"]
    if MODE == "headless":
        args.append("--headless")
    log_path = tmp_path / "stability.log"
    with log_path.open("wb") as output:
        result = subprocess.run(args, cwd=ROOT, stdout=output,
                                stderr=subprocess.STDOUT, timeout=360)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, log[-12000:]
    assert "Done: 900 frames" in log and "fmi=on  physics=on" in log, log[-12000:]
    rows = [json.loads(line) for line in
            (tmp_path / "physics-poses.jsonl").read_text().splitlines()]
    assert len(rows) == 900
    body = "/World/sm_box_corrugated_brown_b13_01/Geometry/sm_box_corrugated_b13_obj_00"
    # Exclude the authored initial drop. Observe physical poses, not pixels.
    positions = np.asarray([row["boxes"][body] for row in rows[60:]])
    assert np.isfinite(positions).all()
    assert np.ptp(positions[:, 0]) > 0.5, "Box did not travel along the conveyor"
    assert np.ptp(positions[:, 2]) < 0.03, (
        f"Box jumped after reversal: world Z range "
        f"{positions[:, 2].min():.6f}..{positions[:, 2].max():.6f} m"
    )
