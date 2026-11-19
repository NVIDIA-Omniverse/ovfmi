# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess helper for opt-in GPU tests; does not alter simulation or camera."""

from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from ovstage_renderer import OvstageRenderer


def run():
    destination = Path(sys.argv.pop(1))
    destination.mkdir(parents=True, exist_ok=True)
    original_step = OvstageRenderer.step
    frame_number = 0
    original_poses = main._write_physics_poses_to_ovstage
    pose_trace = (destination / "physics-poses.jsonl").open("w", buffering=1)
    physics_frame = 0

    def capture_poses(physx, renderer, ordinal):
        nonlocal physics_frame
        matrices = original_poses(physx, renderer, ordinal)
        boxes = {path: matrix[3, :3].tolist() for path, matrix in matrices.items()
                 if "/sm_box_" in path}
        pose_trace.write(json.dumps({"frame": physics_frame, "boxes": boxes}) + "\n")
        physics_frame += 1
        return matrices

    def capture(self, *args, **kwargs):
        nonlocal frame_number
        products = original_step(self, *args, **kwargs)
        if frame_number in (30, 119):
            for product in products.values():
                for frame in product.frames:
                    with main._color_render_var(frame.render_vars).map() as mapped:
                        pixels = np.from_dlpack(mapped).copy()
                    Image.fromarray(pixels).save(destination / f"frame-{frame_number}.png")
        frame_number += 1
        return products

    OvstageRenderer.step = capture
    main._write_physics_poses_to_ovstage = capture_poses
    try:
        main.main()
    finally:
        pose_trace.close()


if __name__ == "__main__":
    run()
