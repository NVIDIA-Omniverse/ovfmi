# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise an installed native wheel from outside its source checkout."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def smoke(fmu: Path) -> None:
    assert "OVFMI_NATIVE_LIBRARY" not in os.environ, "Remove the native library override"
    assert importlib.util.find_spec("fmpy") is None, "Use an environment without FMPy"

    import ovfmi
    from ovfmi import AttributeWrite, FmiHost, FmiHostConfig
    from ovfmi._native_runtime import _library_path

    package = Path(ovfmi.__file__).resolve().parent
    library = _library_path().resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    assert library.is_relative_to(package), library
    assert FmiHostConfig().backend == "native"

    with tempfile.TemporaryDirectory(prefix="ovfmi-smoke-") as directory:
        source = Path(directory) / "integrator.usda"
        source.write_text(f'''#usda 1.0
def Xform "World"
{{
    custom double sim:rate = 2
    custom double sim:integral = 0
    def FmuInstance "Integrator"
    {{
        asset fmi:fmu = @{fmu.as_posix()}@
        def FmuConnection "Values"
        {{
            rel fmi:targets = </World>
            def FmuMapping "Rate"
            {{
                token fmi:fmuAttribute = "rate"
                token fmi:usdAttribute = "sim:rate"
                token fmi:direction = "input"
                int2 fmi:usdMapping = (0, 1)
            }}
            def FmuMapping "Integral"
            {{
                token fmi:fmuAttribute = "integral"
                token fmi:usdAttribute = "sim:integral"
                token fmi:direction = "output"
                int2 fmi:usdMapping = (0, 1)
            }}
        }}
    }}
}}
''', encoding="utf-8")

        with FmiHost() as host:
            report = host.attach_ovstage(object(), source_asset=source)
            assert len(report.instances) == 1, report
            assert report.instances[0].enabled

            def step(dt, expected_time, expected_integral):
                host.step_sync(dt)
                assert host.time == expected_time, host.time
                with host.read(attribute_names=["sim:integral"]) as result:
                    assert len(result.groups) == 1, result.groups
                    group = result.groups[0]
                    assert group.prim_paths == ("/World",), group.prim_paths
                    actual = float(group.tensors[0][0, 0])
                    assert actual == expected_integral, (actual, expected_integral)

            step(0.25, 0.25, 0.5)
            host.wait_op(host.write([AttributeWrite(("/World",), "sim:rate", [-1.0])]))
            step(0.5, 0.75, 0.0)
            step(0.25, 1.0, -0.25)
            host.reset()
            assert host.time == 0.0
            with host.read() as result:
                assert not result.groups
            step(0.25, 0.25, 0.5)
            step(0.25, 0.5, 1.0)

    assert not any(name == "fmpy" or name.startswith("fmpy.") for name in sys.modules)
    print(json.dumps({"package": str(package), "native_library": str(library), "result": "passed"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fmu", type=Path)
    args = parser.parse_args()
    smoke(args.fmu.resolve(strict=True))
