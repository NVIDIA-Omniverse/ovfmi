# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic for first-step inputs and Float64 publication (no rendering).

Run with the demo Python environment and USD_PYTHON from demoapp/.env:
  python tests/integration/probe_first_step_inputs.py --backend native --route stage
  python tests/integration/probe_first_step_inputs.py --backend fmpy --route write
  python tests/integration/probe_first_step_inputs.py --backend direct

Uses our generated PDController FMU: force_x = -50 * pos_x. Reports actual
behavior, not a passing regression assertion; no library implementation is changed.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def installed_versions():
    versions = {}
    for name in ('ovfmi', 'fmpy', 'numpy', 'ovstage'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def scene(fmu, mapping_count=1):
    return f'''#usda 1.0
def Xform "World"
{{
    custom double probe:input = 0
    custom double probe:output = 0
    def FmuInstance "Controller"
    {{
        asset fmi:fmu = @{fmu.as_posix()}@
        def FmuConnection "Signals"
        {{
            rel fmi:targets = </World>
            def FmuMapping "Input"
            {{
                token fmi:direction = "input"
                token fmi:fmuAttribute = "pos_x"
                token fmi:usdAttribute = "probe:input"
                int2 fmi:usdMapping = (0, {mapping_count})
            }}
            def FmuMapping "Output"
            {{
                token fmi:direction = "output"
                token fmi:fmuAttribute = "force_x"
                token fmi:usdAttribute = "probe:output"
                int2 fmi:usdMapping = (0, {mapping_count})
            }}
        }}
    }}
}}
'''


def probe(args):
    fmu = ROOT / 'demoapp/usd' / ('PDController.fmu3' if args.fmi_version == 3 else 'PDController.fmu')
    assert fmu.exists(), 'Run demo setup to generate the PDController FMUs'
    result = {
        'backend': args.backend, 'route': args.route, 'fmi_version': args.fmi_version,
        'input': args.value, 'warmup': args.warmup, 'after_reset': args.after_reset,
        'expected': -50.0 * args.value, 'python': sys.version,
        'versions': installed_versions(), 'mapping_count': args.mapping_count,
    }
    if args.backend == 'direct':
        from fmpy import simulate_fmu
        outputs = simulate_fmu(str(fmu), fmi_type='CoSimulation', start_values={'pos_x': args.value},
                               output=['force_x'], stop_time=0.01)
        result.update(raw=float(outputs['force_x'][-1]), raw_dtype=str(outputs.dtype['force_x']))
        return result

    if args.backend == 'native':
        from importlib.abc import MetaPathFinder
        class BlockFmpy(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'fmpy' or fullname.startswith('fmpy.'):
                    raise ImportError('Native diagnostic must not import FMPy')
        sys.meta_path.insert(0, BlockFmpy())

    from ovfmi import AttributeWrite, FmiHost, FmiHostConfig
    import ovfmi
    result['ovfmi_source'] = ovfmi.__file__
    if args.backend == 'native':
        from ovfmi._native_runtime import _library_path
        result['native_library'] = str(_library_path())
    with tempfile.TemporaryDirectory(prefix='ovfmi-first-step-') as temporary:
        source = Path(temporary)/'probe.usda'
        source.write_text(scene(fmu, args.mapping_count), encoding='utf-8')
        with contextlib.ExitStack() as stack:
            stage = object()
            if args.route == 'stage':
                import ovstage
                from ovstage import population
                stage = stack.enter_context(ovstage.Stage('ovfmi.first-step-probe'))
                population.open_usd(stage, str(source), ordinal=1)
            host = stack.enter_context(FmiHost(FmiHostConfig(backend=args.backend)))
            host.attach_ovstage(stage, source_asset=source)
            if args.after_reset:
                host.step_sync(0.01)
                host.reset()
            if args.warmup:
                host.step_sync(0.01)

            # Test-only instrumentation: observe the backend's original output
            # BEFORE the common routing/snapshot layer, without changing values.
            captured = []
            original = host._output_router.write_outputs
            def capture(instance, outputs, sample):
                captured.append({'value': float(sample['force_x']),
                                 'dtype': str(sample.dtype['force_x'])})
                return original(instance, outputs, sample)
            host._output_router.write_outputs = capture

            if args.route == 'write':
                host.wait_op(host.write([AttributeWrite(('/World',), 'probe:input',
                                                       np.array([[args.value]], dtype=np.float64))]))
            else:
                dictionary = ovstage.PathDictionary(stage)
                paths = dictionary.create_path_list_from_strings(['/World'])
                query = stage.query_from_path_list(paths)
                try:
                    token = dictionary.intern_token('probe:input')
                    stage.write_attribute(query, token, 2, np.array([args.value], dtype=np.float64),
                                          is_array=False).wait()
                    stage.advance_write_floor(2).wait()
                    # Independent read verifies the stage itself holds the new input.
                    with stage.read_attributes(query, [token], ovstage.OrdinalRange.latest(2)) as read:
                        read.wait()
                        result['stage_inputs'] = []
                        for group in read.groups():
                            try:
                                column = group.columns[0]
                                array = np.asarray(column.array(0))
                                result['stage_inputs'].append({'value':array.tolist(), 'dtype':str(array.dtype)})
                            finally:
                                stage.release_group(group)
                    host.update_from_ovstage(2, 2)
                finally:
                    stage.release_query(query).wait()
                    dictionary.destroy_path_list(paths)
                    dictionary.destroy()
            result['router_input_before_step'] = host._input_router._values['/World']['probe:input']
            samples = []
            for _ in range(2):
                host.step_sync(0.01)
                with host.read(attribute_names=['probe:output']) as read:
                    tensor = read.groups[0].tensors[0]
                    samples.append({'value':float(tensor[0, 0]), 'dtype':str(tensor.dtype)})
            result.update(raw_samples=list(captured), public_samples=samples,
                          first_step_applied=captured[0]['value'] == result['expected'])
            host.reset()
            host.step_sync(0.01)
            with host.read(attribute_names=['probe:output']) as read:
                result['output_after_reset_to_authored'] = float(read.groups[0].tensors[0][0, 0])
    result['fmpy_loaded'] = any(n == 'fmpy' or n.startswith('fmpy.') for n in sys.modules)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['native', 'fmpy', 'direct'], required=True)
    parser.add_argument('--route', choices=['stage', 'write'], default='stage')
    parser.add_argument('--fmi-version', type=int, choices=[2, 3], default=3)
    parser.add_argument('--mapping-count', type=int, choices=[0, 1], default=1)
    parser.add_argument('--value', type=float, default=6.0)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--after-reset', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('USD_PYTHON', str(ROOT/'demoapp/.usd_venv/Scripts/python.exe') if os.name == 'nt'
                          else str(ROOT/'demoapp/.usd_venv/bin/python'))
    with contextlib.redirect_stdout(io.StringIO()) as diagnostics:
        result = probe(args)
    print(json.dumps(result))
