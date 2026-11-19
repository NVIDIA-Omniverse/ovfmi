# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real FMUs distinguish initialization inputs from communication-step inputs."""

from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEP = 0.125


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "input-initialized"])
def integrator(request, tmp_path_factory):
    pytest.importorskip("fmpy")
    compiler = "cl" if sys.platform == "win32" else "cc"
    if sys.platform not in ("linux", "win32") or platform.machine().lower() not in ("x86_64", "amd64"):
        pytest.skip("the FMU fixture requires x86-64 Linux or Windows")
    if shutil.which(compiler) is None:
        pytest.skip(f"the FMU fixture requires {compiler}")
    path = tmp_path_factory.mktemp("ssp-integrator") / "Integrator.fmu"
    command = [sys.executable, str(ROOT / "tests/packaging/build_fmu.py"), "--output", str(path)]
    if request.param:
        command.append("--initialize-from-input")
    subprocess.run(command, check=True, capture_output=True, text=True)
    from fmpy.validation import validate_fmu
    assert validate_fmu(str(path)) == []
    return path, request.param


def _build_ssp(path, fmu, count=1, reverse=False, fan_out=False):
    connectors = '''<ssd:Connectors>
      <ssd:Connector name="rate" kind="input"><ssc:Real/></ssd:Connector>
      <ssd:Connector name="integral" kind="output"><ssc:Real/></ssd:Connector>
    </ssd:Connectors>'''
    components = "".join(f'''
      <ssd:Component name="Integrator{i}" type="application/x-fmu-sharedlibrary"
                     source="resources/Integrator.fmu">{connectors}</ssd:Component>'''
                         for i in (reversed(range(count)) if reverse else range(count)))
    connections = ['<ssd:Connection startConnector="rate" endElement="Integrator0" endConnector="rate"/>']
    for i in range(1, count):
        start = 'startConnector="rate"' if fan_out else f'startElement="Integrator{i - 1}" startConnector="integral"'
        connections.append(f'<ssd:Connection {start} endElement="Integrator{i}" endConnector="rate"/>')
    connections.append(f'''<ssd:Connection startElement="Integrator{count - 1}"
                          startConnector="integral" endConnector="integral"/>''')
    ssd = f'''<?xml version="1.0" encoding="UTF-8"?>
<ssd:SystemStructureDescription xmlns:ssd="http://ssp-standard.org/SSP1/SystemStructureDescription"
    xmlns:ssc="http://ssp-standard.org/SSP1/SystemStructureCommon" version="1.0" name="Initialization">
  <ssd:System name="Initialization">
    {connectors}
    <ssd:Elements>{components}</ssd:Elements>
    <ssd:Connections>{''.join(connections)}</ssd:Connections>
  </ssd:System>
</ssd:SystemStructureDescription>'''
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SystemStructure.ssd", ssd)
        archive.write(fmu, "resources/Integrator.fmu")
    return path


@pytest.fixture
def runtime(tmp_path, integrator):
    from ovfmi._ssp_runtime import SspRuntimeInstance

    fmu, _ = integrator
    instances = []

    def create(count=1, reverse=False, fan_out=False):
        path = _build_ssp(tmp_path / f"system-{count}-{reverse}.ssp", fmu, count, reverse, fan_out)
        instance = SspRuntimeInstance(None, str(path))
        instances.append(instance)
        return instance

    yield create
    for instance in instances:
        instance.destroy()


@pytest.mark.parametrize("phase", ["initial", "reset"])
@pytest.mark.parametrize("live_input", [False, True], ids=["start-value", "live-override"])
def test_first_step_inputs_match_direct_fmu(integrator, runtime, phase, live_input):
    from fmpy import simulate_fmu

    fmu, initialize_from_input = integrator
    instance = runtime()
    starts = {"rate": 2.0}
    instance.set_start_values(starts)
    if phase == "reset":
        instance.step("", {"rate": 11.0}, ["integral"], STEP)
        instance.reset()
    rate = (3.0 if phase == "initial" else 5.0) if live_input else starts["rate"]
    inputs = {"rate": rate} if live_input else {}
    expected = (rate if initialize_from_input else 0.0) + rate * STEP
    direct = simulate_fmu(str(fmu), start_values={"rate": rate}, output=["integral"],
                          stop_time=STEP, output_interval=STEP)
    assert direct["integral"][-1] == pytest.approx(expected)
    first = instance.step("", inputs, ["integral"], STEP)
    assert first["integral"] == pytest.approx(direct["integral"][-1])
    running = instance.step("", {"rate": 7.0}, ["integral"], 2 * STEP)
    assert running["integral"] == pytest.approx(expected + 7.0 * STEP)
    assert starts == {"rate": 2.0}


@pytest.mark.parametrize("phase", ["initial", "reset"])
@pytest.mark.parametrize("zero_source", ["absent", "live", "authored"])
def test_steps_distinguish_absent_input_from_zero(integrator, runtime, phase, zero_source):
    _, initialize_from_input = integrator
    instance = runtime()
    if zero_source == "authored":
        instance.set_start_values({"rate": 0.0})
    if phase == "reset":
        instance.step("", {"rate": 11.0}, ["integral"], STEP)
        instance.reset()
    inputs = {"rate": 0.0} if zero_source == "live" else {}
    expected = 1.0 if initialize_from_input and zero_source == "absent" else 0.0
    for step in range(1, 4):
        if zero_source == "absent":
            expected += STEP
        result = instance.step("", inputs, ["integral"], step * STEP)
        assert result["integral"] == pytest.approx(expected)


@pytest.mark.parametrize("phase", ["initial", "reset"])
@pytest.mark.parametrize("reverse,plain,initialized", [
    (False, (0.046875, 0.140625), (3.421875, 3.890625)),
    (True, (0.0, 0.046875), (3.375, 3.796875)),
], ids=["upstream-first", "downstream-first"])
def test_chain_initializes_before_any_component_steps(integrator, runtime, phase, reverse, plain, initialized):
    _, initialize_from_input = integrator
    instance = runtime(count=2, reverse=reverse)
    if phase == "reset":
        instance.step("", {"rate": 11.0}, ["integral"], STEP)
        instance.reset()
    expected = initialized if initialize_from_input else plain
    for step, downstream in enumerate(expected, start=1):
        result = instance.step("", {"rate": 3.0}, ["integral"], step * STEP)
        assert result["integral"] == pytest.approx(downstream), step


@pytest.mark.parametrize("phase", ["initial", "reset"])
def test_cyclic_initialization_keeps_unready_input_default(integrator, runtime, phase):
    _, initialize_from_input = integrator
    instance = runtime(count=2)
    feedback = next(connector for connector in instance._components[1].connectors
                    if connector.name == "integral")
    instance._connections = [
        (feedback if start is instance._system_inputs[0] else start, end)
        for start, end in instance._connections
    ]
    if phase == "reset":
        instance.step("", {}, ["integral"], STEP)
        instance.reset()
    expected = (1.140625, 1.299072265625) if initialize_from_input else (0.0, 0.0)
    for step, downstream in enumerate(expected, start=1):
        result = instance.step("", {}, ["integral"], step * STEP)
        assert result["integral"] == pytest.approx(downstream), step


@pytest.mark.parametrize("phase", ["initial", "reset"])
def test_disconnected_cycle_preserves_chain_initialization(integrator, runtime, phase):
    _, initialize_from_input = integrator
    reference = runtime(count=2, reverse=True)
    instance = runtime(count=4, reverse=True)
    components = {component.name: component for component in instance._components}
    upstream, downstream, cycle_a, cycle_b = [
        {connector.name: connector for connector in components[f"Integrator{i}"].connectors}
        for i in range(4)
    ]
    instance._connections = [
        (instance._system_inputs[0], upstream["rate"]),
        (upstream["integral"], downstream["rate"]),
        (downstream["integral"], instance._system_outputs[0]),
        (cycle_a["integral"], cycle_b["rate"]),
        (cycle_b["integral"], cycle_a["rate"]),
    ]
    if phase == "reset":
        for model in (reference, instance):
            model.step("", {"rate": 11.0}, ["integral"], STEP)
            model.reset()
    expected = (3.375, 3.796875) if initialize_from_input else (0.0, 0.046875)
    for step, expected_output in enumerate(expected, start=1):
        control = reference.step("", {"rate": 3.0}, ["integral"], step * STEP)
        result = instance.step("", {"rate": 3.0}, ["integral"], step * STEP)
        assert control["integral"] == pytest.approx(expected_output), step
        assert result["integral"] == pytest.approx(control["integral"]), step


@pytest.mark.parametrize("phase", ["initial", "reset"])
def test_cycle_consumer_waits_for_initial_outputs(integrator, runtime, phase):
    _, initialize_from_input = integrator
    instance = runtime(count=3, reverse=True)
    consumer, cycle_a, cycle_b = [
        {connector.name: connector for connector in component.connectors}
        for component in instance._components
    ]
    instance._connections = [
        (cycle_a["integral"], cycle_b["rate"]),
        (cycle_b["integral"], cycle_a["rate"]),
        (cycle_a["integral"], consumer["rate"]),
        (consumer["integral"], instance._system_outputs[0]),
    ]
    cycle_fmu = instance._components[1].fmu
    # Give the cycle a distinct input before initialization.
    cycle_fmu.setReal([0], [3.0])
    if phase == "reset":
        instance.step("", {}, ["integral"], STEP)
        instance.reset()
        cycle_fmu.setReal([0], [3.0])
    expected = (3.375, 3.796875) if initialize_from_input else (0.0, 0.0)
    for step, expected_output in enumerate(expected, start=1):
        result = instance.step("", {}, ["integral"], step * STEP)
        assert result["integral"] == pytest.approx(expected_output), step


@pytest.mark.parametrize("phase", ["initial", "reset"])
def test_public_host_uses_input_written_after_attachment(integrator, tmp_path, require_usd_python, phase):
    from ovfmi import AttributeWrite, FmiHost, FmiHostConfig

    fmu, initialize_from_input = integrator
    archive = _build_ssp(tmp_path / "system.ssp", fmu)
    source = tmp_path / "system.usda"
    mappings = "".join(f'''
            def FmuMapping "{direction}"
            {{
                token fmi:direction = "{direction}"
                token fmi:fmuAttribute = "{name}"
                token fmi:usdAttribute = "probe:{name}"
                int2 fmi:usdMapping = (0, 1)
            }}''' for direction, name in (("input", "rate"), ("output", "integral")))
    source.write_text(f'''#usda 1.0
def Xform "World"
{{
    custom double probe:rate = 2
    custom double probe:integral = 0
    def SspInstance "Model"
    {{
        asset fmi:ssp = @{archive.as_posix()}@
        def FmuConnection "Signals"
        {{
            rel fmi:targets = </World>
            {mappings}
        }}
    }}
}}
''', encoding="utf-8")
    with FmiHost(FmiHostConfig(backend="fmpy")) as host:
        report = host.attach_ovstage(object(), source_asset=source)
        assert len(report.instances) == 1
        if phase == "reset":
            host.step_sync(STEP)
            host.reset()
        rate = 3.0 if phase == "initial" else 5.0
        host.wait_op(host.write([AttributeWrite(("/World",), "probe:rate", [[rate]])]))
        for expected_rate in (rate, 2.0):
            host.step_sync(STEP)
            expected = (expected_rate if initialize_from_input else 0.0) + expected_rate * STEP
            with host.read(attribute_names=["probe:integral"]) as result:
                assert result.groups[0].tensors[0][0, 0] == pytest.approx(expected)
            host.reset()


@pytest.mark.parametrize("phase", ["initial", "reset"])
def test_missing_input_accepts_later_writes_and_retains_last_value(integrator, runtime, phase):
    _, initialize_from_input = integrator
    instance = runtime()
    if phase == "reset":
        instance.step("", {"rate": 11.0}, ["integral"], STEP)
        instance.reset()
    expected = 1.0 if initialize_from_input else 0.0
    for step, (inputs, rate) in enumerate([
        ({}, 1.0), ({"rate": 0.0}, 0.0), ({}, 0.0),
        ({"rate": 4.0}, 4.0), ({}, 4.0),
    ], start=1):
        expected += rate * STEP
        result = instance.step("", inputs, ["integral"], step * STEP)
        assert result["integral"] == pytest.approx(expected)


@pytest.mark.parametrize("phase", ["initial", "reset"])
def test_missing_fan_out_source_preserves_each_component_value(integrator, runtime, phase):
    _, initialize_from_input = integrator
    instance = runtime(count=2, fan_out=True)
    if phase == "reset":
        instance.step("", {"rate": 11.0}, [], STEP)
        instance.reset()
    instance._components[1].fmu.setReal([0], [3.0])
    for step in range(1, 4):
        instance.step("", {}, [], step * STEP)
        for component, rate in zip(instance._components, (1.0, 3.0)):
            expected = (rate if initialize_from_input else 0.0) + rate * step * STEP
            assert component.fmu.getReal([1])[0] == pytest.approx(expected)
