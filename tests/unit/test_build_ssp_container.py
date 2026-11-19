# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import zipfile
from xml.etree import ElementTree

import pytest
from fmpy import read_model_description

from tools.build_ssp_container import _normalise_fmpy_metadata, build_ssp_container
import tools.build_ssp_container as builder


def _direct_connection_ssp(tmp_path, elements=""):
    ssp = tmp_path / "direct.ssp"
    with zipfile.ZipFile(ssp, "w") as archive:
        archive.writestr("SystemStructure.ssd", f'''<?xml version="1.0"?>
<ssd:SystemStructureDescription name="Direct" version="1.0"
    xmlns:ssd="http://ssp-standard.org/SSP1/SystemStructureDescription"
    xmlns:ssc="http://ssp-standard.org/SSP1/SystemStructureCommon">
  <ssd:System name="Direct">
    <ssd:Connectors>
      <ssd:Connector name="u" kind="input"><ssc:Real/></ssd:Connector>
      <ssd:Connector name="y" kind="output"><ssc:Real/></ssd:Connector>
    </ssd:Connectors>
    <ssd:Connections>
      <ssd:Connection {elements} startConnector="u" endConnector="y"/>
    </ssd:Connections>
  </ssd:System>
</ssd:SystemStructureDescription>''')
    return ssp


@pytest.mark.parametrize("elements", ["", 'startElement="" endElement=""'])
@pytest.mark.parametrize("existing_output", [False, True])
def test_direct_system_connection_is_rejected_early(tmp_path, monkeypatch, elements, existing_output):
    ssp = _direct_connection_ssp(tmp_path, elements)
    # These are valid SSP documents, but not supported by this converter.
    builder.read_ssd(ssp, validate=True)
    output = tmp_path / "output.native.fmu"
    if existing_output:
        output.write_bytes(b"existing artifact")
    before = set(tmp_path.iterdir())

    def must_not_run(*args, **kwargs):
        pytest.fail("unsupported topology reached dependency resolution or container build")

    monkeypatch.setattr(builder, "get_connections", must_not_run)
    monkeypatch.setattr(builder, "create_container_fmu", must_not_run)
    with pytest.raises(ValueError, match="direct system-input to system-output connections are unsupported"):
        build_ssp_container(ssp, output, step_size=0.01)
    assert set(tmp_path.iterdir()) == before
    if existing_output:
        assert output.read_bytes() == b"existing artifact"


def test_unrelated_connection_typeerror_is_not_relabelled(tmp_path, monkeypatch):
    ssp = _direct_connection_ssp(tmp_path)
    original = builder.read_ssd

    def read_without_direct_edge(*args, **kwargs):
        ssd = original(*args, **kwargs)
        ssd.system.connections.clear()
        return ssd

    def unrelated_error(*args):
        raise TypeError("unrelated resolver failure")

    monkeypatch.setattr(builder, "read_ssd", read_without_direct_edge)
    monkeypatch.setattr(builder, "get_connections", unrelated_error)
    with pytest.raises(TypeError, match="unrelated resolver failure"):
        build_ssp_container(ssp, tmp_path / "output.fmu", step_size=0.01)


@pytest.mark.parametrize("output_first", [False, True])
def test_container_model_structure_validates(tmp_path, output_first):
    component = tmp_path / "Integrator.fmu"
    component_xml = Path(__file__).resolve().parents[1] / "packaging/modelDescription.xml"
    root = ElementTree.parse(component_xml).getroot()
    sources = ElementTree.SubElement(root.find("CoSimulation"), "SourceFiles")
    ElementTree.SubElement(sources, "File", name="integrator.c")
    component_description = ElementTree.tostring(root, encoding="utf-8")
    with zipfile.ZipFile(component, "w") as archive:
        archive.writestr("modelDescription.xml", component_description)
        archive.write(component_xml.with_name("integrator.c"), "sources/integrator.c")
    read_model_description(component, validate=True, validate_model_structure=True)

    connectors = [
        '<ssd:Connector name="rate" kind="input"><ssc:Real/></ssd:Connector>',
        '<ssd:Connector name="integral" kind="output"><ssc:Real/></ssd:Connector>',
    ]
    if output_first:
        connectors.reverse()
    system_connectors = "".join(connectors)
    ssd = f"""<?xml version="1.0" encoding="UTF-8"?>
<ssd:SystemStructureDescription name="IntegratorSystem" version="1.0"
    xmlns:ssd="http://ssp-standard.org/SSP1/SystemStructureDescription"
    xmlns:ssc="http://ssp-standard.org/SSP1/SystemStructureCommon">
  <ssd:System name="IntegratorSystem">
    <ssd:Connectors>{system_connectors}</ssd:Connectors>
    <ssd:Elements>
      <ssd:Component name="Integrator" source="resources/Integrator.fmu"
          type="application/x-fmu-sharedlibrary">
        <ssd:Connectors>{system_connectors}</ssd:Connectors>
      </ssd:Component>
    </ssd:Elements>
    <ssd:Connections>
      <ssd:Connection startConnector="rate" endElement="Integrator" endConnector="rate"/>
      <ssd:Connection startElement="Integrator" startConnector="integral" endConnector="integral"/>
    </ssd:Connections>
  </ssd:System>
</ssd:SystemStructureDescription>"""
    ssp = tmp_path / "integrator.ssp"
    with zipfile.ZipFile(ssp, "w") as archive:
        archive.writestr("SystemStructure.ssd", ssd)
        archive.write(component, "resources/Integrator.fmu")

    output = tmp_path / "integrator.native.fmu"
    build_ssp_container(ssp, output, step_size=0.01)
    description = read_model_description(output, validate=True, validate_model_structure=True)
    assert [unknown.variable.name for unknown in description.outputs] == ["integral"]
    assert [unknown.variable.name for unknown in description.initialUnknowns] == ["integral"]
    with zipfile.ZipFile(output) as archive:
        root = ElementTree.fromstring(archive.read("modelDescription.xml"))
        for section in ("Outputs", "InitialUnknowns"):
            unknowns = root.findall(f"ModelStructure/{section}/Unknown")
            assert [unknown.get("index") for unknown in unknowns] == ["1" if output_first else "2"]
        assert archive.read("resources/Integrator/modelDescription.xml") == component_description


def test_fmpy_metadata_is_normalised_without_replacing_runtime(tmp_path):
    fmu_path = tmp_path / "container.fmu"
    model_description = b"""<?xml version='1.0' encoding='utf-8'?>
<fmiModelDescription><ModelVariables>
<ScalarVariable name='input' valueReference='0'/>
<ScalarVariable name='output' valueReference='1'/>
</ModelVariables></fmiModelDescription>"""
    configuration = {
        "connections": [{"variableType": "Real"}],
    }
    runtime = b"stock FMPy runtime"

    with zipfile.ZipFile(fmu_path, "w") as archive:
        archive.writestr("modelDescription.xml", model_description)
        archive.writestr("resources/container.json", json.dumps(configuration))
        archive.writestr("binaries/win64/container_fmu.dll", runtime)

    _normalise_fmpy_metadata(fmu_path)

    with zipfile.ZipFile(fmu_path) as archive:
        root = ElementTree.fromstring(archive.read("modelDescription.xml"))
        variables = root.find("ModelVariables")
        assert variables is not None
        assert [variable.get("valueReference") for variable in variables] == ["1", "2"]
        configuration = json.loads(archive.read("resources/container.json"))
        assert configuration["connections"][0]["variableType"] == "Float64"
        assert archive.read("binaries/win64/container_fmu.dll") == runtime
