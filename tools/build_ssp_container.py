# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile the ovfmi-supported SSP subset into a native container FMU."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from xml.etree import ElementTree
from pathlib import Path

from fmpy import read_model_description
from fmpy.container_fmu.cli import create_container_fmu
from fmpy.container_fmu.config import Component, Configuration, Connection
from fmpy.model_description import CoSimulation, DefaultExperiment, ModelDescription, ModelVariable, Unknown
from fmpy.ssp.ssd import add_tree_info, find_components, get_connections, read_ssd


def _normalise_fmpy_metadata(fmu_path: Path) -> None:
    """Apply the metadata normalizations required by FMPy 0.3.32 containers."""
    temporary = fmu_path.with_suffix(fmu_path.suffix + ".tmp")
    with zipfile.ZipFile(fmu_path, "r") as source, zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED
    ) as destination:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename == "resources/container.json":
                configuration = json.loads(payload)
                for connection in configuration["connections"]:
                    if connection["variableType"] == "Real":
                        connection["variableType"] = "Float64"
                payload = json.dumps(configuration, indent=4).encode("utf-8")
            elif item.filename == "modelDescription.xml":
                root = ElementTree.fromstring(payload)
                variables = root.find("ModelVariables")
                if variables is not None:
                    for index, variable in enumerate(variables, start=1):
                        variable.set("valueReference", str(index))
                payload = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
            destination.writestr(item, payload)
    temporary.replace(fmu_path)


def _container_variable(connector) -> ModelVariable:
    if connector.kind == "input":
        start = "1" if connector.name.lower() in {"enable", "enabled"} else "0"
        return ModelVariable(
            name=connector.name,
            type="Real",
            causality="input",
            variability="continuous",
            start=start,
        )
    if connector.kind == "output":
        return ModelVariable(
            name=connector.name,
            type="Real",
            causality="output",
            variability="continuous",
            initial="calculated",
        )
    raise ValueError(f"unsupported system connector kind {connector.kind!r}")


def build_ssp_container(ssp_path: Path, output_path: Path, *, step_size: float) -> None:
    """Build an FMI 2 Co-Simulation container for a flat, scalar-Real SSP."""
    ssp_path = ssp_path.resolve()
    output_path = output_path.resolve()
    ssd = read_ssd(ssp_path)
    # Reject this valid-but-unsupported SSP topology before FMPy's resolver
    # tries to concatenate absent component names into paths (TypeError).
    # Do not catch arbitrary resolver errors or reinterpret malformed XML.
    for connection in ssd.system.connections:
        if not connection.startElement and not connection.endElement:
            raise ValueError("direct system-input to system-output connections are unsupported")
    add_tree_info(ssd.system)

    with tempfile.TemporaryDirectory(prefix="ovfmi_ssp_build_") as extracted_name:
        extracted = Path(extracted_name)
        with zipfile.ZipFile(ssp_path) as archive:
            archive.extractall(extracted)

        components_by_ssp = {}
        components = []
        variables_by_component = {}
        for ssp_component in find_components(ssd.system):
            fmu_path = extracted / ssp_component.source
            model_description = read_model_description(fmu_path)
            if model_description.fmiMajorVersion != 2 or model_description.coSimulation is None:
                raise ValueError(
                    f"{ssp_component.name}: only FMI 2 Co-Simulation components are supported"
                )
            variables = {variable.name: variable for variable in model_description.modelVariables}
            for connector in ssp_component.connectors:
                variable = variables.get(connector.name)
                if variable is None or variable.type != "Real":
                    raise ValueError(
                        f"{ssp_component.name}.{connector.name}: only scalar Real connectors are supported"
                    )
            component = Component(
                name=ssp_component.name,
                filename=fmu_path,
                modelDescription=model_description,
            )
            components_by_ssp[id(ssp_component)] = component
            variables_by_component[id(ssp_component)] = variables
            components.append(component)

        system_variables = {
            connector.name: _container_variable(connector)
            for connector in ssd.system.connectors
        }
        mappings = {variable: [] for variable in system_variables.values()}
        internal_connections = []

        for start, end in get_connections(ssd.system):
            start_is_system = start.parent is ssd.system
            end_is_system = end.parent is ssd.system
            if start_is_system and not end_is_system:
                component = components_by_ssp[id(end.parent)]
                variable = variables_by_component[id(end.parent)][end.name]
                mappings[system_variables[start.name]].append((component, variable))
            elif not start_is_system and end_is_system:
                component = components_by_ssp[id(start.parent)]
                variable = variables_by_component[id(start.parent)][start.name]
                mappings[system_variables[end.name]].append((component, variable))
            elif not start_is_system and not end_is_system:
                start_component = components_by_ssp[id(start.parent)]
                end_component = components_by_ssp[id(end.parent)]
                internal_connections.append(
                    Connection(
                        startElement=start_component,
                        startConnectors=[variables_by_component[id(start.parent)][start.name]],
                        endElement=end_component,
                        endConnectors=[variables_by_component[id(end.parent)][end.name]],
                    )
                )
            else:
                raise ValueError("direct system-input to system-output connections are unsupported")

        for variable, variable_mappings in mappings.items():
            if not variable_mappings:
                raise ValueError(f"system connector {variable.name!r} has no component mapping")

        model_description = ModelDescription(
            fmiVersion="2.0",
            modelName=f"{ssd.system.name}Container",
            instantiationToken="",
            generationTool="ovfmi SSP container builder",
            defaultExperiment=DefaultExperiment(stepSize=str(step_size)),
            coSimulation=CoSimulation(
                fixedInternalStepSize=step_size,
                canHandleVariableCommunicationStepSize=True,
            ),
            modelVariables=list(system_variables.values()),
        )
        for variable in model_description.modelVariables:
            if variable.causality == "output":
                model_description.outputs.append(Unknown(variable=variable))
                model_description.initialUnknowns.append(Unknown(variable=variable))
        configuration = Configuration(
            parallelDoStep=False,
            modelDescription=model_description,
            components=components,
            connections=internal_connections,
            variableMappings=mappings,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        build_dir = output_path.parent / f".{output_path.stem}-build"
        build_dir.mkdir(exist_ok=True)
        try:
            create_container_fmu(configuration, build_dir, output_path)
            _normalise_fmpy_metadata(output_path)
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ssp", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step-size", type=float, default=1.0 / 60.0)
    args = parser.parse_args()
    build_ssp_container(args.ssp, args.output, step_size=args.step_size)
    print(f"Built native SSP container: {args.output.resolve()}")


if __name__ == "__main__":
    main()
