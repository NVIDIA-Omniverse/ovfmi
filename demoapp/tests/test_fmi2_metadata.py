# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
import zipfile

import pytest
from fmpy import read_model_description

from demoapp import build_fmu


FMI2_MODELS = [(folder, name) for folder, _, name, version in build_fmu.FMUS if version == 2]


@pytest.mark.parametrize("folder", [folder for folder, _ in FMI2_MODELS])
def test_fmi2_source_model_structure(folder):
    read_model_description(
        build_fmu.FMU_SRC_DIR / folder,
        validate=True,
        validate_model_structure=True,
    )


@pytest.mark.parametrize("name", [name for _, name in FMI2_MODELS])
def test_built_fmi2_model_structure(name):
    read_model_description(
        build_fmu.DATA_DIR / f"{name}.fmu",
        validate=True,
        validate_model_structure=True,
    )


@pytest.mark.parametrize("filename,models", [(filename, models) for _, filename, models in build_fmu.SSPS])
def test_embedded_fmi2_model_structure(filename, models, tmp_path):
    ssp = build_fmu.DATA_DIR / filename
    with zipfile.ZipFile(ssp) as archive:
        for name in models:
            read_model_description(
                BytesIO(archive.read(f"resources/{name}.fmu")),
                validate=True,
                validate_model_structure=True,
            )

    container = ssp.with_suffix(".native.fmu")
    read_model_description(container, validate=True, validate_model_structure=True)
    with zipfile.ZipFile(container) as archive:
        for name in models:
            xml = tmp_path / f"{name}.xml"
            xml.write_bytes(archive.read(f"resources/{name}/modelDescription.xml"))
            read_model_description(str(xml), validate=True, validate_model_structure=True)
