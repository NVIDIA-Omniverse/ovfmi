# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

import ovfmi.api as api_module
import ovfmi.population as population
from ovfmi import FmiHost


def test_parse_source_does_not_forward_renderer_usd_plugin_paths(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "stage.usda"
    source.touch()
    captured = {}

    monkeypatch.setenv("PXR_PLUGINPATH_NAME", "renderer-plugins")
    monkeypatch.setenv("OV_PXR_PLUGINPATH_2511", "renamed-renderer-plugins")
    monkeypatch.setenv("UNRELATED_SETTING", "preserved")

    def fake_run(*args, **kwargs):
        captured.update(kwargs["env"])

        class Result:
            returncode = 0
            stdout = json.dumps({"instances": {}})
            stderr = ""

        return Result()

    monkeypatch.setattr(population.subprocess, "run", fake_run)

    assert population.parse_source(str(source)) == {"instances": {}}
    assert "PXR_PLUGINPATH_NAME" not in captured
    assert "OV_PXR_PLUGINPATH_2511" not in captured
    assert captured["UNRELATED_SETTING"] == "preserved"


def test_parse_source_forwards_schema_warnings(monkeypatch, tmp_path, capsys):
    warnings = (
        "[parse_fmi_schema] WARNING: incomplete mapping at /World/Mapping\n"
        "[parse_fmi_schema] WARNING: no numeric output baseline for /World.result (quatf)\n"
    )
    result = population.subprocess.CompletedProcess(
        args=[], returncode=0, stdout='{"instances": {}}',
        stderr="unrelated subprocess output\n" + warnings,
    )
    monkeypatch.setattr(population.subprocess, "run", lambda *args, **kwargs: result)

    assert population.parse_source(str(tmp_path / "stage.usda")) == {"instances": {}}
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err == warnings


def test_parse_source_keeps_failure_stderr_in_exception(monkeypatch, tmp_path, capsys):
    result = population.subprocess.CompletedProcess(
        args=[], returncode=2, stdout="", stderr="schema failure",
    )
    monkeypatch.setattr(population.subprocess, "run", lambda *args, **kwargs: result)

    with pytest.raises(RuntimeError, match=r"FMI schema parsing failed \(exit 2\):\nschema failure"):
        population.parse_source(str(tmp_path / "stage.usda"))
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


def _instance(kind, *, enabled=True, resolved=""):
    return {
        "enabled": enabled,
        "fmu": resolved if kind == "fmu" else None,
        "ssp": resolved if kind == "ssp" else None,
        "authored_asset": f"./missing.{kind}",
        "path": "/World/Model",
        "connections": [],
    }


@pytest.mark.parametrize("kind", ["fmu", "ssp"])
def test_unresolved_asset_rejected_before_backend_creation(monkeypatch, kind):
    raw = {
        "/World/Valid": dict(_instance("fmu", resolved="valid.fmu"), path="/World/Valid"),
        "/World/Model": _instance(kind),
    }
    monkeypatch.setattr(api_module, "parse_source", lambda _: {"instances": raw})

    def unexpected_backend(*args):
        pytest.fail("asset validation must finish before backend creation")

    with FmiHost(_backend_factory=unexpected_backend) as host:
        with pytest.raises(ValueError) as error:
            host.attach_ovstage(object(), source_asset="stage.usda")
        for value in ("/World/Model", f"fmi:{kind}", f"./missing.{kind}"):
            assert value in str(error.value)
        assert host.instances == ()
        with pytest.raises(RuntimeError, match="not attached"):
            host.step_sync(0.125)


@pytest.mark.parametrize("kind,selection", [
    ("fmu", "disabled"), ("ssp", "disabled"),
    ("fmu", "outside_root"), ("ssp", "outside_root"),
    ("ssp", "ssp_off"),
])
def test_unresolved_asset_validation_respects_selection(kind, selection):
    raw = {"/World/Model": _instance(kind, enabled=selection != "disabled")}
    instances = population.deserialise_instances(
        raw,
        root_prim="/World/Other" if selection == "outside_root" else "/",
        enable_ssp=selection != "ssp_off",
    )
    if selection == "disabled":
        instance = instances["/World/Model"]
        assert not instance.enabled
        assert instance.fmu == f"./missing.{kind}"
        assert instance._is_ssp == (kind == "ssp")
    else:
        assert instances == {}


@pytest.mark.parametrize("kind", ["fmu", "ssp"])
def test_resolved_asset_is_used_without_reinterpreting_authored_path(kind):
    raw = {"/World/Model": _instance(kind, resolved="/resolved/model")}
    instance = population.deserialise_instances(raw, root_prim="/", enable_ssp=True)["/World/Model"]
    assert instance.fmu == "/resolved/model"
    assert instance._is_ssp == (kind == "ssp")
