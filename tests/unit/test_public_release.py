# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "public_release",
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "public-release.py",
)
publisher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publisher)

_REPOSITORY = "NVIDIA-Omniverse/ovfmi"
_API = f"repos/{_REPOSITORY}"
_TAG = "v0.3.0"
_SHA = "a" * 40
_OTHER_SHA = "b" * 40
_TAG_ENDPOINT = f"{_API}/git/ref/tags/{_TAG}"
_RELEASES_ENDPOINT = f"{_API}/releases?per_page=100&page="
_DISTRIBUTIONS = (
    "ovfmi-0.3.0-py3-none-manylinux_2_28_x86_64.whl",
    "ovfmi-0.3.0-py3-none-win_amd64.whl",
    "ovfmi-0.3.0.tar.gz",
)


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "VERSION.md").write_text("0.3.0\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_REPOSITORY", _REPOSITORY)
    monkeypatch.setenv("GITHUB_REF", f"refs/tags/{_TAG}")
    monkeypatch.setenv("GITHUB_SHA", _SHA)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.delenv("EXPECTED_RELEASE_ID", raising=False)
    state = SimpleNamespace(
        checkout_sha=_SHA,
        tag={"object": {"type": "commit", "sha": _SHA}},
        tag_objects={},
        release={"id": 123, "tag_name": _TAG, "draft": True, "assets": []},
        release_pages=None,
        api_calls=[],
        uploads=[],
        tag_checks=0,
        before_tag=None,
        output=tmp_path / "outputs",
    )

    def api(endpoint):
        state.api_calls.append(endpoint)
        if endpoint == _TAG_ENDPOINT:
            state.tag_checks += 1
            if state.before_tag is not None:
                state.before_tag(state)
            return copy.deepcopy(state.tag)
        if endpoint.startswith(_RELEASES_ENDPOINT):
            page = int(endpoint[len(_RELEASES_ENDPOINT):])
            if state.release_pages is not None:
                return copy.deepcopy(state.release_pages.get(page, []))
            assert page == 1
            return [copy.deepcopy(state.release)]
        if endpoint.startswith(f"{_API}/git/tags/"):
            return copy.deepcopy(state.tag_objects[endpoint.rsplit("/", 1)[1]])
        pytest.fail(f"Unexpected GitHub API access: {endpoint}")

    def checkout(command, *, text):
        assert command == ["git", "rev-parse", "HEAD"]
        assert text is True
        return f"{state.checkout_sha}\n"

    def upload(command, *, check):
        assert command[:3] == ["gh", "release", "upload"]
        assert check is True
        state.uploads.append(command)

    monkeypatch.setattr(publisher, "api", api)
    monkeypatch.setattr(publisher.subprocess, "check_output", checkout)
    monkeypatch.setattr(publisher.subprocess, "run", upload)
    return state


@pytest.fixture
def distributions(tmp_path):
    directory = tmp_path / "dist"
    directory.mkdir()
    for name in _DISTRIBUTIONS:
        (directory / name).write_bytes(b"tested distribution")
    return directory


def test_accepts_the_existing_tag_and_draft(release):
    version, draft = publisher.checked_release()

    assert version == "0.3.0"
    assert draft["id"] == 123
    assert release.uploads == []


@pytest.mark.parametrize("name, value", [
    ("GITHUB_REPOSITORY", "NVIDIA-Omniverse/ovfmi-internal"),
    ("GITHUB_REPOSITORY", "a-contributor/ovfmi"),
    ("GITHUB_REF", "refs/heads/main"),
    ("GITHUB_REF", "refs/heads/v0.3.0"),
    ("GITHUB_REF", "refs/tags/v0.2.0"),
    ("GITHUB_REF", "refs/tags/0.3.0"),
])
def test_refuses_other_repositories_and_refs(release, monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()
    assert release.uploads == []


@pytest.mark.parametrize("version", ["0.3.0rc1", "0.3.0+local", "v0.3.0", "0.3", ""])
def test_refuses_non_release_versions(release, version):
    Path("VERSION.md").write_text(version, encoding="utf-8")

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()


def test_refuses_a_checkout_from_another_commit(release):
    release.checkout_sha = _OTHER_SHA

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()


def test_refuses_a_moved_lightweight_tag(release):
    release.tag["object"]["sha"] = _OTHER_SHA

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()


def test_resolves_an_annotated_tag_to_its_commit(release):
    tag_sha = "c" * 40
    release.tag = {"object": {"type": "tag", "sha": tag_sha}}
    release.tag_objects[tag_sha] = {"object": {"type": "commit", "sha": _SHA}}

    version, _ = publisher.checked_release()

    assert version == "0.3.0"
    assert f"{_API}/git/tags/{tag_sha}" in release.api_calls


def test_refuses_an_annotated_tag_for_another_commit(release):
    tag_sha = "c" * 40
    release.tag = {"object": {"type": "tag", "sha": tag_sha}}
    release.tag_objects[tag_sha] = {"object": {"type": "commit", "sha": _OTHER_SHA}}

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()


def test_refuses_a_tag_object_cycle(release):
    tag_sha = "c" * 40
    release.tag = {"object": {"type": "tag", "sha": tag_sha}}
    release.tag_objects[tag_sha] = {"object": {"type": "tag", "sha": tag_sha}}

    with pytest.raises(ValueError):
        publisher.main([])

    assert len(release.api_calls) <= 12
    assert not release.output.exists()


@pytest.mark.parametrize("changes", [
    {"draft": False},
    {"draft": "true"},
    {"draft": 1},
    {"immutable": True},
    {"tag_name": "v0.2.0"},
])
def test_refuses_a_published_immutable_or_mismatched_release(release, changes):
    release.release.update(changes)

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()
    assert release.uploads == []


def test_refuses_a_release_without_draft_state(release):
    del release.release["draft"]

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()


def test_propagates_an_unavailable_release_instead_of_creating_one(release, monkeypatch):
    original_api = publisher.api

    def missing_release(endpoint):
        if endpoint.startswith(_RELEASES_ENDPOINT):
            raise subprocess.CalledProcessError(1, ["gh", "api", endpoint])
        return original_api(endpoint)

    monkeypatch.setattr(publisher, "api", missing_release)

    with pytest.raises(subprocess.CalledProcessError):
        publisher.main([])

    assert not release.output.exists()
    assert release.uploads == []


def test_refuses_a_missing_release(release):
    release.release_pages = {1: []}

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()
    assert release.uploads == []


def test_finds_a_draft_beyond_the_first_page(release):
    release.release_pages = {
        1: [{"id": index, "tag_name": f"v0.2.{index}"} for index in range(100)],
        2: [release.release],
    }

    _, draft = publisher.checked_release()

    assert draft["id"] == 123
    assert f"{_RELEASES_ENDPOINT}2" in release.api_calls


@pytest.mark.parametrize("separate_pages", [False, True])
def test_refuses_duplicate_releases_for_the_tag(release, separate_pages):
    duplicate = dict(release.release, id=456)
    if separate_pages:
        release.release_pages = {
            1: [release.release] + [
                {"id": index, "tag_name": f"v0.2.{index}"} for index in range(99)
            ],
            2: [duplicate],
        }
    else:
        release.release_pages = {1: [release.release, duplicate]}

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()
    assert release.uploads == []


def test_preflight_emits_the_validated_version(release):
    publisher.main([])

    assert release.output.read_text(encoding="utf-8").splitlines() == [
        "version=0.3.0", "release_id=123",
    ]
    assert release.uploads == []


def test_accepts_the_release_validated_by_the_previous_job(release, monkeypatch):
    monkeypatch.setenv("EXPECTED_RELEASE_ID", "123")

    _, draft = publisher.checked_release()

    assert draft["id"] == 123


def test_refuses_a_recreated_release_after_the_build(release, monkeypatch):
    monkeypatch.setenv("EXPECTED_RELEASE_ID", "456")

    with pytest.raises(ValueError):
        publisher.main([])

    assert not release.output.exists()
    assert release.uploads == []


def test_dry_run_with_distributions_never_uploads(release, distributions):
    publisher.main(["--dist", str(distributions)])

    assert release.uploads == []


def test_upload_requires_distributions(release):
    with pytest.raises(ValueError):
        publisher.main(["--upload"])

    assert release.uploads == []


@pytest.mark.parametrize("invalid", [
    "missing-wheel", "extra-file", "wrong-wheel-version", "wrong-source-version", "directory",
])
def test_refuses_incomplete_or_mismatched_distributions(release, distributions, invalid):
    if invalid == "missing-wheel":
        (distributions / _DISTRIBUTIONS[0]).unlink()
    elif invalid == "extra-file":
        (distributions / "unexpected.txt").touch()
    elif invalid == "wrong-wheel-version":
        wheel = distributions / _DISTRIBUTIONS[0]
        wheel.rename(distributions / wheel.name.replace("0.3.0", "0.2.0"))
    elif invalid == "wrong-source-version":
        source = distributions / _DISTRIBUTIONS[2]
        source.rename(distributions / source.name.replace("0.3.0", "0.2.0"))
    else:
        wheel = distributions / _DISTRIBUTIONS[0]
        wheel.unlink()
        wheel.mkdir()

    with pytest.raises(ValueError):
        publisher.main(["--dist", str(distributions), "--upload"])

    assert release.uploads == []


@pytest.mark.parametrize("name", _DISTRIBUTIONS)
def test_refuses_to_replace_an_existing_release_asset(release, distributions, name):
    release.release["assets"] = [{"name": name}]

    with pytest.raises(ValueError):
        publisher.main(["--dist", str(distributions), "--upload"])

    assert release.uploads == []


def test_uploads_only_the_tested_files_to_the_existing_draft(release, distributions):
    publisher.main(["--dist", str(distributions), "--upload"])

    assert len(release.uploads) == 1
    command = release.uploads[0]
    assert command[:4] == ["gh", "release", "upload", _TAG]
    assert command[-2:] == ["--repo", _REPOSITORY]
    assert {Path(path).name for path in command[4:-2]} == set(_DISTRIBUTIONS)
    assert all(Path(path).parent == distributions for path in command[4:-2])
    assert "--clobber" not in command
    assert release.release["draft"] is True


@pytest.mark.parametrize("change", ["tag-moved", "published", "immutable", "recreated", "asset-added"])
def test_rechecks_tag_and_release_before_upload(release, distributions, change):
    def change_after_preflight(state):
        if state.tag_checks < 2:
            return
        if change == "tag-moved":
            state.tag["object"]["sha"] = _OTHER_SHA
        elif change == "published":
            state.release["draft"] = False
        elif change == "immutable":
            state.release["immutable"] = True
        elif change == "recreated":
            state.release["id"] = 456
        else:
            state.release["assets"] = [{"name": _DISTRIBUTIONS[0]}]

    release.before_tag = change_after_preflight

    with pytest.raises(ValueError):
        publisher.main(["--dist", str(distributions), "--upload"])

    assert release.uploads == []
