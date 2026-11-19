# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attach tested distributions to an existing public draft Release."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


REPOSITORY = "NVIDIA-Omniverse/ovfmi"


def api(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))


def find_release(tag):
    matches = []
    page = 1
    while True:
        releases = api(f"repos/{REPOSITORY}/releases?per_page=100&page={page}")
        matches.extend(release for release in releases if release["tag_name"] == tag)
        if len(releases) < 100:
            break
        page += 1
    if len(matches) != 1:
        raise ValueError(f"Expected one existing Release for {tag}, found {len(matches)}")
    return matches[0]


def checked_release():
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise ValueError(f"Release assets can only be uploaded in {REPOSITORY}")
    version = Path("VERSION.md").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("VERSION.md must contain a bare X.Y.Z version")
    tag = f"v{version}"
    if os.environ.get("GITHUB_REF") != f"refs/tags/{tag}":
        raise ValueError(f"Dispatch this workflow from tag {tag}, not a branch")
    sha = os.environ.get("GITHUB_SHA", "")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or head != sha:
        raise ValueError("The checkout does not match the dispatched commit")

    target = api(f"repos/{REPOSITORY}/git/ref/tags/{tag}")["object"]
    for _ in range(10):
        if target["type"] == "commit":
            break
        if target["type"] != "tag":
            raise ValueError("The release tag does not identify a commit")
        target = api(f"repos/{REPOSITORY}/git/tags/{target['sha']}")["object"]
    else:
        raise ValueError("The release tag has too many annotated tag objects")
    if target["sha"] != sha:
        raise ValueError("The remote release tag moved from the dispatched commit")

    release = find_release(tag)
    if release["tag_name"] != tag or release.get("draft") is not True:
        raise ValueError(f"Release {tag} must exist and remain a draft")
    if release.get("immutable"):
        raise ValueError(f"Release {tag} is immutable")
    if str(release["id"]) != os.environ.get("EXPECTED_RELEASE_ID", str(release["id"])):
        raise ValueError("The draft Release changed after validation")
    return version, release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    if args.upload and args.dist is None:
        raise ValueError("--upload requires --dist")

    version, release = checked_release()
    tag = f"v{version}"
    if args.dist is not None:
        assets = sorted(args.dist.iterdir())
        wheels = [path for path in assets if path.suffix == ".whl"]
        source = args.dist / f"ovfmi-{version}.tar.gz"
        if (len(assets) != 3 or len(wheels) != 2 or source not in assets
                or any(not path.is_file() or path.is_symlink() for path in assets)
                or any(not path.name.startswith(f"ovfmi-{version}-") for path in wheels)):
            raise ValueError("Expected the two tested wheels and their shared source archive")
        if args.upload:
            _, latest = checked_release()
            if latest["id"] != release["id"]:
                raise ValueError("The draft Release changed before upload")
            release = latest
        existing = {asset["name"] for asset in release["assets"]}
        duplicate = existing.intersection(path.name for path in assets)
        if duplicate:
            raise ValueError(f"Release assets already exist: {', '.join(sorted(duplicate))}")
        if args.upload:
            subprocess.run(
                ["gh", "release", "upload", tag, *(str(path) for path in assets),
                 "--repo", REPOSITORY],
                check=True,
            )
        for path in assets:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"{digest}  {path.name}")

    if output_path := os.environ.get("GITHUB_OUTPUT"):
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"version={version}\nrelease_id={release['id']}\n")
    state = "Assets uploaded; Release remains a draft" if args.upload else "Release validation passed; no assets uploaded"
    print(f"{tag}: {state}")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(f"{tag}: {state}.\n")


if __name__ == "__main__":
    main()
