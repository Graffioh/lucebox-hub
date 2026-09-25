#!/usr/bin/env python3
"""Find each model's baseline: the newest baseline artifact from a main run.

Baseline runs on main (nightly, or dispatched with update_baseline) upload
their result as the artifact `model-e2e-baseline-<model>-<device>`. This reads
the select job's matrix on stdin and adds to each entry the `baseline_run` that
holds its baseline, or "" when there is none yet.

Only artifacts from scheduled or dispatched model-e2e runs on this
repository's main count: pull request code runs in the same workflow and could
upload an artifact with the same name.

    python3 find_baseline.py --repo OWNER/REPO < matrix.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable

WORKFLOW = ".github/workflows/model-e2e.yml"
TRUSTED_EVENTS = ("schedule", "workflow_dispatch")

Api = Callable[[str], dict]


def artifact_name(entry: dict) -> str:
    return f"model-e2e-baseline-{entry['model']}-{entry['device']}"


def github_api(token: str) -> Api:
    def get(path: str) -> dict:
        request = urllib.request.Request(
            f"https://api.github.com/{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    return get


def trusted(run: dict, repo: str) -> bool:
    return (
        run.get("event") in TRUSTED_EVENTS
        and run.get("head_branch") == "main"
        and (run.get("head_repository") or {}).get("full_name") == repo
        and (run.get("path") or "").split("@")[0] == WORKFLOW
    )


def find_run(api: Api, repo: str, name: str) -> str:
    """The newest trusted run that uploaded `name`, or ""."""
    query = urllib.parse.urlencode({"name": name, "per_page": 50})
    artifacts = api(f"repos/{repo}/actions/artifacts?{query}").get("artifacts", [])
    artifacts = [a for a in artifacts if a.get("name") == name and not a.get("expired")]
    artifacts.sort(key=lambda a: a.get("created_at") or "", reverse=True)
    for artifact in artifacts:
        run_id = (artifact.get("workflow_run") or {}).get("id")
        if run_id and trusted(api(f"repos/{repo}/actions/runs/{run_id}"), repo):
            return str(run_id)
    return ""


def add_baselines(matrix: dict, api: Api, repo: str) -> dict:
    found: dict[str, str] = {}
    for entry in matrix["include"]:
        name = artifact_name(entry)
        if name not in found:
            found[name] = find_run(api, repo, name)
        entry["baseline_run"] = found[name]
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", required=True, help="OWNER/REPO")
    args = parser.parse_args(argv)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    matrix = add_baselines(json.load(sys.stdin), github_api(token), args.repo)
    for entry in matrix["include"]:
        where = f"run {entry['baseline_run']}" if entry["baseline_run"] else "none yet"
        print(f"Baseline for {entry['model']}: {where}", file=sys.stderr)
    print(json.dumps(matrix, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
