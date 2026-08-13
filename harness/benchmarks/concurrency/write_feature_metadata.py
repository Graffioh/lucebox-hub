#!/usr/bin/env python3
"""Write reproducible server and feature configuration metadata for one case."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_digest(
    path: pathlib.Path | None,
    claimed: str | None,
    label: str,
    cache: dict[pathlib.Path, str],
) -> str | None:
    if path is None:
        if claimed is not None:
            raise ValueError(f"{label} SHA-256 supplied without a model file")
        return None
    if not isinstance(claimed, str) or not claimed:
        raise ValueError(f"{label} SHA-256 is required")
    resolved = path.resolve()
    if resolved not in cache:
        cache[resolved] = digest(resolved)
    actual = cache[resolved]
    if claimed != actual:
        raise ValueError(f"{label} SHA-256 does not match {resolved}")
    return actual


def resolved_libraries(binary: pathlib.Path) -> dict[str, str]:
    result = subprocess.run(
        ["ldd", str(binary)], text=True, capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"ldd failed for {binary}: {detail}")
    unresolved = [
        line.strip() for line in result.stdout.splitlines()
        if "=> not found" in line
    ]
    if unresolved:
        raise RuntimeError(f"ldd found unresolved libraries for {binary}: {unresolved}")
    libraries: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.replace("=>", " ").split()
        paths = [pathlib.Path(value) for value in fields if value.startswith("/")]
        for path in paths:
            if path.is_file():
                libraries[str(path.resolve())] = digest(path)
    return libraries


def repository_head(repo: pathlib.Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"git rev-parse failed for {repo}: {detail}")
    head = result.stdout.strip()
    if not head:
        raise RuntimeError(f"git rev-parse returned an empty revision for {repo}")
    return head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--prompt-file", type=pathlib.Path, required=True)
    parser.add_argument("--command-file", type=pathlib.Path, required=True)
    parser.add_argument("--repo", type=pathlib.Path, required=True)
    parser.add_argument("--max-concurrent-prefills", type=int, required=True)
    parser.add_argument("--target-device", default=None)
    parser.add_argument("--draft-device", default=None)
    parser.add_argument("--draft-model", type=pathlib.Path)
    parser.add_argument("--draft-model-sha256")
    parser.add_argument("--ddtree", action="store_true")
    parser.add_argument("--fast-rollback", action="store_true")
    parser.add_argument("--ddtree-budget", type=int)
    parser.add_argument("--prefill-compression", default="off")
    parser.add_argument("--prefill-threshold", type=int)
    parser.add_argument("--prefill-keep-ratio", type=float)
    parser.add_argument("--prefill-drafter", type=pathlib.Path)
    parser.add_argument("--prefill-drafter-sha256")
    parser.add_argument("--draft-residency", default=None)
    parser.add_argument("--kvflash", default="off")
    parser.add_argument("--kvflash-max-pool-tokens", type=int)
    parser.add_argument("--kvflash-scorer-drafter", type=pathlib.Path)
    parser.add_argument("--kvflash-scorer-drafter-sha256")
    parser.add_argument("--launch-env", action="append", default=[])
    args = parser.parse_args()

    launch_env: dict[str, str] = {}
    for item in args.launch_env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"bad --launch-env {item!r}; expected KEY=VALUE")
        launch_env[key] = value

    digest_cache: dict[pathlib.Path, str] = {}
    model_sha256 = validated_digest(
        args.model, args.model_sha256, "target model", digest_cache,
    )
    draft_model_sha256 = validated_digest(
        args.draft_model, args.draft_model_sha256, "draft model", digest_cache,
    )
    prefill_drafter_sha256 = validated_digest(
        args.prefill_drafter, args.prefill_drafter_sha256,
        "prefill drafter", digest_cache,
    )
    kvflash_scorer_drafter_sha256 = validated_digest(
        args.kvflash_scorer_drafter,
        args.kvflash_scorer_drafter_sha256,
        "KVFlash scorer drafter",
        digest_cache,
    )

    libraries = resolved_libraries(args.binary)
    git_head = repository_head(args.repo)
    literal_flags: list[str] = []
    if args.target_device:
        literal_flags += ["--target-device", args.target_device]
    if args.draft_device:
        literal_flags += ["--draft-device", args.draft_device]
    if args.ddtree:
        literal_flags += ["--ddtree"]
    if args.ddtree_budget is not None:
        literal_flags += ["--ddtree-budget", str(args.ddtree_budget)]
    if args.fast_rollback:
        literal_flags += ["--fast-rollback"]
    if args.draft_residency:
        literal_flags += ["--draft-residency", args.draft_residency]
    if args.prefill_compression != "off":
        literal_flags += ["--prefill-compression", args.prefill_compression]
    if args.prefill_drafter:
        literal_flags += ["--prefill-drafter", str(args.prefill_drafter.resolve())]
    if args.kvflash != "off":
        literal_flags += ["--kvflash", args.kvflash]

    obj = {
        "schema_version": 3,
        "variant": args.variant,
        "workload": args.workload,
        "clients": args.clients,
        "repeat": args.repeat,
        "max_concurrent_prefills": args.max_concurrent_prefills,
        "server_binary": str(args.binary.resolve()),
        "server_binary_sha256": digest(args.binary),
        "model": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "prompt_file_sha256": digest(args.prompt_file),
        "server_command": args.command_file.read_text(encoding="utf-8").strip(),
        "launch_environment": launch_env,
        "resolved_shared_library_sha256": libraries,
        "git_head": git_head,
        "literal_screenshot_flags": literal_flags,
        # Populated from fail-closed startup markers after the server is healthy.
        "runtime_observed": None,
        "feature_config": {
            "target_device": args.target_device,
            "draft_device": args.draft_device,
            "draft_model": str(args.draft_model.resolve()) if args.draft_model else None,
            "draft_model_sha256": draft_model_sha256,
            "ddtree": args.ddtree,
            "fast_rollback": args.fast_rollback,
            "ddtree_budget": args.ddtree_budget,
            "prefill_compression": args.prefill_compression,
            "prefill_threshold": args.prefill_threshold,
            "prefill_keep_ratio": args.prefill_keep_ratio,
            "prefill_drafter": (
                str(args.prefill_drafter.resolve()) if args.prefill_drafter else None
            ),
            "prefill_drafter_sha256": prefill_drafter_sha256,
            "draft_residency": args.draft_residency,
            "kvflash": args.kvflash,
            "kvflash_max_pool_tokens": args.kvflash_max_pool_tokens,
            "kvflash_scorer_drafter": (
                str(args.kvflash_scorer_drafter.resolve())
                if args.kvflash_scorer_drafter else None
            ),
            "kvflash_scorer_drafter_sha256": kvflash_scorer_drafter_sha256,
        },
    }
    args.out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
