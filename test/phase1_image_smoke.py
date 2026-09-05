#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "aiohttp>=3.9",
#   "asyncssh>=2.21.1",
# ]
# ///
"""Exercise Phase 1 image import and Dockerfile build against a live API.

Run from any directory with:

    uv run ~/Documents/thunder-sandbox/test/phase1_image_smoke.py

Authentication and endpoint selection use the normal local SDK contract:
``~/.thunder/cli_config.json``, ``TNR_API_TOKEN``, and ``TNR_API_URL``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import re
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

thunder_sandbox = importlib.import_module("thunder_sandbox")
Image = thunder_sandbox.Image
ResolvedImage = thunder_sandbox.ResolvedImage
Client = importlib.import_module("thunder_sandbox.asynchronous.client").Client
_create_canonical_build_context = importlib.import_module(
    "thunder_sandbox.image"
)._create_canonical_build_context


DEFAULT_REGISTRY_IMAGE = "docker.io/library/ubuntu:24.04"
DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60
FIXTURE_DIRECTORY = Path(__file__).with_name("phase1_image_context")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SENSITIVE_KEYS = ("authorization", "credential", "password", "secret", "token")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def emit(event: str, **fields: Any) -> None:
    payload = {"time": timestamp(), "event": event, **fields}
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


def redact(value: Any, key: str = "") -> Any:
    lowered = key.lower()
    if any(sensitive in lowered for sensitive in SENSITIVE_KEYS):
        return "<redacted>" if value not in (None, "") else value
    if isinstance(value, dict):
        return {
            str(item_key): redact(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class VerboseClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self.request_number = 0

    async def _request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        query: dict[str, object] | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> dict[str, Any]:
        self.request_number += 1
        request_number = self.request_number
        started = time.monotonic()
        emit(
            "api.request",
            number=request_number,
            method=method,
            path=path,
            body=redact(body),
            query=redact(query),
        )
        try:
            response = await super()._request(
                method,
                path,
                body=body,
                query=query,
                timeout=timeout,
            )
        except Exception as error:
            emit(
                "api.error",
                number=request_number,
                elapsed_seconds=round(time.monotonic() - started, 3),
                error_type=type(error).__name__,
                error=str(error),
                code=getattr(error, "code", None),
                status=getattr(error, "status", None),
                retry_after=getattr(error, "retry_after", None),
            )
            raise
        emit(
            "api.response",
            number=request_number,
            elapsed_seconds=round(time.monotonic() - started, 3),
            response=redact(response),
        )
        return response

    async def _upload_image_context(
        self,
        context: Any,
        upload: dict[str, Any],
        *,
        private_key: Any,
        deadline: float | None,
    ) -> None:
        emit(
            "dockerfile.upload.started",
            host=upload.get("host"),
            port=upload.get("port"),
            username=upload.get("username"),
            path=upload.get("path"),
            host_public_key=upload.get("host_public_key"),
            expires_at=upload.get("expires_at"),
            archive_path=str(context.archive_path),
            archive_bytes=context.archive_bytes,
            context_hash=context.context_hash,
            recipe_hash=context.recipe_hash,
        )
        started = time.monotonic()
        await super()._upload_image_context(
            context, upload, private_key=private_key, deadline=deadline
        )
        emit(
            "dockerfile.upload.completed",
            elapsed_seconds=round(time.monotonic() - started, 3),
            archive_bytes=context.archive_bytes,
        )


def verify_local_sdk_import() -> None:
    module_path = Path(thunder_sandbox.__file__).resolve()
    if REPOSITORY_ROOT not in module_path.parents:
        raise RuntimeError(
            f"refusing to run against an installed SDK at {module_path}; "
            f"expected the checkout at {REPOSITORY_ROOT}"
        )
    emit(
        "sdk.loaded",
        module_path=str(module_path),
        repository_root=str(REPOSITORY_ROOT),
        reported_version=thunder_sandbox.__version__,
    )


def inspect_dockerfile_fixture(client: Client, fixture: Path) -> str:
    if not fixture.is_dir():
        raise RuntimeError(f"Docker build fixture does not exist: {fixture}")
    context = _create_canonical_build_context(
        fixture,
        client.config.paths.image_build_contexts,
    )
    try:
        with tarfile.open(context.archive_path, "r:") as archive:
            members = sorted(archive.getnames())
        required = {
            ".dockerignore",
            "Dockerfile",
            "artifacts/build-info.json",
            "artifacts/message.txt",
        }
        excluded = {
            "artifacts/ignored.tmp",
            "ignored-directory/ignored.txt",
            "ignored.txt",
        }
        missing = sorted(required.difference(members))
        unexpectedly_included = sorted(excluded.intersection(members))
        if missing or unexpectedly_included:
            raise RuntimeError(
                "canonical Docker context did not honor the fixture contract: "
                f"missing={missing}, unexpectedly_included={unexpectedly_included}"
            )
        emit(
            "dockerfile.context",
            fixture=str(fixture),
            artifact_directory=str(client.config.paths.image_build_contexts),
            archive_bytes=context.archive_bytes,
            archive_members=members,
            context_hash=context.context_hash,
            dockerfile_hash=context.dockerfile_hash,
            build_options_hash=context.build_options_hash,
            recipe_hash=context.recipe_hash,
        )
        return context.recipe_hash
    finally:
        context.close()


def managed_path_hash(image: ResolvedImage) -> str:
    if not SHA256_PATTERN.fullmatch(image.managed_digest):
        raise RuntimeError(f"invalid managed image digest: {image.managed_digest!r}")
    repository, separator, digest = image.managed_reference.partition("@")
    if separator != "@" or digest != image.managed_digest:
        raise RuntimeError(
            "managed image reference is not qualified by its reported digest: "
            f"{image.managed_reference!r}"
        )
    components = repository.split("/")
    if len(components) < 5 or components[-3] != "sandbox":
        raise RuntimeError(
            "managed image is not under <gar>/sandbox/<orgId>/<hash>: "
            f"{image.managed_reference!r}"
        )
    source_hash = components[-1]
    if not HASH_PATTERN.fullmatch(source_hash):
        raise RuntimeError(
            f"managed image path has an invalid source hash: {source_hash!r}"
        )
    return source_hash


async def resolve_case(
    client: VerboseClient,
    name: str,
    image: Image,
    timeout: float,
) -> ResolvedImage:
    emit("case.started", case=name, image=repr(image), timeout_seconds=timeout)
    started = time.monotonic()
    result = await client.resolve_image(image, timeout=timeout)
    path_hash = managed_path_hash(result)
    emit(
        "case.ready",
        case=name,
        elapsed_seconds=round(time.monotonic() - started, 3),
        image_id=result.id,
        managed_reference=result.managed_reference,
        managed_digest=result.managed_digest,
        source_hash=path_hash,
    )
    return result


async def run(args: argparse.Namespace) -> int:
    verify_local_sdk_import()
    failures: list[str] = []
    results: dict[str, ResolvedImage] = {}

    async with VerboseClient() as client:
        emit(
            "configuration",
            api_url=client.config.api_url,
            thunder_home=str(client.config.paths.root),
            image_artifact_directory=str(client.config.paths.image_build_contexts),
            registry_image=args.registry_image,
            dockerfile_fixture=str(args.context),
        )
        recipe_hash = inspect_dockerfile_fixture(client, args.context)
        cases = (
            ("dockerhub_registry", Image.from_registry(args.registry_image)),
            ("dockerfile", Image.from_dockerfile(args.context)),
        )
        for name, image in cases:
            try:
                results[name] = await resolve_case(client, name, image, args.timeout)
            except Exception as error:  # noqa: BLE001
                failures.append(name)
                emit(
                    "case.failed",
                    case=name,
                    error_type=type(error).__name__,
                    error=str(error),
                    code=getattr(error, "code", None),
                    status=getattr(error, "status", None),
                )
                traceback.print_exc()

    registry_result = results.get("dockerhub_registry")
    if registry_result is not None:
        registry_hash = managed_path_hash(registry_result)
        if registry_hash != registry_result.managed_digest.removeprefix("sha256:"):
            failures.append("dockerhub_registry_path")
            emit(
                "validation.failed",
                case="dockerhub_registry",
                expected_hash=registry_result.managed_digest.removeprefix("sha256:"),
                actual_hash=registry_hash,
            )

    dockerfile_result = results.get("dockerfile")
    if dockerfile_result is not None:
        dockerfile_hash = managed_path_hash(dockerfile_result)
        if dockerfile_hash != recipe_hash.removeprefix("sha256:"):
            failures.append("dockerfile_path")
            emit(
                "validation.failed",
                case="dockerfile",
                expected_hash=recipe_hash.removeprefix("sha256:"),
                actual_hash=dockerfile_hash,
            )

    if (
        len(results) == 2
        and results["dockerhub_registry"].id == results["dockerfile"].id
    ):
        failures.append("distinct_image_ids")
        emit(
            "validation.failed",
            error="registry and Dockerfile sources unexpectedly returned the same image ID",
            image_id=results["dockerfile"].id,
        )

    if failures:
        emit("smoke.failed", failures=failures, ready_cases=sorted(results))
        return 1
    emit(
        "smoke.passed",
        cases=sorted(results),
        image_ids={name: image.id for name, image in results.items()},
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-image",
        default=DEFAULT_REGISTRY_IMAGE,
        help=f"public Docker Hub image to import (default: {DEFAULT_REGISTRY_IMAGE})",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=FIXTURE_DIRECTORY,
        help=f"Docker build-context directory (default: {FIXTURE_DIRECTORY})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="timeout in seconds for each image (default: 7200)",
    )
    args = parser.parse_args()
    args.context = args.context.expanduser().resolve()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
