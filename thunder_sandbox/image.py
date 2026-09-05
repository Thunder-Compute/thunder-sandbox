"""Container image definitions and canonical Docker build contexts."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._common.exceptions import InvalidRequestError

MAX_BUILD_CONTEXT_BYTES = 5 * 1024 * 1024 * 1024
_BUILD_OPTIONS = b"thunder-sandbox-build-options-v1\0Dockerfile"
_RECIPE_DOMAIN = b"thunder-sandbox-dockerfile-recipe-v1\0"


@dataclass(frozen=True, repr=False)
class Image:
    """A lazily resolved container image for a sandbox."""

    _source: Literal["registry", "dockerfile"]
    _registry_url: str | None = None
    _registry_username: str | None = None
    _registry_password: str | None = None
    _context_directory: Path | None = None

    @staticmethod
    def from_registry(
        url: str,
        username: str | None = None,
        password: str | None = None,
    ) -> "Image":
        """Create an image definition from a public or private registry image."""
        if not isinstance(url, str) or not url.strip():
            raise InvalidRequestError("registry image URL must be a non-empty string")
        if any(character.isspace() for character in url):
            raise InvalidRequestError("registry image URL cannot contain whitespace")
        if (username is None) != (password is None):
            raise InvalidRequestError(
                "registry username and password must be provided together"
            )
        if username is not None and (not username or not password):
            raise InvalidRequestError("registry username and password cannot be empty")
        return Image(
            _source="registry",
            _registry_url=url.strip(),
            _registry_username=username,
            _registry_password=password,
        )

    @staticmethod
    def from_dockerfile(directory_path: str | Path) -> "Image":
        """Create an image definition from a directory containing a Dockerfile."""
        if not isinstance(directory_path, (str, Path)):
            raise InvalidRequestError("Dockerfile directory must be a path")
        context_directory = Path(directory_path).expanduser().resolve()
        if not context_directory.is_dir():
            raise InvalidRequestError(
                f"Dockerfile directory does not exist: {context_directory}"
            )
        dockerfile = context_directory / "Dockerfile"
        if not dockerfile.is_file():
            raise InvalidRequestError(
                "Dockerfile directory does not contain a Dockerfile: "
                f"{context_directory}"
            )
        return Image(
            _source="dockerfile",
            _context_directory=context_directory,
        )

    def __repr__(self) -> str:
        if self._source == "registry":
            credentials = ", credentials=<redacted>" if self._registry_username else ""
            return f"Image.from_registry({self._registry_url!r}{credentials})"
        return f"Image.from_dockerfile({str(self._context_directory)!r})"


@dataclass(frozen=True)
class ResolvedImage:
    """An immutable Thunder-managed image ready for sandbox creation."""

    id: str
    managed_reference: str
    managed_digest: str


@dataclass(frozen=True)
class _CanonicalBuildContext:
    archive_path: Path
    archive_bytes: int
    context_hash: str
    dockerfile_hash: str
    build_options_hash: str
    recipe_hash: str

    def close(self) -> None:
        self.archive_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _IgnoreRule:
    expression: re.Pattern[str]
    negated: bool


class _DockerIgnore:
    def __init__(self, contents: str) -> None:
        rules: list[_IgnoreRule] = []
        for raw_line in contents.splitlines():
            if raw_line.startswith("#"):
                continue
            pattern = raw_line.strip()
            if not pattern or pattern == ".":
                continue
            negated = pattern.startswith("!")
            if negated:
                pattern = pattern[1:].strip()
                if not pattern:
                    raise InvalidRequestError(".dockerignore contains an empty negation")
            pattern = pattern.strip("/")
            if pattern:
                rules.append(_IgnoreRule(_compile_ignore_pattern(pattern), negated))
        self._rules = rules

    def excludes(self, path: str) -> bool:
        excluded = False
        for rule in self._rules:
            if rule.expression.fullmatch(path):
                excluded = not rule.negated
        return excluded


def _compile_ignore_pattern(pattern: str) -> re.Pattern[str]:
    expression = ""
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 1
                if index + 1 < len(pattern) and pattern[index + 1] == "/":
                    index += 1
                    expression += "(?:.*/)?"
                else:
                    expression += ".*"
            else:
                expression += "[^/]*"
        elif character == "?":
            expression += "[^/]"
        elif character == "[":
            closing = pattern.find("]", index + 1)
            if closing == -1:
                expression += r"\["
            else:
                character_class = pattern[index + 1 : closing]
                if character_class.startswith("!"):
                    character_class = "^" + character_class[1:]
                expression += "[" + character_class.replace("\\", r"\\") + "]"
                index = closing
        elif character == "\\" and index + 1 < len(pattern):
            index += 1
            expression += re.escape(pattern[index])
        else:
            expression += re.escape(character)
        index += 1
    return re.compile(expression + "(?:/.*)?")


def _digest_bytes(contents: bytes) -> str:
    return "sha256:" + hashlib.sha256(contents).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _recipe_hash(
    context_hash: str, dockerfile_hash: str, build_options_hash: str
) -> str:
    payload = (
        _RECIPE_DOMAIN
        + context_hash.encode("ascii")
        + b"\0"
        + dockerfile_hash.encode("ascii")
        + b"\0"
        + build_options_hash.encode("ascii")
    )
    return _digest_bytes(payload)


def _normalized_relative_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    normalized = unicodedata.normalize("NFC", relative.as_posix())
    if not normalized or normalized.startswith("/") or any(
        part in ("", ".", "..") for part in normalized.split("/")
    ):
        raise InvalidRequestError(f"unsafe build context path: {relative}")
    return normalized


def _excluded_context_path(dockerignore: _DockerIgnore, path: str) -> bool:
    return path not in ("Dockerfile", ".dockerignore") and dockerignore.excludes(path)


def _context_files(
    root: Path, dockerignore: _DockerIgnore, artifact_directory: Path
) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    normalized_paths: dict[str, Path] = {}
    try:
        for directory, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            directory_names.sort()
            file_names.sort()
            parent = Path(directory)
            directory_names[:] = [
                name
                for name in directory_names
                if parent / name != artifact_directory
                and not _excluded_context_path(
                    dockerignore, _normalized_relative_path(root, parent / name)
                )
            ]
            for name in [*directory_names, *file_names]:
                path = parent / name
                normalized = _normalized_relative_path(root, path)
                if _excluded_context_path(dockerignore, normalized):
                    continue
                metadata = path.lstat()
                existing = normalized_paths.get(normalized)
                if existing is not None:
                    raise InvalidRequestError(
                        f"build context paths normalize to the same name: "
                        f"{existing.relative_to(root)} and {path.relative_to(root)}"
                    )
                normalized_paths[normalized] = path
                if stat.S_ISLNK(metadata.st_mode):
                    raise InvalidRequestError(
                        f"build context cannot contain symbolic links: {normalized}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise InvalidRequestError(
                        f"build context can contain only regular files: {normalized}"
                    )
                files.append((normalized, path))
    except OSError as error:
        raise InvalidRequestError(f"could not read build context: {error}") from error
    return sorted(files, key=lambda item: item[0])


def _dockerignore(root: Path) -> _DockerIgnore:
    path = root / ".dockerignore"
    if not path.exists():
        return _DockerIgnore("")
    try:
        return _DockerIgnore(path.read_text(encoding="utf-8-sig"))
    except UnicodeError as error:
        raise InvalidRequestError(".dockerignore must be UTF-8") from error


def _create_canonical_build_context(
    root: Path, artifact_directory: Path
) -> _CanonicalBuildContext:
    root = root.resolve()
    try:
        artifact_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not stat.S_ISDIR(artifact_directory.lstat().st_mode):
            raise InvalidRequestError(
                f"image artifact path is not a directory: {artifact_directory}"
            )
        artifact_directory.chmod(0o700)
        artifact_directory = artifact_directory.resolve()
    except InvalidRequestError:
        raise
    except OSError as error:
        raise InvalidRequestError(
            f"could not create image artifacts under {artifact_directory}: {error}"
        ) from error

    files = _context_files(root, _dockerignore(root), artifact_directory)
    if not any(name == "Dockerfile" for name, _ in files):
        raise InvalidRequestError("build context does not contain a regular Dockerfile")

    try:
        descriptor, archive_name = tempfile.mkstemp(
            prefix="thunder-build-context-",
            suffix=".tar",
            dir=artifact_directory,
        )
    except OSError as error:
        raise InvalidRequestError(
            f"could not create image artifacts under {artifact_directory}: {error}"
        ) from error
    os.close(descriptor)
    archive_path = Path(archive_name)
    try:
        with tarfile.open(archive_path, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for name, path in files:
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                try:
                    file_descriptor = os.open(path, flags)
                except OSError as error:
                    raise InvalidRequestError(
                        f"could not read build context file {name}: {error}"
                    ) from error
                with os.fdopen(file_descriptor, "rb") as source:
                    metadata = os.fstat(source.fileno())
                    if not stat.S_ISREG(metadata.st_mode):
                        raise InvalidRequestError(
                            f"build context file changed while being archived: {name}"
                        )
                    entry = tarfile.TarInfo(name)
                    entry.size = metadata.st_size
                    entry.mode = 0o644
                    entry.uid = 0
                    entry.gid = 0
                    entry.uname = ""
                    entry.gname = ""
                    entry.mtime = 0
                    archive.addfile(entry, source)
        archive_bytes = archive_path.stat().st_size
        if archive_bytes > MAX_BUILD_CONTEXT_BYTES:
            raise InvalidRequestError(
                f"canonical build context exceeds the {MAX_BUILD_CONTEXT_BYTES}-byte limit"
            )
        context_hash = _digest_file(archive_path)
        with tarfile.open(archive_path, mode="r:") as archive:
            dockerfile = archive.extractfile("Dockerfile")
            if dockerfile is None:
                raise InvalidRequestError("canonical build context has no Dockerfile")
            dockerfile_hash = _digest_bytes(dockerfile.read())
        build_options_hash = _digest_bytes(_BUILD_OPTIONS)
        return _CanonicalBuildContext(
            archive_path=archive_path,
            archive_bytes=archive_bytes,
            context_hash=context_hash,
            dockerfile_hash=dockerfile_hash,
            build_options_hash=build_options_hash,
            recipe_hash=_recipe_hash(
                context_hash, dockerfile_hash, build_options_hash
            ),
        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise


__all__ = ["Image", "ResolvedImage"]
